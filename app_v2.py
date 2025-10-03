from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import the AI signal detector
from ai_signal_detector import AISignalDetector
import os

app = FastAPI(title="FAERS Signal Detection API", version="2.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")
origins = [
    "http://localhost:8080",
    "http://localhost:4173",
    ALLOWED_ORIGINS
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Allow cookies, authorization headers, etc.
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ------------------------------ Global State ------------------------------


class DataCache:
    def __init__(self) -> None:
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_at: Optional[str] = None
        self.stats: Optional[pd.DataFrame] = None  # drug-event table with PRR, ROR, IC
        self.ai_predictions: Optional[pd.DataFrame] = None  # AI model predictions
        self.ai_detector: Optional[AISignalDetector] = None
        self.status: str = "idle"
        self.column_map: Dict[str, str] = {}


CACHE = DataCache()


# ------------------------------ Column Schema ------------------------------


EXPECTED_COLUMNS = [
    "primaryid",
    "caseid",
    "caseversion",
    "fda_dt",
    "age",
    "sex",
    "rpt_country",
    "rpsr_cod",
    "drug_seq",
    "role_cod",
    "drugname",
    "prod_ai",
    "pt",
    "outc_cod",
    "indi_drug_seq",
    "indi_pt",
    "rpsr_cod",
    "start_dt",
    "end_dt",
    "dur",
    "dur_cod",
]

# Internal canonical names
CANON_DRUG = "drug_name"
CANON_EVENT = "adverse_event"

FAERS_COLUMN_MAP = {
    # FAERS originals -> canonical
    "drugname": CANON_DRUG,
    "prod_ai": "prod_ai",
    "pt": CANON_EVENT,
    "indi_pt": "indication",
    "indi_drug_seq": "indi_drug_seq",
    "age": "age",
    "sex": "sex",
    "outc_cod": "outcome",
    "rpt_country": "country",
    "role_cod": "role_cod",
    "dur_cod": "dur_cod",
    "dur": "dur",
    "fda_dt": "fda_dt",
    "caseid": "case_id",
    "caseversion": "case_version",
    "primaryid": "primary_id",
    "drug_seq": "drug_sequence",
    "start_dt": "start_dt",
    "end_dt": "end_dt",
    # Alternate headers from provided CSV -> canonical
    "drug_name": CANON_DRUG,
    "adverse_event": CANON_EVENT,
    "active_ingredient": "prod_ai",
    "indication_sequence": "indi_drug_seq",
    "fda_date": "fda_dt",
    "start_date": "start_dt",
    "end_date": "end_dt",
}

# Minimum required source columns to proceed (either FAERS or canonical)
REQUIRES_ANY_OF = {
    "drug": ["drugname", "drug_name"],
    "event": ["pt", "adverse_event"],
}


# ------------------------------ Utilities ------------------------------


def _ensure_uploaded() -> None:
    if CACHE.raw_df is None or CACHE.stats is None:
        raise HTTPException(
            status_code=400, detail="No dataset uploaded or processed yet."
        )


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(buffer, low_memory=False)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Validate presence of minimally required columns (accept aliases)
    has_drug = any(col in df.columns for col in REQUIRES_ANY_OF["drug"])
    has_event = any(col in df.columns for col in REQUIRES_ANY_OF["event"])
    if not (has_drug and has_event):
        missing = []
        if not has_drug:
            missing.append(REQUIRES_ANY_OF["drug"])  # type: ignore[arg-type]
        if not has_event:
            missing.append(REQUIRES_ANY_OF["event"])  # type: ignore[arg-type]
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns.",
                "expect_any_of": missing,
            },
        )

    # Map to canonical names
    for source_col, canon in FAERS_COLUMN_MAP.items():
        if source_col in df.columns:
            df[canon] = df[source_col]

    # Types and parsing
    for date_col in ["fda_dt", "start_dt", "end_dt"]:
        if date_col in df.columns:
            # handle yyyymmdd or other string formats gracefully
            parsed = pd.to_datetime(
                df[date_col].astype(str), format="%Y%m%d", errors="coerce"
            )
            # Fallback general parse if all NaT
            if parsed.isna().all():
                parsed = pd.to_datetime(df[date_col], errors="coerce")
            df[date_col] = parsed

    for col in ["age", "dur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure required canonical columns exist
    for col in [CANON_DRUG, CANON_EVENT, "outcome", "sex", "country", "dur_cod"]:
        if col not in df.columns:
            df[col] = "missing"

    return df


def _compute_contingency(df: pd.DataFrame) -> pd.DataFrame:
    # a: reports with drug and event
    contingency = df.groupby([CANON_DRUG, CANON_EVENT]).size().reset_index(name="a")
    total_drug = df.groupby(CANON_DRUG).size().reset_index(name="drug_total")
    total_event = df.groupby(CANON_EVENT).size().reset_index(name="event_total")
    total = len(df)

    out = contingency.merge(total_drug, on=CANON_DRUG)
    out = out.merge(total_event, on=CANON_EVENT)
    out["b"] = out["drug_total"] - out["a"]
    out["c"] = out["event_total"] - out["a"]
    out["d"] = total - (out["a"] + out["b"] + out["c"])  # remaining
    # Guard against negatives due to any inconsistencies
    for col in ["a", "b", "c", "d"]:
        out[col] = out[col].clip(lower=0)
    return out[[CANON_DRUG, CANON_EVENT, "a", "b", "c", "d"]]


def _compute_prr_ror_ic(cont: pd.DataFrame) -> pd.DataFrame:
    df = cont.copy()
    eps = 1e-6
    # PRR
    df["PRR"] = (df["a"] / (df["a"] + df["b"] + eps)) / (
        (df["c"] / (df["c"] + df["d"] + eps)) + eps
    )
    # ROR
    df["ROR"] = (df["a"] / (df["b"] + eps)) / ((df["c"] / (df["d"] + eps)) + eps)
    # IC (observed/expected)
    n = (df["a"] + df["b"] + df["c"] + df["d"]).astype(float)
    expected = (df["a"] + df["b"]) * (df["a"] + df["c"]) / n.clip(lower=eps)
    df["IC"] = np.log2((df["a"].astype(float) + eps) / (expected + eps))

    # 95% CI lower bounds
    # PRR CI on log scale
    se_log_prr = np.sqrt(
        (1.0 / (df["a"] + eps))
        - (1.0 / (df["a"] + df["b"] + eps))
        + (1.0 / (df["c"] + eps))
        - (1.0 / (df["c"] + df["d"] + eps))
    )
    df["PRR_LCL"] = np.exp(np.log(df["PRR"] + eps) - 1.96 * se_log_prr)

    # ROR CI on log scale
    se_log_ror = np.sqrt(
        (1.0 / (df["a"] + eps))
        + (1.0 / (df["b"] + eps))
        + (1.0 / (df["c"] + eps))
        + (1.0 / (df["d"] + eps))
    )
    df["ROR_LCL"] = np.exp(np.log(df["ROR"] + eps) - 1.96 * se_log_ror)

    # Set PRR, ROR, IC to NaN if a < 3 (or your chosen threshold)
    min_count = 3
    mask = df["a"] < min_count
    df.loc[mask, ["PRR", "ROR", "IC", "PRR_LCL", "ROR_LCL"]] = np.nan

    # Flags
    df["meets_signal"] = (
        (df["PRR"] >= 2.0) & (df["a"] >= min_count) & (df["PRR_LCL"] > 1.0)
    ).astype(int)

    return df


def _severity_score(series: pd.Series) -> pd.Series:
    # Map FAERS outcome codes to severity 1-5
    mapping = {
        "DE": 5,  # Death
        "LT": 5,  # Life Threatening
        "HO": 4,  # Hospitalization
        "DS": 4,  # Disability
        "CA": 4,  # Congenital Anomaly
        "RI": 3,  # Required Intervention
        "OT": 1,  # Other
    }
    return series.astype(str).str.upper().map(mapping).fillna(1).astype(int)


def _prepare_cache(df: pd.DataFrame) -> None:
    CACHE.status = "processing"
    df = _normalize_columns(df)
    # add severity
    df["severity"] = _severity_score(
        df.get("outcome", pd.Series(index=df.index, dtype=str))
    )

    # contingency and stats
    cont = _compute_contingency(df)
    stats = _compute_prr_ror_ic(cont)

    # merge severity mean per pair
    sev = (
        df.groupby([CANON_DRUG, CANON_EVENT])["severity"]
        .mean()
        .reset_index(name="mean_severity")
    )
    stats = stats.merge(sev, on=[CANON_DRUG, CANON_EVENT], how="left")

    # Initialize AI detector and run predictions
    try:
        ai_detector = AISignalDetector()
        ai_detector.load_model()

        # Run AI predictions on the dataset
        ai_predictions = ai_detector.predict_many(df)

        # Merge AI predictions with stats
        present = set(ai_predictions.columns)
        agg_spec: Dict[str, str] = {}
        if "is_signal" in present:
            agg_spec["is_signal"] = "max"
        if "probability" in present:
            agg_spec["probability"] = "mean"
        if "PRR" in present:
            agg_spec["PRR"] = "first"
        if "ROR" in present:
            agg_spec["ROR"] = "first"
        if "is_disproportional_signal" in present:
            agg_spec["is_disproportional_signal"] = "max"

        if agg_spec:
            ai_stats = (
                ai_predictions.groupby([CANON_DRUG, CANON_EVENT])
                .agg(agg_spec)
                .reset_index()
            )
            # Merge AI predictions with statistical results
            stats = stats.merge(
                ai_stats, on=[CANON_DRUG, CANON_EVENT], how="left", suffixes=("", "_ai")
            )

            # Update signal detection to include AI predictions (only if columns exist)
            meets = stats["meets_signal"].astype(int)
            if "is_signal" in stats.columns:
                meets = (meets == 1) | (stats["is_signal"].fillna(0).astype(int) == 1)
                meets = meets.astype(int)
            if "is_disproportional_signal" in stats.columns:
                meets = (meets == 1) | (
                    stats["is_disproportional_signal"].fillna(0).astype(int) == 1
                )
                meets = meets.astype(int)
            stats["meets_signal"] = meets

        CACHE.ai_detector = ai_detector
        CACHE.ai_predictions = ai_predictions

    except Exception as e:
        print(f"AI model inference failed: {e}")
        # Continue with statistical analysis only
        CACHE.ai_detector = None
        CACHE.ai_predictions = None

    # cache
    CACHE.raw_df = df
    CACHE.stats = stats
    CACHE.processed_at = datetime.utcnow().isoformat()
    CACHE.status = "ready"


def _jsonify_df(df: pd.DataFrame, orient: str = "records") -> List[Dict[str, Any]]:
    # Ensure JSON serializable (convert Timestamps and numpy types)
    converted = df.copy()
    for col in converted.columns:
        if np.issubdtype(converted[col].dtype, np.datetime64):
            converted[col] = (
                converted[col].astype("datetime64[ns]").dt.strftime("%Y-%m-%d")
            )
    return converted.to_dict(orient=orient)  # type: ignore[return-value]


# ------------------------------ API Endpoints ------------------------------


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await file.read()
    df = _read_csv(content)
    _prepare_cache(df)
    return JSONResponse(
        {
            "message": "Dataset uploaded and processed.",
            "rows": int(len(CACHE.raw_df) if CACHE.raw_df is not None else 0),
            "processed_at": CACHE.processed_at,
        }
    )


@app.get("/total-reports")
def total_reports() -> Dict[str, Any]:
    _ensure_uploaded()
    return {"total_reports": int(len(CACHE.raw_df))}


@app.get("/detected-signals")
def detected_signals() -> Dict[str, Any]:
    _ensure_uploaded()
    count = int(CACHE.stats["meets_signal"].sum())
    return {"detected_signals": count}


@app.get("/critical-signals")
def critical_signals() -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.raw_df
    stats = CACHE.stats
    sev = (
        df.groupby([CANON_DRUG, CANON_EVENT])["severity"]
        .max()
        .reset_index(name="max_severity")
    )
    merged = stats.merge(sev, on=[CANON_DRUG, CANON_EVENT], how="left")
    crit = merged[(merged["meets_signal"] == 1) & (merged["max_severity"] >= 5)]
    return {"critical_signals": int(len(crit))}


@app.get("/high-risk")
def high_risk() -> Dict[str, Any]:
    _ensure_uploaded()
    hr = CACHE.stats[(CACHE.stats["PRR"] >= 5) | (CACHE.stats["ROR"] >= 5)]
    return {"high_risk_pairs": int(len(hr))}


@app.get("/top-drugs-signals")
def top_drugs_signals(limit: int = 10) -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.stats[CACHE.stats["meets_signal"] == 1]
    top = (
        df.groupby(CANON_DRUG)
        .size()
        .reset_index(name="signal_count")
        .sort_values("signal_count", ascending=False)
        .head(limit)
    )
    return {"top_drugs": _jsonify_df(top)}


@app.get("/top-adverse-events")
def top_adverse_events(limit: int = 10) -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.stats[CACHE.stats["meets_signal"] == 1]
    top = (
        df.groupby(CANON_EVENT)
        .size()
        .reset_index(name="signal_count")
        .sort_values("signal_count", ascending=False)
        .head(limit)
    )
    return {"top_adverse_events": _jsonify_df(top)}


@app.get("/quick-actions")
def quick_actions() -> Dict[str, Any]:
    _ensure_uploaded()
    total = int(len(CACHE.raw_df))
    signals = int(CACHE.stats["meets_signal"].sum())
    high_risk = int(((CACHE.stats["PRR"] >= 5) | (CACHE.stats["ROR"] >= 5)).sum())
    actions: List[str] = []
    if high_risk > 0:
        actions.append(
            "Initiate immediate safety review for high-risk pairs (PRR/ROR ≥ 5)."
        )
    if signals > 50:
        actions.append("Prioritize top 20 signal pairs for medical assessment.")
    if total > 10000 and signals / max(total, 1) > 0.002:
        actions.append(
            "Increase surveillance frequency; elevated signal rate for large cohort."
        )
    if not actions:
        actions.append("No urgent actions; continue routine monitoring.")
    return {"suggestions": actions}


@app.get("/system-status")
def system_status() -> Dict[str, Any]:
    ready = CACHE.raw_df is not None and CACHE.stats is not None
    ai_ready = CACHE.ai_detector is not None and CACHE.ai_predictions is not None

    return {
        "status": CACHE.status,
        "ready": ready,
        "ai_model_ready": ai_ready,
        "processed_at": CACHE.processed_at,
        "total_reports": int(len(CACHE.raw_df)) if CACHE.raw_df is not None else 0,
        "signal_pairs": (
            int(CACHE.stats["meets_signal"].sum()) if CACHE.stats is not None else 0
        ),
        "ai_signal_pairs": (
            int(CACHE.ai_predictions["is_signal"].sum())
            if CACHE.ai_predictions is not None
            and "is_signal" in CACHE.ai_predictions.columns
            else 0
        ),
    }


@app.get("/signal-rate")
def signal_rate() -> Dict[str, Any]:
    _ensure_uploaded()
    rate = float(CACHE.stats["meets_signal"].mean()) if len(CACHE.stats) else 0.0
    return {"signal_rate": rate}


@app.get("/signal-criteria")
def signal_criteria() -> Dict[str, Any]:
    return {
        "criteria": {
            "PRR_threshold": 2.0,
            "minimum_frequency_a": 3,
            "lower_CI_PRR_gt": 1.0,
        }
    }


@app.get("/risk-assessment")
def risk_assessment(limit: int = 50) -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.stats.copy()
    df["risk_score"] = (
        (df["PRR"].fillna(0)).clip(lower=0) * 0.5
        + (df["ROR"].fillna(0)).clip(lower=0) * 0.5
        + (df["mean_severity"].fillna(0))
    )
    ranked = df.sort_values(
        ["meets_signal", "risk_score"], ascending=[False, False]
    ).head(limit)
    cols = [
        CANON_DRUG,
        CANON_EVENT,
        "a",
        "PRR",
        "ROR",
        "IC",
        "PRR_LCL",
        "ROR_LCL",
        "mean_severity",
        "meets_signal",
        "risk_score",
    ]
    return {"risk_ranking": _jsonify_df(ranked[cols])}


@app.get("/statistical-measures")
def statistical_measures(limit: int = 100) -> Dict[str, Any]:
    _ensure_uploaded()
    cols = [
        CANON_DRUG,
        CANON_EVENT,
        "a",
        "b",
        "c",
        "d",
        "PRR",
        "ROR",
        "IC",
        "PRR_LCL",
        "ROR_LCL",
        "meets_signal",
    ]
    return {"measures": _jsonify_df(CACHE.stats[cols].head(limit))}


@app.get("/average-severity")
def average_severity() -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.stats[CACHE.stats["meets_signal"] == 1]
    avg = float(df["mean_severity"].mean()) if len(df) else 0.0
    return {"average_severity": avg}


@app.get("/total-signal-pairs")
def total_signal_pairs() -> Dict[str, Any]:
    _ensure_uploaded()
    return {"total_signal_pairs": int(CACHE.stats["meets_signal"].sum())}


@app.get("/drug-event-signal-pairs")
def drug_event_signal_pairs(limit: Optional[int] = None) -> Dict[str, Any]:
    _ensure_uploaded()
    cols = [CANON_DRUG, CANON_EVENT, "a", "PRR", "ROR", "IC", "mean_severity"]
    # Only include pairs with a >= 3
    df = CACHE.stats[(CACHE.stats["meets_signal"] == 1) & (CACHE.stats["a"] >= 3)][cols].sort_values("PRR", ascending=False)
    if limit is None:
        limit = len(df)
    df = df.head(limit)
    return {"signal_pairs": _jsonify_df(df)}


# ------------------------------ AI Model Endpoints ------------------------------


@app.get("/ai-signals")
def ai_signals(limit: int = 100) -> Dict[str, Any]:
    """Get signals detected by AI model only"""
    _ensure_uploaded()
    if CACHE.ai_predictions is None:
        return {"ai_signals": [], "message": "AI model not available"}

    df_ai = CACHE.ai_predictions
    if "is_signal" not in df_ai.columns:
        return {"ai_signals": [], "message": "AI signals not provided by model"}

    ai_signals_df = df_ai[df_ai["is_signal"] == 1]
    cols = [CANON_DRUG, CANON_EVENT]
    for c in ["probability", "PRR", "ROR", "is_disproportional_signal"]:
        if c in ai_signals_df.columns:
            cols.append(c)
    result = ai_signals_df[cols]
    sort_key = "probability" if "probability" in result.columns else None
    if sort_key:
        result = result.sort_values(sort_key, ascending=False)
    result = result.head(limit)
    return {"ai_signals": _jsonify_df(result)}


@app.get("/ai-signal-count")
def ai_signal_count() -> Dict[str, Any]:
    """Count of signals detected by AI model"""
    _ensure_uploaded()
    if CACHE.ai_predictions is None or "is_signal" not in CACHE.ai_predictions.columns:
        return {"ai_signal_count": 0, "message": "AI signals not available"}

    count = int(CACHE.ai_predictions["is_signal"].sum())
    return {"ai_signal_count": count}


@app.get("/ai-high-confidence-signals")
def ai_high_confidence_signals(
    threshold: float = 0.8, limit: int = 50
) -> Dict[str, Any]:
    """Get AI signals with high confidence (probability >= threshold)"""
    _ensure_uploaded()
    if CACHE.ai_predictions is None:
        return {"high_confidence_signals": [], "message": "AI model not available"}

    if "is_signal" not in CACHE.ai_predictions.columns:
        return {"high_confidence_signals": [], "message": "AI signals not available"}

    if "probability" not in CACHE.ai_predictions.columns:
        return {"high_confidence_signals": [], "message": "Probability not available"}

    high_conf = CACHE.ai_predictions[
        (CACHE.ai_predictions["is_signal"] == 1)
        & (CACHE.ai_predictions["probability"] >= threshold)
    ]
    cols = [CANON_DRUG, CANON_EVENT, "probability"]
    for c in ["PRR", "ROR"]:
        if c in high_conf.columns:
            cols.append(c)
    result = high_conf[cols].sort_values("probability", ascending=False).head(limit)
    return {"high_confidence_signals": _jsonify_df(result)}


@app.get("/ai-model-status")
def ai_model_status() -> Dict[str, Any]:
    """Check if AI model is loaded and available"""
    _ensure_uploaded()
    return {
        "ai_model_available": CACHE.ai_detector is not None,
        "ai_predictions_available": CACHE.ai_predictions is not None,
        "total_ai_predictions": (
            len(CACHE.ai_predictions) if CACHE.ai_predictions is not None else 0
        ),
    }


@app.get("/hybrid-signals")
def hybrid_signals(limit: int = 100) -> Dict[str, Any]:
    """Get signals detected by both statistical and AI methods"""
    _ensure_uploaded()
    if CACHE.ai_predictions is None:
        return {"hybrid_signals": [], "message": "AI model not available"}

    if "is_signal" not in CACHE.ai_predictions.columns:
        return {"hybrid_signals": [], "message": "AI signals not available"}

    # Find pairs that are signals by both methods
    ai_pairs = set(
        zip(
            CACHE.ai_predictions[CACHE.ai_predictions["is_signal"] == 1][CANON_DRUG],
            CACHE.ai_predictions[CACHE.ai_predictions["is_signal"] == 1][CANON_EVENT],
        )
    )

    stat_pairs = set(
        zip(
            CACHE.stats[CACHE.stats["meets_signal"] == 1][CANON_DRUG],
            CACHE.stats[CACHE.stats["meets_signal"] == 1][CANON_EVENT],
        )
    )

    hybrid_pairs = ai_pairs.intersection(stat_pairs)

    if not hybrid_pairs:
        return {"hybrid_signals": [], "message": "No signals detected by both methods"}

    # Get details for hybrid signals
    hybrid_df = CACHE.stats[
        CACHE.stats[[CANON_DRUG, CANON_EVENT]].apply(
            lambda x: (x[CANON_DRUG], x[CANON_EVENT]) in hybrid_pairs, axis=1
        )
    ]

    cols = [
        CANON_DRUG,
        CANON_EVENT,
        "a",
        "PRR",
        "ROR",
        "IC",
        "mean_severity",
        "meets_signal",
    ]
    result = hybrid_df[cols].sort_values("PRR", ascending=False).head(limit)
    return {"hybrid_signals": _jsonify_df(result)}


@app.get("/ai-prediction-distribution")
def ai_prediction_distribution() -> Dict[str, Any]:
    """Distribution of AI prediction probabilities"""
    _ensure_uploaded()
    if CACHE.ai_predictions is None:
        return {"distribution": [], "message": "AI model not available"}

    if "probability" not in CACHE.ai_predictions.columns:
        return {"distribution": [], "message": "Probability not available"}

    # Create probability bins
    probs = CACHE.ai_predictions["probability"]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, edges = np.histogram(probs, bins=bins)

    return {
        "distribution": {
            "bins": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)],
            "counts": [int(x) for x in hist],
        }
    }


@app.post("/predict-single")
async def predict_single(
    drug_name: str,
    adverse_event: str,
    age: int = 50,
    sex: str = "F",
    country: str = "US",
    outcome: str = "OT",
    indication: str = "Unknown",
) -> Dict[str, Any]:
    """Predict signal for a single drug-event pair using AI model"""
    if CACHE.ai_detector is None:
        raise HTTPException(
            status_code=400, detail="AI model not available. Upload dataset first."
        )

    # Create single record DataFrame
    single_record = pd.DataFrame(
        [
            {
                "drug_name": drug_name,
                "adverse_event": adverse_event,
                "age": age,
                "sex": sex,
                "country": country,
                "outcome": outcome,
                "indication": indication,
                "fda_dt": datetime.now().strftime("%Y%m%d"),
                "dur": 30,
                "dur_cod": "DY",
            }
        ]
    )

    try:
        # Use AI detector for prediction
        result = CACHE.ai_detector.predict_many(single_record)
        prediction = result.iloc[0]

        return {
            "drug_name": drug_name,
            "adverse_event": adverse_event,
            "is_signal": bool(prediction.get("is_signal", False)),
            "probability": float(prediction.get("probability", 0.0)),
            "PRR": float(prediction.get("PRR", 1.0)),
            "ROR": float(prediction.get("ROR", 1.0)),
            "is_disproportional_signal": bool(
                prediction.get("is_disproportional_signal", False)
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predicted-dataset")
def predicted_dataset(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Returns the uploaded dataset with AI predictions (important columns only).
    """
    _ensure_uploaded()
    if CACHE.ai_detector is None or CACHE.raw_df is None:
        raise HTTPException(status_code=400, detail="AI model or dataset not available.")

    # Run predict_many if not already cached
    if CACHE.ai_predictions is None:
        CACHE.ai_predictions = CACHE.ai_detector.predict_many(CACHE.raw_df)

    df = CACHE.ai_predictions.copy()
    if limit is None:
        limit = len(df)
    important_cols = [
        CANON_DRUG,
        CANON_EVENT,
        "age",
        "sex",
        "country",
        "outcome",
        "indication",
        "dur",
        "dur_cod",
        "fda_dt",
        "is_signal",
        "probability",
        "PRR",
        "ROR",
        "is_disproportional_signal",
    ]
    # Only keep columns that exist
    cols = [c for c in important_cols if c in df.columns]
    result = df[cols].head(limit)
    return {"predicted_dataset": _jsonify_df(result)}


# ------------------------------ Visualization Data ------------------------------


@app.get("/distribution-analysis")
def distribution_analysis() -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.raw_df
    # Age histogram (0-100 by decade)
    ages = df["age"].clip(lower=0, upper=100)
    bins = list(range(0, 110, 10))
    hist, edges = np.histogram(ages, bins=bins)
    sex_counts = df["sex"].astype(str).str.upper().value_counts().to_dict()
    country_top = df["country"].astype(str).value_counts().head(10).to_dict()
    return {
        "age_histogram": {
            "bins": list(map(int, edges[:-1])),
            "counts": list(map(int, hist)),
        },
        "sex_distribution": sex_counts,
        "top_countries": country_top,
    }


@app.get("/temporal-trends")
def temporal_trends() -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.raw_df.copy()
    if "fda_dt" not in df.columns:
        return {"trends": []}
    df["fda_month"] = df["fda_dt"].dt.to_period("M").dt.to_timestamp()
    trends = df.groupby("fda_month").size().reset_index(name="count")
    return {"trends": _jsonify_df(trends.rename(columns={"fda_month": "month"}))}


@app.get("/risk-heatmaps")
def risk_heatmaps(limit_drugs: int = 20, limit_events: int = 20) -> Dict[str, Any]:
    _ensure_uploaded()
    stats = CACHE.stats
    # pick top drugs and events by signal frequency
    top_drugs = (
        stats.groupby(CANON_DRUG)["meets_signal"]
        .sum()
        .sort_values(ascending=False)
        .head(limit_drugs)
        .index
    )
    top_events = (
        stats.groupby(CANON_EVENT)["meets_signal"]
        .sum()
        .sort_values(ascending=False)
        .head(limit_events)
        .index
    )
    sub = stats[stats[CANON_DRUG].isin(top_drugs) & stats[CANON_EVENT].isin(top_events)]
    pivot = sub.pivot_table(
        index=CANON_DRUG, columns=CANON_EVENT, values="PRR", fill_value=0
    )
    data = {
        "rows": list(pivot.index),
        "cols": list(pivot.columns),
        "values": [[float(x) for x in row] for row in pivot.values],
    }
    return {"prr_heatmap": data}


@app.get("/top-drugs-detected")
def top_drugs_detected(limit: int = 10) -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.stats[CACHE.stats["meets_signal"] == 1]
    top = (
        df.groupby(CANON_DRUG)
        .size()
        .reset_index(name="signals")
        .sort_values("signals", ascending=False)
        .head(limit)
    )
    return {"top_drugs_detected": _jsonify_df(top)}


@app.get("/most-frequent-adverse-events")
def most_frequent_adverse_events(limit: int = 10) -> Dict[str, Any]:
    _ensure_uploaded()
    top = (
        CACHE.raw_df.groupby(CANON_EVENT)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(limit)
    )
    return {"most_frequent_adverse_events": _jsonify_df(top)}


@app.get("/signal-trends")
def signal_trends() -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.raw_df.copy()
    if "fda_dt" not in df.columns:
        return {"signal_trends": []}
    df["fda_month"] = df["fda_dt"].dt.to_period("M").dt.to_timestamp()
    # mark if record belongs to any signaled pair
    pairs = set(
        zip(
            CACHE.stats.loc[CACHE.stats["meets_signal"] == 1, CANON_DRUG],
            CACHE.stats.loc[CACHE.stats["meets_signal"] == 1, CANON_EVENT],
        )
    )
    df["is_signal_record"] = df[[CANON_DRUG, CANON_EVENT]].apply(
        lambda r: (r[CANON_DRUG], r[CANON_EVENT]) in pairs, axis=1
    )
    trend = (
        df.groupby("fda_month")["is_signal_record"]
        .mean()
        .reset_index(name="signal_rate")
    )
    return {"signal_trends": _jsonify_df(trend.rename(columns={"fda_month": "month"}))}


@app.get("/drug-event-severity-heatmap")
def drug_event_severity_heatmap(
    limit_drugs: int = 50, limit_events: int = 50
) -> Dict[str, Any]:
    _ensure_uploaded()
    stats = CACHE.stats
    top_drugs = (
        stats.groupby(CANON_DRUG)["meets_signal"]
        .sum()
        .sort_values(ascending=False)
        .head(limit_drugs)
        .index
    )
    top_events = (
        stats.groupby(CANON_EVENT)["meets_signal"]
        .sum()
        .sort_values(ascending=False)
        .head(limit_events)
        .index
    )
    sub = stats[stats[CANON_DRUG].isin(top_drugs) & stats[CANON_EVENT].isin(top_events)]
    pivot = sub.pivot_table(
        index=CANON_DRUG, columns=CANON_EVENT, values="mean_severity", fill_value=0
    )
    data = {
        "rows": list(pivot.index),
        "cols": list(pivot.columns),
        "values": [[float(x) for x in row] for row in pivot.values],
    }
    return {"severity_heatmap": data}


@app.get("/key-insights")
def key_insights() -> Dict[str, Any]:
    _ensure_uploaded()
    total = int(len(CACHE.raw_df))
    pairs = int(len(CACHE.stats))
    signals = int(CACHE.stats["meets_signal"].sum())
    high_risk = int(((CACHE.stats["PRR"] >= 5) | (CACHE.stats["ROR"] >= 5)).sum())
    avg_sev = float(
        CACHE.stats.loc[CACHE.stats["meets_signal"] == 1, "mean_severity"].mean() or 0.0
    )
    return {
        "insights": {
            "total_reports": total,
            "total_pairs": pairs,
            "signals": signals,
            "high_risk_pairs": high_risk,
            "avg_severity_signals": avg_sev,
        }
    }


@app.get("/temporal-patterns")
def temporal_patterns() -> Dict[str, Any]:
    _ensure_uploaded()
    df = CACHE.raw_df.copy()
    if "fda_dt" not in df.columns:
        return {"temporal_patterns": []}
    df["fda_week"] = df["fda_dt"].dt.to_period("W").dt.start_time
    patt = df.groupby(["fda_week", "sex"]).size().reset_index(name="count")
    return {"temporal_patterns": _jsonify_df(patt.rename(columns={"fda_week": "week"}))}


@app.get("/signal-distribution")
def signal_distribution() -> Dict[str, Any]:
    _ensure_uploaded()
    stats = CACHE.stats
    by_drug = (
        stats.groupby(CANON_DRUG)["meets_signal"]
        .sum()
        .reset_index(name="signals")
        .sort_values("signals", ascending=False)
    )
    by_event = (
        stats.groupby(CANON_EVENT)["meets_signal"]
        .sum()
        .reset_index(name="signals")
        .sort_values("signals", ascending=False)
    )
    return {
        "by_drug": _jsonify_df(by_drug.head(50)),
        "by_event": _jsonify_df(by_event.head(50)),
    }


# ------------------------------ Run (optional) ------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_v2:app", host="0.0.0.0", port=8000, reload=False)
