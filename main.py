from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parent / "data"

print(f"Data path: {DATA_DIR}")

DATA_PATH = DATA_DIR / "synthetic_reports.csv"

print(f"Data path: {DATA_PATH}")

# --- App setup ---
app = FastAPI(title="Signal Detector API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ensure data file exists ---
def ensure_data_file():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        df = pd.DataFrame(
            {
                "id": [1],
                "sex": ["M"],
                "drug_name": ["Aspirin"],
                "indication": ["Headache"],
                "age": [34],
                "adverse_event": ["Rash"],
                "outcome": ["Recovered"],
                "date_reported": [pd.Timestamp.today().strftime("%Y-%m-%d")],
            }
        )
        df.to_csv(DATA_PATH, index=False)

ensure_data_file()

def load_df():
    return pd.read_csv(DATA_PATH, parse_dates=["date_reported"])


# --- API endpoints ---

@app.get("/api/kpis")
def get_kpis():
    df = load_df()
    total_reports = int(len(df))

    flagged_events = int(
        (
            (df["adverse_event"].str.lower().isin(["seizure", "liver toxicity"]))
            | (df["outcome"].str.lower().isin(["fatal", "hospitalized"]))
        ).sum()
    )

    # Compute top risk drugs by weighted severity
    severity_weight = (
        (df["outcome"].str.lower().map({"fatal": 3, "hospitalized": 2}).fillna(0))
        + df["adverse_event"].str.lower().map({"seizure": 2, "liver toxicity": 3}).fillna(0)
    )
    df_scores = df.assign(score=severity_weight)
    drug_scores = (
        df_scores.groupby("drug_name")["score"].mean().sort_values(ascending=False).head(5).reset_index()
    )

    # Estimate new signals by slope over the last 6 months and count positive slopes
    signals = compute_signal_trends(df)
    new_signals = len(signals["top_new_signals"]) if signals else 0

    return {
        "total_reports": total_reports,
        "flagged_events": flagged_events,
        "new_signals_detected": int(new_signals),
        "top_risk_drugs": drug_scores.to_dict(orient="records"),
    }


@app.get("/api/events")
def get_events(
    drug: str | None = None,
    event: str | None = None,
    sex: str | None = None,
    min_age: int | None = Query(None, ge=0, le=120),
    max_age: int | None = Query(None, ge=0, le=120),
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(100, ge=1, le=2000),
):
    df = load_df()
    
    # Apply filters
    if drug:
        df = df[df["drug_name"].str.contains(drug, case=False, na=False)]
    if event:
        df = df[df["adverse_event"].str.contains(event, case=False, na=False)]
    if sex:
        df = df[df["sex"].str.upper() == sex.upper()]
    if min_age is not None:
        df = df[df["age"] >= min_age]
    if max_age is not None:
        df = df[df["age"] <= max_age]
    if start_date:
        try:
            sd = pd.to_datetime(start_date)
            df = df[df["date_reported"] >= sd]
        except Exception:
            pass
    if end_date:
        try:
            ed = pd.to_datetime(end_date)
            df = df[df["date_reported"] <= ed]
        except Exception:
            pass

    # Compute a simple signal score per record: severity + rarity + recency
    severity = (
        df["outcome"].str.lower().map({"fatal": 3, "hospitalized": 2, "not recovered": 1}).fillna(0)
        + df["adverse_event"].str.lower().map({"seizure": 2, "liver toxicity": 3, "anaphylaxis": 3}).fillna(0)
    )
    # Rarity: inverse frequency of event per drug
    pair_counts = df.groupby(["drug_name", "adverse_event"]).size().rename("pair_count")
    df = df.join(pair_counts, on=["drug_name", "adverse_event"])
    rarity = 1.0 / (1.0 + df["pair_count"].astype(float))
    # Recency: scale 0..1 over last 365 days
    max_date = df["date_reported"].max()
    recency = 1.0 - ((max_date - df["date_reported"]).dt.days.clip(lower=0, upper=365) / 365.0)
    signal_score = (severity * 2.0) + (rarity * 3.0) + (recency * 2.0)
    df = df.assign(signal_score=signal_score)

    cols = [
        "patient_id",
        "sex",
        "age",
        "drug_name",
        "indication",
        "adverse_event",
        "outcome",
        "date_reported",
        "signal_score",
    ]
    df = df[cols].sort_values("signal_score", ascending=False).head(limit)
    items = df.assign(date_reported=df["date_reported"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    return {"items": items}

@app.get("/api/heatmap")
def get_heatmap():
    df = load_df()
    grp = df.groupby(["drug_name", "adverse_event"]).size().reset_index(name="count")
    max_count = int(grp["count"].max()) if len(grp) else 1
    grp["intensity"] = grp["count"].astype(float) / float(max_count)
    return {"cells": grp.to_dict(orient="records")}

def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def compute_signal_trends(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"series": {}, "top_new_signals": []}
    # Build monthly counts for last 12 months per (drug, event)
    max_date = df["date_reported"].max()
    last_12 = pd.date_range(month_floor(max_date) - pd.DateOffset(months=11), month_floor(max_date), freq="MS")
    df_m = df.copy()
    df_m["month"] = df_m["date_reported"].dt.to_period("M").dt.to_timestamp()
    counts = (
        df_m.groupby(["drug_name", "adverse_event", "month"]).size().rename("count").reset_index()
    )
    # Ensure all months present per pair
    all_idx = pd.MultiIndex.from_product(
        [counts["drug_name"].unique(), counts["adverse_event"].unique(), last_12],
        names=["drug_name", "adverse_event", "month"],
    )
    counts = counts.set_index(["drug_name", "adverse_event", "month"]).reindex(all_idx, fill_value=0).reset_index()

    # Compute slope via linear fit over months index 0..11
    month_to_x = {m: i for i, m in enumerate(last_12)}
    counts["x"] = counts["month"].map(month_to_x)
    slopes = (
        counts.groupby(["drug_name", "adverse_event"]).apply(
            lambda g: float(np.polyfit(g["x"].values, g["count"].values, 1)[0]) if len(g) >= 2 else 0.0
        ).rename("slope").reset_index()
    )
    # Pick top positive slopes
    top = slopes.sort_values("slope", ascending=False).head(5)
    top_pairs = set(zip(top["drug_name"], top["adverse_event"]))

    series = {}
    for (drug_name, adverse_event), grp in counts.groupby(["drug_name", "adverse_event"]):
        key = f"{drug_name}__{adverse_event}"
        series[key] = {
            "label": f"{drug_name} - {adverse_event}",
            "points": [
                {"month": pd.Timestamp(m).strftime("%Y-%m"), "count": int(c)} for m, c in zip(grp["month"].values, grp["count"].values)
            ],
        }

    top_new_signals = (
        slopes.sort_values("slope", ascending=False)
        .head(5)
        .assign(drug_name=lambda d: d["drug_name"], adverse_event=lambda d: d["adverse_event"], slope=lambda d: d["slope"].astype(float))
        [["drug_name", "adverse_event", "slope"]]
        .to_dict(orient="records")
    )
    return {"series": series, "top_new_signals": top_new_signals}

@app.get("/api/signals")
def get_signals():
    df = load_df()
    return compute_signal_trends(df)

@app.get("/api/nlp")
def get_nlp_summary(limit: int = Query(20, ge=1, le=200)):
    df = load_df()
    
    def extract_spans(text, terms):
        """Extract spans for highlighting key terms in text"""
        spans = []
        text_lower = text.lower()
        for term in terms:
            term_lower = term.lower()
            start = 0
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                spans.append({
                    "start": pos,
                    "end": pos + len(term),
                    "term": term
                })
                start = pos + 1
        return sorted(spans, key=lambda x: x["start"])
    
    def extract_key_terms(row):
        """Extract key terms from the row data"""
        terms = []
        for field in ["adverse_event", "drug_name", "indication"]:
            value = str(row.get(field, ""))
            if value and value != "nan":
                terms.append(value)
        return terms

    severity_map = {"fatal": 1.0, "hospitalized": 0.7, "not recovered": 0.5, "recovered": 0.2}
    out = []
    
    for _, r in df.head(limit).iterrows():
        # Create a narrative note from available data
        note_parts = []
        if r.get("adverse_event"):
            note_parts.append(f"Adverse event: {r['adverse_event']}")
        if r.get("drug_name"):
            note_parts.append(f"Drug: {r['drug_name']}")
        if r.get("indication"):
            note_parts.append(f"Indication: {r['indication']}")
        if r.get("outcome"):
            note_parts.append(f"Outcome: {r['outcome']}")
        
        note = ". ".join(note_parts) + "."
        key_terms = extract_key_terms(r)
        spans = extract_spans(note, key_terms)
        
        out.append(
            {
                "patient_id": r.get("patient_id"),
                "date_reported": pd.Timestamp(r.get("date_reported")).strftime("%Y-%m-%d"),
                "severity_score": float(severity_map.get(str(r.get("outcome", "")).lower(), 0.3)),
                "note": note,
                "spans": spans,
            }
        )
    return {"insights": out}

@app.get("/api/event_stats")
def get_event_stats(event: str = Query(..., description="Event term to analyze")):
    """Get detailed statistics for a specific event term"""
    df = load_df()
    
    # Filter data for the specific event
    event_data = df[df["adverse_event"].str.contains(event, case=False, na=False)]
    
    if event_data.empty:
        return {
            "event": event,
            "total": 0,
            "recent_30d": 0,
            "by_drug": [],
            "cases": []
        }
    
    # Calculate total cases
    total = len(event_data)
    
    # Calculate recent cases (last 30 days)
    if not event_data.empty:
        max_date = event_data["date_reported"].max()
        cutoff = max_date - pd.Timedelta(days=30)
        recent_30d = len(event_data[event_data["date_reported"] >= cutoff])
    else:
        recent_30d = 0
    
    # Group by drug
    by_drug = event_data.groupby("drug_name").size().reset_index(name="count")
    by_drug = by_drug.sort_values("count", ascending=False).head(10)
    by_drug_list = by_drug.to_dict(orient="records")
    
    # Get recent cases
    recent_cases = event_data.sort_values("date_reported", ascending=False).head(10)
    cases_list = recent_cases[["patient_id", "drug_name", "date_reported"]].to_dict(orient="records")
    
    return {
        "event": event,
        "total": total,
        "recent_30d": recent_30d,
        "by_drug": by_drug_list,
        "cases": cases_list
    }

@app.get("/api/alerts")
def get_alerts():
    df = load_df()
    sig = compute_signal_trends(df)
    alerts = []
    # Alert on strong slopes
    for s in sig.get("top_new_signals", [])[:3]:
        sev = "high" if s["slope"] >= 0.5 else "medium" if s["slope"] >= 0.2 else "low"
        alerts.append({
            "message": f"Emerging signal: {s['drug_name']} — {s['adverse_event']} (slope {s['slope']:.2f})",
            "severity": sev,
        })

    # Alert on volume/severity in last 30 days
    if not df.empty:
        cutoff = df["date_reported"].max() - pd.Timedelta(days=30)
        recent = df[df["date_reported"] >= cutoff]
        severe = recent[recent["outcome"].str.lower().isin(["fatal", "hospitalized"])]
        if len(severe) >= 10:
            alerts.append({
                "message": f"High severity volume: {len(severe)} serious outcomes in 30 days",
                "severity": "high",
            })

    return {"alerts": alerts}


@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process a CSV file"""
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Parse CSV content
        import io
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['drug_name', 'adverse_event', 'outcome', 'date_reported']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing required columns: {missing_columns}"}
        
        # Add patient_id if not present
        if 'patient_id' not in df.columns:
            df['patient_id'] = range(1, len(df) + 1)
        
        # Add missing columns with default values if not present
        if 'sex' not in df.columns:
            df['sex'] = 'Unknown'
        if 'age' not in df.columns:
            df['age'] = 0
        if 'indication' not in df.columns:
            df['indication'] = 'Unknown'
        
        # Ensure date_reported is in datetime format
        df['date_reported'] = pd.to_datetime(df['date_reported'], errors='coerce')
        
        # Save to the data file (append or replace based on your preference)
        # For now, we'll replace the existing data
        df.to_csv(DATA_PATH, index=False)
        
        return {
            "message": "File uploaded successfully",
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}
    
if __name__ == "__main__":
    app.run(debug=False)