import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path
import joblib
from datetime import datetime

# ---------------- CONFIG ----------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "faers_sampled_dataset.csv"
MODEL_PATH = DATA_DIR / "ai_model.pkl"
INFERENCE_BATCH_PATH = "faers_sampled_dataset_20k_set2.csv"

# Enhanced FAERS column mapping with additional mappings
FAERS_COLUMN_MAP = {
    "drugname": "drug_name",
    "prod_ai": "prod_ai",
    "pt": "adverse_event",
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
    # Additional mappings from normalize_columns_mappings
    "caseid": "case_id",
    "caseversion": "case_version",
    "primaryid": "primary_id",
    "drug_seq": "drug_sequence",
    "start_dt": "start_date",
    "end_dt": "end_date",
}

INTERNAL_CAT_COLS = [
    "sex",
    "drug_name",
    "indication",
    "adverse_event",
    "outcome",
    "country",
    "role_cod",
    "dur_cod",
]
INTERNAL_NUM_COLS = ["age", "dur", "exposure_days", "report_recency_days"]

CURRENT_DATE = datetime(2025, 9, 24)


class AISignalDetector:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cat_features = INTERNAL_CAT_COLS.copy()
        self.num_features = INTERNAL_NUM_COLS + [
            "PRR",
            "ROR",
            "is_disproportional_signal",
            "is_rare_pair",
        ]
        self.prr_ror_table = None
        # Configuration flags
        self.enable_synthetic_injection = False
        self.prediction_threshold = 0.6

    def load_data(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False, index_col=None)
        df = self.normalize_columns(df)
        df = df.reset_index(drop=True)
        print(f"Loaded data shape: {df.shape}, Index is unique: {df.index.is_unique}")
        print(f"Loaded data columns: {df.columns.tolist()}")
        return df

    def save_model(self):
        if self.model:
            joblib.dump(
                {
                    "model": self.model,
                    "label_encoders": self.label_encoders,
                    "scaler": self.scaler,
                    "prr_ror_table": self.prr_ror_table,
                },
                self.model_path,
            )

    def load_model(self):
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data.get("model")
            self.label_encoders = data.get("label_encoders", {})
            self.scaler = data.get("scaler", StandardScaler())
            self.prr_ror_table = data.get("prr_ror_table")
        else:
            print("Model file not found, will retrain on first predict.")
            self.model = None
            self.prr_ror_table = None

    def normalize_columns(self, df):
        """
        Enhanced normalize_columns function that consolidates the functionality
        from both the original normalize_columns and normalize_columns_mappings.

        Normalizes FAERS-style input DataFrame to a canonical schema
        with consistent column names and datatypes.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Apply FAERS column mapping
        for faers_col, internal_col in FAERS_COLUMN_MAP.items():
            if faers_col.lower() in df.columns:
                df[internal_col] = df[faers_col.lower()]

        # Handle date fields with enhanced parsing
        date_fields = ["fda_dt", "start_date", "end_date"]
        for col in date_fields:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col].astype(str), format="%Y%m%d", errors="coerce"
                )

        # Ensure required categorical columns exist
        for col in INTERNAL_CAT_COLS:
            if col not in df.columns:
                df[col] = "missing"

        # Ensure required numerical columns exist
        for col in INTERNAL_NUM_COLS:
            if col not in df.columns:
                df[col] = 0

        # Handle duration NaNs
        if "dur" in df.columns:
            df["dur"] = df["dur"].fillna(0)

        # Ensure canonical signal columns exist (even if empty)
        required_cols = ["drug_name", "adverse_event"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "missing"

        # Drop duplicate columns if any slipped in
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def compute_additional_features(self, df):
        df = df.copy()
        if "fda_dt" in df.columns:
            df["fda_dt_parsed"] = pd.to_datetime(
                df["fda_dt"].astype(str), format="%Y%m%d", errors="coerce"
            )
            df["report_recency_days"] = (
                (CURRENT_DATE - df["fda_dt_parsed"]).dt.days.fillna(0).astype(int)
            )
        else:
            df["report_recency_days"] = 0

        dur_multiplier = (
            df.get("dur_cod", pd.Series(dtype=str))
            .str.upper()
            .map({"YR": 365, "MO": 30, "WK": 7, "DY": 1, "HR": 1 / 24, "MI": 1 / 1440})
            .fillna(1)
        )
        df["exposure_days"] = df.get("dur", 0) * dur_multiplier

        pair_counts = (
            df.groupby(["drug_name", "adverse_event"])
            .size()
            .reset_index(name="pair_count")
        )
        df = df.merge(pair_counts, on=["drug_name", "adverse_event"], how="left")
        df["is_rare_pair"] = (df["pair_count"].fillna(0) < 3).astype(int)
        df = df.drop(columns=["pair_count"], errors="ignore")
        return df

    def inject_synthetic_signals_realistic(
        self,
        df,
        fraction=0.05,
        min_inject=1,
        prr_boost_range=(1.2, 3.0),
        ror_boost_range=(1.2, 3.0),
        noise_std=0.05,
        partial_signal_prob=0.7,
        fraction_novel=0.1,
        random_state=42,
    ):
        np.random.seed(random_state)
        df = df.copy()

        if "drug_name" not in df.columns or "adverse_event" not in df.columns:
            df["is_injected_signal"] = 0
            return df

        df["drug_name"] = df["drug_name"].astype(str).fillna("missing")
        df["adverse_event"] = df["adverse_event"].astype(str).fillna("missing")
        df = df.reset_index(drop=True)
        print(f"Input df shape: {df.shape}, Index is unique: {df.index.is_unique}")

        unique_pairs = df[["drug_name", "adverse_event"]].drop_duplicates()
        n_inject = max(min_inject, int(len(unique_pairs) * fraction))
        n_novel = max(1, int(n_inject * fraction_novel))

        print(f"Unique pairs shape: {unique_pairs.shape}")
        print(f"Sample unique pairs:\n{unique_pairs.head()}")

        ae_counts = df["adverse_event"].value_counts()
        weights = unique_pairs["adverse_event"].map(
            lambda ae: 1 / (ae_counts.get(ae, 0) + 1)
        )
        inject_pairs = unique_pairs.sample(
            n=n_inject, weights=weights, random_state=random_state
        )

        if "is_injected_signal" not in df.columns:
            df["is_injected_signal"] = 0

        all_duplicates = []
        for idx, row in inject_pairs.iterrows():
            drug_name = str(row["drug_name"])
            adverse_event = str(row["adverse_event"])
            mask = (df["drug_name"] == drug_name) & (
                df["adverse_event"] == adverse_event
            )
            print(f"Row {idx}: drug_name={type(drug_name)}, value={drug_name}")
            print(
                f"Row {idx}: adverse_event={type(adverse_event)}, value={adverse_event}"
            )
            print(f"Mask type: {type(mask)}, shape: {mask.shape}, dtype: {mask.dtype}")

            original_rows = df[mask]
            if len(original_rows) > 0:
                num_duplicates = np.random.randint(1, 5)
                duplicates = original_rows.copy()
                duplicates = pd.concat([duplicates] * num_duplicates, ignore_index=True)
                all_duplicates.append(duplicates)
                is_signal_flag = np.random.rand() < partial_signal_prob
                df.loc[mask, "is_injected_signal"] = int(is_signal_flag)

        if all_duplicates:
            df = pd.concat([df] + all_duplicates, ignore_index=True)
            print(
                f"Concatenated {len(all_duplicates)} duplicate sets, new shape: {df.shape}"
            )

        for i in range(n_novel):
            base_row = df.sample(1).copy()
            if np.random.rand() < 0.5:
                base_row["drug_name"] = f"novel_drug_{i}"
            else:
                base_row["adverse_event"] = f"novel_ae_{i}"
            base_row["is_injected_signal"] = int(np.random.rand() < partial_signal_prob)
            df = pd.concat([df, base_row], ignore_index=True)

        for col in ["age", "exposure_days"]:
            if col in df.columns:
                df[col] += np.random.normal(0, noise_std * df[col].std(), size=len(df))

        print(
            f"Injected {n_inject} synthetic signals (including {n_novel} novel), final shape: {df.shape}"
        )
        return df

    def compute_prr_ror(self, df, prr_threshold=2.0, min_a=3):
        if "drug_name" not in df.columns or "adverse_event" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "drug_name",
                    "adverse_event",
                    "PRR",
                    "ROR",
                    "is_disproportional_signal",
                ]
            )
        contingency = (
            df.groupby(["drug_name", "adverse_event"]).size().reset_index(name="a")
        )
        total_drug_counts = (
            df.groupby("drug_name").size().reset_index(name="drug_total")
        )
        total_ae_counts = (
            df.groupby("adverse_event").size().reset_index(name="ae_total")
        )
        total_reports = len(df)

        df_stats = contingency.merge(total_drug_counts, on="drug_name")
        df_stats = df_stats.merge(total_ae_counts, on="adverse_event")

        df_stats["b"] = df_stats["drug_total"] - df_stats["a"]
        df_stats["c"] = df_stats["ae_total"] - df_stats["a"]
        df_stats["d"] = total_reports - (df_stats["a"] + df_stats["b"] + df_stats["c"])

        df_stats["PRR"] = (df_stats["a"] / (df_stats["a"] + df_stats["b"] + 1e-6)) / (
            (df_stats["c"] / (df_stats["c"] + df_stats["d"] + 1e-6)) + 1e-6
        )
        df_stats["ROR"] = (df_stats["a"] / (df_stats["b"] + 1e-6)) / (
            (df_stats["c"] / (df_stats["d"] + 1e-6)) + 1e-6
        )

        df_stats["is_disproportional_signal"] = (
            (df_stats["PRR"] > prr_threshold) & (df_stats["a"] >= min_a)
        ).astype(int)

        return df_stats[
            ["drug_name", "adverse_event", "PRR", "ROR", "is_disproportional_signal"]
        ]

    def _create_target(self, df):
        df = df.copy()
        severity_score = (
            df.get("outcome", pd.Series(dtype=str))
            .str.upper()
            .map({"DE": 3, "LT": 3, "HO": 1, "DS": 2, "CA": 2, "RI": 2, "OT": 1})
            .fillna(0)
        )
        df["is_signal"] = (
            (severity_score > 2)
            | (df.get("is_injected_signal", 0) == 1)
            | (df.get("is_disproportional_signal", 0) == 1)
        ).astype(int)
        return df

    def fit_preprocess(self, df):
        df = self.normalize_columns(df)
        df = self.compute_additional_features(df)
        if self.enable_synthetic_injection:
            df = self.inject_synthetic_signals_realistic(df)
        self.prr_ror_table = self.compute_prr_ror(df)
        df = df.merge(self.prr_ror_table, on=["drug_name", "adverse_event"], how="left")
        df = df.fillna({"PRR": 1.0, "ROR": 1.0, "is_disproportional_signal": 0})
        df = self._create_target(df)

        X = df[self.cat_features + self.num_features].copy()
        y = df["is_signal"]

        for col in self.cat_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("missing"))
            self.label_encoders[col] = le

        X[self.num_features] = self.scaler.fit_transform(X[self.num_features].fillna(0))
        return X, y

    def transform_preprocess(self, df, prr_ror_table=None):
        df = self.normalize_columns(df)
        df = self.compute_additional_features(df)

        if prr_ror_table is None:
            prr_ror_table = self.prr_ror_table
        if prr_ror_table is None:
            raise ValueError("PRR/ROR table is missing. Train the model first.")

        df = df.merge(prr_ror_table, on=["drug_name", "adverse_event"], how="left")
        df = df.fillna(
            {"PRR": 1.0, "ROR": 1.0, "is_disproportional_signal": 0, "is_rare_pair": 1}
        )

        X = df[self.cat_features + self.num_features].copy()
        for col in self.cat_features:
            le = self.label_encoders.get(col)
            if le:
                X[col] = X[col].astype(str).fillna("missing")
                unseen_mask = ~X[col].isin(le.classes_)
                if unseen_mask.any():
                    X.loc[unseen_mask, col] = "unknown"
                    if "unknown" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "unknown")
                X[col] = le.transform(X[col])
            else:
                X[col] = -1

        X[self.num_features] = self.scaler.transform(X[self.num_features].fillna(0))
        return X

    def train_model(self, test_size=0.2, random_state=42):
        df = self.load_data()
        X, y = self.fit_preprocess(df)

        if len(np.unique(y)) < 2:
            print(
                "Only one class found in training data; consider enabling synthetic injection or providing more varied data."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        self.model = RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred, zero_division=0))

        self.save_model()

    def predict(self, input_data):
        if self.model is None or self.prr_ror_table is None:
            print("Model missing or outdated, retraining...")
            self.train_model()
        input_df = pd.DataFrame([input_data])
        X = self.transform_preprocess(input_df)
        probs = self.model.predict_proba(X)[0]
        probability = probs[1] if len(probs) > 1 else probs[0]
        is_signal = bool(probability >= self.prediction_threshold)
        return {"is_signal": is_signal, "probability": float(probability)}

    def predict_many(self, df_input, export_csv_path=None):
        # Ensure model and PRR/ROR are available
        if self.model is None or self.prr_ror_table is None:
            print("Model missing or PRR/ROR table missing, training now...")
            self.train_model()

        # Normalize input data
        df_input = df_input.copy()
        df_input = self.normalize_columns(df_input).reset_index(drop=True)

        print(f"df_input index unique: {df_input.index.is_unique}")

        # Compute features on input only (avoid leakage)
        df_features = self.compute_additional_features(df_input)

        # Transform with training PRR/ROR table
        X = self.transform_preprocess(df_features, prr_ror_table=self.prr_ror_table)

        # Run model predictions
        probs = self.model.predict_proba(X)
        probability = [p[1] if len(p) > 1 else p[0] for p in probs]
        predictions = [int(p >= self.prediction_threshold) for p in probability]

        # Merge input with training PRR/ROR for transparency
        df_merged = df_input.merge(
            self.prr_ror_table.loc[:, ~self.prr_ror_table.columns.duplicated()],
            on=["drug_name", "adverse_event"],
            how="left",
        )

        df_merged["is_signal"] = predictions
        df_merged["probability"] = probability

        result_cols = df_input.columns.tolist() + [
            "PRR",
            "ROR",
            "is_disproportional_signal",
            "is_rare_pair",
            "is_signal",
            "probability",
        ]
        for col in ["PRR", "ROR", "is_disproportional_signal", "is_rare_pair"]:
            if col not in df_merged.columns:
                df_merged[col] = None
        df_result = df_merged[result_cols]

        if export_csv_path:
            df_result.to_csv(export_csv_path, index=False)
            print(f"Results exported to {export_csv_path}")

        return df_result


# ---------------------- Usage ----------------------
if __name__ == "__main__":
    detector = AISignalDetector(model_path=MODEL_PATH)
    detector.load_model()

    sample4 = pd.DataFrame(
        {
            "primary_id": list(range(1101, 1116)),
            "case_id": list(range(2101, 2116)),
            "case_version": [1] * 15,
            "fda_date": [
                20250601,
                20250602,
                20250603,
                20250604,
                20250605,
                20250606,
                20250607,
                20250608,
                20250609,
                20250610,
                20250611,
                20250612,
                20250613,
                20250614,
                20250615,
            ],
            "age": [45, 52, 28, 33, 60, 40, 29, 55, 38, 47, 50, 31, 27, 44, 36],
            "sex": [
                "F",
                "M",
                "F",
                "M",
                "F",
                "M",
                "F",
                "M",
                "F",
                "M",
                "F",
                "M",
                "F",
                "M",
                "F",
            ],
            "country": [
                "CA",
                "US",
                "DE",
                "FR",
                "UK",
                "IN",
                "JP",
                "AU",
                "BR",
                "IT",
                "CN",
                "ES",
                "MX",
                "KR",
                "ZA",
            ],
            "role_cod": [
                "SS",
                "PS",
                "SS",
                "PS",
                "SS",
                "PS",
                "SS",
                "PS",
                "SS",
                "PS",
                "SS",
                "PS",
                "SS",
                "PS",
                "SS",
            ],
            "drug_sequence": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
                150,
            ],
            "drug_name": [
                "OBSCUREDRUGX",
                "RAREDRUGY",
                "NEWDRUGZ",
                "EXPERIMENTAL1",
                "MEDICINEA",
                "SUPPLEMENTB",
                "THERAPYC",
                "CUREALLD",
                "NOVELDRUGE",
                "REMEDYF",
                "TREATMENTG",
                "HEALINGH",
                "PILLX",
                "TABLETY",
                "CAPSULEZ",
            ],
            "active_ingredient": [
                "X-CHEMICAL",
                "Y-SUBSTANCE",
                "Z-COMPOUND",
                "A-COMPOUND",
                "B-CHEMICAL",
                "C-SUBSTANCE",
                "D-CHEMICAL",
                "E-COMPOUND",
                "F-SUBSTANCE",
                "G-CHEMICAL",
                "H-COMPOUND",
                "I-SUBSTANCE",
                "J-CHEMICAL",
                "K-COMPOUND",
                "L-SUBSTANCE",
            ],
            "adverse_event": [
                "Unusual rash",
                "Liver failure",
                "Unexpected pregnancy",
                "Headache",
                "Dizziness",
                "Nausea",
                "Fatigue",
                "Joint pain",
                "Insomnia",
                "Fever",
                "Allergic reaction",
                "Blurred vision",
                "Vomiting",
                "Cough",
                "Palpitations",
            ],
            "outcome": [
                "DS",
                "DE",
                "HO",
                "DS",
                "LT",
                "RC",
                "DS",
                "HO",
                "DE",
                "LT",
                "RC",
                "DS",
                "HO",
                "LT",
                "RC",
            ],
            "indication_sequence": [5, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "indication": [
                "Unknown",
                "Cancer",
                "Contraception",
                "Pain",
                "Diabetes",
                "Hypertension",
                "Flu",
                "Cold",
                "Allergy",
                "Arthritis",
                "Infection",
                "Migraine",
                "Asthma",
                "Depression",
                "Anxiety",
            ],
            "start_date": [
                20250101,
                20250110,
                20250115,
                20250201,
                20250210,
                20250215,
                20250301,
                20250310,
                20250315,
                20250401,
                20250410,
                20250415,
                20250501,
                20250510,
                20250515,
            ],
            "end_date": [
                20250501,
                "",
                "",
                20250601,
                "",
                20250615,
                20250701,
                "",
                20250715,
                20250801,
                "",
                20250815,
                "",
                "",
                20250901,
            ],
            "dur": [
                120,
                np.nan,
                60,
                90,
                np.nan,
                120,
                75,
                np.nan,
                60,
                100,
                np.nan,
                80,
                np.nan,
                np.nan,
                105,
            ],
            "dur_cod": [
                "DY",
                "",
                "DY",
                "DY",
                "",
                "DY",
                "DY",
                "",
                "DY",
                "DY",
                "",
                "DY",
                "",
                "",
                "DY",
            ],
        }
    )

    # load_df = pd.read_csv(INFERENCE_BATCH_PATH, low_memory=False)

    df_norm = detector.normalize_columns(sample4)

    results = detector.predict_many(
        df_norm, export_csv_path="test_results_20K_sample4.csv"
    )
    print(results)
