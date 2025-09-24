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
INFERENCE_BATCH_PATH = DATA_DIR / "faers_inference.csv"

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

    def load_data(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False, index_col=None)
        df = self.normalize_columns(df)
        df = df.reset_index(drop=True)
        print(f"Loaded data shape: {df.shape}, Index is unique: {df.index.is_unique}")
        print(f"Loaded data columns: {df.columns.tolist()}")
        return df

    def normalize_columns(self, df):
        df = df.copy()
        df.columns = df.columns.str.lower()
        for faers_col, internal_col in FAERS_COLUMN_MAP.items():
            if faers_col.lower() in df.columns:
                df[internal_col] = df[faers_col.lower()]
        for col in INTERNAL_CAT_COLS:
            if col not in df.columns:
                df[col] = "missing"
        for col in INTERNAL_NUM_COLS:
            if col not in df.columns:
                df[col] = 0
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
            .map({"DE": 3, "LT": 3, "HO": 2, "DS": 2, "CA": 2, "RI": 2, "OT": 1})
            .fillna(0)
        )
        df["is_signal"] = (
            (severity_score > 1)
            | (df.get("is_injected_signal", 0) == 1)
            | (df.get("is_disproportional_signal", 0) == 1)
        ).astype(int)
        return df

    def fit_preprocess(self, df):
        df = self.normalize_columns(df)
        df = self.compute_additional_features(df)
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
            print("Only one class found in training data, forcing some signals...")
            y.iloc[:5] = 1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
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
        prediction = self.model.predict(X)

        probs = self.model.predict_proba(X)[0]
        probability = (
            probs[1] if len(probs) > 1 else (probs[0] if prediction[0] == 1 else 0.0)
        )

        return {"is_signal": bool(prediction[0]), "probability": float(probability)}

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

    def predict_many(self, df_input, export_csv_path=None):
        # Load and normalize training data
        train_df = self.load_data()
        train_df = self.normalize_columns(train_df).reset_index(drop=True)

        # Normalize input data separately
        df_input = df_input.copy()
        df_input = self.normalize_columns(df_input).reset_index(drop=True)

        print(f"train_df index unique: {train_df.index.is_unique}")
        print(f"df_input index unique: {df_input.index.is_unique}")

        # Concatenate normalized data
        combined_df = pd.concat([train_df, df_input], ignore_index=True)

        # 🔍 Ensure no duplicate columns
        if not combined_df.columns.is_unique:
            dupes = combined_df.columns[combined_df.columns.duplicated()].tolist()
            print("⚠️ Duplicate columns detected in combined_df:", dupes)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        print(f"combined_df index unique: {combined_df.index.is_unique}")

        # Compute features and PRR/ROR
        combined_df = self.compute_additional_features(combined_df)

        # 🔍 Ensure no duplicate columns after feature computation
        if not combined_df.columns.is_unique:
            dupes = combined_df.columns[combined_df.columns.duplicated()].tolist()
            print(
                "⚠️ Duplicate columns detected after compute_additional_features:", dupes
            )
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        prr_ror_cumulative = self.compute_prr_ror(combined_df)

        # Already-normalized input
        df_input_norm = df_input.loc[:, ~df_input.columns.duplicated()]
        X = self.transform_preprocess(df_input_norm, prr_ror_table=prr_ror_cumulative)

        # Run model predictions
        predictions = self.model.predict(X)
        probs = self.model.predict_proba(X)
        probability = [
            p[1] if len(p) > 1 else (p[0] if pred == 1 else 0.0)
            for p, pred in zip(probs, predictions)
        ]

        # Merge with PRR/ROR results
        df_merged = df_input_norm.merge(
            prr_ror_cumulative.loc[:, ~prr_ror_cumulative.columns.duplicated()],
            on=["drug_name", "adverse_event"],
            how="left",
        )

        df_merged["is_signal"] = predictions
        df_merged["probability"] = probability

        result_cols = df_input_norm.columns.tolist() + [
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

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize FAERS-style input DataFrame to a canonical schema
        with consistent column names and datatypes.
        """

        # Define mapping from raw FAERS names → canonical names
        col_map = {
            "drugname": "drug_name",
            "pt": "adverse_event",
            "indi_pt": "indication",
            "prod_ai": "active_ingredient",
            "rpt_country": "country",  # 👈 match pipeline
            "outc_cod": "outcome",  # 👈 match pipeline
            "role_cod": "role_cod",  # 👈 unchanged
            "caseid": "case_id",
            "caseversion": "case_version",
            "primaryid": "primary_id",
            "fda_dt": "fda_date",
            "drug_seq": "drug_sequence",
            "indi_drug_seq": "indication_sequence",
            "sex": "sex",
            "age": "age",
            "dur": "dur",  # 👈 keep as pipeline expects
            "dur_cod": "dur_cod",  # 👈 keep as pipeline expects
            "start_dt": "start_date",
            "end_dt": "end_date",
        }
        # Rename columns if they exist in the DataFrame
        df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

        # Coerce date fields to datetime (invalid → NaT)
        for col in ["start_date", "end_date", "fda_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y%m%d")

        # Handle duration NaNs
        if "duration" in df.columns:
            df["duration"] = df["duration"].fillna(0)

        # Ensure canonical signal columns exist (even if empty)
        required_cols = ["drug_name", "adverse_event"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None  # placeholder if missing

        # Drop duplicate columns if any slipped in
        df = df.loc[:, ~df.columns.duplicated()]

        return df


# ---------------------- Usage ----------------------
if __name__ == "__main__":
    detector = AISignalDetector(model_path=MODEL_PATH)
    detector.load_model()

    sample_data = pd.DataFrame(
        {
            "primaryid": [247547526, 246890264, 247894922],
            "caseid": [24754752, 24689026, 24789492],
            "caseversion": [6, 4, 2],
            "fda_dt": [20250623, 20250619, 20250611],
            "age": [59.0, 43.0, 44.0],
            "sex": ["F", "F", "F"],
            "rpt_country": ["CA", "CA", "CA"],
            "rpsr_cod": ["EXP", "EXP", "EXP"],
            "drug_seq": [138, 495, 70],
            "role_cod": ["SS", "SS", "SS"],
            "drugname": ["DESOXIMETASONE", "PHTHALYLSULFATHIAZOLE", "ORENCIA"],
            "prod_ai": ["DESOXIMETASONE", "PHTHALYLSULFATHIAZOLE", "ABATACEPT"],
            "pt": ["Foot deformity", "Pregnancy", "Colitis ulcerative"],
            "outc_cod": ["CA", "DS", "DE"],
            "indi_drug_seq": [301.0, 1.0, 157.0],
            "indi_pt": [
                "Product used for unknown indication",
                "Product used for unknown indication",
                "Rheumatoid arthritis",
            ],
            "start_dt": ["", "", ""],
            "end_dt": ["", "", ""],
            "dur": [np.nan, np.nan, np.nan],
            "dur_cod": ["", "", ""],
        }
    )

    normalize_columns = detector.normalize_columns(sample_data)

    results = detector.predict_many(
        normalize_columns, export_csv_path="test_results.csv"
    )
    print(results)
