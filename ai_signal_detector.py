import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path
import joblib

# ---------------- CONFIG ----------------

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "fears_sampled_dataset.csv"  # dataset CSV
MODEL_PATH = DATA_DIR / "ai_model.pkl"  # saved model file
INFERENCE_BATCH_PATH = DATA_DIR / "faers_inference.csv"  # for batch predictions

FAERS_COLUMN_MAP = {
    "drugname": "drug_name",
    "prod_ai": "prod_ai",
    "pt": "adverse_event",
    "indi_pt": "indication",
    "indi_drug_seq": "indi_drug_seq",
    "age": "age",
    "sex": "sex",
    "outcome": "outcome"
}

INTERNAL_CAT_COLS = ["sex", "drug_name", "indication", "adverse_event"]
INTERNAL_NUM_COLS = ["age"]


class AISignalDetector:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        # new: include disproportionality flag
        self.cat_features = INTERNAL_CAT_COLS.copy()
        self.num_features = INTERNAL_NUM_COLS + ["PRR", "ROR", "is_disproportional_signal"]
        self.prr_ror_table = None

    # ---------------- Data Loading ----------------
    def load_data(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        df.columns = df.columns.str.lower()
        for faers_col, internal_col in FAERS_COLUMN_MAP.items():
            if faers_col.lower() in df.columns:
                df[internal_col] = df[faers_col.lower()]
        missing_cols = [col for col in INTERNAL_CAT_COLS + INTERNAL_NUM_COLS if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in CSV: {missing_cols}")
        return df

    # ---------------- Signal Injection ----------------
    def inject_signals(self, df, fraction=0.05, random_state=42):
        np.random.seed(random_state)
        if "drug_name" not in df.columns or "adverse_event" not in df.columns:
            print("Skipping signal injection: required columns missing")
            df["is_injected_signal"] = 0
            return df
        unique_pairs = df[["drug_name", "adverse_event"]].drop_duplicates()
        n_inject = max(1, int(len(unique_pairs) * fraction))
        inject_pairs = unique_pairs.sample(n=n_inject)
        df["is_injected_signal"] = 0
        for _, row in inject_pairs.iterrows():
            df.loc[
                (df["drug_name"] == row["drug_name"]) & (df["adverse_event"] == row["adverse_event"]),
                "is_injected_signal"
            ] = 1
        return df
    
    def compute_prr_ror_for_injection(self, df):
        """
        Compute PRR, ROR, and is_disproportional_signal for all unique drug-event pairs.
        Returns a DataFrame suitable for merging with the main dataset.
        """
        contingency = df.groupby(["drug_name", "adverse_event"]).size().reset_index(name="a")
        total_drug_counts = df.groupby("drug_name").size().reset_index(name="drug_total")
        total_ae_counts = df.groupby("adverse_event").size().reset_index(name="ae_total")
        total_reports = len(df)

        df_stats = contingency.merge(total_drug_counts, on="drug_name")
        df_stats = df_stats.merge(total_ae_counts, on="adverse_event")

        df_stats["b"] = df_stats["drug_total"] - df_stats["a"]
        df_stats["c"] = df_stats["ae_total"] - df_stats["a"]
        df_stats["d"] = total_reports - (df_stats["a"] + df_stats["b"] + df_stats["c"])

        # PRR/ROR calculation with small epsilon to avoid div by 0
        df_stats["PRR"] = (df_stats["a"] / (df_stats["a"] + df_stats["b"] + 1e-6)) / \
                        ((df_stats["c"] / (df_stats["c"] + df_stats["d"] + 1e-6)) + 1e-6)
        df_stats["ROR"] = (df_stats["a"] / (df_stats["b"] + 1e-6)) / \
                        ((df_stats["c"] / (df_stats["d"] + 1e-6)) + 1e-6)

        # Optional: flag disproportional signal (simple threshold)
        df_stats["is_disproportional_signal"] = ((df_stats["PRR"] > 2) & (df_stats["a"] >= 3)).astype(int)

        return df_stats[["drug_name", "adverse_event", "PRR", "ROR", "is_disproportional_signal"]]


    # ---------------- Compute PRR / ROR ----------------
    def compute_prr_ror(self, df, prr_threshold=2.0, ror_threshold=2.0):
        if "drug_name" not in df.columns or "adverse_event" not in df.columns:
            print("Skipping PRR/ROR computation: required columns missing")
            return pd.DataFrame(columns=["drug_name", "adverse_event", "PRR", "ROR", "is_disproportional_signal"])
        contingency = df.groupby(["drug_name", "adverse_event"]).size().reset_index(name="a")
        total_drug_counts = df.groupby("drug_name").size().reset_index(name="drug_total")
        total_ae_counts = df.groupby("adverse_event").size().reset_index(name="ae_total")
        total_reports = len(df)

        df_stats = contingency.merge(total_drug_counts, on="drug_name")
        df_stats = df_stats.merge(total_ae_counts, on="adverse_event")

        df_stats["b"] = df_stats["drug_total"] - df_stats["a"]
        df_stats["c"] = df_stats["ae_total"] - df_stats["a"]
        df_stats["d"] = total_reports - (df_stats["a"] + df_stats["b"] + df_stats["c"])

        df_stats["PRR"] = (df_stats["a"] / (df_stats["a"] + df_stats["b"] + 1e-6)) / \
                          ((df_stats["c"] / (df_stats["c"] + df_stats["d"] + 1e-6)) + 1e-6)
        df_stats["ROR"] = (df_stats["a"] / (df_stats["b"] + 1e-6)) / \
                          ((df_stats["c"] / (df_stats["d"] + 1e-6)) + 1e-6)

        # New binary signal flag
        df_stats["is_disproportional_signal"] = ((df_stats["PRR"] > prr_threshold) &
                                                 (df_stats["ROR"] > ror_threshold)).astype(int)

        return df_stats[["drug_name", "adverse_event", "PRR", "ROR", "is_disproportional_signal"]]

    # ---------------- Create Target ----------------
    def _create_target(self, df):
        df = df.copy()
        severity_score = df.get("outcome", pd.Series(dtype=str)).str.lower().map(
            {"fatal": 3, "hospitalized": 2, "not recovered": 1}
        ).fillna(0)
        df["is_signal"] = ((severity_score > 1) |
                           (df.get("is_injected_signal", 0) == 1) |
                           (df.get("is_disproportional_signal", 0) == 1)).astype(int)
        return df

    # ---------------- Preprocessing ----------------
    def fit_preprocess(self, df):
        # Step 1: Inject synthetic signals
        df = self.inject_synthetic_signals_realistic(df)

        # Step 2: Compute PRR/ROR for all pairs (existing + injected)
        self.prr_ror_table = self.compute_prr_ror_for_injection(df)

        # Step 3: Merge PRR/ROR values into main dataframe
        df = df.merge(
            self.prr_ror_table,
            on=["drug_name", "adverse_event"],
            how="left"
        )

        # Step 4: Create target variable: combines injected + severe outcome
        df = self._create_target(df)

        # Step 5: Ensure all required categorical and numeric columns exist
        for col in self.cat_features:
            if col not in df.columns:
                df[col] = "missing"
        for col in self.num_features:
            if col not in df.columns:
                df[col] = 0

        # Step 6: Select features and target
        X = df[self.cat_features + self.num_features].copy()
        y = df["is_signal"]

        # Step 7: Encode categorical features
        for col in self.cat_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Step 8: Scale numeric features
        X[self.num_features] = self.scaler.fit_transform(X[self.num_features])

        return X, y


    def transform_preprocess(self, df):
        df = df.copy()
        if self.prr_ror_table is None:
            raise ValueError("PRR/ROR table is missing. Train the model first.")
        df = df.merge(self.prr_ror_table, on=["drug_name", "adverse_event"], how="left")

        for col in self.cat_features:
            if col not in df.columns:
                df[col] = "missing"
        for col in self.num_features:
            if col not in df.columns:
                df[col] = 0

        X = df[self.cat_features + self.num_features].copy()
        for col in self.cat_features:
            le = self.label_encoders.get(col)
            if le:
                X[col] = X[col].map(lambda val: le.transform([val])[0] if val in le.classes_ else -1)

        X[self.num_features] = self.scaler.transform(X[self.num_features])
        return X

    # ---------------- Train ----------------
    def train_model(self, test_size=0.2, random_state=42):
        df = self.load_data()
        X, y = self.fit_preprocess(df)

        if len(np.unique(y)) < 2:
            print("Only one class found in training data, forcing some signals...")
            y.iloc[:5] = 1  # force a few positives

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred, zero_division=0))

        self.save_model()

    # ---------------- Predict ----------------
    def predict(self, input_data):
        if self.model is None or self.prr_ror_table is None:
            print("Model missing or outdated, retraining...")
            self.train_model()
        input_df = pd.DataFrame([input_data])
        X = self.transform_preprocess(input_df)
        prediction = self.model.predict(X)

        probs = self.model.predict_proba(X)[0]
        probability = probs[1] if len(probs) > 1 else (probs[0] if prediction[0] == 1 else 0.0)

        return {"is_signal": bool(prediction[0]), "probability": float(probability)}

    # ---------------- Persistence ----------------
    def save_model(self):
        if self.model:
            joblib.dump({
                "model": self.model,
                "label_encoders": self.label_encoders,
                "scaler": self.scaler,
                "prr_ror_table": self.prr_ror_table
            }, self.model_path)

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

    # ---------------- Batch Inference ----------------
    def predict_many(self, df_input, export_csv_path=None):
        """
        df_input: pd.DataFrame with columns ["sex","age","drug_name","indication","adverse_event"]
        export_csv_path: optional, path to save results CSV
        """
        # Step 1: Normalize column names and ensure required columns
        df_input = df_input.copy()
        df_input.columns = df_input.columns.str.lower()
        
        # Map FAERS columns to internal names if needed
        for faers_col, internal_col in FAERS_COLUMN_MAP.items():
            if faers_col.lower() in df_input.columns:
                df_input[internal_col] = df_input[faers_col.lower()]
        
        for col in INTERNAL_CAT_COLS + INTERNAL_NUM_COLS:
            if col not in df_input.columns:
                df_input[col] = "" if col in INTERNAL_CAT_COLS else 0

        # Step 2: Compute PRR/ROR for the batch if model has a table
        if self.prr_ror_table is None:
            raise ValueError("PRR/ROR table missing. Train the model first.")

        df_merged = df_input.merge(
            self.prr_ror_table,
            on=["drug_name", "adverse_event"],
            how="left"
        )

        # Step 3: Fill missing PRR/ROR/disproportional_signal with 0
        for col in ["PRR", "ROR", "is_disproportional_signal"]:
            if col not in df_merged.columns or df_merged[col].isnull().any():
                df_merged[col] = 0.0 if col != "is_disproportional_signal" else 0

        # Step 4: Preprocess features
        X = self.transform_preprocess(df_merged)

        # Step 5: Predict
        predictions = self.model.predict(X)
        probs = self.model.predict_proba(X)
        probability = [p[1] if len(p) > 1 else (p[0] if pred==1 else 0.0)
                    for p, pred in zip(probs, predictions)]

        # Step 6: Add results
        df_merged["is_signal"] = predictions
        df_merged["probability"] = probability

        # Step 7: Select output columns
        result_cols = df_input.columns.tolist() + ["PRR", "ROR", "is_disproportional_signal", "is_signal", "probability"]
        df_result = df_merged[result_cols]

        # Step 8: Export CSV if requested
        if export_csv_path:
            df_result.to_csv(export_csv_path, index=False)
            print(f"Results exported to {export_csv_path}")

        return df_result

  # ---------------- Robust Synthetic Signal Injector ----------------
    def inject_synthetic_signals_realistic(
        self,
        df,
        fraction=0.05,
        min_inject=1,
        prr_boost_range=(1.2, 3.0),
        ror_boost_range=(1.2, 3.0),
        noise_std=0.05,
        partial_signal_prob=0.7,
        random_state=42
    ):
        """
        Injects synthetic signals into the dataset with noise and partial signals.

        Parameters:
            df (pd.DataFrame): input dataset
            fraction (float): fraction of unique drug-event pairs to inject
            min_inject (int): minimum number of signals
            prr_boost_range (tuple): range for boosting PRR
            ror_boost_range (tuple): range for boosting ROR
            noise_std (float): std deviation for Gaussian noise
            partial_signal_prob (float): probability injected pair is treated as actual signal
            random_state (int): reproducibility
        """
        np.random.seed(random_state)
        df = df.copy()

        if "drug_name" not in df.columns or "adverse_event" not in df.columns:
            df["is_injected_signal"] = 0
            return df

        unique_pairs = df[["drug_name", "adverse_event"]].drop_duplicates()
        n_inject = max(min_inject, int(len(unique_pairs) * fraction))

        # Weight rarer AE higher
        ae_counts = df["adverse_event"].value_counts()
        weights = unique_pairs["adverse_event"].map(lambda ae: 1 / (ae_counts.get(ae, 0) + 1))
        inject_pairs = unique_pairs.sample(n=n_inject, weights=weights, random_state=random_state)

        if "is_injected_signal" not in df.columns:
            df["is_injected_signal"] = 0

        for _, row in inject_pairs.iterrows():
            mask = (df["drug_name"] == row["drug_name"]) & (df["adverse_event"] == row["adverse_event"])

            # Decide if this injected pair becomes an actual signal
            is_signal_flag = np.random.rand() < partial_signal_prob
            df.loc[mask, "is_injected_signal"] = int(is_signal_flag)

            # Boost PRR/ROR randomly in specified range
            if self.prr_ror_table is not None:
                prr_mask = (self.prr_ror_table["drug_name"] == row["drug_name"]) & \
                        (self.prr_ror_table["adverse_event"] == row["adverse_event"])
                prr_boost = np.random.uniform(*prr_boost_range)
                ror_boost = np.random.uniform(*ror_boost_range)
                df.loc[mask, "PRR"] = df.loc[mask, "PRR"].fillna(1.0) * prr_boost
                df.loc[mask, "ROR"] = df.loc[mask, "ROR"].fillna(1.0) * ror_boost

        # Add Gaussian noise to PRR/ROR for all rows
        for col in ["PRR", "ROR"]:
            if col in df.columns:
                df[col] = df[col].fillna(1.0) + np.random.normal(0, noise_std, size=len(df))

        # Recompute PRR/ROR table including new injections
        self.prr_ror_table = self.compute_prr_ror_for_injection(df)

        print(f"Injected {n_inject} synthetic signals with noise into dataset.")
        return df



    def normalize_columns(self, df):
        """
        Ensures the DataFrame has the expected lowercase column names for processing.
        Maps FAERS columns if needed.
        """
        df = df.copy()
        # Lowercase all columns
        df.columns = df.columns.str.lower()

        # Map FAERS columns to internal names
        for faers_col, internal_col in FAERS_COLUMN_MAP.items():
            if faers_col.lower() in df.columns:
                df[internal_col] = df[faers_col.lower()]

        # Ensure all required columns exist
        for col in INTERNAL_CAT_COLS + INTERNAL_NUM_COLS:
            if col not in df.columns:
                df[col] = "" if col in INTERNAL_CAT_COLS else 0

        return df



# ---------------------- Usage ----------------------
if __name__ == "__main__":
    detector = AISignalDetector(model_path=MODEL_PATH)
    detector.train_model()
    

    # # Example batch
    batch_df = pd.read_csv(INFERENCE_BATCH_PATH)
    batch_df_normalised = detector.normalize_columns(batch_df)
    
    df_with_signals = detector.inject_synthetic_signals_realistic(
        batch_df_normalised,
        fraction=0.9,   # inject into 10% of unique pairs
        prr_boost=2.5,
        ror_boost=2.5
    )
    # detector.prr_ror_table = detector.compute_prr_ror_for_injection(df_with_signals)
    

    results = detector.predict_many(df_with_signals, export_csv_path="synthetic_signal_results_new.csv")
    print(results)


