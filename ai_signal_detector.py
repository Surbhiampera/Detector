import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path
import joblib

# ---------------- CONFIG ----------------

# Define base data directory (adjust as needed)
# DATA_DIR = Path("/home/ampara/Downloads/Signal Detector/data/FAERS_merged_25Q2_20250922_161904.csv")  # <-- UPDATE this path to your actual data folder

# DATA_PATH = DATA_DIR / "/home/ampara/Downloads/Signal Detector/data/FAERS_merged_25Q2_20250922_161904.csv"  # dataset CSV
# MODEL_PATH = DATA_DIR / "/home/ampara/Downloads/Signal Detector/data/ai_model.pkl"  # saved model file

DATA_DIR = Path(__file__).resolve().parent / "data"
# DATA_PATH = DATA_DIR / "synthetic_reports.csv"
print(DATA_DIR)


DATA_PATH =  DATA_DIR / "FAERS_merged_25Q2_20250922_161904.csv"  # dataset CSV
print(DATA_PATH)

MODEL_PATH =  DATA_DIR / "ai_model.pkl"  # saved model file
print(MODEL_PATH)

# Expected columns in lowercase
INTERNAL_CAT_COLS = ["sex", "drug_name", "indication", "adverse_event"]
INTERNAL_NUM_COLS = ["age"]

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
        self.cat_features = INTERNAL_CAT_COLS.copy()
        self.num_features = INTERNAL_NUM_COLS + ["PRR", "ROR"]
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
        missing_cols = [col for col in self.cat_features + INTERNAL_NUM_COLS if col not in df.columns]
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

    # ---------------- Compute PRR / ROR ----------------
    def compute_prr_ror(self, df):
        if "drug_name" not in df.columns or "adverse_event" not in df.columns:
            print("Skipping PRR/ROR computation: required columns missing")
            return pd.DataFrame(columns=["drug_name", "adverse_event", "PRR", "ROR"])
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

        return df_stats[["drug_name", "adverse_event", "PRR", "ROR"]]

    # ---------------- Create Target ----------------
    def _create_target(self, df):
        df = df.copy()
        severity_score = df.get("outcome", pd.Series(dtype=str)).str.lower().map(
            {"fatal": 3, "hospitalized": 2, "not recovered": 1}
        ).fillna(0)
        df["is_signal"] = ((severity_score > 1) | (df.get("is_injected_signal", 0) == 1)).astype(int)
        return df

    # ---------------- Preprocessing ----------------
    def fit_preprocess(self, df):
        df = self.inject_signals(df)
        self.prr_ror_table = self.compute_prr_ror(df)
        df = df.merge(self.prr_ror_table, on=["drug_name", "adverse_event"], how="left")
        df = self._create_target(df)

        for col in self.cat_features:
            if col not in df.columns:
                df[col] = "missing"
        for col in self.num_features:
            if col not in df.columns:
                df[col] = 0

        X = df[self.cat_features + self.num_features].copy()
        y = df["is_signal"]

        for col in self.cat_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

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

        # Ensure both classes exist
        if len(np.unique(y)) < 2:
            print("Only one class found in training data, forcing some signals...")
            y.iloc[:5] = 1  # force a few positives

        # Stratified split to keep class balance
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

        # Handle single-class case
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


# ---------------------- Usage ----------------------
if __name__ == "__main__":
    detector = AISignalDetector(model_path=MODEL_PATH)
    detector.load_model()

    sample_input = {
        "sex": "F",
        "age": 39,
        "drug_name": "AZACITIDINE",
        "indication": "",
        "adverse_event": "Fatigue",
        "PRR": 0.0,
        "ROR": 0.0
    }

    required_features = INTERNAL_CAT_COLS + INTERNAL_NUM_COLS + ["PRR", "ROR"]
    missing_features = [feat for feat in required_features if feat not in sample_input]

    if missing_features:
        print(f"Missing features in sample_input: {missing_features}")
    else:
        print("All required features are present in sample_input.")

    result = detector.predict(sample_input)
    print(f"Prediction: {result}")