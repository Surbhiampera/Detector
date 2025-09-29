import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path
import joblib

# Estimate severity per Adverse Event Report - Using basic Random Forest Model - Need to update weights for serverity calc

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "synthetic_reports.csv"


class AICaseSeverityClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path or (DATA_DIR / "ai_model.pkl")
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cat_features = ["sex", "drug_name", "indication", "adverse_event"]
        self.num_features = ["age"]

    # Data Loading
    def load_data(self):
        """Load the dataset from CSV"""
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["date_reported"])
        return df

    # Preprocessing
    def _create_target(self, df):
        """Create binary target variable"""
        severity_score = df["outcome"].str.lower().map(
            {"fatal": 3, "hospitalized": 2, "not recovered": 1}
        ).fillna(0) + df["adverse_event"].str.lower().map(
            {"seizure": 2, "liver toxicity": 3, "anaphylaxis": 3}
        ).fillna(0)
        df = df.assign(severity_score=severity_score)
        df["is_signal"] = (df["severity_score"] > 1).astype(int)
        return df

    def fit_preprocess(self, df):
        """Preprocess data for training (fit encoders & scaler)"""
        df = self._create_target(df)

        X = df[self.cat_features + self.num_features].copy()
        y = df["is_signal"]

        # Fit encoders
        for col in self.cat_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Fit scaler
        X[self.num_features] = self.scaler.fit_transform(X[self.num_features])

        return X, y

    def transform_preprocess(self, df):
        """Preprocess data for inference (use fitted encoders & scaler)"""
        X = df[self.cat_features + self.num_features].copy()

        # Transform categories (handle unseen labels safely)
        for col in self.cat_features:
            le = self.label_encoders[col]
            X[col] = X[col].map(
                lambda val: le.transform([val])[0] if val in le.classes_ else -1
            )

        # Scale numeric
        X[self.num_features] = self.scaler.transform(X[self.num_features])

        return X

    # Training
    def train_model(self, test_size=0.2, random_state=42):
        """Train the AI model"""
        df = self.load_data()
        X, y = self.fit_preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))

        # Save model
        self.save_model()

    # Prediction
    def predict(self, input_data):
        """Predict if new data indicates a signal"""
        if self.model is None:
            self.load_model()

        input_df = pd.DataFrame([input_data])
        X = self.transform_preprocess(input_df)

        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)[0][1]

        return {"is_signal": bool(prediction[0]), "probability": float(probability)}

    # Persistence
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if self.model:
            joblib.dump(
                {
                    "model": self.model,
                    "label_encoders": self.label_encoders,
                    "scaler": self.scaler,
                },
                self.model_path,
            )

    def load_model(self):
        """Load the trained model and preprocessing objects"""
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.label_encoders = data["label_encoders"]
            self.scaler = data["scaler"]
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")


if __name__ == "__main__":
    detector = AICaseSeverityClassifier()

    if not detector.model_path.exists():
        detector.train_model()

    sample_input = {
        "sex": "M",
        "age": 45,
        "drug_name": "Atorvastatin", # Aspirin
        "indication": "Headache",
        "adverse_event": "Liver toxicity", # Anaphylaxisamper@123.
        
    }
    result = detector.predict(sample_input)
    print(f"Prediction: {result}")
 