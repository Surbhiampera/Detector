import pandas as pd
from collections import defaultdict
from report_severity_model import AICaseSeverityClassifier

# Simple aggregation of 

detector = AICaseSeverityClassifier()
if not detector.model_path.exists():
    detector.train_model()
else:
    detector.load_model()


class SignalAggregator:
    def __init__(self):
        self.reports = []

    def add_report(self, case, prediction):
        """Store triaged case with model prediction"""
        if prediction["is_signal"]:
            self.reports.append({**case, **prediction})

    def aggregate_signals(self):
        """Aggregate counts of serious cases by drug-event pair"""
        df = pd.DataFrame(self.reports)
        agg = (
            df.groupby(["drug_name", "adverse_event"])
            .agg(count=("is_signal", "sum"), avg_prob=("probability", "mean"))
            .reset_index()
        )
        return agg

    def compute_prr(self, df, drug_name, event):
        """Compute PRR for a drug-event pair"""
        a = len(df[(df["drug_name"] == drug_name) & (df["adverse_event"] == event)])
        b = len(df[(df["drug_name"] == drug_name) & (df["adverse_event"] != event)])
        c = len(df[(df["drug_name"] != drug_name) & (df["adverse_event"] == event)])
        d = len(df[(df["drug_name"] != drug_name) & (df["adverse_event"] != event)])

        prr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else None
        return prr
