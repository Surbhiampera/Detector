from disproportionality_analysis import SignalAggregator
from report_severity_model import AICaseSeverityClassifier

sample_reports = [
    {"sex": "M", "age": 45, "drug_name": "Atorvastatin", "indication": "Headache", "adverse_event": "Liver toxicity"},
    {"sex": "F", "age": 62, "drug_name": "Atorvastatin", "indication": "Cholesterol", "adverse_event": "Seizure"},
    {"sex": "M", "age": 33, "drug_name": "Ibuprofen", "indication": "Pain", "adverse_event": "Liver toxicity"},
    {"sex": "F", "age": 70, "drug_name": "Ibuprofen", "indication": "Arthritis", "adverse_event": "Headache"},
]

detector = AICaseSeverityClassifier()
if not detector.model_path.exists():
    detector.train_model()
else:
    detector.load_model()


aggregator = SignalAggregator()

for report in sample_reports:
    prediction = detector.predict(report)
    aggregator.add_report(report, prediction)

agg_df = aggregator.aggregate_signals()
print("Aggregated signals:")
print(agg_df)

prr = aggregator.compute_prr(agg_df, drug_name="Atorvastatin", event="Liver toxicity")
print(f"PRR for Atorvastatin - Liver toxicity: {prr:.2f}" if prr else "PRR not computable")