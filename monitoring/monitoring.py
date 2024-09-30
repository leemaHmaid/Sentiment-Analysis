import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.pipelines.training_pipeline import run as training_run
from src.utils.config import Config

def generate_data_drift_report(reference_data_path, current_data_path, report_path):
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    # Prepare the report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    data_drift_report.save_html(report_path)
    print(f"Data drift report saved to {report_path}")

    drift_result = data_drift_report.as_dict()
    dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']
    drift_score = drift_result['metrics'][0]['result']['drift_share']
    print(f"Dataset Drift Detected: {dataset_drift}, Drift Score: {drift_score}")

    return dataset_drift, drift_score

def check_for_drift_and_retrain(reference_data_path, current_data_path, threshold=0.5):
    report_path = 'reports/data_drift_report.html'
    dataset_drift, drift_score = generate_data_drift_report(reference_data_path, current_data_path, report_path)

    if dataset_drift:
        print("Significant data drift detected. Triggering retraining.")
        # Trigger retraining
        config = Config('config/config.yaml')
        training_run(config)
        print("Retraining completed.")
        return True
    else:
        print("Data drift within acceptable range.")
        return False
