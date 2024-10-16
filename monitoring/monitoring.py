import os
import pandas as pd
import json
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.utils.config import load_config
from src.utils.logger import setup_logger
import logging
from typing import Tuple, Optional


def generate_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path_html: str,
    report_path_json: str
) -> Tuple[bool, float]:
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    # Ensure the report directories exist
    report_dir_html = os.path.dirname(report_path_html)
    if report_dir_html and not os.path.exists(report_dir_html):
        os.makedirs(report_dir_html, exist_ok=True)
        logging.info(f"Created report directory at {report_dir_html}")

    report_dir_json = os.path.dirname(report_path_json)
    if report_dir_json and not os.path.exists(report_dir_json):
        os.makedirs(report_dir_json, exist_ok=True)
        logging.info(f"Created report directory at {report_dir_json}")

    # Save the report as an HTML file
    data_drift_report.save_html(report_path_html)
    logging.info(f"Data drift report saved to {report_path_html}")

    # Save the report as a JSON file
    drift_result = data_drift_report.as_dict()
    with open(report_path_json, 'w') as f:
        json.dump(drift_result, f, indent=4)
    logging.info(f"Data drift report saved to {report_path_json}")

    # Extract drift results
    dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']
    drift_score = drift_result['metrics'][0]['result']['drift_share']
    logging.info(f"Dataset Drift Detected: {dataset_drift}, Drift Score: {drift_score}")

    return dataset_drift, drift_score


def check_for_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path_html: str,
    report_path_json: str,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    
    # Generate the data drift report
    dataset_drift, drift_score = generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_path_html=report_path_html,
        report_path_json=report_path_json
    )

    # Determine if drift is significant based on the threshold
    significant_drift = dataset_drift and drift_score > threshold
    if significant_drift:
        logging.info(f"Significant data drift detected. Drift score: {drift_score}")
    else:
        logging.info(f"Data drift within acceptable range. Drift score: {drift_score}")

    return significant_drift, drift_score


def simulate_data_drift_and_monitor(
    training_data_path: str,
    config: dict,
    report_path_html: str,
    report_path_json: str
) -> Optional[Tuple[bool, float]]:
    try:
        # Load the training data
        data = pd.read_csv(training_data_path)
        logging.info(f"Loaded training data from {training_data_path}")

        # Define the number of rows for reference and current datasets
        reference_size = config.get('monitoring', {}).get('reference_size', 500)
        current_size = config.get('monitoring', {}).get('current_size', 500)

        # Ensure there is enough data
        total_required = reference_size + current_size
        if len(data) < total_required:
            logging.error(f"Not enough data to simulate drift. Required: {total_required}, Available: {len(data)}")
            return None

        # Split the data into reference and current datasets
        reference_data = data.iloc[:reference_size].reset_index(drop=True)
        current_data = data.iloc[reference_size:reference_size + current_size].reset_index(drop=True)
        logging.info(
            f"Simulated data drift by using first {reference_size} rows as reference and next {current_size} rows as current data."
        )

        # Check for drift
        return check_for_drift(reference_data, current_data, report_path_html, report_path_json)

    except FileNotFoundError:
        logging.error(f"Training data file not found at: {training_data_path}")
    except pd.errors.EmptyDataError:
        logging.error(f"Training data file is empty: {training_data_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during data drift simulation: {e}")
    
    return None


def main():
    """
    Main function to execute the monitoring process.
    """
    # Determine the absolute path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    config_path = os.path.join(base_dir, 'backend', 'configs', 'config.yaml')

    # Debugging: Print base_dir and config_path
    print(f"Base directory: {base_dir}")
    print(f"Config path: {config_path}")

    # Load configuration
    if not os.path.exists(config_path):
        print(f"Configuration file does not exist at: {config_path}")
        exit(1)

    config = load_config(config_path)
    
    # Set up logging
    log_file = config['logging']['log_file']
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory at {log_dir}")  # Initial setup before logging is configured
    logger = setup_logger(log_file, config['logging']['level'])
    logging.info("Starting monitoring for data drift.")
    
    # Define the path to your training data
    training_data_path = config['data']['raw_data_path']
    if not os.path.exists(training_data_path):
        logging.error(f"Training data file does not exist at: {training_data_path}")
        exit(1)
    
    # Define report paths inside the monitoring directory
    report_dir = os.path.join(script_dir, 'reports')
    report_path_html = os.path.join(report_dir, 'data_drift_report.html')
    report_path_json = os.path.join(report_dir, 'data_drift_report.json')
    
    # Simulate data drift and monitor
    result = simulate_data_drift_and_monitor(training_data_path, config, report_path_html, report_path_json)
    
    if result:
        drift_detected, drift_score = result
        if drift_detected:
            logging.info(f"Drift detected with a score of {drift_score}. Consider retraining the model.")
        else:
            logging.info(f"No significant drift detected. Drift score: {drift_score}.")
    else:
        logging.warning("Monitoring process encountered issues and drift detection was not completed.")


if __name__ == "__main__":
    main()
