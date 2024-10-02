from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from scripts.prepare_current_data import prepare_current_data
from src.monitoring.monitoring import check_for_drift_and_retrain

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 2),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
}

dag = DAG(
    'data_drift_monitoring',
    default_args=default_args,
    description='Monitor data drift daily',
    schedule_interval=timedelta(days=1),
)

def prepare_current_data_task():
    prepare_current_data()

def check_drift_task():
    drift_detected = check_for_drift_and_retrain(
        reference_data_path='data/reference_data.csv',
        current_data_path='data/current_data.csv',
        threshold=0.5
    )
    return drift_detected

prepare_current_data_operator = PythonOperator(
    task_id='prepare_current_data',
    python_callable=prepare_current_data_task,
    dag=dag,
)

check_drift_operator = PythonOperator(
    task_id='check_for_drift_and_retrain',
    python_callable=check_drift_task,
    dag=dag,
)

prepare_current_data_operator >> check_drift_operator
