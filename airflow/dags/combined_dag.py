from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
import os

# Adding src to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.pipelines.data_ingestion import run_data_ingestion
from src.pipelines.data_validation import run_data_validation
from src.pipelines.data_transformation import run_data_transformation
from src.pipelines.model_training import run_model_training
from src.pipelines.model_evaluation import run_model_evaluation
from monitoring import run_drift_monitoring

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'combined_pipeline_with_monitoring',
    default_args=default_args,
    description='DAG for full model lifecycle including monitoring and retraining',
    schedule_interval='@daily',
    catchup=False,
)

# Define tasks
data_ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    dag=dag,
)

data_validation_task = PythonOperator(
    task_id='data_validation',
    python_callable=run_data_validation,
    dag=dag,
)

data_transformation_task = PythonOperator(
    task_id='data_transformation',
    python_callable=run_data_transformation,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    dag=dag,
)

def run_drift_monitoring_and_check(**kwargs):
    """
    Executes drift monitoring and decides whether to retrain the model.
    
    Returns:
        str: Task ID to follow based on drift detection ('model_retraining_task' or 'no_retraining_needed_task').
    """
    drift_detected = run_drift_monitoring()
    if drift_detected:
        return 'model_retraining_task'
    else:
        return 'no_retraining_needed_task'

drift_monitoring_task = BranchPythonOperator(
    task_id='check_drift_and_decide',
    python_callable=run_drift_monitoring_and_check,
    provide_context=True,
    dag=dag,
)

model_retraining_task = PythonOperator(
    task_id='model_retraining_task',
    python_callable=run_model_training,
    dag=dag,
)

no_retraining_needed_task = DummyOperator(
    task_id='no_retraining_needed_task',
    dag=dag,
)
data_ingestion_task >> data_validation_task >> data_transformation_task >> model_training_task >> model_evaluation_task >> drift_monitoring_task
drift_monitoring_task >> model_retraining_task
drift_monitoring_task >> no_retraining_needed_task
