from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Adding src to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.pipelines.data_ingestion import run_data_ingestion
from src.pipelines.data_validation import run_data_validation

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 5),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_ingestion_validation_dag',
    default_args=default_args,
    description='DAG for data ingestion and validation',
    schedule_interval='@daily',  # Run daily
)

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

data_ingestion_task >> data_validation_task
