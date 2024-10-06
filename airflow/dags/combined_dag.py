from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os

from src.pipelines.data_ingestion import run_data_ingestion
from src.pipelines.data_validation import run_data_validation
from src.pipelines.data_transformation import run_data_transformation
from src.pipelines.model_training import run_model_training
from src.pipelines.model_evaluation import run_model_evaluation

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
    'combined_pipeline_dag',
    default_args=default_args,
    description='DAG for ingestion, validation, transformation, training, and evaluation',
    schedule_interval=timedelta(days=1),
)

# Define pipeline tasks
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

data_ingestion_task >> data_validation_task >> data_transformation_task >> model_training_task >> model_evaluation_task
