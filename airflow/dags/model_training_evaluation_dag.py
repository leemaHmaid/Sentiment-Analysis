from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Adding src to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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
    'model_training_evaluation_dag',
    default_args=default_args,
    description='DAG for data transformation, model training, and evaluation',
    schedule_interval='@weekly',  # Run weekly
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

data_transformation_task >> model_training_task >> model_evaluation_task
