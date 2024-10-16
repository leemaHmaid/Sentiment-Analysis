# Configurations
# source mlops-lab1/bin/activate

export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.10.2
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install 'apache-airflow==2.10.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.9.txt"

# Install Airflow (may need to upgrade pip)
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Initialize DB (SQLite by default)
airflow db init

# Make sure AIRFLOW_HOME is exported as variable in the terminal you use
# Inside airflow.cfg set load_examples=False and run airflow db reset -y

#   File "/Users/princemensah/Desktop/sent_analysis_project/sent_venv/lib/python3.9/site-packages/airflow/providers/fab/__init__.py", line 37, in <module>
#     raise RuntimeError(
# RuntimeError: The package `apache-airflow-providers-fab:1.4.0` needs Apache Airflow 2.9.0+