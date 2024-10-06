# Sentiment Aanalysis on Movie Reviews

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Requirements](#project-requirements)
3. [Installation Instruction](#installation-instruction)
    - [Clone the Repository](#clone-the-repository)
    - [Set Up a Virtual Environment](#set-up-a-virtual-environment)
    - [Install the Required Packages](#install-the-required-packages)
4. [Running the Project](#running-the-project)
    - [MLflow Tracking](#mlflow-tracking)
    - [Model Training](#model-training)
5. [Airflow Setup](#airflow-setup)
    - [Install Apache Airflow](#install-apache-airflow)
    - [Setup Airflow Database](#setup-airflow-database)
    - [Start Airflow Scheduler and Webserver](#start-airflow-scheduler-and-webserver)
6. [Model Deployment](#model-deployment)
7. [Screenshots](#screenshots)

## Project Overview
This project is designed to analyze sentiment from movie reviews using a BERT-based model. The pipeline includes data ingestion, preprocessing, model training, evaluation, and deployment, orchestrated with Apache Airflow. The project utilizes DVC for data versioning and model tracking, MLflow for experiment tracking, and FastAPI for serving the model through an API.

## Project Requirements

- Python 3.9.16, but latest version should also work, might be incompatible with libraries like airflow etc.
- Virtualenv or Conda for environment management 
- DVC for data version control
- MLflow for experiment tracking
- Apache Airflow for orchestrating workflows
- FastAPI for serving the model through an API
- AWS as our preferred cloud platform and MongoDB for data storage
- Other packages listed in requirements.txt

### Installation Instruction

1. **Clone the repository**:
   ```bash
   git clone https://github.com/leemaHmaid/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. **Set up a virtual environment**:

     ```bash
     python3 -m venv sent_venv
     source sent_venv/bin/activate
     ```
    - **Using Conda**

    ```bash
    python3 -m venv sent_venv
    source sent_venv/bin/activate
    ```
3. **Install the required packages**:

     ```bash
     pip install -r requirements.txt

     ```
### Running the Project

1. **MLflow Tracking**:

    - During training, MLflow is used to track experiments automatically. You need to ensure the MLflow tracking server is running:

    ```bash
    mlflow server --backend-store-uri file://$(pwd)/mlruns --default-artifact-root file://$(pwd)/mlruns --host 0.0.0.0 -p 5050
    ```
    You can access the MLflow UI to see the experiment metrics at `http://localhost:5050`

2. **Model Training**:

    - To initiate the entire workflow, run the `main.py` script. Ensure that the MLflow server is running to track all experiments
    ```bash
    python main.py
    ```

    This will execute the data ingestion, validation, transformation, model training, and evaluation processes consecutively. All generated logs will be saved in the `logs` folder, and the entire workflow will be tracked on the MLflow UI.


<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/linear_reg_error.png" width="540"\>
</p>

<p align="center">
    Description
</p>
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/linear_reg_pred.png" width="540"\>
</p>
<p align="center">
   Description
</p>

   ## Airflow Setup
Alternately, we can us Apache Airflow to manage the entire workflow of the project, from data ingestion to model training and evaluation for scheduling and orchestrating the different tasks of the project.
1. **Install Apache Airflow**:
    - First, install Apache Airflow using the following script:
    ```bash
    run chmod 777 install.sh
    run ./install.sh
    ```
    This will download airflow and initialize it. Make sure AIRFLOW_HOME is exported as variable in the terminal you use. Inside airflow.cfg set load_examples=False and run airflow db reset -y.

2. **Setup Airflow**
    - Initialize the Airflow database to set up the necessary tables and users:
    ```bash
    run chmod 777 setup.sh
    run ./setup.sh 
    ```
    This will initialize the airflow database, make sure the script contains the following: `airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin`

3. **Start Airflow Scheduler and Webserver**
    - Start the scheduler to manage your tasks and the webserver to access the Airflow UI:
    ```bash
    run chmod 777 scheduler.sh
    run ./scheduler.sh 

    run chmod 777 scheduler.sh
    run ./webserver.sh 
    ```
   You can access the Airflow UI at `http://localhost:8080` to view and manage DAGs (Directed Acyclic Graphs) for the project.

<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/linear_reg_error.png" width="540"\>
</p>

<p align="center">
    Description
</p>
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/linear_reg_pred.png" width="540"\>
</p>
<p align="center">
   Description
</p>

3. **Model Deployment**:

    - To serve the model, use FastAPI:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```
    This will launch the API, and you can make predictions by sending POST requests to http://`localhost:8000/predict`




## Contact
If you have any comments, suggestions or anything you'd like to be clarify on, feel free
to reach me via [email](mailto:pmensah@aimsammi.org) or let's connect on [LinkedIn](https://www.linkedin.com/in/prince-mensah/).



