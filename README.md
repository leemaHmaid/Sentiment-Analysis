# Sentiment Aanalysis on Movie Reviews

## Table of Contents

- [Sentiment Aanalysis on Movie Reviews](#sentiment-aanalysis-on-movie-reviews)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [System Design Overview](#system-design-overview)
  - [Project Requirements](#project-requirements)
    - [Installation Instruction](#installation-instruction)
    - [Running the Project](#running-the-project)
  - [Airflow Setup](#airflow-setup)
  - [Running the Application](#running-the-application)
    - [Run the Application Locally](#run-the-application-locally)
      - [Important Note](#important-note)
  - [Contact](#contact)

## Project Overview

This project is designed to analyze sentiment from movie reviews using a BERT-based model. The pipeline includes data ingestion, preprocessing, model training, evaluation, and deployment, orchestrated with Apache Airflow. The project utilizes DVC for data versioning and model tracking, MLflow for experiment tracking, and FastAPI for serving the model through an API.

For more information on the product design and vision, you can refer to the [**Sentiment Aanalysis Product Design Template**](https://drive.google.com/file/d/18RES-XwarehqIulJUXCvlwDqdowN9Ojf/view?usp=sharing).

## System Design Overview

Below is a high-level architecture diagram of the **Sentiment Aanalysis**. This diagram provides an overview of how the components are structured, including the **web client**, **API server**, **ML core template**, **REST API template**, and the **deployment pipeline** for hosting the application in a cloud environment.

![System Design](img/system-design.png)

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

   This will execute the data ingestion, validation, transformation, model training, and evaluation processes consecutively. All generated logs will be saved in the `logs` folder, and the entire workflow will be tracked on the MLflow UI. Here is how it looks like (from our run):

<p align="center">
    <img src="https://github.com/leemaHmaid/Sentiment-Analysis/blob/development/img/mlflow_ui.png" width="7500"\>
</p>

<p align="center">
</p>
<p align="center">
    <img src="https://github.com/leemaHmaid/Sentiment-Analysis/blob/development/img/mlflow1.png" width="7500"\>
</p>
<p align="center">
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

   You can access the Airflow UI at `http://localhost:8080` to view and manage DAGs (Directed Acyclic Graphs) for the project. Here is how it looks like (from our run):

<p align="center">
    <img src="https://github.com/leemaHmaid/Sentiment-Analysis/blob/development/img/aiflow_ui.png" width="750"\>
</p>

<p align="center">
</p>
<p align="center">
    <img src="https://github.com/leemaHmaid/Sentiment-Analysis/blob/development/img/airflow1.png" width="750"\>
</p>
<p align="center">
</p>

## Running the Application

### Run the Application Locally

1. **Set Up Environment Variables**
   To run the application, you'll need to set up some environment variables. You can create a .env file in the root of your project with the following placeholders:

   ```bash
   # .env file

   MONGO_DB_URL=mongodb://your-mongo-url

   TOKEN_SECRET_KEY=your-secret-key

   ACCESS_TOKEN_EXPIRE_MINUTES=30

   # Base URL for the API service
   API_BASE_URL=http://127.0.0.1:8000
   ```

   **Note**: Make sure to replace the placeholders with your actual values. The `API_BASE_URL` should match the URL you use when running the API service. For local development, it should be set to http://127.0.0.1:8000.

   #### Important Note

   First of all, to run anything related to the API, navigate to the backend directory:

   ```bash
   cd backend
   ```

1. **Run the Setup Script**: Before starting the server, run the `setup.sh` script in the `scripts` folder to create dummy users in the database:
   ```bash
   sh scripts/setup.sh
   ```
1. **Start the FastAPI Server**: To run the API, execute the `server.sh` script in the `scripts` folder:

   ```bash
   sh scripts/server.sh
   ```

   The API should now be running at `http://127.0.0.1:8001`. Navigate to `http://127.0.0.1:8001/docs` and you'll see an interface like this:

   <p align="center">
    <img src="https://github.com/leemaHmaid/Sentiment-Analysis/blob/development/img/api.png" width="7500"\>
</p>

1. **Run the Streamlit App**: Navigate to the frontend directory and use the following command:
   ```bash
   cd frontend
   streamlit run app_ui.py
   ```
1. **Access the Application**: Open your web browser and go to `http://localhost:8501`. You see a ui like this when you click on register:

<p align="center">
    <img src="https://github.com/leemaHmaid/Sentiment-Analysis/blob/development/img/api_ui.png" width="7500"\>
</p>

## Contact

If you have any comments, suggestions or anything you'd like to be clarify on, feel free
to reach us via email [Prince](mailto:pmensah@aimsammi.org), [Leema](mailto:lhamid@aimsammi.org), [Asim](mailto:amohamed@aimsammi.org) or let's connect on linkedin, [Leema](https://www.linkedin.com/in/leema-hamid/),[Prince](https://www.linkedin.com/in/prince-mensah/), [Asim](https://www.linkedin.com/in/asim-mohamed-9a2047135/).
