\# Breast Cancer Classification — End-to-End MLOps Pipeline



An end-to-end MLOps project for breast cancer classification, combining machine learning, MLflow experiment/model management, FastAPI model serving, Docker containerization, automated testing, and GitHub Actions CI.



The project demonstrates how a trained machine learning model can be transformed into a reproducible and testable API service using modern MLOps practices.



\---



\## Project Overview



This project builds a machine learning pipeline for breast cancer classification and exposes the trained model through a REST API.



The system follows an MLOps workflow:



```text

Dataset

&#x20;  │

&#x20;  ▼

Model Training

&#x20;  │

&#x20;  ▼

MLflow Tracking

&#x20;  │

&#x20;  ▼

MLflow Model Registry

&#x20;  │

&#x20;  ▼

Saved Model Artifact

&#x20;  │

&#x20;  ▼

FastAPI Prediction Service

&#x20;  │

&#x20;  ▼

Docker Container

&#x20;  │

&#x20;  ▼

Automated Tests

&#x20;  │

&#x20;  ▼

GitHub Actions CI



The goal is to demonstrate the complete lifecycle of a machine learning model from training to deployment-ready serving.



Architecture

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Breast Cancer     │

&#x20;                        │       Dataset       │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Model Training    │

&#x20;                        │      src/train.py   │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │       MLflow        │

&#x20;                        │ Experiment Tracking │

&#x20;                        │  \& Model Registry   │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ BreastCancerClassifier│

&#x20;                        │      Version 1      │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │    Model Artifact   │

&#x20;                        │ models/breast\_cancer│

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │      FastAPI        │

&#x20;                        │     src/app.py      │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                             REST API

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │       Docker        │

&#x20;                        │  breast-cancer-api  │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ Automated Testing   │

&#x20;                        │       pytest        │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │   GitHub Actions    │

&#x20;                        │        CI           │

&#x20;                        └─────────────────────┘

Technology Stack

Component	Technology

Programming Language	Python 3.11

Machine Learning	Scikit-learn

Experiment Tracking	MLflow

Model Serialization	skops

API Framework	FastAPI

API Server	Uvicorn

Containerization	Docker

Testing	pytest

CI/CD Automation	GitHub Actions

Version Control	Git / GitHub

Data Processing	Pandas

Project Structure

mlops-capstone/

│

├── .github/

│   └── workflows/

│       └── ci.yml

│

├── data/

│   └── breast\_cancer.csv

│

├── models/

│   └── breast\_cancer/

│       ├── MLmodel

│       ├── model.skops

│       ├── conda.yaml

│       ├── python\_env.yaml

│       ├── registered\_model\_meta

│       └── requirements.txt

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── train.py

│   └── app.py

│

├── tests/

│   ├── \_\_init\_\_.py

│   └── test\_api.py

│

├── .dockerignore

├── .gitignore

├── Dockerfile

├── mlflow.db

├── requirements.txt

└── README.md

Machine Learning Pipeline



The training pipeline is implemented in:



src/train.py



The training process includes:



Loading the breast cancer dataset.

Preparing the input features and target.

Training the machine learning pipeline.

Tracking the model using MLflow.

Registering the trained model.

Saving the model artifact.

Making the trained model available for serving.



The resulting registered model is:



Model Name: BreastCancerClassifier

Version: 1

Status: READY

MLflow



MLflow is used to manage the machine learning lifecycle.



The project uses MLflow for:



Experiment tracking

Model artifacts

Model metadata

Model registration

Model versioning



The local MLflow tracking database is:



mlflow.db



The registered model is:



BreastCancerClassifier



with the current model version:



1



The model artifact is stored under:



models/breast\_cancer/



The model is serialized using skops.



This provides a reproducible way to store and load the trained scikit-learn pipeline.



FastAPI Model Serving



The prediction API is implemented in:



src/app.py



The API uses FastAPI and Uvicorn to expose the machine learning model through HTTP endpoints.



Start the API locally



Make sure the virtual environment is activated:



.\\venv\\Scripts\\Activate.ps1



Then start the server:



uvicorn src.app:app --host 0.0.0.0 --port 8000



The API will be available on:



http://localhost:8000

API Endpoints

Health Check

Request

GET /

Example response

{

&#x20; "message": "Breast Cancer Classification API is running",

&#x20; "model": "BreastCancerClassifier",

&#x20; "version": "1"

}



This endpoint can be used to verify that the API is running and serving the expected model version.



Prediction

Request

POST /predict



The endpoint expects exactly 30 numerical feature values.



Request body

{

&#x20; "features": \[

&#x20;   17.99,

&#x20;   10.38,

&#x20;   122.8,

&#x20;   1001.0,

&#x20;   0.1184,

&#x20;   0.2776,

&#x20;   0.3001,

&#x20;   0.1471,

&#x20;   0.2419,

&#x20;   0.07871,

&#x20;   1.095,

&#x20;   0.9053,

&#x20;   8.589,

&#x20;   153.4,

&#x20;   0.006399,

&#x20;   0.04904,

&#x20;   0.05373,

&#x20;   0.01587,

&#x20;   0.03003,

&#x20;   0.006193,

&#x20;   25.38,

&#x20;   17.33,

&#x20;   184.6,

&#x20;   2019.0,

&#x20;   0.1622,

&#x20;   0.6656,

&#x20;   0.7119,

&#x20;   0.2654,

&#x20;   0.4601,

&#x20;   0.1189

&#x20; ]

}

Example response

{

&#x20; "prediction": 0,

&#x20; "model\_name": "BreastCancerClassifier",

&#x20; "model\_version": "1"

}



The API validates that the request contains exactly 30 feature values.



Interactive API Documentation



FastAPI automatically provides interactive API documentation.



After starting the application, open:



http://localhost:8000/docs



The Swagger UI can be used to:



View available endpoints

Inspect request schemas

Send prediction requests

View API responses

Docker



The application is containerized using Docker.



Build the Docker image

docker build -t breast-cancer-api .

Run the container

docker run -d --name breast-cancer-container -p 8000:8000 breast-cancer-api

Check running containers

docker ps

View container logs

docker logs breast-cancer-container



A successful startup should show:



Uvicorn running on http://0.0.0.0:8000

Test the Dockerized API



Once the container is running:



Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get



The API should return information about the running service and model.



The interactive API documentation is available at:



http://localhost:8000/docs

Testing



The project uses pytest for automated API testing.



The test suite is located at:



tests/test\_api.py



The current test suite validates:



API health/root endpoint

Prediction endpoint

Invalid feature count handling

Run tests locally

pytest -v



Expected result:



3 passed



Warnings may appear from third-party libraries, but they do not indicate test failures.



Continuous Integration



GitHub Actions is used to automatically validate changes pushed to the repository.



The CI workflow is:



.github/workflows/ci.yml



The pipeline performs the following steps:



Git Push

&#x20;  │

&#x20;  ▼

Checkout Repository

&#x20;  │

&#x20;  ▼

Set up Python 3.11

&#x20;  │

&#x20;  ▼

Install Dependencies

&#x20;  │

&#x20;  ▼

Run pytest

&#x20;  │

&#x20;  ▼

Build Docker Image



The workflow installs project dependencies using:



pip install -r requirements.txt



Then runs:



pytest -v



Finally, it validates that the Docker image can be built:



docker build -t breast-cancer-api .



This prevents broken code or dependency issues from being merged without detection.



Reproducibility



The project aims to make model serving reproducible by maintaining:



Python dependency definitions

MLflow model metadata

Versioned model artifacts

Docker configuration

Automated tests

CI validation



The model can therefore be loaded independently of the original training session and served through the FastAPI application.



MLOps Workflow



The complete workflow demonstrated by this project is:



1\. Data

&#x20;  ↓

2\. Model Development

&#x20;  ↓

3\. Model Training

&#x20;  ↓

4\. MLflow Experiment Tracking

&#x20;  ↓

5\. Model Registration

&#x20;  ↓

6\. Model Versioning

&#x20;  ↓

7\. Model Serialization

&#x20;  ↓

8\. FastAPI Model Serving

&#x20;  ↓

9\. Docker Containerization

&#x20;  ↓

10\. Automated API Testing

&#x20;  ↓

11\. Git Version Control

&#x20;  ↓

12\. GitHub Actions Continuous Integration

Key MLOps Concepts Demonstrated

Experiment Tracking



MLflow records information related to the machine learning workflow and model artifacts.



Model Registry



The trained model is registered as:



BreastCancerClassifier



and maintained using model versions.



Model Serving



FastAPI exposes the trained model through a REST API.



Containerization



Docker packages the API and its runtime environment into a portable container.



Automated Testing



pytest validates the API behavior automatically.



Continuous Integration



GitHub Actions automatically installs dependencies, executes tests, and builds the Docker image whenever changes are pushed.



Future Improvements



The current project establishes the CI stage of the MLOps lifecycle.



Potential future improvements include:



Continuous Deployment

Docker image publishing to a container registry

Cloud deployment

Model monitoring

Data drift detection

Model performance monitoring

Automated model retraining

MLflow server deployment

API authentication

Prediction logging

Production health monitoring

Project Status

Model Training          ✅

MLflow Tracking         ✅

Model Registry          ✅

Model Serialization     ✅

FastAPI                 ✅

Docker                  ✅

API Testing             ✅

GitHub                  ✅

GitHub Actions CI       ✅

Continuous Deployment   🔜

Model Monitoring        🔜

Author



Deepika R



Information Technology Undergraduate Student



Interested in Software Engineering, Machine Learning, Data Structures \& Algorithms, and MLOps.





\---



\## 2. Save it



In Notepad:



\*\*Ctrl + S\*\*



Close Notepad.



Then check:



```powershell

git status

