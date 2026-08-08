# Breast Cancer Classification — End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

An end-to-end MLOps project for breast cancer classification combining machine learning, MLflow experiment & model management, FastAPI model serving, Docker containerization, automated testing, and GitHub Actions CI.

---

## 📌 Project Overview

This repository demonstrates the complete lifecycle of a machine learning model—from training to production-ready deployment serving.

Dataset ➔ Model Training ➔ MLflow Tracking ➔ Model Registry ➔ Saved Artifact ➔ FastAPI Service ➔ Docker ➔ Pytest ➔ GitHub Actions CI
### 🏗 Architecture
                    ┌─────────────────────┐
                    │   Breast Cancer     │
                    │       Dataset       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Training    │
                    │      src/train.py   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       MLflow        │
                    │ Experiment Tracking │
                    │  & Model Registry   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ BreastCancerClassifier│
                    │      Version 1      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Model Artifact   │
                    │ models/breast_cancer│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │     src/app.py      │
                    └──────────┬──────────┘
                               │
                         REST API
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Docker        │
                    │  breast-cancer-api  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Automated Testing   │
                    │       pytest        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   GitHub Actions    │
                    │        CI           │
                    └─────────────────────┘

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Programming Language** | Python 3.11 |
| **Machine Learning** | Scikit-learn |
| **Experiment Tracking & Registry** | MLflow |
| **Model Serialization** | skops |
| **API Framework** | FastAPI + Uvicorn |
| **Containerization** | Docker |
| **Testing** | pytest |
| **CI/CD Automation** | GitHub Actions |
| **Data Processing** | Pandas |

---

## 📂 Project Structure

```text
mlops-capstone/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── breast_cancer.csv
├── models/
│   └── breast_cancer/
│       ├── MLmodel
│       ├── model.skops
│       ├── conda.yaml
│       ├── python_env.yaml
│       ├── registered_model_meta
│       └── requirements.txt
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── app.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── mlflow.db
├── requirements.txt
└── README.md
🚀 Quick Start1. Local API ServingActivate your virtual environment and start the Uvicorn server:PowerShell.\venv\Scripts\Activate.ps1
uvicorn src.app:app --host 0.0.0.0 --port 8000
API Base: http://localhost:8000Interactive Swagger Docs: http://localhost:8000/docs2. Docker ContainerizationBash# Build Docker image
docker build -t breast-cancer-api .

# Run container
docker run -d --name breast-cancer-container -p 8000:8000 breast-cancer-api

# Check running container & logs
docker ps
docker logs breast-cancer-container
🧪 Testing & CI PipelineRun unit tests locally using pytest:Bashpytest -v
GitHub Actions WorkflowOn every git push, the CI pipeline (.github/workflows/ci.yml) automatically:Checks out repository code.Sets up Python 3.11.Installs dependencies (requirements.txt).Runs automated tests via pytest.Verifies container buildability via docker build.📊 Project StatusMilestoneStatusModel Training✅MLflow Tracking & Registry✅FastAPI Serving✅Docker Containerization✅Automated API Testing✅GitHub Actions CI✅Continuous Deployment (CD)🔜Model & Data Drift Monitoring🔜👤 AuthorDeepika RInformation Technology Undergraduate StudentGitHub Profile
