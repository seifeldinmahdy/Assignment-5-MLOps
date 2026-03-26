# Assignment-5-MLOps

[![CI/CD Pipeline](https://github.com/seifeldinmahdy/Assignment-5-MLOps/actions/workflows/mlops-pipeline.yml/badge.svg)](https://github.com/seifeldinmahdy/Assignment-5-MLOps/actions/workflows/mlops-pipeline.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-orange)](https://dvc.org/)

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Data Tracking with DVC](#data-tracking-with-dvc)
  - [Model Training](#model-training)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Project Structure](#-project-structure)

---

## Getting Started

### Prerequisites

- Python >= 3.10
- Git
- pip / virtualenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/seifeldinmahdy/Assignment-5-MLOps.git
   cd Assignment-5-MLOps
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Tracking with DVC

This project uses the public `penguins.csv` dataset from seaborn-data. Metadata is handled by DVC.

Fetch the dataset locally before your first training run:
```bash
dvc pull
```

*Note: The actual dataset isn't stored in GitHub. DVC knows how to fetch it precisely from the source.*

### Model Training

Execute the robust training script. This will output a run ID and an accuracy score:
```bash
python train.py
```
You can view the resulting runs in the MLflow UI by running:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## CI/CD Pipeline

The GitHub Action located at `.github/workflows/mlops-pipeline.yml` executes on every `push` to the `main` branch. 

- **Job 1: Validate Model**
  - Synthesizes the environment and pulls data via DVC.
  - Trains the model.
  - Uploads the SQLite tracking backend and Run ID as build artifacts.
  
- **Job 2: Deploy Model** (Depends on `Validate`)
  - Downloads the MLflow states.
  - Validates if the saved run achieved an accuracy of strictly &ge; 0.85.
  - Builds a mock Docker image carrying the trained weights/Run ID upon successful validation.

---

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml   # CI/CD orchestration
├── data/
│   ├── .gitignore
│   └── penguins.csv.dvc         # DVC tracking metadata for the dataset
├── check_threshold.py           # Evaluation gatekeeper script
├── Dockerfile                   # Mock container for final model packaging
├── requirements.txt             # Environment dependencies 
├── train.py                     # ML pipeline, model training, and MLflow logging
└── README.md                    # Project documentation
```