## 🧠 Cancer Severity Prediction – End-to-End MLOps Project

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-red)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-yellow)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightblue)

---

## 🚀 Project Overview

This project is a complete end-to-end Machine Learning system designed to predict cancer severity levels based on patient data.

It demonstrates a full MLOps workflow including:
- Data preprocessing & feature engineering  
- Model training (Scikit-learn)  
- Experiment tracking with MLflow  
- Model versioning & logging  
- REST API deployment with Flask  
- Frontend interface with React  
- Docker containerization  

---

## 🎯 Problem Statement

Predict cancer severity into 3 classes:

- 0 → Low Risk  
- 1 → Medium Risk  
- 2 → High Risk  

Using features such as:
- Age  
- Smoking  
- Obesity level  
- Genetic risk  
- Cancer stage  

---

## 🏗️ System Architecture


Data → Preprocessing → Feature Engineering → Model Training → MLflow Tracking → Flask API → React Frontend → Docker Deployment

 
 
## 📊 Machine Learning Pipeline

### 🔹 Data Processing
- Missing value handling (SimpleImputer)  
- Feature scaling (StandardScaler)  
- Categorical encoding (OneHotEncoder)  
- ColumnTransformer pipeline  

### 🔹 Models Used
- Decision Tree Classifier  
- Random Forest Classifier (best performance)  

### 🔹 Evaluation Metrics
- Accuracy Score  
- Cross-validation Score  
- Confusion Matrix  

---

## 📈 MLflow Tracking

All experiments are tracked using MLflow for reproducibility and experiment management.

✔ Logged Items:
- Model parameters  
- Accuracy metrics  
- Cross-validation results  
- Confusion matrix artifact  
- Saved trained model  

🔬 Experiment Name: Cancer_Severity_Experiment  

---

## 🌐 Flask API (Model Serving)

The trained model is deployed using a Flask REST API for real-time predictions.

🔹 Endpoint:
POST /predict

🔹 Request Example:
{
  "Age": 45,
  "Smoking": 1,
  "Obesity_Level": 3,
  "Genetic_Risk": 2,
  "Cancer_Stage": "Stage II"
}

🔹 Response Example:
{
  "prediction": 2,
  "risk_level": "High Risk"
}

---

## 💻 React Frontend

The frontend provides a simple interface to:
- Input patient data  
- Send request to API  
- Display prediction results  

---

## 🐳 Docker Setup

📦 Services:
- Flask Backend (ML API)  
- React Frontend  

▶️ Run Project:
docker-compose up --build  

---

## 📁 Project Structure

project/
│
├── ml/
│   ├── train.py
│   ├── preprocessing.py
│
├── backend/
│   ├── app.py
│   ├── model_loader.py
│
├── frontend/
│   ├── React App
│
├── outputs/
│   └── confusion_matrix.png
│
├── mlruns/
├── docker-compose.yml
├── Dockerfile
└── README.md

---

## 📊 Results

- High accuracy achieved using Random Forest / Decision Tree  
- Full MLflow experiment tracking  
- Real-time predictions via REST API  
- End-to-end production ML pipeline  

---

## 💡 Key Skills Demonstrated

✔ Machine Learning Engineering  
✔ MLOps (MLflow, Experiment Tracking)  
✔ REST API Development (Flask)  
✔ Frontend Integration (React)  
✔ Docker & Deployment  
✔ End-to-end System Design  
