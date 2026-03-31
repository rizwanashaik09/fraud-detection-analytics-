# 🛡️ Fraud Detection Analytics using Machine Learning

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using Python. The project includes data preprocessing, model training, evaluation, and an interactive browser-based dashboard.

## 📌 Project Overview

Credit card fraud is a critical issue in financial systems. With millions of daily transactions and only a small fraction being fraudulent, detecting anomalies accurately is challenging.

This project builds a robust ML-based solution to identify fraudulent transactions efficiently.

- **Dataset:** Synthetic dataset inspired by the Kaggle Credit Card Fraud dataset (10,000 transactions, 200 fraud cases)
- **Models Used:** Logistic Regression, Random Forest Classifier
- **Best Accuracy:** 99.80%
- **ROC-AUC Score:** 0.9866

## 🚀 Features

- Full data preprocessing and feature scaling
- Machine Learning model training and comparison
- Interactive browser dashboard with animations
- Hover tooltips for better data insights
- Count-up animations for key metrics
- Dark-themed professional UI
- Visualizations:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance
  - Transaction Amount Analysis

## 📊 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 99.80% | 0.9889 |
| Random Forest | 99.80% | 0.9866 |

## 🛠️ Tech Stack

- **Python Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Frontend Visualization:** Chart.js  

## ⚙️ How to Run

### 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

### 2. Run the project
python fraud_detection_analytics.py

### 3. View the dashboard
The interactive dashboard will automatically open in your browser.

## 🧠 Key Concepts

- **Class Imbalance:** Only 2% transactions are fraudulent, making detection challenging  
- **Feature Scaling:** StandardScaler used to normalize features  
- **Random Forest:** Ensemble learning using multiple decision trees  
- **ROC-AUC Score:** Measures model performance (closer to 1 = better)  
- **Confusion Matrix:** Evaluates classification performance  

## 🌐 Live Interactive Dashboard

👉 https://rizwanashaik09.github.io/fraud-detection-analytics/fraud_detection_dashboard.html

## 👩‍💻 Author

**Rizwana Shaik**  
Built using Python and Scikit-learn
