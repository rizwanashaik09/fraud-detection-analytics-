# 🛡️ Fraud Detection Analytics using Machine Learning

A complete end-to-end Machine Learning project that detects fraudulent credit card transactions using Python. Features an interactive browser dashboard with animations and hover effects.

## 📌 Project Overview

Credit card fraud is a major problem — millions of transactions happen daily and only a tiny fraction are fraudulent. This project builds a smart system that can automatically detect suspicious transactions using Machine Learning.

- **Dataset:** Synthetic dataset mimicking the Kaggle Credit Card Fraud dataset (10,000 transactions, 200 fraud cases)
- **Models Used:** Logistic Regression & Random Forest Classifier
- **Best Accuracy:** 99.80%
- **ROC-AUC Score:** 0.9866 (near perfect)

## 🚀 Features

- ✅ Full data preprocessing and feature scaling
- ✅ Two ML models trained and compared
- ✅ Interactive browser dashboard with animations
- ✅ Hover tooltips on all charts
- ✅ Count-up animations on key metrics
- ✅ Dark themed professional UI
- ✅ Confusion matrix, ROC curve, Feature importance, Amount analysis

## 📊 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 99.80% | 0.9889 |
| Random Forest | 99.80% | 0.9866 |

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Static visualizations |
| `seaborn` | Advanced heatmaps |
| `scikit-learn` | ML models and evaluation |
| `Chart.js` | Interactive browser charts |

## ⚙️ How to Run

### 1. Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn

### 2. Run the project
python fraud_detection_analytics.py

### 3. View the dashboard
The interactive dashboard will automatically open in your browser!

## 🧠 Key Concepts

- **Class Imbalance** — Only 2% of transactions are fraud, making detection challenging
- **StandardScaler** — Normalizes features so no single feature dominates
- **Random Forest** — An ensemble of 100 decision trees voting together
- **ROC-AUC** — Measures how well the model separates fraud from legitimate (1.0 = perfect)
- **Confusion Matrix** — Shows True Positives, False Positives, True Negatives, False Negatives

## 👨‍💻 Author
Made with ❤️ using Python and scikit-learn
