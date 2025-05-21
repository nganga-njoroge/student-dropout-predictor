# 🎯 Student Dropout Predictor

Predicting student dropout risk using demographic and academic features. This is a binary classification project designed to evaluate students likely to leave school early.

## 📊 Dataset
- Source: UCI / Kaggle (TBD)
- Features: study time, failures, absences, parental education, etc.
- Target: Dropout (Yes/No)

## 🛠 Tools & Stack
- Python, Pandas, Scikit-learn
- TensorFlow/Keras
- Matplotlib & Seaborn
- Imbalanced-learn (optional)

## 🗓️ Timeline
| Day | Focus                    |
|-----|--------------------------|
| 1   | Setup, load dataset, EDA |
| 2   | Feature analysis, encoding |
| 3   | Baseline classifiers     |
| 4   | Deep model + metrics     |
| 5   | Results + documentation  |

## 📁 Structure
- `data/`: CSV files
- `src/`: Scripts
- `models/`: Saved `.keras` models
- `outputs/`: Confusion matrix, ROC, logs
- `notebooks/`: EDA + evaluation

---

## ✅ Status: Ready to Start

## 🧠 Day 3 – Baseline Model Training

### ✅ Tasks Completed:
- Loaded and prepared the dataset using `prepare_data.py`
- Encoded categorical features and split data using stratified train-test split
- Trained two models:
    - Logistic Regression (linear baseline)
    - Decision Tree (nonlinear, interpretable)
- Evaluated both using:
    - Classification report (Precision, Recall, F1)
    - Confusion matrix

### 🔍 Key Observations:
- Dropout class is reasonably separable even with basic features
- Decision Tree may overfit — worth testing regularization
- Logistic Regression provides solid baseline for ROC and interpretability

### 🛠 Next Steps:
- Try Random Forest or XGBoost
- Tune hyperparameters
- Handle class imbalance (class weights or resampling)