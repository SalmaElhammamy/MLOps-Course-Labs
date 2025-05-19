# Customer Churn Prediction Project

This project aims to predict customer churn using machine learning models. It is part of a research workflow leveraging MLflow for tracking experiments and managing model lifecycle.

---

## 🔍 Objective

To build and evaluate models that can predict whether a customer will churn (exit) based on historical customer data.

---

## 🧠 Models Used

We trained and compared the following models:

1. **Logistic Regression**  
   - Simple, interpretable baseline model  
   - Accuracy: ~73%

2. **XGBoost Classifier**  
   - Gradient boosting model known for high performance  
   - Accuracy: ~75%  
   - Registered in **Staging** for potential promotion after further testing

3. **Random Forest Classifier**  
   - Ensemble model with high generalization ability  
   - Accuracy: **~77%**  
   - Registered in **Production** due to best performance

---

## 📊 Model Comparison

| Model               | Accuracy | Status     |
|--------------------|----------|------------|
| Logistic Regression| 73%      | Experiment |
| XGBoost            | 75%      | Staging    |
| Random Forest      | 77%      | Production |

---

## 📁 Project Structure

- `train.py` – Handles preprocessing, training, evaluation, and MLflow logging  
- `data/Churn_Modelling.csv` – Dataset used  
- `mlruns/` – MLflow tracking directory  
- `confusion_matrix.png` – Visualized evaluation result  
- `README.md` – Project overview  

---

## 📦 MLflow Model Registry

Models are tracked, logged, and versioned using MLflow:

- **Production**: Random Forest  
- **Staging**: XGBoost  
- **Experiment**: Logistic Regression

---

## 🚀 How to Run

```bash
# Run Logistic Regression
python train.py logistic

# Run Random Forest
python train.py random_forest

# Run XGBoost
python train.py xgboost
