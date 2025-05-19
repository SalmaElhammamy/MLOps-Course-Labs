import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier

# -------------------------------
def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj, churn_min = churn_0, churn_1
    else:
        churn_maj, churn_min = churn_1, churn_0
    churn_maj_downsample = resample(churn_maj, n_samples=len(churn_min), replace=False, random_state=1234)
    return pd.concat([churn_maj_downsample, churn_min])

# -------------------------------
def preprocess(df):
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]

    data = df.loc[:, filter_feat]
    data_bal = rebalance(data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough"
    )

    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    return col_transf, X_train, X_test, y_train, y_test

# -------------------------------
def train(X_train, y_train, model_type="logistic"):
    if model_type == "logistic":
        # Changed parameters
        model = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1500)
    elif model_type == "random_forest":
        # Changed parameters
        model = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=3, random_state=42
        )
    elif model_type == "xgboost":
        # Changed parameters
        model = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    else:
        raise ValueError("Unsupported model_type. Use 'logistic', 'random_forest', or 'xgboost'.")
    
    model.fit(X_train, y_train)
    return model

# -------------------------------
def main(model_type="logistic"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn_prediction_experiment_v3")

    with mlflow.start_run(run_name=f"{model_type}_run"):
        df = pd.read_csv("data/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        mlflow.log_param("model_type", model_type)

        if model_type == "logistic":
            mlflow.log_param("C", 0.5)
            mlflow.log_param("max_iter", 1500)
        elif model_type == "random_forest":
            mlflow.log_param("n_estimators", 150)
        elif model_type == "xgboost":
            mlflow.log_param("n_estimators", 150)
            mlflow.log_param("max_depth", 5)


        model = train(X_train, y_train, model_type=model_type)

        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.set_tag("model_type", model_type)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0, 1])
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

# -------------------------------
if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    main(model_type)
