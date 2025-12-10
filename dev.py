import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    # Load dataset
    data = pd.read_csv("data/iris.csv")
    X = data.drop(columns=["species"])
    y = data["species"]

    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters
    params = {
        "C": 1.0,
        "max_iter": 200,
        "solver": "lbfgs",
        "multi_class": "auto"
    }

    # Train model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Evaluate model
    accuracy = lr.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(lr, "logistic_regression_model")

    print("Model training and logging complete.")
    