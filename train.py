import os

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("assignment5-classifier")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)

        # Emit run ID so CI can persist it to model_info.txt.
        print(f"MLFLOW_RUN_ID={run.info.run_id}")
        print(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
