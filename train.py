import os

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("assignment5-classifier")

    dataset_path = "data/penguins.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "Dataset not found at data/penguins.csv. Run `dvc pull` before training."
        )

    data = pd.read_csv(dataset_path)
    data = data.dropna()
    y = data["species"]
    X = pd.get_dummies(data.drop(columns=["species"]), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run() as run:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        
        # Intentionally hardcoding a low accuracy (< 0.85) to trigger a pipeline failure
        accuracy = 0.50 # accuracy_score(y_test, predictions)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("dataset", dataset_path)
        mlflow.log_metric("accuracy", accuracy)

        # Emit run ID so CI can persist it to model_info.txt.
        print(f"MLFLOW_RUN_ID={run.info.run_id}")
        print(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
