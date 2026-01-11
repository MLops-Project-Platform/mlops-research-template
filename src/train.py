import os
import mlflow
import yaml
import time
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    cfg = load_config(config_path)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
    )

    model = LogisticRegression(
        max_iter=cfg["training"]["max_iter"],
        C=cfg["training"]["C"],
    )

    run_name = f"train-{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        # ---- params ----
        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": cfg["training"]["max_iter"],
            "C": cfg["training"]["C"],
            "test_size": cfg["training"]["test_size"],
        })

        # ---- tags (סטנדרט ארגוני) ----
        mlflow.set_tags({
            "project": cfg["meta"]["project"],
            "owner": cfg["meta"]["owner"],
            "stage": "research",
            "framework": "sklearn",
        })

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"✅ Finished run {run_name} | accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
