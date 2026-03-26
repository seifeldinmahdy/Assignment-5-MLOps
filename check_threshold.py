import argparse
import os
import sys

from mlflow.tracking import MlflowClient


def read_run_id(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()
    if not run_id:
        raise ValueError(f"No run ID found in {path}")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-info", default="model_info.txt")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    run_id = read_run_id(args.model_info)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(run_id)
    value = run.data.metrics.get(args.metric)
    if value is None:
        print(
            f"Metric '{args.metric}' was not logged for run {run_id}.",
            file=sys.stderr,
        )
        return 3

    print(f"Run ID: {run_id}")
    print(f"{args.metric}: {value:.4f}")
    print(f"Threshold: {args.threshold:.4f}")

    if value < args.threshold:
        print(
            f"Validation failed: {args.metric}={value:.4f} < {args.threshold:.4f}",
            file=sys.stderr,
        )
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
