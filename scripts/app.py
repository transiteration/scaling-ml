import argparse
from http import HTTPStatus
from typing import Dict

import ray
import uvicorn
from fastapi import FastAPI
from starlette.requests import Request

from scripts import evaluate, predict
from scripts.config import MLFLOW_TRACKING_URI, mlflow

app = FastAPI(title="Scaling-ML", description="Classify News Headlines.", version="0.1")


@app.get("/")
def home() -> Dict:
    """
    Home endpoint.

    Returns:
        Dict: A dictionary containing information about the response.
    """
    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}
    return response


@app.get("/run_id")
def run_id() -> Dict:
    """
    Run ID endpoint.

    Returns:
        Dict: A dictionary containing the run ID of MLflow experiment.
    """
    return {
        "run_id": args.run_id,
    }


@app.post("/evaluate/")
async def evaluate_(request: Request) -> Dict:
    """
    Evaluate endpoint.

    Args:
        request (Request): The HTTP request object containing the JSON payload.

    Returns:
        Dict: A dictionary containing the evaluation results of given dataset.
    """
    data = await request.json()
    results = evaluate.evaluate(run_id=args.run_id, dataset_loc=data.get("dataset"))
    return {"results": results}


@app.post("/predict/")
async def predict_(request: Request):
    """
    Predict endpoint.

    Args:
        request (Request): The HTTP request object containing the JSON payload.

    Returns:
        Dict: A dictionary containing the prediction results.
    """
    data = await request.json()
    results = predict.predict_category(run_id=args.run_id, headline=data.get("headline", ""), keywords=data.get("keywords", ""))

    for result in results:
        probabilities = result.get("probabilities")
        if probabilities:
            result["probabilities"] = {key: float(value) for key, value in probabilities.items()}

    for i, result in enumerate(results):
        pred = result["prediction"]
        prob = result["probabilities"]
        if prob[pred] < args.threshold:
            results[i]["prediction"] = "other"

    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="Run ID of MLflow experiment to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for class prediction.")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use.")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to use.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    uvicorn.run(app, host="0.0.0.0", port=8010)
