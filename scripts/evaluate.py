import argparse
import datetime
import json
from typing import Dict

import numpy as np
import ray
import ray.train.torch
from sklearn.metrics import precision_recall_fscore_support

from scripts import predict, utils
from scripts.config import logger
from scripts.predict import TorchPredictor


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute overall metrics based on true and predicted labels.

    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels

    Returns:
        dict: Overall metrics including precision, recall, F1 score, and the number of samples
    """
    metrics = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="weighted")
    overall_metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics


def evaluate(
    run_id: str = None,
    dataset_loc: str = None,
    results_fp: str = None,
) -> Dict:
    """Evaluate Model Performance.

    Args:
        run_id (str): ID of the specific MLflow run to load from
        dataset_loc (str): Path to dataset to evaluate on
        results_fp (str): JSON file path to save results

    Returns:
        dict: Overall evaluation metrics including precision, recall, F1 score, and the number of samples
    """
    ds = ray.data.read_csv(dataset_loc)
    ds = ray.data.from_items(ds.take(200))
    checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(checkpoint)

    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
    y_true = np.stack([item["targets"] for item in values])

    predictions = preprocessed_ds.map_batches(predictor).take_all()
    y_pred = np.array([d["output"] for d in predictions])
    metrics = get_overall_metrics(y_true=y_true, y_pred=y_pred)
    logger.info(json.dumps(metrics, indent=2))
    if results_fp:
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, help="ID of the specific MLflow run to load from.")
    parser.add_argument("--dataset_loc", default=None, help="Path to dataset to evaluate on.")
    parser.add_argument("--results_fp", default=None, help="JSON file path to save results.")
    args = parser.parse_args()
    ray.init(num_cpus=8, num_gpus=1)

    evaluate(
        run_id=args.run_id,
        dataset_loc=args.dataset_loc,
        results_fp=args.results_fp,
    )
