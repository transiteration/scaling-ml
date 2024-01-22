import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlparse

import numpy as np
import ray
from numpyencoder import NumpyEncoder
from ray.air import Result
from ray.train.torch.torch_checkpoint import TorchCheckpoint

from scripts.config import logger, mlflow
from scripts.data import Preprocessor
from scripts.model import LLM
from scripts.utils import collate_fn


class TorchPredictor:
    """
    This class used for making predictions using a PyTorch-based model.

    Parameters:
        preprocessor: The preprocessor used for data transformation
        model: The PyTorch model for making predictions

    Methods:
        __call__(batch): Make predictions on a batch of input data
        predict_proba(batch): Get class probabilities for a batch of input data
        get_preprocessor(): Get the preprocessor associated with the predictor
        from_checkpoint(checkpoint): Instantiate a TorchPredictor from a given checkpoint
    """

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch):
        results = self.model.predict(collate_fn(batch))
        return {"output": results}

    def predict_proba(self, batch):
        results = self.model.predict_proba(collate_fn(batch))
        return {"output": results}

    def get_preprocessor(self):
        return self.preprocessor

    @classmethod
    def from_checkpoint(cls, checkpoint):
        metadata = checkpoint.get_metadata()
        preprocessor = Preprocessor(class_to_index=metadata["class_to_index"])
        model = LLM.load(Path(checkpoint.path, "args.json"), Path(checkpoint.path, "model.pt"))
        return cls(preprocessor=preprocessor, model=model)


def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    """
    Format class probabilities into a dictionary.

    Args:
        prob (Iterable): Iterable containing class probabilities
        index_to_class (Dict): Mapping of indices to class names

    Returns:
        Dict: Formatted dictionary with class names as keys and corresponding probabilities as values
    """
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = item
    return d


def predict_proba(ds: ray.data.dataset.Dataset, predictor: TorchPredictor) -> List:
    """
    Perform batch prediction with probabilities.

    Args:
        ds (ray.data.dataset.Dataset): Input Ray Dataset for prediction
        predictor (TorchPredictor): TorchPredictor instance for making predictions

    Returns:
        List: List of dictionaries containing predictions and associated class probabilities
    """
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    outputs = preprocessed_ds.map_batches(predictor.predict_proba)
    y_prob = np.array([d["output"] for d in outputs.take_all()])
    results = []
    for i, prob in enumerate(y_prob):
        tag = preprocessor.index_to_class[prob.argmax()]
        results.append({"prediction": tag, "probabilities": format_prob(prob, preprocessor.index_to_class)})
    return results


def get_best_checkpoint(run_id: str) -> TorchCheckpoint:
    """
    Get the best checkpoint from an MLflow run.

    Args:
        run_id (str): MLflow run ID

    Returns:
        TorchCheckpoint: Best checkpoint from the specified run
    """
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path
    results = Result.from_path(artifact_dir)
    return results.best_checkpoints[0][0]


def predict_category(
    run_id: str = None,
    headline: str = None,
    keywords: str = None,
) -> Dict:
    """
    Predict tags for a news based on its headline and keyword.

    Args:
        run_id (str): ID of the specific MLflow run to load from
        headline (str): News headline
        keywords (str): News keywords

    Returns:
        Dict: Predicted tags and associated class probabilities
    """
    checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(checkpoint)
    sample_ds = ray.data.from_items([{"headline": headline, "keywords": keywords, "category": ""}])
    results = predict_proba(ds=sample_ds, predictor=predictor)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, help="ID of the specific MLflow run to load from.")
    parser.add_argument("--headline", default=None, help="News headline.")
    parser.add_argument("--keyword", default="", help="News keywords.")
    args = parser.parse_args()

    predict_category(run_id=args.run_id, headline=args.headline, keyword=args.keyword)
