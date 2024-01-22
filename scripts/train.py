import argparse
import datetime
import json
import tempfile
from typing import Tuple

import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertModel

from scripts import data, utils
from scripts.config import MLFLOW_TRACKING_URI, logger
from scripts.model import LLM


def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Perform a single training step for a PyTorch model on a Ray Dataset batch.

    Args:
        ds (Dataset): Ray Dataset for training
        batch_size (int): Batch size for training
        model (nn.Module): PyTorch model for training
        num_classes (int): Number of output classes
        loss_fn (torch.nn.modules.loss._WeightedLoss): Loss function for training
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights

    Returns:
        float: Cumulative loss for the training step
    """
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss


def eval_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
) -> Tuple[float, np.array, np.array]:
    """
    Perform a single evaluation step for a PyTorch model on a Ray Dataset batch.

    Args:
        ds (Dataset): Ray dataset for evaluation
        batch_size (int): Batch size for evaluation
        model (nn.Module): PyTorch model for evaluation
        num_classes (int): Number of output classes
        loss_fn (torch.nn.modules.loss._WeightedLoss): Loss function for evaluation

    Returns:
        Tuple[float, np.array, np.array]: Tuple containing evaluation loss, true labels, and predicted labels
    """
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict) -> None:
    """
    Training loop for a distributed training setup per worker.

    Args:
        config (dict): Dictionary containing hyperparameters for training

    Returns:
        None
    """
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # Get datasets
    utils.set_seeds()
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = LLM(llm=llm, dropout_p=dropout_p, embedding_dim=llm.config.hidden_size, num_classes=num_classes)
    model = train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Training
    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        with tempfile.TemporaryDirectory() as dp:
            if isinstance(model, DistributedDataParallel):
                model.module.save(dp=dp)
            else:
                model.save(dp=dp)
            metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss)
            checkpoint = Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)


def train_model(
    experiment_name: str = None,
    dataset_loc: str = None,
    train_loop_config: str = None,
    num_workers: int = 1,
    cpu_per_worker: int = 1,
    gpu_per_worker: int = 0,
    num_samples: int = None,
    num_epochs: int = 1,
    batch_size: int = 128,
    results_fp: str = None,
) -> ray.air.result.Result:
    """
    Training process for a Language Model using the specified configuration.

    Args:
        experiment_name (str): Name of the experiment for this training workload
        dataset_loc (str): Path to the dataset to train
        train_loop_config (str): JSON string containing parameters to use for training
        num_workers (int): Number of workers to use for training
        cpu_per_worker (int): Number of CPUs to use per worker
        gpu_per_worker (int): Number of GPUs to use per worker
        num_samples (int): Number of samples to use from the dataset
        num_epochs (int): Number of epochs to train for
        batch_size (int): Batch size of the dataset to train
        results_fp (str): JSON file path to save results

    Returns:
        ray.air.result.Result: Results of the training process
    """
    # Set up
    train_loop_config = json.loads(train_loop_config)
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers, use_gpu=bool(gpu_per_worker), resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker}
    )

    # Checkpoint config
    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment_name, save_artifact=True)

    # Run config
    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config)

    # Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="category", test_size=0.2)
    tags = train_ds.unique(column="category")
    train_loop_config["num_classes"] = len(tags)

    # Dataset config
    options = ray.data.ExecutionOptions(preserve_order=True)
    dataset_config = DataConfig(datasets_to_split=["train"], execution_options=options)

    # Preprocess
    preprocessor = data.Preprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    # Train
    results = trainer.fit()
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=results.metrics["trial_id"]),
        "params": results.config["train_loop_config"],
        "metrics": utils.dict_to_list(results.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:
        utils.save_dict(d, results_fp)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default=None, help="Name of the experiment for this training workload.")
    parser.add_argument("--dataset_loc", default=None, help="Path to the dataset to train.")
    parser.add_argument("--train_loop_config", default=None, help="JSON string containing parameters to use for training.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for training.")
    parser.add_argument("--cpu_per_worker", type=int, default=1, help="Number of CPUs to use per worker.")
    parser.add_argument("--gpu_per_worker", type=int, default=0, help="Number of GPUs to use per worker.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from the dataset.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of the dataset to train.")
    parser.add_argument("--results_fp", default=None, help="JSON file path to save results.")
    args = parser.parse_args()

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    train_model(
        experiment_name=args.experiment_name,
        dataset_loc=args.dataset_loc,
        train_loop_config=args.train_loop_config,
        num_workers=args.num_workers,
        cpu_per_worker=args.cpu_per_worker,
        gpu_per_worker=args.gpu_per_worker,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        results_fp=args.results_fp,
    )
