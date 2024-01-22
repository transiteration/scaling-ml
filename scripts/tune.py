import argparse
import datetime
import json

import ray
from ray import tune
from ray.train import CheckpointConfig, DataConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.logger.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

from scripts import data, train, utils
from scripts.config import MLFLOW_TRACKING_URI, logger


def tune_models(
    experiment_name: str = None,
    dataset_loc: str = None,
    initial_params: str = None,
    num_workers: int = 1,
    cpu_per_worker: int = 1,
    gpu_per_worker: int = 0,
    num_runs: int = 1,
    grace_period: int = 1,
    num_samples: int = None,
    num_epochs: int = 1,
    batch_size: int = 128,
    results_fp: str = None,
) -> ray.tune.result_grid.ResultGrid:
    """
    Hyperparameter tuning for Language models using Ray Tune.

    Args:
        experiment_name (str): Name of the experiment for this training workload
        dataset_loc (str): Path to the dataset to train
        initial_params (str): Initial parameters to use for tuning
        num_workers (int): Number of workers to use for training
        cpu_per_worker (int): Number of CPUs to use per worker
        gpu_per_worker (int): Number of GPUs to use per worker
        num_runs (int): Number of runs in tuning experiment
        grace_period (int): Minimum time units (e.g., epochs) for the grace period
        num_samples (int): Number of samples to use from the dataset
        num_epochs (int): Number of epochs to train for
        batch_size (int): Batch size of the dataset to train
        results_fp (str): JSON file path to save results

    Returns:
        ray.tune.result_grid.ResultGrid: Results of the tuning process
    """
    # Set up
    utils.set_seeds()
    train_loop_config = {}
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers, use_gpu=bool(gpu_per_worker), resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker}
    )

    # Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config.get("num_samples", None))
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
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")

    # Run configuration
    mlflow_callback = MLflowLoggerCallback(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment_name, save_artifact=True)
    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config)

    # Hyperparameters to start with
    initial_params = json.loads(initial_params)
    initial_params_format = [
        {
            "train_loop_config": {
                "dropout_p": initial_params["dropout_p"],
                "lr": initial_params["lr"],
                "lr_factor": initial_params["lr_factor"],
                "lr_patience": initial_params["lr_patience"],
            }
        }
    ]

    search_alg = HyperOptSearch(points_to_evaluate=initial_params_format)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    # Parameter space
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    # Scheduler
    scheduler = AsyncHyperBandScheduler(max_t=train_loop_config["num_epochs"], grace_period=grace_period)

    # Tune config
    tune_config = tune.TuneConfig(metric="val_loss", mode="min", search_alg=search_alg, scheduler=scheduler, num_samples=num_runs)

    # Tuner
    tuner = Tuner(trainable=trainer, run_config=run_config, param_space=param_space, tune_config=tune_config)

    # Tune
    results = tuner.fit()
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=best_trial.metrics["trial_id"]),
        "params": best_trial.config["train_loop_config"],
        "metrics": utils.dict_to_list(best_trial.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:
        utils.save_dict(d, results_fp)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default=None, help="Name of the experiment for this training workload.")
    parser.add_argument("--dataset_loc", default=None, help="Path to the dataset to train.")
    parser.add_argument("--initial_params", default=None, help="Initial parameters to use for tuning.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for training.")
    parser.add_argument("--cpu_per_worker", type=int, default=1, help="Number of CPUs to use per worker.")
    parser.add_argument("--gpu_per_worker", type=int, default=0, help="Number of GPUs to use per worker.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs in Tuning experiment.")
    parser.add_argument("--grace_period", type=int, default=1, help="Minimum time units (e.g., epochs) for the grace period.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from dataset.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of the dataset to train.")
    parser.add_argument("--results_fp", default=None, help="JSON file path to save results.")
    args = parser.parse_args()

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    tune_models(
        experiment_name=args.experiment_name,
        dataset_loc=args.dataset_loc,
        initial_params=args.initial_params,
        num_workers=args.num_workers,
        cpu_per_worker=args.cpu_per_worker,
        gpu_per_worker=args.gpu_per_worker,
        num_runs=args.num_runs,
        grace_period=args.grace_period,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        results_fp=args.results_fp,
    )
