import json

import pytest
import utils

from scripts import tune


@pytest.mark.training
def test_tune_models(dataset_loc):
    num_runs = 2
    experiment_name = utils.generate_experiment_name(prefix="test_tune")
    initial_params = {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}
    results = tune.tune_models(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        initial_params=json.dumps(initial_params),
        num_workers=1,
        cpu_per_worker=10,
        gpu_per_worker=1,
        num_runs=num_runs,
        grace_period=1,
        num_epochs=1,
        batch_size=128,
    )
    utils.delete_experiment(experiment_name=experiment_name)
    assert len(results.get_dataframe()) == num_runs
