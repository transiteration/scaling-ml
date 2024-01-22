import json

import pytest
import utils

from scripts import train


@pytest.mark.training
def test_train_model(dataset_loc):
    experiment_name = utils.generate_experiment_name(prefix="test_train")
    train_loop_config = {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}
    result = train.train_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        train_loop_config=json.dumps(train_loop_config),
        num_workers=1,
        cpu_per_worker=10,
        gpu_per_worker=1,
        num_epochs=2,
        batch_size=128,
    )
    train_loss_list = result.metrics_dataframe.to_dict()["train_loss"]
    utils.delete_experiment(experiment_name=experiment_name)
    assert train_loss_list[0] > train_loss_list[1]
