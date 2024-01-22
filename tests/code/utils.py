import shutil
import uuid
from pathlib import Path

from scripts.config import EFS_DIR, mlflow


def generate_experiment_name(prefix: str = "test") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def delete_experiment(experiment_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    experiment_path = Path(EFS_DIR, "mlflow", experiment_id)
    shutil.rmtree(experiment_path)
