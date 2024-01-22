---
license: mit
tags:
- pytorch
- mlflow
- ray
- fastapi
- nlp
---
## Scaling-ML
Scaling-ML is a project that classifies news headlines into 10 groups.
The main part of the project fine-tuning of the [BERT](https://huggingface.co/allenai/scibert_scivocab_uncased) model and including tools like MLflow for tracking experiments, Ray for scaling and distibuted computing, and MLOps components for seamless management of machine learning workflows.

### Set Up

1. Clone the repository:
```bash
git clone https://github.com/your-username/scaling-ml.git
cd scaling-ml
```
2. Set up your virtual environment and install dependencies:
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
pip install -r requirements.txt
```
### Scripts Overview
```bash
scripts
├── app.py
├── config.py
├── data.py
├── evaluate.py
├── model.py
├── predict.py
├── train.py
├── tune.py
└── utils.py
```
- `app.py` - Implementation of FastAPI web service for serving a model.
- `config.py` - Configuration of logging settings, directory structures, and MLflow registry.
- `data.py`- Functions and a class for data preprocessing tasks in a scalable machine learning project.
- `evaluate.py` - Evaluating the performance of a model, calculating precision, recall and F1 score.
- `model.py` - Finetuned language model by adding a fully connected layer for classification tasks.
- `predict.py` - TorchPredictor class for making predictions using a PyTorch-based model.
- `train.py` - Training process using Ray for distributed training.
- `tune.py` -  Hyperparameter tuning for Language Model using Ray Tune.
- `utils.py` - Various utility functions for handling data, setting random seeds, saving and loading dictionaries, etc.
#### Dataset
For training, small portion of the [News Category Dataset](https://www.kaggle.com/datasets/setseries/news-category-dataset) was used, which contains numerous headlines and descriptions of various articles.

### How to Train
```bash
export DATASET_LOC="path/to/dataset"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 5}'
python3 scripts/train.py \
--experiment_name "llm_train" \
--dataset_loc $DATASET_LOC \
--train_loop_config "$TRAIN_LOOP_CONFIG" \
--num_workers 1 \
--cpu_per_worker 1 \
--gpu_per_worker 0 \
--num_epochs 1 \
--batch_size 128 \
--results_fp results.json 
```
- experiment_name: A name for the experiment or run, in this case, "llm".
- dataset_loc: The location of the training dataset, replace with the actual path.
- train_loop_config: The configuration for the training loop, replace with the actual configuration.
- num_workers: The number of workers used for parallel processing. Adjust based on available CPU resources.
- cpu_per_worker: The number of CPU cores assigned to each worker. Adjust based on available CPU resources.
- gpu_per_worker: The number of GPUs assigned to each worker. Adjust based on available GPU resources.
- num_epochs: The number of training epochs.
- batch_size: The batch size used during training.
- results_fp: The file path to save the results.

### How to Tune
```bash
export DATASET_LOC="path/to/dataset"
export INITIAL_PARAMS='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 5}'
python3 scripts/tune.py \
--experiment_name "llm_tune" \
--dataset_loc "$DATASET_LOC" \
--initial_params "$INITIAL_PARAMS" \
--num_workers 1 \
--cpu_per_worker 1 \
--gpu_per_worker 0 \
--num_runs 1 \
--grace_period 1 \
--num_epochs 1 \
--batch_size 128 \
--results_fp results.json 
```
- num_runs: The number of tuning runs to perform.
- grace_period: The grace period for early stopping during hyperparameter tuning.

**Note**: modify the values of the `--num-workers`, `--cpu-per-worker`, and `--gpu-per-worker` input parameters below according to the resources available on your system.

### Experiment Tracking with MLflow
```bash
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri /path/to/mlflow/folder
```

### Evaluation
```bash
export RUN_ID=YOUR_MLFLOW_EXPERIMENT_RUN_ID
python3 evaluate.py --run_id $RUN_ID --dataset_loc "path/to/dataset" --results_fp results.json
```
```json
{                                                                                                                                                                                                           
  "timestamp": "January 22, 2024 09:57:12 AM",
  "precision": 0.9163323229539818,
  "recall": 0.9124083769633508,
  "f1": 0.9137224104301406,
  "num_samples": 1000.0
}
```
- run_id: ID of the specific MLflow run to load from.
### Inference
```
python3 predict.py --run_id $RUN_ID --headline "Airport Guide: Chicago O'Hare" --keyword "destination" 
```
```json
[
  {
    "prediction": "TRAVEL",
    "probabilities": {
      "BUSINESS": 0.0024151806719601154,
      "ENTERTAINMENT": 0.002721842611208558,
      "FOOD & DRINK": 0.001193400239571929,
      "PARENTING": 0.0015436559915542603,
      "POLITICS": 0.0012392215430736542,
      "SPORTS": 0.0020724297501146793,
      "STYLE & BEAUTY": 0.0018642042996361852,
      "TRAVEL": 0.9841892123222351,
      "WELLNESS": 0.0013303911546245217,
      "WORLD NEWS": 0.0014305398799479008
    }
  }
]
```
### Application
```bash
python3 app.py --run_id $RUN_ID --num_cpus 2
```
Now, we can send requests to our application:
```python
import json
import requests
headline = "Reboot Your Skin For Spring With These Facial Treatments"
keywords = "skin-facial-treatments"
json_data = json.dumps({"headline": headline, "keywords": keywords})
out = requests.post("http://127.0.0.1:8010/predict", data=json_data).json()
print(out["results"][0])
```
```json
{
  "prediction": "STYLE & BEAUTY",
  "probabilities": {
      "BUSINESS": 0.002265132963657379,
      "ENTERTAINMENT": 0.008689943701028824,
      "FOOD & DRINK": 0.0011296054581180215,
      "PARENTING": 0.002621663035824895,
      "POLITICS": 0.002141285454854369,
      "SPORTS": 0.0017548275645822287,
      "STYLE & BEAUTY": 0.9760453104972839,
      "TRAVEL": 0.0024237297475337982,
      "WELLNESS": 0.001382972695864737,
      "WORLD NEWS": 0.0015455639222636819
}
```
### Testing the Code
How to test the written code for asserted inputs and outputs:
```bash
python3 -m pytest tests/code --verbose --disable-warnings
```
How to test the Model behaviour:
```bash
python3 -m pytest --run-id $RUN_ID tests/model --verbose --disable-warnings
```

### Workload
To execute all stages of this project with a single command, `workload.sh` script has been provided, change the resource(cpu_nums, gpu_nums, etc.) parameters to suit your needs.
```bash
bash workload.sh
```

### Extras
Makefile to clean caces from the directories and format scripts:
```bash
make style && make clean
```
Served documentation for functions and classes:
```bash
python3 -m mkdocs serve
```
