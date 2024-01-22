#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir results

# Test code
export RESULTS_FILE=results/test_code_results.txt
python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Train
export EXPERIMENT_NAME="llm_workload"
export RESULTS_FILE=results/training_results.json
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python scripts/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --dataset_loc "$DATASET_LOC" \
    --train_loop_config "$TRAIN_LOOP_CONFIG" \
    --num_workers 1 \
    --cpu_per_worker 1 \
    --gpu_per_worker 1 \
    --num_epochs 1 \
    --batch_size 128 \
    --results_fp $RESULTS_FILE

# Get and save run ID
export RUN_ID=$(python -c "import os; from scripts import utils; d = utils.load_dict(os.getenv('RESULTS_FILE')); print(d['run_id'])")
export RUN_ID_FILE=results/run_id.txt
echo $RUN_ID > $RUN_ID_FILE  # used for serving later

# Evaluate
export RESULTS_FILE=results/evaluation_results.json
export TEST_DATASET_LOC="datasets/test.csv"
python scripts/evaluate.py \
    --run_id $RUN_ID \
    --dataset_loc $TEST_DATASET_LOC \
    --results_fp $RESULTS_FILE

#Test Model
RESULTS_FILE=results/test_model_results.txt
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE