#!/bin/bash
#SBATCH --job-name=model_pos
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=40000

# Loads conda environment. Sets WANDB and BRT_ENV env variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML
MASTER_CSV_PATH="${DATASET_DIR}/master_annotations.csv"
DATASET_NAME=halo_images_for_train_implement_dust_puddle_small
CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_halo/20676_r2_rgb_bigdecay_biglr/checkpoints/epoch=49-val_loss=0.069474.ckpt


DATASET_PATH=/data/jupiter/datasets/$DATASET_NAME
OUTPUT_DIR="/mnt/sandbox1/${USER}/model_positives/${DATASET_NAME}_repro_bug"
OUTPUT_CSV="${OUTPUT_DIR}/output.csv"

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

# srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
#     --config_path /home/alex.li/git/scripts/training/halo_7_class_pred.yml \
#     --data.test_set.csv master_annotations.csv \
#     --data.test_set.dataset_path $DATASET_PATH \
#     --ckpt_path $CHECKPOINT_FULL_PATH \
#     --output_dir /mnt/sandbox1/alex.li/model_positives/$OUTPUT_MODEL_NAME/${DATASET_NAME}_repro_bug \
#     --metrics.run-productivity-metrics \
#     --inputs.with_semantic_label false

echo "Model inference completed"

# Runs Image similarity on model positives and creates facets
python europa/base/src/europa/dl/scripts/model_positives_local.py --model_output_csv_path $OUTPUT_CSV \
    --pipeline_outputs_base_path $OUTPUT_DIR \
    --pack_perception_output_csv_path $DATASET_PATH/master_annotations.csv \
    --cache_pp_output_path $DATASET_PATH

echo "Model positives run completed"
