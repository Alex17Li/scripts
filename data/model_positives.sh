#!/bin/bash
#SBATCH --job-name=model➕
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40000
#SBATCH --mail-type=ALL
##SBATCH --nodelist=stc01spplmdanl006

# Loads conda environment. Sets WANDB and BRT_ENV env variables
source ~/.bashrc
conda activate cvml

MASTER_CSV_PATH="${DATASET_DIR}/master_annotations.csv"
# DATASET_NAME=halo_images_for_train_implement_dust_puddle_small
DATASET_NAME=halo_vehicles_on_path_test_v6_1
CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/train_halo/20676_r2_rgb_bigdecay_biglr/checkpoints/epoch=49-val_loss=0.069474.ckpt


DATASET_PATH=/data/jupiter/datasets/$DATASET_NAME
OUTPUT_DIR="/mnt/sandbox1/${USER}/model_positives/$DATASET_NAME/run_prod"
OUTPUT_CSV="${OUTPUT_DIR}/output.csv"

cd /home/$USER/git/JupiterCVML

srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
    --config_path /home/alex.li/git/scripts/training/halo_7_class_pred.yml \
    --data.test_set.csv master_annotations.csv \
    --data.test_set.dataset_path $DATASET_PATH \
    --ckpt_path $CHECKPOINT_FULL_PATH \
    --output_dir $OUTPUT_DIR \
    --states_to_save 'false_positive' \
    --metrics.run-productivity-metrics \
    --inputs.with_semantic_label false
echo "Model inference completed"

# Runs Image similarity on model positives and creates facets
python europa/base/src/europa/dl/scripts/model_positives_local.py --model_output_csv_path $OUTPUT_CSV \
    --pipeline_outputs_base_path $OUTPUT_DIR \
    --pack_perception_output_csv_path $DATASET_PATH/master_annotations.csv \
    --cache_pp_output_path $DATASET_DIR

echo "Model positives run completed"
