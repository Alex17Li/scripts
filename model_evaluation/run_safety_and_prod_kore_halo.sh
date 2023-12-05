#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

DUST_OUTPUT_PARAMS='{"dust_head_output":false}'
LABEL_MAP_FILE=\$EUROPA_DIR/dl/config/label_maps/eight_class_train_dust_light_as_sky_birds_as_driveable.csv
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/wandb/run-17680/files/
# CHECKPOINT=last.ckpt
CHECKPOINT_FULL_DIR=/home/alex.li/logs
CHECKPOINT=17902.ckpt

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

SAFETY_DATASETS=("halo_rgb_stereo_test_v6_0")
for DATASET in ${SAFETY_DATASETS[@]}
do
    echo ----------------------------RUN ON ${DATASET}-----------------------------------
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --inputs.input_mode RECTIFIED_RGB \
        --config_path kore/configs/options/seg_no_dust_head.yml \
        --data.test_set.csv_name master_annotations.csv \
        --data.test_set.dataset_path /data2/jupiter/datasets/$DATASET \
        --inputs.label.label_map_file $LABEL_MAP_FILE \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --metrics.use_depth_threshold true \
        --output_dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
        --states_to_save 'human_false_negative' 'human_false_positive' \
        --batch_size 32
done
echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------
