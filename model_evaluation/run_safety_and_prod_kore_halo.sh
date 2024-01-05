#!/bin/bash
#SBATCH --job-name=kore_halo_test
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

DUST_OUTPUT_PARAMS='{"dust_head_output":false}'
LABEL_MAP_FILE=\$EUROPA_DIR/dl/config/label_maps/eight_class_train_dust_light_as_sky_birds_as_driveable.csv
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/wandb/run-17680/files/
# CHECKPOINT=last.ckpt
# CHECKPOINT_FULL_DIR=/home/alex.li/logs
# CHECKPOINT=17902.ckpt
# OUTPUT_MODEL_NAME=17902

CHECKPOINT_FULL_DIR=/mnt/sandbox1/pooja.mehta/jupiter/models/pmehta_2023/rgb_baseline_sample_a_v3_birds_driveable_train_v6_0_1130
CHECKPOINT=pmehta_2023_val_bestmodel.pth
OUTPUT_MODEL_NAME=pmehta_2023_val_bestmodel

# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/logs/driveable_terrain_model/example_halo_rgb
# CHECKPOINT=driveable_terrain_model_99_epoch_model.pth
# OUTPUT_MODEL_NAME=driveable_terrain_model_99_epoch_model

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

# echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
    --config_path kore/configs/options/seg_no_dust_head.yml kore/configs/options/halo_seg_prediction.yml \
    --data.test_set.csv_name master_annotations.csv \
    --data.test_set.dataset_path /data/jupiter/datasets/halo_vehicles_on_path_test_v6_1 \
    --inputs.label.label_map_file $LABEL_MAP_FILE \
    --ckpt_path $CHECKPOINT_FULL_PATH \
    --metrics.gt_stop_classes_to_consider 'Non-driveable' 'Trees_Weeds' 'Humans' 'Vehicles' \
    --output_dir /mnt/sandbox1/alex.li/introspection/$OUTPUT_MODEL_NAME/halo_vehicles_on_path_test_v6_1 \
    --states_to_save 'human_false_negative' 'false_negative' 'vehicle_false_negative'
srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
    --config_path kore/configs/options/seg_no_dust_head.yml kore/configs/options/halo_seg_prediction.yml \
    --data.test_set.csv_name master_annotations.csv \
    --data.test_set.dataset_path /data/jupiter/datasets/halo_humans_on_path_test_v6_1 \
    --inputs.label.label_map_file $LABEL_MAP_FILE \
    --ckpt_path $CHECKPOINT_FULL_PATH \
    --metrics.gt_stop_classes_to_consider 'Non-driveable' 'Trees_Weeds' 'Humans' 'Vehicles' \
    --output_dir /mnt/sandbox1/alex.li/introspection/$OUTPUT_MODEL_NAME/halo_humans_on_path_test_v6_1 \
    --states_to_save 'human_false_negative' 'false_negative' 'vehicle_false_negative'

# echo ----------------------------RUN_PRODUCTIVITY_COMPLETE-----------------------------------

# srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
#     --config_path kore/configs/options/seg_no_dust_head.yml kore/configs/options/halo_seg_prediction.yml \
#     --data.test_set.csv_name master_annotations.csv \
#     --data.test_set.dataset_path /data2/jupiter/datasets/halo_humans_on_path_v3 \
#     --inputs.label.label_map_file $LABEL_MAP_FILE \
#     --ckpt_path $CHECKPOINT_FULL_PATH \
#     --output_dir /mnt/sandbox1/alex.li/introspection/$OUTPUT_MODEL_NAME/halo_humans_on_path_v3 \
#     --states_to_save 'human_false_negative' 
# echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------
    # --metrics.gt_stop_classes_to_consider Humans \
