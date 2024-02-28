#!/bin/bash
#SBATCH --job-name=r2_eval_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=40G

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

# CHECKPOINT_FULL_DIR=/data/jupiter/alex.li/models
# CHECKPOINT=22263_bencomp.ckpt
# RUN_ID=22263_ben_compare
CHECKPOINT_FULL_DIR=/mnt/sandbox1/ben.cline/output/bc_sandbox_2023/ds_v6_1_4x_human
CHECKPOINT=bc_sandbox_2023_val_bestmodel.pth
RUN_ID=ben_ds_v6_1_4x_human

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

# echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
SAFETY_DATASETS=(
    halo_humans_on_path_test_v6_1_unofficial_cleaned
    halo_rgb_stereo_test_v6_1_unofficial_cleaned

)
for DATASET_NAME in ${SAFETY_DATASETS[@]}
do

    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path /home/alex.li/git/scripts/training/halo_8_class_pred.yml \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_test_v6_1_unofficial_cleaned \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/halo_rgb_stereo_test_v6_1_unofficial_cleaned \
        --states_to_save '' \
        --run-id $RUN_ID \
        --pred-tag $CHECKPOINT
done
#     # --metrics.gt_stop_classes_to_consider 'Vehicles' 'Humans' \
# #    --states_to_save 'false_negative'

# echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------


# PRODUCITIVY_DATASETS=(
#     Jupiter_20230803_HHH2_1400_1430
#     Jupiter_20230803_HHH2_2030_2100
#     Jupiter_20230814_HHH1_1415_1445
#     Jupiter_20230825_HHH1_1730_1800
#     Jupiter_20230926_HHH1_1815_1845
#     Jupiter_20230927_HHH1_0100_0130
#     Jupiter_20231007_HHH1_2350_0020
#     Jupiter_20231019_HHH6_1800_1830
#     Jupiter_20231026_HHH8_1515_1545
#     Jupiter_20231121_HHH2_1800_1830
# )
# for DATASET_NAME in ${PRODUCITIVY_DATASETS[@]}
# do
#     srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
#         --config_path /home/alex.li/git/scripts/training/halo_8_class_pred.yml \
#         --data.test_set.csv master_annotations.csv \
#         --data.test_set.dataset_path /data/jupiter/datasets/$DATASET_NAME \ # TODO data or data2
#         --ckpt_path $CHECKPOINT_FULL_PATH \
#         --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET_NAME \
#         --states_to_save 'false_positive' \
#         --run-id $RUN_ID \
#         --pred-tag $CHECKPOINT \
#         --inputs.with_semantic_label false \
#         --metrics.run_productivity_metrics
# done
