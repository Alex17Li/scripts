#!/bin/bash
#SBATCH --job-name=r2_prod_eval
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=40G

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

RUN_ID=20676
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/highres/bc_sandbox_2024/repvit_M0.9_512/bc_sandbox_2024_val_bestmodel.pth
CHECKPOINT_FULL_PATH=/data/jupiter/alex.li/models/20676.ckpt

CHECKPOINT=$(basename $CHECKPOINT_FULL_PATH)

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

PROD_DATASETS=(
    Jupiter_20231019_HHH6_1615_1700
    Jupiter_20231121_HHH2_1800_1830
    Jupiter_20231026_HHH8_1515_1545
    Jupiter_20231007_HHH1_2350_0020
    Jupiter_20230926_HHH1_1815_1845
    Jupiter_20230927_HHH1_0100_0130
    Jupiter_20230814_HHH1_1415_1445
    Jupiter_20230825_HHH1_1730_1800
    Jupiter_20230803_HHH2_2030_2100
    Jupiter_20230803_HHH2_1400_1430
    Jupiter_20230803_HHH3_2115_2145
    Jupiter_20230720_HHH3_1805_1835
    Jupiter_20230823_HHH3_1815_1845
)
for DATASET in ${PROD_DATASETS[@]}
do
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path /home/alex.li/git/scripts/training/halo_7_class_pred.yml \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/$DATASET \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET \
        --states_to_save 'false_positive' \
        --metrics.run-productivity-metrics \
        --inputs.with_semantic_label false \
        --metrics.use-depth-threshold \
        --run-id $RUN_ID \
        --pred-tag $CHECKPOINT
done
