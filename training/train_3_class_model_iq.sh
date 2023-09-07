#!/bin/bash
#SBATCH --job-name=train_3cls
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CVML_PATH=/home/$USER/git/JupiterCVML
EXP=seg_$SLURM_JOB_ID
# EXP=seg_12688
SNAPSHOT_DIR=/mnt/sandbox1/$USER
OUTPUT_DIR=${OUTPUT_PATH}/${EXP}

# --tqdm \
# --augmentations CustomCrop SmartCrop HorizontalFlip TorchColorJitter Resize \
# --restore-from /mnt/sandbox1/alex.li/results/dust/dust_trivial_augment_1/dust_val_bestmodel.pth \
# --trivial-augment '{"use": true}' \
# --color-jitter '{"use": false}' \
# --cutnpaste-augmentations "{}" \
# --tqdm \
# fl_iq : 0.4
python -m dl.scripts.trainer \
    --use-albumentation-transform \
    --batch-size 64 \
    --num-workers 16 \
    --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_11/epoch0_5_30_focal05_master_annotations.csv \
    --data-dir /data/jupiter/datasets/Jupiter_train_v5_11/ \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/three_class_train_iq.csv \
    --exp-name dust \
    --model-params "{\"activation\": \"relu\"}" \
    --optimizer adamw \
    --weight-decay 1e-5 \
    --learning-rate 1e-3 \
    --lr-scheduler-name exponentiallr \
    --lr-scheduler-parameters '{"exponential_gamma": 0.95}' \
    --epochs 1 \
    --model brtresnetpyramid_lite12 \
    --early-stop-patience 12 \
    --resume-from-snapshot True \
    --val-set-ratio 0.05 \
    --losses '{"msl": 1.0, "tv": 1.0, "prodl": 0.02, "fl_iq": 0.2}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.6], "fn_weight":[0.4,0.7,0.4], "class_weight":[1.0,2.0,1.0], "gamma":1.0}' \
    --productivity-loss-params '{"depth_thresh": 0.35, "prob_thresh": 0.01}' \
    --night-model '{"use": false, "dark_pix_threshold": 10}' \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --weighted-sampling '{"birds": 1.0,
                        "tiny_humans": 0.0, "tiny_human_pixels": 30,
                        "tiny_vehicles": 0.0, "tiny_vehicle_pixels": 100,
                        "humans": 1.0, "human_pixels": [100, 5000],
                        "occluded_humans": 5.0, "occluded_human_pixels": [100, 2000],
                        "reverse_humans": 5.0, "reverse_human_pixels": [50, 2000],
                        "triangle_humans": 5.0, "triangle_human_pixels": [50, 2000],
                        "day_vehicles": 2.0, "day_vehicle_pixels": [3000, 100000],
                        "night_vehicles": 5.0, "night_vehicle_pixels": [3000, 100000],
                        "airborne_debris": 2.0, "airborne_debris_pixels": [100, 100000]}' \
    --run-id ${EXP};

# https://us-east-2.console.aws.amazon.com/s3/object/blueriver-jupiter-data?region=us-west-2&prefix=model_training/dust_trivial_augment_1/dust_val_bestmodel.pth
# aws s3 cp /mnt/sandbox1/alex.li/dust/dust_trivial_augment_1/dust_val_bestmodel.pth s3://blueriver-jupiter-data/model_training/dust_trivial_augment_1/dust_val_bestmodel.pth