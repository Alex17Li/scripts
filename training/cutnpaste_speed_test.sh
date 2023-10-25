#!/bin/bash
#SBATCH --job-name=time_preprocessing
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

#--SBATCH --partition=cpu
source /home/$USER/.bashrc
conda activate cvml
cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CVML_PATH=/home/$USER/git/JupiterCVML
# EXP=14904
EXP=${SLURM_JOB_ID}
SNAPSHOT_DIR=/mnt/sandbox1/$USER
OUTPUT_DIR=${OUTPUT_PATH}/${EXP}
wandb disabled

# --lr-scheduler-name exponentiallr \
# --lr-scheduler-parameters '{"exponential_gamma": 0.96}' \
python -m dl.scripts.trainer \
    --csv-path /data/jupiter/datasets/Jupiter_train_v5_11_20230508//master_annotations_v481.csv \
    --data-dir /data/jupiter/datasets/Jupiter_train_v5_11_20230508/ \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
    --dust-output-params "{\"dust_seg_output\": true}" \
    --exp-name dust \
    --model-params "{\"activation\": \"relu\"}" \
    --optimizer adamw \
    --weight-decay 1e-5 \
    --learning-rate 5e-4 \
    --lr-scheduler-name cosinelr \
    --lr-scheduler-parameters '{"cosinelr_T_max": 40, "cosinelr_eta_min": 1e-6}' \
    --epochs 40 \
    --model brtresnetpyramid_lite12 \
    --early-stop-patience 999 \
    --batch-size 72 \
    --val-set-ratio 0.05 \
    --ignore_dust_with_stop_class \
    --losses '{"msl": 1.0, "tv": 1.0, "prodl": 0.1, "hardsoft_iq": 1.0}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.6,0.6,0.6,0.3,0.3], "fn_weight":[0.4,0.7,0.4,0.4,0.4,0.7,0.7], "class_weight":[1.0,2.0,1.0,1.0,2.0,10.0,5.0], "gamma":1.0, "use_pixel_count_mask": true, "normalize_class_weights": true, "depth_thresh": -1}' \
    --hardsoft-loss-params '{"class_weight": [0.2, 1.0], "focal_gamma": 1.0, "soft_weight": 0.5}' \
    --multiscalemixedloss-parameters '{"scale_weight":0.2, "dust_weight":0.1, "dust_scale_weight":0.02}' \
    --productivity-loss-params '{"depth_thresh": 0.35, "prob_thresh": 0.01}' \
    --night-model '{"use": false, "dark_pix_threshold": 10}' \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --resume-from-snapshot False \
    --restore-from /mnt/sandbox1/alex.li/dust/15154/dust_20_epoch_model.pth \
    --output-dir ${OUTPUT_DIR} \
    --color-jitter '{"use": false}' \
    --trivial-augment '{"use": true}' \
    --num-steps 3 \
    --cutnpaste-augmentations '{"Humans": {"sample_ratio": 0.40, "use_standing_human": true, "standing_min_pixels": 20, "standing_max_pixels": 100000, "use_laying_down_human": true, "laying_down_min_pixels": 50, "laying_down_max_pixels": 15000, "use_multi_human": true, "only_non_occluded": false, "blend_mode": "vanilla", "rotate_object": true, "rotate_degree": 30, "jitter_object": false, "jitter_range": 0.15, "depth_aware": false, "cutout_rate": 0.5, "max_cutout": 0.6},
                                "Tractors or Vehicles": {"sample_ratio": 0.01, "min_pixels": 3000, "max_pixels": 100000, "blend_mode": "vanilla", "rotate_object": false, "rotate_degree": 30, "jitter_object": false, "jitter_range": 0.15, "depth_aware": false, "cutout_rate": 0.0, "max_cutout": 0.6}}' \
    --weighted-sampling '{"birds": 1.0,
                    "mis_labeled_driveable": 0.0, "mis_labeled_driveable_pixels": 10000,
                    "tiny_humans": 0.3, "tiny_human_pixels": 30,
                    "tiny_vehicles": 0.3, "tiny_vehicle_pixels": 100,
                    "humans": 2.0, "human_pixels": [100, 5000],
                    "occluded_humans": 3.0,
                    "reverse_humans": 3.0,
                    "day_front_vehicles": 1.0,
                    "day_rear_vehicles": 1.0,
                    "night_front_vehicles": 2.0,
                    "night_rear_vehicles": 2.0,
                    "airborne_debris": 6.0}' \
    --run-id ${EXP};
# Run again to check how the cache did
python -m dl.scripts.trainer \
    --csv-path /data/jupiter/datasets/Jupiter_train_v5_11_20230508//master_annotations_v481.csv \
    --data-dir /data/jupiter/datasets/Jupiter_train_v5_11_20230508/ \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
    --dust-output-params "{\"dust_seg_output\": true}" \
    --exp-name dust \
    --model-params "{\"activation\": \"relu\"}" \
    --optimizer adamw \
    --weight-decay 1e-5 \
    --learning-rate 5e-4 \
    --lr-scheduler-name cosinelr \
    --lr-scheduler-parameters '{"cosinelr_T_max": 40, "cosinelr_eta_min": 1e-6}' \
    --epochs 40 \
    --model brtresnetpyramid_lite12 \
    --early-stop-patience 999 \
    --batch-size 72 \
    --val-set-ratio 0.05 \
    --ignore_dust_with_stop_class \
    --losses '{"msl": 1.0, "tv": 1.0, "prodl": 0.1, "hardsoft_iq": 1.0}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.6,0.6,0.6,0.3,0.3], "fn_weight":[0.4,0.7,0.4,0.4,0.4,0.7,0.7], "class_weight":[1.0,2.0,1.0,1.0,2.0,10.0,5.0], "gamma":1.0, "use_pixel_count_mask": true, "normalize_class_weights": true, "depth_thresh": -1}' \
    --hardsoft-loss-params '{"class_weight": [0.2, 1.0], "focal_gamma": 1.0, "soft_weight": 0.5}' \
    --multiscalemixedloss-parameters '{"scale_weight":0.2, "dust_weight":0.1, "dust_scale_weight":0.02}' \
    --productivity-loss-params '{"depth_thresh": 0.35, "prob_thresh": 0.01}' \
    --night-model '{"use": false, "dark_pix_threshold": 10}' \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --resume-from-snapshot False \
    --restore-from /mnt/sandbox1/alex.li/dust/15154/dust_20_epoch_model.pth \
    --output-dir ${OUTPUT_DIR} \
    --color-jitter '{"use": false}' \
    --trivial-augment '{"use": true}' \
    --num-steps 3 \
    --cutnpaste-augmentations '{"Humans": {"sample_ratio": 0.40, "use_standing_human": true, "standing_min_pixels": 20, "standing_max_pixels": 100000, "use_laying_down_human": true, "laying_down_min_pixels": 50, "laying_down_max_pixels": 15000, "use_multi_human": true, "only_non_occluded": false, "blend_mode": "vanilla", "rotate_object": true, "rotate_degree": 30, "jitter_object": false, "jitter_range": 0.15, "depth_aware": false, "cutout_rate": 0.5, "max_cutout": 0.6},
                                "Tractors or Vehicles": {"sample_ratio": 0.01, "min_pixels": 3000, "max_pixels": 100000, "blend_mode": "vanilla", "rotate_object": false, "rotate_degree": 30, "jitter_object": false, "jitter_range": 0.15, "depth_aware": false, "cutout_rate": 0.0, "max_cutout": 0.6}}' \
    --weighted-sampling '{"birds": 1.0,
                    "mis_labeled_driveable": 0.0, "mis_labeled_driveable_pixels": 10000,
                    "tiny_humans": 0.3, "tiny_human_pixels": 30,
                    "tiny_vehicles": 0.3, "tiny_vehicle_pixels": 100,
                    "humans": 2.0, "human_pixels": [100, 5000],
                    "occluded_humans": 3.0,
                    "reverse_humans": 3.0,
                    "day_front_vehicles": 1.0,
                    "day_rear_vehicles": 1.0,
                    "night_front_vehicles": 2.0,
                    "night_rear_vehicles": 2.0,
                    "airborne_debris": 6.0}' \
    --run-id ${EXP};
