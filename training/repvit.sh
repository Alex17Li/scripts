#!/bin/bash
#SBATCH --job-name=brthighresnet
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=7 
#SBATCH --mem-per-gpu=50G
#SBATCH --exclude=stc01sppamxnl018,stc01sppamxnl005
#SBATCH --time=240:00:00

# module load anaconda/2022.05
# eval "$('/home/ben.cline/anaconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
# conda activate /mnt/sandbox1/ben.cline/conda_envs/transformer_2
source /home/alex.li/.bashrc
conda activate highreseuropa

RUN_ID="brthighresnet"

DATA_DIR="/data2/jupiter/datasets"
# DATASET="halo_rgb_stereo_train_v6_2"
DATASET="halo_rgb_stereo_train_v6_2_full_res"
# DATASET="halo_rgb_stereo_train_v6_2_768"

EXP_NAME="bc_sandbox_2024"
SNAPSHOT_DIR=/mnt/sandbox1/alex.li/highres
MASTER_CSV="master_annotations_dedup_clean_ocal_20240208_50k_intersection.csv"

OUTPUT_DIR=${SNAPSHOT_DIR}/${EXP_NAME}/${RUN_ID}

# M 1.5
# --model-params '{"version": "timm/repvit_m1_5.dist_450e_in1k", "fixed_size_aux_output": false, "upsample_mode": "nearest", "in_features": [[4, 64], [8, 128], [16, 256], [32, 512]]}' \

# M 0.9
# --model-params '{"version": "timm/repvit_m0_9.dist_450e_in1k", "upsample_mode": "nearest", "in_features": [[4, 48], [8, 96], [16, 192], [32, 384]], "widening_factor": 2}' \

# M 2.3
# --model-params '{"version": "timm/repvit_m2_3.dist_450e_in1k", "fixed_size_aux_output": false, "upsample_mode": "nearest", "in_features": [[4, 80], [8, 160], [16, 320], [32, 640]]}' \
# 'repvit_lite12' 'brthighresnet' 'brtresnetnus'
# BRT model params
# --model-params '{"num_block_layers":2, "upsample_mode": "bilinear", "widening_factor": 2, "activation": "hardswish"}' \

cd ~/git/JupiterCVML/europa/base/src/europa
python dl/scripts/trainer.py \
    --csv-path ${DATA_DIR}/${DATASET}/${MASTER_CSV} \
    --dataset ${DATASET} \
    --data-dir ${DATA_DIR}/${DATASET} \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --exp-name ${EXP_NAME} \
    --run-id ${RUN_ID} \
    --val-set-ratio 0.05 \
    --gpu 0,1,2,3 \
    --resume-from-snapshot True \
    --input-mode rectifiedRGB \
    --early-stop-patience 100 \
    --input-dims 3 \
    --batch-size 48 \
    --num-workers 28 \
    --epochs 100 \
    --tqdm \
    --save-pred-every 2500000 \
    --num-steps 20000000 \
    --model brthighresnet \
    --val-csv dl/config/val_ids/halo_rgb_stereo_train_v6_2_val_by_geohash_6_for_50k_subset.csv \
    --notes "Highresnet training run" \
    --human-augmentation "{\"use\": false}" \
    --hard-sampling '{"use": false, "warmup_epochs": 0, "min_pix_count": 200, "gamma": 1.0, "multiplier": 7.0, "safety_max_weight": 30.0, "productivity": true, "depth_threshold": 0.3, "max_area_denominator": 2000, "productivity_max_weight": 30.0, "use_max_weights": false, "running_weight": 0.5}' \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --lr-scheduler-name cosinelr \
    --lr-scheduler-parameters '{"steplr_step_size": 7, "steplr_gamma": 0.1, "cosinelr_T_max": 100, "cosinelr_eta_min": 1e-6, "cycliclr_base_lr": 1e-5, "cycliclr_max_lr": 1e-3, "cycliclr_step_size_epoch": 2, "cycliclr_mode": "exp_range", "cycliclr_gamma": 0.97}' \
    --weight-decay 0.001 \
    --model-params '{"num_block_layers":2, "upsample_mode": "nearest", "widening_factor": 2, "activation": "hardswish"}' \
    --use-albumentation-transform \
    --loss tvmsl \
    --focalloss-parameters '{"alpha":[4.0,0.01,1.0,0.01,5.0,10.0,1.0,1.0,1.0], "gamma":2.0, "normalize_class_weights": false}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.6,0.6,0.6,0.7,0.3,0.6,0.6], "fn_weight":[0.7,0.007,0.4,0.004,2.0,7.0,0.7,0.4,0.4], "class_weight":[1.0,2.0,1.0,1.0,2.0,2.0,5.0,1.0,1.0], "gamma":1.0, "tversky_loss_weight": 1.0, "focal_loss_weight": 1.0, "use_pixel_count_mask": true, "image_wise": true, "normalize_class_weights": true, "use_sum_method": true, "depth_thresh": -1, "productivity_weight": 7.0, "productivity_prob_thresh": -1, "productivity_depth_thresh": -1}' \
    --label-map-file dl/config/label_maps/label_map_nine_class_birds_as_birds.csv \
    --learning-rate 1e-3 \
    --fine-tune-lr 0 \
    --num-classes 9 \
    --crop-transform-str "[[\"imgaug.augmenters.CropToFixedSize\", {\"width\": 1200, \"height\": 1024, 'position': 'uniform'}]]" \
    --weighted-sampling '""' \
    --input-size '1024,1200'

