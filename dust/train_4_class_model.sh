#!/bin/bash
#SBATCH --job-name=train_4cls
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

#--SBATCH --partition=cpu
source /home/$USER/.bashrc
module load pytorch/1.12.0+cuda11.3
conda activate cvml

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CVML_PATH=/home/$USER/git/JupiterCVML
EXP=alex_4cls_$SLURM_JOB_ID
SNAPSHOT_DIR=/mnt/sandbox1/$USER
OUTPUT_DIR=${OUTPUT_PATH}/${EXP}

# --tqdm \
# --augmentations CustomCrop SmartCrop HorizontalFlip TorchColorJitter Resize \

# --restore-from /mnt/sandbox1/alex.li/results/dust/dust_trivial_augment_1/dust_val_bestmodel.pth \
python -m dl.scripts.trainer \
    --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_11/epoch0_5_30_focal05_master_annotations.csv \
    --data-dir /data/jupiter/datasets/Jupiter_train_v5_11/ \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
    --exp-name dust \
    --model-params "{\"activation\": \"gelu\"}" \
    --optimizer adamw \
    --weight-decay 1e-5 \
    --activation-reg 0 \
    --learning-rate 1e-3 \
    --lr-scheduler-name cosinelr \
    --lr-scheduler-parameters '{"cosinelr_T_max": 60, "cosinelr_eta_min": 1e-6}' \
    --epochs 60 \
    --model brtresnetpyramid_lite12 \
    --early-stop-patience 12 \
    --batch-size 64 \
    --val-set-ratio 0.05 \
    --losses '{"tv": 0.2, "prodl": 0.02}' \
    --multiscalemixedloss-parameters '{"scale_weight":0.1, "dust_weight":0.5, "dust_scale_weight":0.05}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.3,0.6], "fn_weight":[0.4,0.7,0.7,0.4], "class_weight":[1.5,3.0,2.0,1.0], "gamma":1.0}' \
    --productivity-loss-params '{"depth_thresh": 0.35, "prob_thresh": 0.01}' \
    --trivial-augment '{"use": true}' \
    --night-model '{"use": false, "dark_pix_threshold": 10}' \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --resume-from-snapshot False \
    --output-dir ${OUTPUT_DIR} \
    --color-jitter '{"use": false}' \
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

    # --use-albumentation-transform \
    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_7/epoch0_5_30_focal05_notiny_onlyleft_master_annotations.csv \
    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_8/epoch0_5_30_focal05_master_annotations.csv \
    # --multiscalemixedloss-parameters '{"scale_weight":0.1, "dust_weight":0.5, "dust_scale_weight":0.05}' \
    # --focalloss-parameters '{"alpha":[1.0,1.0,0.5,0.5,1.0,2.0,1.0], "gamma":2.0}' \
    # --tversky-parameters '{"fp_weight":[0.1,0.0,0.0,0.1], "fn_weight":[0.9,1.0,1.0,0.9], "class_weight":[0.0,1.0,1.0,0.0], "gamma":1.0, "tversky_weight":0.01, "use_msl":false}' \
    # --use-albumentation-transform \
    # --imgaug-transform-str "[[\"imgaug.augmenters.OneOf\", [[\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"translate_percent\": {\"x\": (-0.2, 0.2),\"y\": (-0.2, 0.2)},\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"scale\": (0.7, 1.5),\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"rotate\": (-10, 10),\"mode\": \"reflect\",\"order\": 0}]]], [\"imgaug.augmenters.OneOf\", [[\"imgaug.augmenters.Multiply\", {\"mul\": (0.8,1.25),\"per_channel\": 0.2}], [\"imgaug.augmenters.LogContrast\", {\"gain\": (0.6,1.4)}], [\"imgaug.augmenters.SigmoidContrast\", {\"gain\": (3,10), \"cutoff\": (0.4,0.6)}]]], [\"dl.augmentations.augmentations.Clip\",{\"lower\": 0,\"upper\": 1}]]" \
    # --restore-from '/data/jupiter/li.yu/exps/driveable_terrain_model/v51rd_7cls_imgaug_highbz_100epoch_0123/vehicle_cls_75_epoch_model.pth' \

# mv ${OUTPUT_DIR} /data/jupiter/alex.li/exps/
# mv ${OUTPUT_DIR}/* /data/jupiter/alex.li/exps/${EXP}/

# https://us-east-2.console.aws.amazon.com/s3/object/blueriver-jupiter-data?region=us-west-2&prefix=model_training/dust_trivial_augment_1/dust_val_bestmodel.pth
# aws s3 cp /mnt/sandbox1/alex.li/dust/dust_trivial_augment_1/dust_val_bestmodel.pth s3://blueriver-jupiter-data/model_training/dust_trivial_augment_1/dust_val_bestmodel.pth