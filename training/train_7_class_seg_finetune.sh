#!/bin/bash
#SBATCH --job-name=dusthead_only
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
EXP=${SLURM_JOB_ID}
SNAPSHOT_DIR=/mnt/sandbox1/$USER
OUTPUT_DIR=${OUTPUT_PATH}/${EXP}
wandb enabled

python -m dl.scripts.trainer \
    --csv-path /data/jupiter/datasets/Jupiter_train_v5_11_20230508//master_annotations_v481.csv \
    --data-dir /data/jupiter/datasets/Jupiter_train_v5_11_20230508/ \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
    --dust-output-params "{\"dust_seg_output\": true, \"train_only_dust_head\": true}" \
    --exp-name dust \
    --model-params "{\"activation\": \"relu\"}" \
    --optimizer adamw \
    --weight-decay 1e-5 \
    --trivial-augment '{"use": true}' \
    --learning-rate 1e-3 \
    --n-images-to-sample 160000 \
    --lr-scheduler-name exponentiallr \
    --lr-scheduler-parameters '{"exponential_gamma": 0.9}' \
    --epochs 20 \
    --model brtresnetpyramid_lite12 \
    --early-stop-patience 999 \
    --batch-size 64 \
    --val-set-ratio 0.05 \
    --losses '{"hardsoft_iq": 1.0}' \
    --night-model '{"use": false, "dark_pix_threshold": 10}' \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --resume-from-snapshot False \
    --restore-from /mnt/sandbox1/alex.li/results/dust_51_v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30/dust_val_bestmodel.pth \
    --output-dir ${OUTPUT_DIR} \
    --color-jitter '{"use": false}' \
    --num-steps 3000000000 \
    --cutnpaste-augmentations "{}" \
    --run-id ${EXP};
    # --use-albumentation-transform \
    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_7/epoch0_5_30_focal05_notiny_onlyleft_master_annotations.csv \
    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_8/epoch0_5_30_focal05_master_annotations.csv \
    # --multiscalemixedloss-parameters '{"scale_weight":0.1, "dust_weight":0.5, "dust_scale_weight":0.05}' \
    # --focalloss-parameters '{"alpha":[1.0,1.0,0.5,0.5,1.0,2.0,1.0], "gamma":2.0}' \
    # --use-albumentation-transform \
    # --imgaug-transform-str "[[\"imgaug.augmenters.OneOf\", [[\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"translate_percent\": {\"x\": (-0.2, 0.2),\"y\": (-0.2, 0.2)},\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"scale\": (0.7, 1.5),\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"rotate\": (-10, 10),\"mode\": \"reflect\",\"order\": 0}]]], [\"imgaug.augmenters.OneOf\", [[\"imgaug.augmenters.Multiply\", {\"mul\": (0.8,1.25),\"per_channel\": 0.2}], [\"imgaug.augmenters.LogContrast\", {\"gain\": (0.6,1.4)}], [\"imgaug.augmenters.SigmoidContrast\", {\"gain\": (3,10), \"cutoff\": (0.4,0.6)}]]], [\"dl.augmentations.augmentations.Clip\",{\"lower\": 0,\"upper\": 1}]]" \
# mv ${OUTPUT_DIR} /data/jupiter/alex.li/exps/
# mv ${OUTPUT_DIR}/* /data/jupiter/alex.li/exps/${EXP}/

# https://us-east-2.console.aws.amazon.com/s3/object/blueriver-jupiter-data?region=us-west-2&prefix=model_training/dust_trivial_augment_1/dust_val_bestmodel.pth
# aws s3 cp /mnt/sandbox1/alex.li/dust/dust_trivial_augment_1/dust_val_bestmodel.pth s3://blueriver-jupiter-data/model_training/dust_trivial_augment_1/dust_val_bestmodel.pth