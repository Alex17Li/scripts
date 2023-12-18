#!/bin/bash
#SBATCH --job-name=example_halo_rgb
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48000

# TODO: Update the {username} in the sbatch commands above and create the logs folder mentioned in output.

# TODO: Load your conda environment here
source ~/.bashrc
# TODO: Update CVML_PATH to point to the local path of europa folder inside JupiterCVML repo
export CVML_PATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
set -e

RUN_ID="example_halo_rgb"

DATASET="halo_rgb_stereo_train_v6_0"
DATA_DIR="/data2/jupiter/datasets/${DATASET}"
EXP_NAME="driveable_terrain_model"
MASTER_CSV="master_annotations.csv"

SNAPSHOT_DIR=/mnt/sandbox1/${USER}/logs
OUTPUT_DIR=${SNAPSHOT_DIR}/${EXP_NAME}/${RUN_ID}

cd ${CVML_PATH}

python dl/scripts/trainer.py \
    --csv-path ${DATA_DIR}/${MASTER_CSV} \
    --data-dir ${DATA_DIR} \
    --run-id ${RUN_ID} \
    --exp-name ${EXP_NAME} \
    --snapshot-dir ${SNAPSHOT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --label-map-file dl/config/label_maps/eight_class_train_dust_light_as_sky_birds_as_driveable.csv \
    --val-set-ratio 0.05 \
    --input-mode rectifiedRGB \
    --input-dims 3 \
    --early-stop-patience 12 \
    --num-workers 24 \
    --batch-size 72 \
    --epochs 100 \
    --save-pred-every 25000 \
    --num-steps 200000 \
    --model brtresnetpyramid_lite12 \
    --use-albumentation-transform \
    --weighted-sampling '{}' \
    --fp16 \
    --half-res-output \
    --ignore-deprecation-crash \
    --lr-scheduler-parameters '{"cosinelr_T_max": 100, "cosinelr_eta_min": 1e-6}' \
    --losses '{"msl": 1.0, "tv": 1.0, "prodl": 0.1}' \
    --focalloss-parameters '{"alpha":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], "gamma":2.0}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.6,0.6,0.6,0.3,0.3,0.6], "fn_weight":[0.4,0.7,0.4,0.4,0.4,0.7,0.7,0.4], "class_weight":[1.0,2.0,1.0,1.0,2.0,10.0,5.0,1.0], "gamma":1.0}' \
    --productivity-loss-params '{"depth_thresh": 0.35, "prob_thresh": 0.01}' \
    --notes 'Example end-to-end training run for training RGB models on halo dataset' \
    --tqdm \
    --albumentation-transform-str '{"__version__": "1.3.0", "transform": {"__class_fullname__": "Compose", "p": 1.0, "transforms": [{"__class_fullname__": "OneOf", "p": 1.0, "transforms": [{"__class_fullname__": "Resize", "always_apply": false, "p": 0.5, "height": 512, "width": 640, "interpolation": 1}, {"__class_fullname__": "RandomCrop", "always_apply": false, "p": 0.5, "height": 512, "width": 640}]}, {"__class_fullname__": "Compose", "p": 0.95, "transforms": [{"__class_fullname__": "HorizontalFlip", "always_apply": false, "p": 0.5}, {"__class_fullname__": "ColorJitter", "always_apply": false, "p": 0.5, "brightness": [0.7, 1.3], "contrast": [0.7, 1.3], "saturation": [0.7, 1.3], "hue": [-0.3, 0.3]}, {"__class_fullname__": "OneOf", "p": 0.95, "transforms": [{"__class_fullname__": "MultiplicativeNoise", "always_apply": false, "p": 0.25, "multiplier": [0.8, 1.25], "per_channel": true, "elementwise": true}, {"__class_fullname__": "RandomGamma", "always_apply": false, "p": 0.25, "gamma_limit": [80, 120], "eps": null}, {"__class_fullname__": "RGBShift", "always_apply": false, "p": 0.25, "r_shift_limit": [-0.2, 0.2], "g_shift_limit": [-0.2, 0.2], "b_shift_limit": [-0.2, 0.2]}]}, {"__class_fullname__": "ShiftScaleRotate", "always_apply": false, "p": 0.95, "shift_limit_x": [-0.2, 0.2], "shift_limit_y": [-0.2, 0.2], "scale_limit": [-0.30000000000000004, 0.5], "rotate_limit": [-10, 10], "interpolation": 0, "border_mode": 0, "value": 0.0, "mask_value": 255, "rotate_method": "largest_box"}], "bbox_params": null, "keypoint_params": null, "additional_targets": {}}], "bbox_params": null, "keypoint_params": null, "additional_targets": {}}}' \
    --input-size '512,640'


echo "Training completed"
echo "Testing starting now..."

# TODO: point correctly to the below scripts
cd /home/$USER/scripts

# Test on the standard datasets
./run_safety_pl.sh ${RUN_ID} val_bestmodel 0,1,2,3 20230913_halo_RGB_stereo_test_v2 '--side-left-tire-mask none --side-right-tire-mask none --input-dims 3 --label-map-file dl/config/label_maps/eight_class_train_dust_light_as_sky.csv --input-size 0,0'
./run_safety_pl.sh ${RUN_ID} val_bestmodel 0,1,2,3 20230913_halo_RGB_stereo_test_v2 '--side-left-tire-mask none --side-right-tire-mask none --input-dims 3 --label-map-file dl/config/label_maps/eight_class_train_dust_light_as_sky.csv --input-size 0,0 --gt-stop-classes-to-consider Non-driveable Trees_Weeds Humans Vehicles' '_all'
./run_productivity_pl.sh ${RUN_ID} val_bestmodel 0,1,2,3 20230912_halo_rgb_productivity_day_candidate_1_no_ocal '--side-left-tire-mask none --side-right-tire-mask none --input-dims 3 --label-map-file dl/config/label_maps/eight_class_train_dust_light_as_sky.csv --input-size 0,0'

echo "All Testing runs completed"
exit
