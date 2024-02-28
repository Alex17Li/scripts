#!/bin/bash
#SBATCH --job-name=brthighresnetbesteval
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
## SBATCH --exclude=stc01sppamxnl018,stc01sppamxnl005
#SBATCH --time=240:00:00

source /home/$USER/.bashrc
conda activate cvml
cd /home/$USER/git/JupiterCVML/europa/base/src/europa
DLPATH=/home/alex.li/git/JupiterCVML/europa/base/src/europa
BATCH_SIZE=1
EXP_NAME="bc_sandbox_2024"  
GPUS="0"
NUM_WORKERS=8

# SAVED_MODEL_BASELINE_STR="latest_epoch_model"
SAVED_MODEL_BASELINE_STR="val_bestmodel"
# SAFETY_DATASET=halo_humans_on_path_test_v6_2_2_test_dataset_768
# UPDATE /home/alex.li/git/JupiterCVML/europa/base/src/europa/dl/utils/config.py
SAFETY_DATASET=halo_humans_on_path_test_v6_2_2_test_dataset
PRODUCTIVITY_DATASET=halo_rgb_stereo_test_v6_2
RUN_ID="repvit_M0.9_512"
# 'repvit_lite12' 'brthighresnet' 'brtresnetnus' 'brtresnetpyramid_lite12'
MODEL="repvit_lite12"
MODEL_PATH_BASELINE="/mnt/sandbox1/alex.li/highres/"${EXP_NAME}"/${RUN_ID}/"${EXP_NAME}"_"${SAVED_MODEL_BASELINE_STR}".pth"
# MODEL_PATH_BASELINE=/mnt/sandbox1/ben.cline/output/bc_sandbox_2023/ds_v6_1_4x_human/bc_sandbox_2023_val_bestmodel.pth
# MODEL_PATH_BASELINE=/mnt/sandbox1/alex.li/23229_label_resize/checkpoints/epoch=53.ckpt
MODEL_PARAMS='{"version":"timm/repvit_m0_9.dist_450e_in1k","upsample_mode":"nearest","in_features":[[4,48],[8,96],[16,192],[32,384]],"widening_factor":2}'
# MODEL_PARAMS='{"version":"timm/repvit_m1_5.dist_450e_in1k","fixed_size_aux_output":false,"upsample_mode":"nearest","in_features":[[4,64],[8,128],[16,256],[32,512]]}'
# MODEL_PARAMS='{"version":"timm/repvit_m2_3.dist_450e_in1k","fixed_size_aux_output":false,"upsample_mode":"nearest","in_features":[[4,80],[8,160],[16,320],[32,640]]}'
# MODEL_PARAMS='{"num_block_layers":2,"upsample_mode":"bilinear","widening_factor":2,"activation":"hardswish"}'
# MODEL_PARAMS='{"num_block_layers":2,"upsample_mode":"nearest","widening_factor":2,"activation":"relu"}'


LABEL_MAP_FILE=$DLPATH/dl/config/label_maps/label_map_nine_class_birds_as_birds.csv
# LABEL_MAP_FILE=$DLPATH/dl/config/label_maps/eight_class_train_dust_light_as_sky_birds_as_driveable.csv
DATA_DIR=/data2/jupiter/datasets
# save model output visualization for the following states
STATES_SAFETY="none"
STATES_PRODUCTIVITY="none"
SUFFIX=''H
OUTPUT_DIR_PRODUCTIVITY_BASELINE=/mnt/sandbox1/alex.li/highres/${EXP_NAME}/${RUN_ID}/${PRODUCTIVITY_DATASET}/${SAVED_MODEL_BASELINE_STR}${SUFFIX}
OUTPUT_DIR_SAFETY_BASELINE=/mnt/sandbox1/alex.li/highres/${EXP_NAME}/${RUN_ID}/${SAFETY_DATASET}/${SAVED_MODEL_BASELINE_STR}${SUFFIX}
MASTER_CSV="master_annotations.csv"

echo ""
echo $SAFETY_DATASET

set -ex

python ${DLPATH}/dl/scripts/predictor.py --csv-path $DATA_DIR/$SAFETY_DATASET/$MASTER_CSV \
    --dataset ${SAFETY_DATASET} \
    --data-dir $DATA_DIR/${SAFETY_DATASET} \
    --gpu $GPUS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --run-id $RUN_ID \
    --output-dir $OUTPUT_DIR_SAFETY_BASELINE \
    --restore-from $MODEL_PATH_BASELINE \
    --use-depth-threshold \
    --tqdm \
    --states-to-save $STATES_SAFETY \
    --pred-stop-classes-to-consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
    --ignore-deprecation-crash \
    --input-mode RGBD \
    --input-dims 3 \
    --model $MODEL \
    --model-params $MODEL_PARAMS \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --label-map-file $LABEL_MAP_FILE \
    --input-size '0,0' \
    --merge-stop-class-confidence -1

python ${DLPATH}/dl/scripts/predictor.py --csv-path $DATA_DIR/$PRODUCTIVITY_DATASET/$MASTER_CSV \
    --dataset ${PRODUCTIVITY_DATASET} \
    --data-dir $DATA_DIR/${PRODUCTIVITY_DATASET} \
    --gpu $GPUS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --run-id $RUN_ID \
    --output-dir $OUTPUT_DIR_PRODUCTIVITY_BASELINE \
    --restore-from $MODEL_PATH_BASELINE \
    --use-depth-threshold \
    --tqdm \
    --states-to-save $STATES_PRODUCTIVITY \
    --pred-stop-classes-to-consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
    --ignore-deprecation-crash \
    --input-mode RGBD \
    --input-dims 3 \
    --model $MODEL \
    --model-params $MODEL_PARAMS \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --label-map-file $LABEL_MAP_FILE \
    --input-size '0,0' \
    --gt-stop-classes-to-consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
    --merge-stop-class-confidence -
