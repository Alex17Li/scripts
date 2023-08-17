#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=20G
#SBATCH --time=10:00:00

module load pytorch/1.12.0+cuda11.3
# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

# experiment name
EXP=dust_trivial_augment_1
# CHECKPOINT_PREFIX='vehicle_cls'  # driveable_terrain_model or job_quality or vehicle_cls or bc_sandbox_2023
CHECKPOINT_PREFIX='dust'
# check if dir and file exists
# CHECKPOINT_FULL_DIR=/data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/rakhil.immidisetti/logs/driveable_terrain_model/${EXP}
CHECKPOINT_FULL_DIR=${OUTPUT_PATH}/${CHECKPOINT_PREFIX}/${EXP}
CHECKPOINT=${CHECKPOINT_PREFIX}_val_bestmodel.pth

if [ ! -d $CHECKPOINT_FULL_DIR ]; then
    echo checkpoint $CHECKPOINT_FULL_DIR does not exist
    exit 1
fi

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

if [ ! -f $CHECKPOINT_FULL_PATH ]; then
    echo checkpoint does not exist, please run the cmd below to download
    # echo aws s3 cp s3://mesa-states/prod/jupiter/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    # aws s3 cp s3://mesa-states/prod/jupiter/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    echo aws s3 cp s3://blueriver-jupiter-data/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    aws s3 cp s3://blueriver-jupiter-data/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
        # exit 1
fi

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

# SAFETY_DATASETS=("humans_on_path_test_set_2023_v9_anno" "humans_off_path_test_set_2023_v3_anno")
# for DATASET in ${SAFETY_DATASETS[@]}
# do

# echo ----------------------------RUN ON ${DATASET}-----------------------------------
# python dl/scripts/predictor.py \
#     --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/datasets/${DATASET} \
#     --label-map-file dl/config/label_maps/four_class_train.csv \
#     --restore-from  ${CHECKPOINT_FULL_PATH} \
#     --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence 0.35 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --model-params "{\"activation\": \"gelu\"}" \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --batch-size 20 \
#     --tqdm \
# done

# echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------

# main productivity test sets
# PROD_DATASETS=("Jupiter_productivity_test_2023_v1_cleaned" "Jupiter_productivity_test_spring_2023_v2_cleaned" "Jupiter_productivity_airborne_debris_first_night")
PROD_DATASETS=("Jupiter_productivity_test_spring_2023_v2_cleaned")
for DATASET in ${PROD_DATASETS[@]}
do

echo ----------------------------RUN ON ${DATASET}-----------------------------------
python dl/scripts/predictor.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --label-map-file dl/config/label_maps/four_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
    --model brtresnetpyramid_lite12 \
    --merge-stop-class-confidence 0.35 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --states-to-save '' \
    --use-depth-threshold \
    --run-productivity-metrics \
    --batch-size 20 \
    --tqdm;
done


echo --------------------------RUN_PRODUCTIVITY_COMPLETE-------------------------------