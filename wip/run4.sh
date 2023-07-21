#!/bin/bash
#SBATCH --job-name=rgbd_testing
#SBATCH --output=/home/li.yu/code/scripts/dust_51_v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30_dust.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=20G
#SBATCH --time=8:00:00

# activate virtual env
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
# conda activate shank
# conda activate knowledgedistillation
# conda activate pytorch1.10
conda activate pytorchlightning

# add working dir
export PYTHONPATH=/home/alex.li/git/JupiterCVML/europa/base/src/europa

# enter working directory
cd /home/alex.li/git/JupiterCVML/europa/base/src/europa/dl/scripts

# export EXP=v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30
# export CHECKPOINT_PREFIX='driveable_terrain_model'
export EXP=v57rd_4cls_tiny0occluded5reverse5triangle5_msml_0305
export CHECKPOINT_PREFIX='vehicle_cls'
export CHECKPOINT_FULL_DIR=/mnt/sandbox1/rakhil.immidisetti/logs/driveable_terrain_model/${EXP}
export OUTPUT_DIR=/data/jupiter/alex.li/results

if [ ! -d $CHECKPOINT_FULL_DIR ]; then
    echo checkpoint dir does not exist
    exit 1
fi

export EPOCH=-1

# decide file name
if [ ${EPOCH} -ge 0 ]; then
    export CHECKPOINT=${CHECKPOINT_PREFIX}_${EPOCH}_epoch_model.pth
    export SAVE_PRED_SUFFIX=_epoch${EPOCH}${EXTRA_SUFFIX}
elif [ ${EPOCH} -eq -1 ]; then
    export CHECKPOINT=${CHECKPOINT_PREFIX}_val_bestmodel.pth
    export SAVE_PRED_SUFFIX=${EXTRA_SUFFIX}
else
    echo Unknown checkpoint specified: ${EPOCH}
    exit 1
fi

export CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}
if [ ! -f $CHECKPOINT_FULL_PATH ]; then
    echo checkpoint does not exist, please run the cmd below to download
    # echo aws s3 cp s3://mesa-states/prod/jupiter/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    # aws s3 cp s3://mesa-states/prod/jupiter/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    echo aws s3 cp s3://blueriver-jupiter-data/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    aws s3 cp s3://blueriver-jupiter-data/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    # exit 1
fi

echo -------------------------------------------------
echo $CHECKPOINT_FULL_PATH
echo -------------------------------------------------
# # dust datasets, master branch
# # DUST_DATASETS=("manny_in_dust_8_10mph_Jan2023_labeled" "Jupiter_2022_Dust_Humans_Unfiltered_partiallabeled" "Jupiter_2023_0215_0302_human_dust_labeled")
# # DUST_DATASETS=("Jupiter_2023_03_28_Loamy_619_stops_stereo_2" "Jupiter_2023_03_28_Loamy_812_stops_stereo" "Jupiter_2023_03_29_Loamy_812_stops_after_2pm_left_images_rear_pod_stereo")  # --run-productivity-metrics \
# DUST_DATASETS=("Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2")  # --run-productivity-metrics \
# for DATASET in ${DUST_DATASETS[@]}
# do
# python predictor.py \
#     --csv-path /data/jupiter/li.yu/data/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/li.yu/data/${DATASET} \
#     --dataset ${DATASET}/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/human_vehicle_detector.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --run-productivity-metrics \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 12;
# done


# # dust datasets, GRETZKY_1385_dust_head branch
# # DUST_DATASETS=("manny_in_dust_8_10mph_Jan2023_labeled" "Jupiter_2022_Dust_Humans_Unfiltered_partiallabeled" "Jupiter_2023_0215_0302_human_dust_labeled")
# # DUST_DATASETS=("Jupiter_2023_03_28_Loamy_619_stops_stereo_2" "Jupiter_2023_03_28_Loamy_812_stops_stereo" "Jupiter_2023_03_29_Loamy_812_stops_after_2pm_left_images_rear_pod_stereo")  # --run-productivity-metrics \
# DUST_DATASETS=("Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2" "Jupiter_2023_04_05_loamy869_dust_collection_stereo" "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo")  # --run-productivity-metrics \
# # DUST_DATASETS=("Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2")  # --run-productivity-metrics \
# for DATASET in ${DUST_DATASETS[@]}
# do
# python predictor.py \
#     --csv-path /data/jupiter/li.yu/data/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/li.yu/data/ \
#     --dataset ${DATASET}/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --additional-head-output \
#     --use-depth-threshold \
#     --run-productivity-metrics \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-classes 7 \
#     --num-head-classes 1;
# done


# # dust datasets, GRETZKY_2126_4cls_model branch, dust as a separate class
# # labeled datasets
# # DUST_DATASETS=("manny_in_dust_8_10mph_Jan2023_labeled" "Jupiter_2022_Dust_Humans_Unfiltered_partiallabeled" "Jupiter_2023_0215_0302_human_dust_labeled")
DUST_DATASETS=("Jupiter_2023_03_02_and_2930_human_vehicle_in_dust_labeled" "Jupiter_2023_March_29th30th_human_vehicle_in_dust_front_pod_labeled" "Jupiter_2023_04_05_loamy869_dust_collection_stereo_labeled" "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo_labeled")
for DATASET in ${DUST_DATASETS[@]}
do
python predictor.py \
    --csv-path /data/jupiter/li.yu/data/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/li.yu/data/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${OUTPUT_DIR}/${DATASET}${SAVE_PRED_SUFFIX} \
    --dust-class-metrics \
    --dust-mask NO \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --model-params "{\"num_block_layers\": 2, \"widening_factor\": 2, \"upsample_mode\": \"nearest\", \"bias\": false}" \
    --states-to-save '' \
    --use-depth-threshold \
    --input-dims 4 \
    --batch-size 20 \
    --num-workers 12;
done

# unlabeled datasets
DUST_DATASETS=("Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2" "Jupiter_2023_04_05_loamy869_dust_collection_stereo" "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo")  # --run-productivity-metrics \
# DUST_DATASETS=("Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2")
for DATASET in ${DUST_DATASETS[@]}
do
python predictor.py \
    --csv-path /data/jupiter/li.yu/data/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/li.yu/data/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${CHECKPOINT_FULL_PATH}/${DATASET}${SAVE_PRED_SUFFIX} \
    --dust-class-metrics \
    --dust-mask NO \
    --merge-stop-class-confidence 0.35 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --states-to-save '' \
    --use-depth-threshold \
    --run-productivity-metrics \
    --input-dims 4 \
    --batch-size 20 \
    --num-workers 12;
done


# deactivate virtual env
conda deactivate
