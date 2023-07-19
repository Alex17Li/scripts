#!/bin/bash
#SBATCH --job-name=test_productivity
#SBATCH --output=/home/alex.li/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=20G
#SBATCH --time=8:00:00

# activate virtual env
source /home/alex.li/.bashrc
# conda activate shank
# conda activate knowledgedistillation
# conda activate pytorch1.10
module load pytorch/1.12.0+cuda11.3
conda activate cvml


cd /home/alex.li/workspace/JupiterCVML/europa/base/src/europa

# experiment name
# EXP='dust_test_small_prod'
EXP=v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30
CHECKPOINT_PREFIX='drivable_terrain_model'
CHECKPOINT_FULL_DIR=/mnt/sandbox1/rakhil.immidisetti/logs/driveable_terrain_model/${EXP}
if [ ! -d $CHECKPOINT_FULL_DIR ]; then
    echo checkpoint dir does not exist
    exit 1
fi


for EPOCH in -1
do 

# decide file name
if [ ${EPOCH} -ge 0 ]; then
    CHECKPOINT=${CHECKPOINT_PREFIX}_${EPOCH}_epoch_model.pth
    SAVE_PRED_SUFFIX=_epoch${EPOCH}
elif [ ${EPOCH} -eq -1 ]; then
    CHECKPOINT=${CHECKPOINT_PREFIX}_val_bestmodel.pth
    SAVE_PRED_SUFFIX=""
else
    echo Unknown checkpoint specified: ${EPOCH}
    exit 1
fi

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}
if [ ! -f $CHECKPOINT_FULL_PATH ]; then
    echo checkpoint does not exist, please run the cmd below to download
    echo aws s3 cp s3://mesa-states/prod/jupiter/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    aws s3 cp s3://mesa-states/prod/jupiter/model_training/${EXP}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    # exit 1
fi

echo start epoch $EPOCH with $CHECKPOINT_FULL_PATH

# main productivity test sets
PROD_DATASETS=("2022_productivity_ts_v2_hdr_tone25" "Jupiter_productivity_test_2023_v1_cleaned" "Jupiter_productivity_test_spring_2023_v2_cleaned" "Jupiter_productivity_airborne_debris_first_night")
# PROD_DATASETS=("Jupiter_productivity_test_2023_v1_cleaned" "Jupiter_productivity_test_spring_2023_v2_cleaned")
for DATASET in ${PROD_DATASETS[@]}
do
python predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
    --restore-from $CHECKPOINT_FULL_PATH \
    --output-dir /data/jupiter/alex.li//datasets/spring_dust_data_test/results \
    --model brtresnetpyramid_lite12 \
    --merge-stop-class-confidence 0.35 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --states-to-save '' \
    --use-depth-threshold \
    --input-dims 4 \
    --run-productivity-metrics \
    --batch-size 20 \
    --num-workers 4;
done



# # run testing on dusty dataset v4, on dust head GRETZKY_1385_dust_head
# python predictor.py \
#     --csv-path /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST/master_annotations_labeled.csv \
#     --data-dir /data/jupiter/datasets/ \
#     --dataset dust_test_2022_v4_anno_HEAVY_DUST/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#     --restore-from $CHECKPOINT_FULL_PATH \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --use-depth-threshold \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --input-dims 4 \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --batch-size 20 \
#     --num-workers 12 \
#     --gpu 0,1 \
#     --freeze-encoder \
#     --additional-head-output \
#     --num-classes 8 \
#     --num-head-classes 1;
# mv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv.labeled;
# python predictor.py \
#     --csv-path /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST/master_annotations_skipped.csv \
#     --data-dir /data/jupiter/datasets/ \
#     --dataset dust_test_2022_v4_anno_HEAVY_DUST/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --use-depth-threshold \
#     --run-productivity-metrics \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --input-dims 4 \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --batch-size 20 \
#     --num-workers 12 \
#     --freeze-encoder \
#     --additional-head-output \
#     --num-classes 8 \
#     --num-head-classes 1;
# mv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv.skipped;

# # run testing on dust labeled dataset, dust as a separate class
# python predictor.py \
#     --csv-path /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST/master_annotations_labeled.csv \
#     --data-dir /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST \
#     --dataset dust_test_2022_v4_anno_HEAVY_DUST/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence 0.35 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --gt-stop-classes-to-consider Humans \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 12;
# mv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv.labeled;
# python predictor.py \
#     --csv-path /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST/master_annotations_skipped.csv \
#     --data-dir /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST \
#     --dataset dust_test_2022_v4_anno_HEAVY_DUST/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence 0.35 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --run-productivity-metrics \
#     --gt-stop-classes-to-consider Humans \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 12;
# mv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}/output.csv.skipped;



# # run testing on IQ test set, master branch
# python predictor.py \
#     --csv-path /data/jupiter/avinash.raju/iq_2022_v8_anno/master_annotations.csv \
#     --data-dir /data/jupiter/avinash.raju/iq_2022_v8_anno \
#     --dataset iq_2022_v8_anno/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/iq_2022_v8_anno${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence 0.35 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --input-dims 4 \
#     --run-productivity-metrics \
#     --batch-size 20 \
#     --num-workers 12;
# # run testing on dust test set v4, master branch
# python predictor.py \
#     --csv-path /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST/master_annotations_labeled.csv \
#     --data-dir /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST \
#     --dataset dust_test_2022_v4_anno_HEAVY_DUST/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence 0.35 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --input-dims 4 \
#     --run-productivity-metrics \
#     --batch-size 20 \
#     --num-workers 12;


# # run testing on IQ testset, on dust head GRETZKY_1385_dust_head
# python predictor.py \
#     --csv-path /data/jupiter/avinash.raju/iq_2022_v8_anno/master_annotations.csv \
#     --data-dir /data/jupiter/avinash.raju/ \
#     --dataset iq_2022_v8_anno/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/iq_2022_v8_anno${SAVE_PRED_SUFFIX}_dust \
#     --model brtresnetpyramid_lite12 \
#     --use-depth-threshold \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --input-dims 4 \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --batch-size 20 \
#     --num-workers 12 \
#     --freeze-encoder \
#     --additional-head-output \
#     --num-classes 7 \
#     --num-head-classes 1;
# # run testing on dusty dataset v4, on dust head GRETZKY_1385_dust_head
# python predictor.py \
#     --csv-path /data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST/master_annotations_labeled.csv \
#     --data-dir /data/jupiter/datasets/ \
#     --dataset dust_test_2022_v4_anno_HEAVY_DUST/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/dust_test_2022_v4_anno_HEAVY_DUST${SAVE_PRED_SUFFIX}_dust \
#     --model brtresnetpyramid_lite12 \
#     --use-depth-threshold \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --input-dims 4 \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --batch-size 20 \
#     --num-workers 12 \
#     --freeze-encoder \
#     --additional-head-output \
#     --num-classes 7 \
#     --num-head-classes 1;


echo -------------------------------------------------------------------
echo
echo
done


# # run stand-alone inference 
# cd /home/li.yu/code/notebooks/dust_seg_head
# python3 seg_infer.py

# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
