#!/bin/bash
#SBATCH --job-name=sup_dust_analysis
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=8:00:00

source /home/$USER/.bashrc
module load pytorch/1.12.0+cuda11.3
conda activate cvml

JCVML_PATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
cd $JCVML_PATH
echo ----CURRENT GIT BRANCH AND DIFF----
git branch --show-current
git diff
echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/vehicle_cls_43_epoch_model.pth
# DATASET="suv_driving_through_rear_dust_anno"
# echo ----------------------------RUN ON ${DATASET}-----------------------------------
# # ANNOTATIONS_PATH=64b0197137e915581adec2d5_master_annotations.csv
# # SUBSAMPLE=1000000
# # --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/v471_rd_2cls_dustseghead_0808/job_quality_val_bestmodel.pth \
# python dl/scripts/predictor.py \
#     --csv-path ${DATASET_PATH}/${DATASET}/master_annotations.csv \
#     --data-dir ${DATASET_PATH}/${DATASET} \
#     --label-map-file ${JCVML_PATH}/dl/config/label_maps/four_class_train.csv \
#     --restore-from $CHECKPOINT_FULL_PATH \
#     --output-dir ${OUTPUT_PATH}/${DATASET}/4class_prod/results \
#     --merge-stop-class-confidence 0.35 \
#     --dust-output-params '{"dust_head_output": false, "dust_class_ratio": false, "dust_class_confidence_map": true, "zero_dust_ratio": false}' \
#     --input-dims 4 \
#     --states-to-save  "vehicle_false_negative" "vehicle_false_positive" \
#     --batch-size 1 \
#     --dust-class-metrics \
#     --half-res-output \
#     --dust-mask "NO" \
#     --model brtresnetpyramid_lite12 \
#     --input-mode RGBD;
# echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/results/dust_51_v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30/model.pth
DATASET="mannequin_in_dust"
echo ----------------------------RUN ON ${DATASET} li run-----------------------------------
python dl/scripts/predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${OUTPUT_PATH}/${DATASET}/7class_prod/results \
    --model brtresnetpyramid_lite12 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --states-to-save '' \
    --use-depth-threshold \
    --input-dims 4 \
    --batch-size 20 \
    --num-workers 4 \
    --gpu 0,1;
echo ----------------------------RUN ON ${DATASET} use pl-----------------------------------
# ANNOTATIONS_PATH=64b0197137e915581adec2d5_master_annotations.csv
# SUBSAMPLE=1000000
# --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/v471_rd_2cls_dustseghead_0808/job_quality_val_bestmodel.pth \
python dl/scripts/predictor_pl.py \
    --csv-path ${DATASET_PATH}/${DATASET}/master_annotations.csv \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/seven_class_train.csv \
    --restore-from $CHECKPOINT_FULL_PATH \
    --output-dir ${OUTPUT_PATH}/${DATASET}/7class_prod/results \
    --dust-output-params '{"dust_head_output": true, "dust_class_ratio": false, "dust_class_confidence_map": false, "zero_dust_ratio": false}' \
    --input-dims 4 \
    --states-to-save "human_false_negative" "human_false_positive" \
    --batch-size 1 \
    --dust-class-metrics \
    --half-res-output \
    --dust-mask "NO" \
    --model brtresnetpyramid_lite12 \
    --input-mode RGBD;

echo ----------------------------RUN ON ${DATASET} large bs-----------------------------------

python dl/scripts/predictor_pl.py \
    --csv-path ${DATASET_PATH}/${DATASET}/master_annotations.csv \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/seven_class_train.csv \
    --restore-from $CHECKPOINT_FULL_PATH \
    --output-dir ${OUTPUT_PATH}/${DATASET}/7class_prod/results \
    --dust-output-params '{"dust_head_output": true, "dust_class_ratio": false, "dust_class_confidence_map": false, "zero_dust_ratio": false}' \
    --input-dims 4 \
    --states-to-save "human_false_negative" "human_false_positive" \
    --batch-size 20 \
    --dust-class-metrics \
    --half-res-output \
    --dust-mask "NO" \
    --model brtresnetpyramid_lite12 \
    --input-mode RGBD;

echo ----------------------------RUN ON ${DATASET} fullres-----------------------------------

python dl/scripts/predictor_pl.py \
    --csv-path ${DATASET_PATH}/${DATASET}/master_annotations.csv \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/seven_class_train.csv \
    --restore-from $CHECKPOINT_FULL_PATH \
    --output-dir ${OUTPUT_PATH}/${DATASET}/7class_prod/results \
    --dust-output-params '{"dust_head_output": true, "dust_class_ratio": false, "dust_class_confidence_map": false, "zero_dust_ratio": false}' \
    --input-dims 4 \
    --states-to-save "human_false_negative" "human_false_positive" \
    --batch-size 1 \
    --dust-class-metrics \
    --dust-mask "NO" \
    --model brtresnetpyramid_lite12 \
    --input-mode RGBD;