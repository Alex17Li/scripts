#!/bin/bash
#SBATCH --job-name=rgbd_testing
#SBATCH --output=/home/li.yu/code/scripts/dust_51_v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30_mannydustseg.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
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
export PYTHONPATH=/home/li.yu/code/JupiterCVML/europa/base/src/europa

# enter working directory
cd /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/scripts

# experiment name
EXP='dust_51_v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30'
CHECKPOINT_PREFIX='job_quality'  # driveable_terrain_model or job_quality or vehicle_cls or bc_sandbox_2023
EXTRA_SUFFIX=''  # default empty str or _newmask

for EPOCH in -1
do 

# decide file name
if [ ${EPOCH} -ge 0 ]; then
    CHECKPOINT=${CHECKPOINT_PREFIX}_${EPOCH}_epoch_model.pth
    SAVE_PRED_SUFFIX=_epoch${EPOCH}${EXTRA_SUFFIX}
elif [ ${EPOCH} -eq -1 ]; then
    CHECKPOINT=${CHECKPOINT_PREFIX}_val_bestmodel.pth
    SAVE_PRED_SUFFIX=${EXTRA_SUFFIX}
else
    echo Unknown checkpoint specified: ${EPOCH}
    exit 1
fi

# check if dir and file exists
CHECKPOINT_FULL_DIR=/data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}
if [ ! -d $CHECKPOINT_FULL_DIR ]; then
    echo checkpoint dir does not exist, will create the dir
    mkdir $CHECKPOINT_FULL_DIR
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

echo start epoch $EPOCH with $CHECKPOINT_FULL_PATH


# run testing on human safety, on master
# SAFETY_DATASETS=("humans_on_path_test_set_2023_v9_anno" "humans_off_path_test_set_2023_v3_anno")
SAFETY_DATASETS=("mannequin_in_dust")
for DATASET in ${SAFETY_DATASETS[@]}
do
python predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
    --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
    --model brtresnetpyramid_lite12 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --states-to-save '' \
    --use-depth-threshold \
    --input-dims 4 \
    --batch-size 20 \
    --num-workers 4 \
    --gpu 0,1;
done

# # run testing on vehicle safety 2023 v1.1, on master
# python predictor_pl.py \
#     --csv-path /data/jupiter/datasets/vehicles_on_path_test_set_2023_v1_1_anno/master_annotations.csv \
#     --data-dir /data/jupiter/datasets/vehicles_on_path_test_set_2023_v1_1_anno \
#     --dataset vehicles_on_path_test_set_2023_v1_1_anno/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/vehicles_on_path_test_set_2023_v1_1_anno${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --merge-stop-class-confidence 0.35 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --states-to-save '' \
#     --gt-stop-classes-to-consider Vehicles \
#     --use-depth-threshold \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 4 \
#     --gpu 0,1;



# # run testing on halo test set, on GRETZKY_2106_halo_seg_color_correction branch
# SAFETY_DATASETS=("Jupiter_halo_implement_labeled_data_test_06162023_stereo" "Jupiter_halo_implement_labeled_data_train_06162023_stereo_v2")
# for DATASET in ${SAFETY_DATASETS[@]}
# do
# python predictor.py \
#     --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/datasets/${DATASET} \
#     --dataset ${DATASET}/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/eight_class_train.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --model-params '{"num_block_layers": 2, "widening_factor": 2, "upsample_mode": "nearest", "bias": true}' \
#     --imgaug-transform-str "" \
#     --dust-class-metrics \
#     --dust-mask /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/tire_masks/dust_rear_triangle_mask_fullres.png \
#     --input-size 512,640 \
#     --use-depth-threshold \
#     --merge-stop-class-confidence -1 \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 4 \
#     --gpu 0,1;
# done


# # run testing on halo test set, on GRETZKY_2106_halo_seg_color_correction branch
# SAFETY_DATASETS=("Jupiter_halo_labeled_data_20230503_test_stereo_640" "Jupiter_halo_labeled_data_20230512_test_stereo_640_oc_finetune")
# # SAFETY_DATASETS=("Jupiter_halo_labeled_data_20230512_test_stereo_768_oc")
# for DATASET in ${SAFETY_DATASETS[@]}
# do
# python predictor.py \
#     --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/datasets/${DATASET} \
#     --dataset ${DATASET}/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/eight_class_train_dust_light_as_sky.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --model-params '{"num_block_layers": 2, "widening_factor": 2, "upsample_mode": "nearest", "bias": true}' \
#     --states-to-save '' \
#     --imgaug-transform-str "" \
#     --input-size 512,640 \
#     --use-depth-threshold \
#     --merge-stop-class-confidence -1 \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 4 \
#     --gpu 0,1;
# done


# # run ddp label count or focal loss, on GRETZKY_1645_slice_finetune_vehicle_cls branch
# # SAFETY_DATASETS=("Jupiter_train5_11_driveable_not_labeled_on_harvest" "Jupiter_train5_11_less_than_half_labeled")
# SAFETY_DATASETS=("Jupiter_train5_11_less_than_half_labeled")
# for DATASET in ${SAFETY_DATASETS[@]}
# do
# python predictor_pl.py \
#     --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/datasets/${DATASET} \
#     --dataset ${DATASET}/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/for_label_count.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --model-params '{"num_block_layers": 2, "widening_factor": 2, "upsample_mode": "nearest", "bias": true}' \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 4 \
#     --gpu 0,1;
# done

# # run HALO ddp label count or focal loss, on GRETZKY_2106_halo_seg_color_correction branch
# SAFETY_DATASETS=("Jupiter_halo_labeled_data_20230517_train_stereo_640_768_single_ds_pmehta_oc_correctscale")
# # SAFETY_DATASETS=("Jupiter_halo_labeled_data_20230512_test_stereo_768_oc_finetune")
# for DATASET in ${SAFETY_DATASETS[@]}
# do
# python predictor_pl.py \
#     --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
#     --data-dir /data/jupiter/datasets/${DATASET} \
#     --dataset ${DATASET}/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/eight_class_train_dust_light_as_sky.csv \
#     --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${CHECKPOINT} \
#     --output-dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET}${SAVE_PRED_SUFFIX} \
#     --model brtresnetpyramid_lite12 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --model-params '{"num_block_layers": 2, "widening_factor": 2, "upsample_mode": "nearest", "bias": true}' \
#     --input-size 512,640 \
#     --imgaug-transform-str "" \
#     --states-to-save '' \
#     --use-depth-threshold \
#     --merge-stop-class-confidence -1 \
#     --input-dims 4 \
#     --batch-size 20 \
#     --num-workers 4 \
#     --gpu 0,1;
# done


echo -------------------------------------------------------------------
echo
echo
done

# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
