#!/bin/bash
#SBATCH --job-name=sup_dust_analysis
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
module load pytorch/1.12.0+cuda11.3
conda activate cvml

OUTPUT_PATH=/data/jupiter/$USER/results
JCVML_PATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
cd $JCVML_PATH

# DATASET="hhh_field_data_stratified"
# DATASET=Jupiter_halo_rgbnir_stereo_train_20230710/
DATASET=Jupiter_halo_rgbnir_stereo_train_20230710_cleaned_20230712_tractor_only_withArtifactFix
ANNOTATIONS_PATH=64b0197137e915581adec2d5_master_annotations.csv
# SUBSAMPLE=1000000
# --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/v471_rd_2cls_dustseghead_0808/job_quality_val_bestmodel.pth \
python dl/scripts/predictor.py \
    --csv-path ${DATASET_PATH}/${DATASET}/$ANNOTATIONS_PATH \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --dataset "Yo does the value of this variable actually even matter" \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/seven_class_train.csv \
    --restore-from /mnt/sandbox1/rakhil.immidisetti/logs/driveable_terrain_model/v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30/driveable_terrain_model_val_bestmodel.pth \
    --output-dir ${OUTPUT_PATH}/${DATASET}/results_0808 \
    --merge-stop-class-confidence 0.35 \
    --input-dims 4 \
    --run-productivity-metrics \
    --batch-size 32 \
    --dust-class-metrics \
    --dust-mask "NO MASK INVALID PATH" \
    --input-mode RGBD;
