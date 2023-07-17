#!/bin/bash
#SBATCH --job-name=dust_analysis
#SBATCH --output=/home/alex.li/workspace/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=150:00:00

source /home/alex.li/.bashrc
module load pytorch/1.12.0+cuda11.3
conda activate cvml

DATASET_PATH=/data/jupiter/datasets
OUTPUT_PATH=/data/jupiter/alex.li/results
#DATASET="Spring_hitchhiker_random"
DATASET=Jupiter_halo_rgbnir_stereo_train_20230710/
JCVML_PATH=/home/alex.li/workspace/JupiterCVML/europa/base/src/europa
# SUBSAMPLE=1000000
cd $JCVML_PATH
# python /home/alex.li/workspace/scripts/data/fake_master.py $SUBSAMPLE
python dl/scripts/predictor.py \
    --csv-path ${DATASET_PATH}/${DATASET}/64ac85c10c98f3c1acb8a408_master_annotations.csv \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --dataset ${DATASET} \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/binary_dust.csv \
    --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/v471_rd_2cls_dustseghead_0808/job_quality_val_bestmodel.pth \
    --output-dir ${OUTPUT_PATH}/${DATASET}/results_0808 \
    --side-left-tire-mask /home/li.yu/code/JupiterEmbedded/src/dnn_engine/data/side_left_mask.png \
    --side-right-tire-mask /home/li.yu/code/JupiterEmbedded/src/dnn_engine/data/side_right_mask.png \
    --model brtresnetpyramid_lite12 \
    --merge-stop-class-confidence 0.35 \
    --states-to-save '' \
    --input-dims 3 \
    --run-productivity-metrics \
    --batch-size 20 \
    --num-workers 32 \
    --dust-class-metrics \
    --dust-mask "NO MASK INVALID PATH" \
    --input-mode debayeredRGB \
    --gpu all;

# --dust-mask ${JCVML_PATH}/dl/config/tire_masks/dust_rear_triangle_mask_fullres.png \
# --use-depth-threshold \
