#!/bin/bash
#SBATCH --job-name=unsup_dust_analysis
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
module load pytorch/1.12.0+cuda11.3
conda activate cvml

cd /home/${USER}/git/scripts/data

OUTPUT_PATH=/data/jupiter/$USER/results
JCVML_PATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
cd $JCVML_PATH

DATASET=all_jupiter_data_stratified
ANNOTATIONS_PATH=cleaned_cleaned_annotations.csv

# python /home/${USER}/git/scripts/data/clean_dataset.py $DATASET_PATH/$DATASET annotations.csv
# python /home/$USER/git/scripts/data/fake_master.py ${DATASET_PATH}/$DATASET

python dl/scripts/predictor.py \
    --csv-path ${DATASET_PATH}/${DATASET}/$ANNOTATIONS_PATH \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --dataset whatever \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/binary_dust.csv \
    --restore-from /data/jupiter/li.yu/exps/driveable_terrain_model/v471_rd_2cls_dustseghead_0808/job_quality_val_bestmodel.pth \
    --output-dir ${OUTPUT_PATH}/${DATASET}/results_0808_3 \
    --merge-stop-class-confidence 0.35 \
    --input-dims 3 \
    --batch-size 32 \
    --dust-class-metrics \
    --run-productivity-metrics \
    --num-workers 12 \
    --model brtresnetpyramid_lite12 \
    --dust-mask "NO MASK" \
    --input-mode debayeredRGB \
    --tqdm \
    --states-to-save \
    --gpu all;
