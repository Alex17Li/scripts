#!/bin/bash
#SBATCH --job-name=pack_perception
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48000

source ~/.bashrc

DATA_FOLDER=/data/jupiter/datasets
DATASET_NAME=halo_humans_on_path_test_v6_1
JUPITERCVML_DIR=~/git/JupiterCVML
module load singularity

# TODO: Update WANDB_API_KEY to your key, if you want to
#export WANDB_API_KEY="xxxx"
#export WANDB_DIR="/mnt/sandbox1/${USER}"


#############################
# End of configuration.     #
# Main script starts below. #
#############################

export BRT_ENV=prod
export AWS_DEFAULT_REGION=us-west-2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ -n "$JUPITERCVML_DIR" ]
then
    EUROPA_DIR=$JUPITERCVML_DIR/europa/base/src/europa/
    FILES_DIR=$JUPITERCVML_DIR/europa/base/files/
    EXECUTABLES_DIR=$JUPITERCVML_DIR/europa/base/executables
    _ADDITIONAL_BINDS=$_ADDITIONAL_BINDS,$EUROPA_DIR:/src/europa,$FILES_DIR:/files,$EXECUTABLES_DIR:/executables
fi

DATA=$DATA_FOLDER/$DATASET_NAME
echo $DATA
# python /home/${USER}/git/scripts/data/data_downloader.py  $DATASET_NAME -d $DATA_FOLDER

# TODO: Choose the appropriate command and modify to your tastes/needs

# For rev1
# srun --kill-on-bad-exit --gpu-bind=verbose,single:1 \
# singularity run \
#     --nv --bind /data,/data2$_ADDITIONAL_BINDS \
#     /data2/jupiter/singularity/jupiter-pack-perception/libs_halo_kf-cvml_master.sif \
# python3 -m dl.dataset.pack_perception.ml_pack_perception \
#     --data-dir $DATA --csv-path $DATA/annotations.csv \
#     --calib-tracker-csv /files/calibration/motec_calibration_tracker_2019.csv \
#     --cam-calibration-path /files/calibration \
#     --batch-size 64 --multiprocess-workers 64 --pandarallel-workers 64 --gpu 0 \
#     --models 512,1024=20230710_depth_model_1xCvxUp_LHFeat_ep35_v3.ckpt --model-type lite --max-disp 192

#For Halo   
srun --kill-on-bad-exit --gpu-bind=verbose,single:1 \
singularity run \
    --nv --bind /data,/data2$_ADDITIONAL_BINDS \
    /data2/jupiter/singularity/jupiter-pack-perception/libs_halo_kf-cvml_master.sif \
python3 -m dl.dataset.pack_perception.ml_pack_perception \
    --data-dir $DATA --csv-path $DATA/annotations.csv \
    --calib-tracker-csv /files/calibration/motec_calibration_tracker_2019.csv \
    --cam-calibration-path /files/calibration \
    --batch-size 64 --multiprocess-workers 64 --pandarallel-workers 64 --gpu 0 \
    --models 512,768=ml_512x768_v3_full_rgb_halo_depth_10062023.ckpt 512,640=ml_512x640_v3_full_rgb_halo_depth_10062023.ckpt \
    --model-type full --max-disp 384 \
    --run-oc
