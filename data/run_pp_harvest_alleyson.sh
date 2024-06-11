#!/bin/bash
#SBATCH --job-name=pack_perception
#SBATCH --output=/mnt/sandbox1/%u/logs/%j_%x.batch.txt
#SBATCH --error=/mnt/sandbox1/%u/logs/%j_%x.batch.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=96000
#SBATCH --requeue

# TODO: In the line above, replace {USER} with your username.
# If you wish, change the number of tasks per node, but using 8 GPUs is
# probably the most optimal.

# TODO: Change the data folder and dataset that you are trying to run PP on
# Possible values for RESUME_MODE:
#   fresh: delete existing partitioning and restart all of PP
#   redo-ocal: delete ocal results and restart all of PP
#   redo-depth: keep ocal results, delete PP artifacts, and restart depth inference
#   existing: continue running PP with the current partitioning and partial results
DATA_FOLDER=/data2/jupiter/datasets
DATASET_NAME=halo_productivity_combined_no_mislocalization
JUPITERCVML_DIR=~/git/JupiterCVML
RESUME_MODE=existing


#############################
# End of configuration.     #
# Main script starts below. #
#############################

# Everything below this point must pass (except for deletion of old files) for
# PP to run smoothly.
set -e

module load anaconda singularity

# Change your conda environment if you want to use a different one, but there
# generally isn't any need to. This is only used for downloading and paritioning
# a dataset.
conda activate /mnt/sandbox1/anirudh.vegesana/conda_envs/pp/

# This command is needed for halo_kf for now. It will no longer be a
# dependency in the future, but is included here to avoid issues for now:
pip install --user numpy-quaternion

# Set important environmental variables
export BRT_ENV=prod
export AWS_PROFILE=jupiter_prod_engineer-425642425116
export AWS_DEFAULT_REGION=us-west-2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Get useful locations inside of JupiterCVML if the variable is set
if [ -n "$JUPITERCVML_DIR" ]
then
    EUROPA_DIR=$JUPITERCVML_DIR/europa/base/src/europa/
    export PYTHONPATH=$EUROPA_DIR:$PYTHONPATH
    FILES_DIR=$JUPITERCVML_DIR/europa/base/files/
    EXECUTABLES_DIR=$JUPITERCVML_DIR/europa/base/executables
    _ADDITIONAL_BINDS=$_ADDITIONAL_BINDS,$EUROPA_DIR:/src/europa,$FILES_DIR:/files,$EXECUTABLES_DIR:/executables
    DOWNLOAD_DATASET_CMD="python3 $EUROPA_DIR/dl/dataset/download.py"
    PARITION_DATASET_CMD="python3 $EUROPA_DIR/dl/dataset/pack_perception/partition_dataset.py"
else
    DOWNLOAD_DATASET_CMD=/data2/jupiter/download_ds
    PARITION_DATASET_CMD=/data2/jupiter/partition_dataset
fi

DATA=$DATA_FOLDER/$DATASET_NAME
NUM_PARTITIONS=$SLURM_NTASKS

echo Data folder: $DATA

# Download supporting ocal data.
# TODO: Remove this if statement if not using ocal.
if [ ! -d "$DATA/online_calibration_data/images" ]
then
    mkdir -p $DATA/online_calibration_data/images

    if [ -n "$JUPITERCVML_DIR" ]
    then
        python3 $EUROPA_DIR/dl/dataset/pack_perception/download_ocal_data.py $DATA
    fi
fi

# Delete PP outputs if requested.
set +e
if [ -n "$SLURM_RESTART_COUNT" ] && [ $SLURM_RESTART_COUNT -ne 0 ]
then
    echo SLURM reset. Ignoring resume mode and continuing run.
elif [ "$RESUME_MODE" = "fresh" ]
then
    echo Deleting existing partitions.
    rm -r \
        $DATA/partitions \
        $DATA/processed \
        $DATA/master_annotations.csv
elif [ "$RESUME_MODE" = "redo-ocal" ]
then
    echo Deleting ocal results.
    rm -r \
        $DATA/partitions/*/master_annotations.csv \
        $DATA/partitions/*/annotations_ocal.csv \
        $DATA/partitions/*/online_calibration_data/ocal_df.csv \
        $DATA/processed \
        $DATA/master_annotations.csv
elif [ "$RESUME_MODE" = "redo-depth" ]
then
    echo Deleting depth inference results.
    rm -r \
        $DATA/partitions/*/master_annotations.csv \
        $DATA/processed \
        $DATA/master_annotations.csv
elif [ "$RESUME_MODE" = "existing" ]
then
    if [ -d "$DATA/partitions" ]
    then
        echo Resuming existing PP run.
    else
        echo Starting new PP run.
    fi
else
    echo Unknown resume mode $RESUME_MODE. Using existing.
fi
set -e

# Create processed folder if it doesn't exist.
mkdir -p $DATA/processed

# Copy the calibration data into the right place if it isn't there already.
if [ -n "$JUPITERCVML_DIR" -a ! -d "$DATA/processed/calibration" ]
then
    cp -r $FILES_DIR/calibration $DATA/processed/
fi

# Parition the dataset.
if [ ! -d "$DATA/partitions" ]
then
    $PARITION_DATASET_CMD \
        --dataset-folder $DATA \
        --partitions-folder $DATA/partitions \
        --num-partitions $NUM_PARTITIONS \
        partition \
        --use-relative-symlinks false
else
    echo Using existing partitioning.

    $PARITION_DATASET_CMD \
        --dataset-folder $DATA \
        --partitions-folder $DATA/partitions \
        --num-partitions $NUM_PARTITIONS \
        verify
fi

# TODO: Choose the appropriate command and modify to your tastes/needs
# MOST IMPORTANTLY divide the batch size (and maybe workers) to prevent
# OOMs.
# This includes dividing the batch size by 4 if you are using a full
# resolution model.
# Also, delete the code segment above that downloads ocal data if you
# Are not running ocal.

# For Halo (MUST BE RUN WITH THE GIT BRANCH halo_kf)
srun \
    --output=/mnt/sandbox1/%u/logs/%j_%x.%t.txt \
    --error=/mnt/sandbox1/%u/logs/%j_%x.%t.txt \
    --unbuffered \
singularity run \
    --nv --bind /data,/data2$_ADDITIONAL_BINDS \
    /data2/jupiter/singularity/jupiter-pack-perception/main.sif \
python3 -m dl.dataset.pack_perception.ml_pack_perception \
    --data-dir $DATA/partitions/\$SLURM_PROCID --csv-path \\\$DATA_DIR/annotations.csv \
    --calib-tracker-csv /files/calibration/motec_calibration_tracker_2019.csv \
    --cam-calibration-path /files/calibration --ignore-slurm-variables \
    --batch-size 24 --multiprocess-workers 36 --pandarallel-workers 48 --gpu 0 --dataloader-workers 6 \
    --models 512,768=depth_full_finetuned_512x768_Jan25_2024_fixed.ptp \
             512,640=depth_full_finetuned_512x640_Jan25_2024_fixed.ptp \
    --run-oc \
    --backend adk --debayer alleysson --image-only

# Combine the partitions back into the master_annotations.csv
$PARITION_DATASET_CMD \
    --dataset-folder $DATA \
    --partitions-folder $DATA/partitions \
    --num-partitions $NUM_PARTITIONS \
    combine

