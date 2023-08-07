#!/bin/bash
#SBATCH --job-name=onnx_convert
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --time=00:01:00

#--SBATCH --partition=cpu
source /home/$USER/.bashrc
# conda create -n onnx -c conda-forge -c pytorch python==3.8 onnx=1.6.0 pytorch=1.4.0 pandas=1.3.* yacs torchvision matplotlib wandb
conda activate onnx

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CVML_PATH=/home/$USER/git/JupiterCVML
EXP=dust_trivial_augment_1
CHECKPOINT_PREFIX='dust'
CHECKPOINT_FULL_DIR=${OUTPUT_PATH}/${CHECKPOINT_PREFIX}/${EXP}
CHECKPOINT=${CHECKPOINT_PREFIX}_val_bestmodel.pth

if [ ! -d $CHECKPOINT_FULL_DIR ]; then
    echo checkpoint $CHECKPOINT_FULL_DIR does not exist
    exit 1
fi

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}
echo $CHECKPOINT_FULL_PATH

# --tqdm \
# --augmentations CustomCrop SmartCrop HorizontalFlip TorchColorJitter Resize \

python -m dl.scripts.onnx_converter \
    --csv-path NO \
    --half-res-output \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --model-params "{\"activation\": \"gelu\"}" \
    --dust-output-params '{"dust_head_output": false, "dust_class_ratio": false, "dust_class_confidence_map": true, "zero_dust_ratio": false}' \
    --model brtresnetpyramid_lite12 \
    --batch-size 4 \
    --output-dir ${CHECKPOINT_FULL_DIR};

cd -
conda activate cvml
