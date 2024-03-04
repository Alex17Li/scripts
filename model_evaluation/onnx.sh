#!/bin/bash
#SBATCH --job-name=onnx_convert
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --time=00:01:00

#--SBATCH --partition=cpu
source /home/$USER/.bashrc
# conda create -n onnx -c conda-forge -c pytorch python==3.8 onnx=1.6.0 pytorch=1.4.0 pandas=1.3.* yacs torchvision matplotlib wandb
# conda activate onnx

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CVML_PATH=/home/$USER/git/JupiterCVML
CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/highres/bc_sandbox_2024/repvit_M1.5_512 
CHECKPOINT=bc_sandbox_2024_val_bestmodel.pth

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
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/label_map_nine_class_birds_as_birds.csv  \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --model-params '{"version": "timm/repvit_m1_5.dist_450e_in1k", "fixed_size_aux_output": false, "upsample_mode": "nearest", "in_features": [[4, 64], [8, 128], [16, 256], [32, 512]]}' \
    --dust-output-params '{"dust_head_output": false, "dust_class_ratio": false, "dust_seg_output": false}' \
    --model repvit_lite12 \
    --input-mode rectifiedRGB \
    --batch-size 4 \
    --ignore-deprecation-crash \
    --output-dir ${CHECKPOINT_FULL_DIR};

cd -
conda activate cvml
