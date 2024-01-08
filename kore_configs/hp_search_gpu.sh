set -x

python /home/${USER}/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path /home/${USER}/git/scripts/kore_configs/gpu_seg.yml /home/${USER}/git/scripts/kore_configs/seg_gsam.yml \
    --optimizer.rho 0.025 \
    --optimizer.adaptive false
python /home/${USER}/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path /home/${USER}/git/scripts/kore_configs/gpu_seg.yml /home/${USER}/git/scripts/kore_configs/seg_gsam.yml \
    --optimizer.rho 0.05 \
    --optimizer.adaptive false

python /home/${USER}/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path /home/${USER}/git/scripts/kore_configs/gpu_seg.yml /home/${USER}/git/scripts/kore_configs/seg_gsam.yml \
    --optimizer.rho 0.25 \
    --optimizer.adaptive true
python /home/${USER}/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path /home/${USER}/git/scripts/kore_configs/gpu_seg.yml /home/${USER}/git/scripts/kore_configs/seg_gsam.yml \
    --optimizer.rho 0.1 \
    --optimizer.adaptive false

python /home/${USER}/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path /home/${USER}/git/scripts/kore_configs/gpu_seg.yml /home/${USER}/git/scripts/kore_configs/seg_gsam.yml \
    --optimizer.rho 0.1 \
    --optimizer.adaptive true \
    --optimizer.lr 1e-4
