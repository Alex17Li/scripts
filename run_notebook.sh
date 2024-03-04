#!/bin/bash
#SBATCH --job-name=jup_alex
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00
#SBATCH --mem-per-gpu=10G

# SBATCH --partition=cpu
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=16

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git
# C=stc01sppamxnl003 (Node the notebook is running on)
# Command does not seem to work no pasword?
# ssh -t -t alex.li@harvestlogin001.stc01.bluerivertech.info -L 3989:localhost:3989 ssh $C -L 3989:localhost:3989
jupyter notebook --no-browser --port=3989 --ip=0.0.0.0
# Or run on login node
# jupyter notebook --no-browser --port=8989 --ip=0.0.0.0
