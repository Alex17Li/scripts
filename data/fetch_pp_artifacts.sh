#!/bin/bash
#SBATCH --job-name=fetch_pp_artifacts
#SBATCH --output=/home/%u/workspace/logs/%A_%x.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@bluerivertech.com

# TODO: load your environment here. This assumes that you have loaded your environment in .bashrc
source ~/.bashrc

# TODO: Update the local path of JupiterCVML repo
cd /home/${USER}/workspace/JupiterCVML/europa/base/src/europa/dl/dataset

python fetch_pp_artifacts.py \
--output-path /data/jupiter/alex.li/datasets/spring_dust_data_test \
--master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/3ea359f0c3fd28d093bf41b44a016d15_d76b3d3f580bfc71d0f9e6ad8d992f2f/64a87abee30a2c394883bc62_master_annotations.csv

