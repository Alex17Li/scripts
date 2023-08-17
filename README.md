# Scripts

Expected directory struture

```
/home/$USER/
           /git/JupiterCVML
               /scripts
           /logs
           /data
           /.bashrc
```

Example .bashrc contents

```
export WANDB_API_KEY=...
export PYTHONPATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
export OUTPUT_PATH=/mnt/sandbox1/$USER/results/
export DATASET_PATH=/data/jupiter/datasets
shopt -s direxpand # Can use path variables with autocomplete for ls, cd, etc
export AWS_PROFILE=jupiter_prod
export AWS_CA_BUNDLE="/home/$USER/.certs/bundle.crt"
set -o vi
# conda setup #
conda activate cvml
```
