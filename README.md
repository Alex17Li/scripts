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
export OUTPUT_PATH=/mnt/sandbox1/$USER/results/
export DATASET_PATH=/data/jupiter/datasets
source /home/$USER/git/scriptsnotebook_config.bashrc
# conda setup #
conda activate cvml
```
