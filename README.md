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

.bashrc contents

```
export WANDB_API_KEY=...
export PYTHONPATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
export AWS_PROFILE=jupiter_prod
set -o vi
# conda setup #
conda activate cvml
```
