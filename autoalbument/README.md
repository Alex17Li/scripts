# Find augmentation policy

https://albumentations.ai/docs/autoalbument/how_to_use/


## Running

docker run -it --rm --gpus all --ipc=host -v ~/git/JupiterCVML/autoalbument:/config -v /mnt/datasets/halo_rgb_stereo_train_v6_1/halo_rgb_stereo_train_v6_1:/home/autoalbument/data -u $(id -u ${USER}):$(id -g ${USER}) ghcr.io/albumentations-team/autoalbument:latest
