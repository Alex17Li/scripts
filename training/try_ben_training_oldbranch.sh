#!/bin/bash
#SBATCH --job-name=oldeuropa
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH --exclude=stc01sppamxnl004
#SBATCH --mem-per-gpu=60G
source /home/$USER/.bashrc
conda activate cvml

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CILK_NWORKERS=1
export TBB_MAX_NUM_THREADS=1
export PL_GLOBAL_SEED=304
export COLUMNS=100

EXP=${SLURM_JOB_ID}

cd /home/$USER/git/JupiterCVML/europa/base/src/europa
# git checkout -b "ds_v6_1_4x_human_train" b5a3d760cd825958eb5c64e571bfdd2dff8b2902
python dl/scripts/trainer.py --csv-path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_1/master_annotations_dedup.csv --dataset halo_rgb_stereo_train_v6_1 --data-dir /data2/jupiter/datasets/halo_rgb_stereo_train_v6_1 --snapshot-dir /mnt/sandbox1/ben.cline/output --output-dir /mnt/sandbox1/alex.li/ds_v6_1_4x_human_$EXP --exp-name bc_sandbox_2023 --run-id ds_v6_1_4x_human_$EXP --val-set-ratio 0.05 --gpu 0,1,2,3 --input-mode rectifiedRGB --early-stop-patience 100 --input-dims 3 --batch-size 72 --num-workers 28 --epochs 100 --tqdm --save-pred-every 250000 --num-steps 20000000 --model brtresnetpyramid_lite12 --val-csv dl/config/val_ids/Jupiter_val_ids_v4_63_geohash.csv --notes "Bias true. Remove black bar images. Powerful alb. Fix train bug. Road mapped to driveable. New Scheduler and Optimizer. CnP no rotate multi-human and no paste on human" --human-augmentation "{\"use\": false}" --hard-sampling "{\"use\": false, \"warmup_epochs\": 0, \"min_pix_count\": 200, \"gamma\": 1.0, \"multiplier\": 7.0, \"safety_max_weight\": 30.0, \"productivity\": true, \"depth_threshold\": 0.3, \"max_area_denominator\": 2000, \"productivity_max_weight\": 30.0, \"use_max_weights\": false, \"running_weight\": 0.5}" --normalization-params "{\"policy\": \"tonemap\", \"alpha\": 0.25, \"beta\": 0.9, \"gamma\": 0.9, \"eps\": 1e-6}" --lr-scheduler-name cosinelr --lr-scheduler-parameters "{\"steplr_step_size\": 7, \"steplr_gamma\": 0.1, \"cosinelr_T_max\": 100, \"cosinelr_eta_min\": 1e-6, \"cycliclr_base_lr\": 1e-5, \"cycliclr_max_lr\": 1e-3, \"cycliclr_step_size_epoch\": 2, \"cycliclr_mode\": \"exp_range\", \"cycliclr_gamma\": 0.97}" --fp16 --weight-decay 0.001 --model-params "{\"num_block_layers\": 2, \"widening_factor\": 2, \"upsample_mode\": \"nearest\", \"bias\": true}" --use-albumentation-transform --half-res-output --loss tvmsl --focalloss-parameters "{\"alpha\":[4.0,1.0,1.0,1.0,5.0,4.0,1.0,1.0], \"gamma\":2.0, \"normalize_class_weights\": false}" --tversky-parameters "{\"fp_weight\":[0.6,0.3,0.6,0.6,0.6,0.7,0.3,0.6], \"fn_weight\":[0.7,0.7,0.4,0.4,2.0,2.8,0.7,0.4], \"class_weight\":[1.0,2.0,1.0,1.0,2.0,2.0,5.0,1.0], \"gamma\":1.0, \"tversky_loss_weight\": 1.0, \"focal_loss_weight\": 1.0, \"use_pixel_count_mask\": true, \"image_wise\": true, \"normalize_class_weights\": true, \"use_sum_method\": true, \"depth_thresh\": -1, \"productivity_weight\": 7.0, \"productivity_prob_thresh\": -1, \"productivity_depth_thresh\": -1}" --label-map-file dl/config/label_maps/eight_class_train_dust_light_as_sky_birds_as_driveable_new_classes.csv --learning-rate 1e-3 --num-classes 8 --crop-transform-str "[[\"imgaug.augmenters.CropToFixedSize\", {\"width\": 640, \"height\": 512, 'position': 'uniform'}]]" --weighted-sampling "\"\"" --restore-from /mnt/sandbox1/ben.cline/output/bc_sandbox_2023/cls_dust_light_as_sky_512_640_rgb_no_human_augs_2/bc_sandbox_2023_val_bestmodel.pth --input-size 512,640
