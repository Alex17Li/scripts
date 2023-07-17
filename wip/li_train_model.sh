#!/bin/bash
#SBATCH --job-name=rgbd_training
#SBATCH --output=/home/li.yu/code/scripts/v511rd_4cls_msmltv02prodl002_0614.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=150:00:00

# activate virtual env
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
# conda activate shank
conda activate pytorchlightning
# conda activate knowledgedistillation
# conda activate pytorch1.10

# add working dir
export PYTHONPATH=/home/li.yu/code/JupiterCVML/europa/base/src/europa

# enter working directory
cd /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/scripts

# login in to wandb
export WANDB_API_KEY=d4bf3100cfc878f1936280afad8461b0717f340f
wandb login

# experiment name
EXP='v511rd_4cls_msmltv02prodl002_0614'



# train seg model, on master branch
python trainer.py \
    --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_11/epoch0_5_30_focal05_master_annotations.csv \
    --dataset Jupiter_train_v5_11 \
    --data-dir /data/jupiter/datasets/Jupiter_train_v5_11/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/four_class_train.csv \
    --exp-name vehicle_cls \
    --model brtresnetpyramid_lite12 \
    --model-params "{\"num_block_layers\": 2, \"widening_factor\": 2, \"upsample_mode\": \"nearest\", \"bias\": true}" \
    --optimizer adamw \
    --weight-decay 1e-5 \
    --learning-rate 1e-3 \
    --lr-scheduler-name cosinelr \
    --lr-scheduler-parameters '{"cosinelr_T_max": 60, "cosinelr_eta_min": 1e-6}' \
    --epochs 60 \
    --early-stop-patience 12 \
    --batch-size 64 \
    --val-set-ratio 0.05 \
    --losses '{"msml": 1.0, "tv": 0.2, "prodl": 0.02}' \
    --multiscalemixedloss-parameters '{"scale_weight":0.1, "dust_weight":0.5, "dust_scale_weight":0.05}' \
    --focalloss-parameters '{"alpha":[1.0,1.0,1.0,1.0], "gamma":2.0}' \
    --tversky-parameters '{"fp_weight":[0.6,0.3,0.3,0.6], "fn_weight":[0.4,0.7,0.7,0.4], "class_weight":[1.5,3.0,2.0,1.0], "gamma":1.0}' \
    --productivity-loss-params '{"depth_thresh": 0.35, "prob_thresh": 0.01}' \
    --use-albumentation-transform \
    --night-model '{"use": false, "dark_pix_threshold": 10}' \
    --gpu 0,1,2,3 \
    --num-workers 16 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --snapshot-dir /mnt/sandbox1/li.yu/exps \
    --resume-from-snapshot False \
    --restore-from '' \
    --num-steps 2000000 \
    --save-pred-every 2000000 \
    --output-dir /mnt/sandbox1/li.yu/exps/vehicle_cls/${EXP} \
    --val-csv /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/val_ids/Jupiter_val_ids_v4_63_geohash.csv \
    --weighted-sampling '{"birds": 1.0,
                        "tiny_humans": 0.0, "tiny_human_pixels": 30,
                        "tiny_vehicles": 0.0, "tiny_vehicle_pixels": 100,
                        "humans": 1.0, "human_pixels": [100, 5000],
                        "occluded_humans": 5.0, "occluded_human_pixels": [100, 2000],
                        "reverse_humans": 5.0, "reverse_human_pixels": [50, 2000],
                        "triangle_humans": 5.0, "triangle_human_pixels": [50, 2000],
                        "day_vehicles": 2.0, "day_vehicle_pixels": [3000, 100000],
                        "night_vehicles": 5.0, "night_vehicle_pixels": [3000, 100000],
                        "airborne_debris": 2.0, "airborne_debris_pixels": [100, 100000]}' \
    --human-augmentation '{"use": true, "sample_ratio": 0.30, "non_standing_aspect_ratio_threshold": 0.5,
                      "load_from_human_aug_csv": "/data/jupiter/li.yu/data/Jupiter_train_v5_11/trainrd05_humanaug.csv", 
                      "save_to_human_aug_csv": "/data/jupiter/li.yu/data/Jupiter_train_v5_11/trainrd05_humanaug.csv",
                      "same_operation_time": false, "same_brightness": true, "brightness_range": 0.05,
                      "use_standing_human": true, "standing_min_pixels": 50, "standing_max_pixels": 20000,
                      "use_laying_down_human": true, "laying_down_min_pixels": 50, "laying_down_max_pixels": 15000,
                      "use_multi_human": true, "only_non_occluded": true, "blend_mode": "vanilla",
                      "rotate_human": true, "rotate_degree": 30, "jitter_human": false, "jitter_range": 0.15,
                      "depth_aware": false, "cutout_rate": 0.20, "max_cutout": 0.6,
                      "use_vehicle": false, "vehicle_sample_ratio": 0.05, "vehicle_min_pixels": 3000, "vehicle_max_pixels": 100000}' \
    --run-id ${EXP};

    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_7/epoch0_5_30_focal05_notiny_onlyleft_master_annotations.csv \
    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_8/epoch0_5_30_focal05_master_annotations.csv \
    # --multiscalemixedloss-parameters '{"scale_weight":0.1, "dust_weight":0.5, "dust_scale_weight":0.05}' \
    # --focalloss-parameters '{"alpha":[1.0,1.0,0.5,0.5,1.0,2.0,1.0], "gamma":2.0}' \
    # --tversky-parameters '{"fp_weight":[0.1,0.0,0.0,0.1], "fn_weight":[0.9,1.0,1.0,0.9], "class_weight":[0.0,1.0,1.0,0.0], "gamma":1.0, "tversky_weight":0.01, "use_msl":false}' \
    # --use-albumentation-transform \
    # --imgaug-transform-str "[[\"imgaug.augmenters.OneOf\", [[\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"translate_percent\": {\"x\": (-0.2, 0.2),\"y\": (-0.2, 0.2)},\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"scale\": (0.7, 1.5),\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"rotate\": (-10, 10),\"mode\": \"reflect\",\"order\": 0}]]], [\"imgaug.augmenters.OneOf\", [[\"imgaug.augmenters.Multiply\", {\"mul\": (0.8,1.25),\"per_channel\": 0.2}], [\"imgaug.augmenters.LogContrast\", {\"gain\": (0.6,1.4)}], [\"imgaug.augmenters.SigmoidContrast\", {\"gain\": (3,10), \"cutoff\": (0.4,0.6)}]]], [\"dl.augmentations.augmentations.Clip\",{\"lower\": 0,\"upper\": 1}]]" \
    # --restore-from '/data/jupiter/li.yu/exps/driveable_terrain_model/v51rd_7cls_imgaug_highbz_100epoch_0123/vehicle_cls_75_epoch_model.pth' \


# # train seg model, on master branch, ddp
# python trainer_pl.py \
#     --seed 42 \
#     --csv-path /data/jupiter/datasets/Jupiter_train_v5_5_devised/master_annotations.csv \
#     --dataset Jupiter_train_v5_5_devised \
#     --data-dir /data/jupiter/datasets/Jupiter_train_v5_5_devised \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
#     --exp-name vehicle_cls \
#     --model brtresnetpyramid_lite12 \
#     --optimizer adamw \
#     --weight-decay 1e-5 \
#     --learning-rate 1e-3 \
#     --lr-scheduler-name cosinelr \
#     --epochs 60 \
#     --early-stop-patience 12 \
#     --batch-size 24 \
#     --val-set-ratio 0.05 \
#     --loss msl \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --gpu 0,1,2,3 \
#     --num-workers 8 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --imgaug-transform-str "[[\"imgaug.augmenters.OneOf\", [[\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"translate_percent\": {\"x\": (-0.2, 0.2),\"y\": (-0.2, 0.2)},\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"scale\": (0.7, 1.5),\"mode\": \"reflect\",\"order\": 0}], [\"dl.augmentations.augmentations.AffineLabelAugmentation\", {\"rotate\": (-10, 10),\"mode\": \"reflect\",\"order\": 0}]]], [\"imgaug.augmenters.OneOf\", [[\"imgaug.augmenters.Multiply\", {\"mul\": (0.8,1.25),\"per_channel\": 0.2}], [\"imgaug.augmenters.LogContrast\", {\"gain\": (0.6,1.4)}], [\"imgaug.augmenters.SigmoidContrast\", {\"gain\": (3,10), \"cutoff\": (0.4,0.6)}]]], [\"dl.augmentations.augmentations.Clip\",{\"lower\": 0,\"upper\": 1}]]" \
#     --snapshot-dir /mnt/sandbox1/li.yu/exps \
#     --resume-from-snapshot False \
#     --restore-from '' \
#     --num-steps 2000000 \
#     --save-pred-every 2000000 \
#     --output-dir /mnt/sandbox1/li.yu/exps/vehicle_cls/${EXP} \
#     --val-csv /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/val_ids/Jupiter_val_ids_v4_63_geohash.csv \
#     --human-augmentation '{"use": true, "sample_ratio": 0.35, "non_standing_aspect_ratio_threshold": 0.5,
#                       "load_from_human_aug_csv": "/data/jupiter/li.yu/data/Jupiter_train_v5_5_devised/train_humanaug_ddp.csv", 
#                       "save_to_human_aug_csv": "/data/jupiter/li.yu/data/Jupiter_train_v5_5_devised/train_humanaug_ddp.csv",
#                       "same_operation_time": false, "same_brightness": true, "brightness_range": 0.05,
#                       "use_standing_human": true, "standing_min_pixels": 50, "standing_max_pixels": 20000,
#                       "use_laying_down_human": true, "laying_down_min_pixels": 50, "laying_down_max_pixels": 15000,
#                       "use_multi_human": true, "only_non_occluded": true, "blend_mode": "vanilla",
#                       "rotate_human": true, "rotate_degree": 30, "jitter_human": false, "jitter_range": 0.15,
#                       "depth_aware": false, "cutout_rate": 0.20, "max_cutout": 0.6,
#                       "use_vehicle": false, "vehicle_sample_ratio": 0.05, "vehicle_min_pixels": 50, "vehicle_max_pixels": 1000000}' \
#     --run-id ${EXP};


# # train seg model, on GRETZKY_2106_halo_seg_color_correction branch
# python trainer.py \
#     --csv-path /data/jupiter/datasets/Jupiter_halo_implement_labeled_data_train_06162023_stereo_v2/master_annotations.csv \
#     --dataset Jupiter_halo_implement_labeled_data_train_06162023_stereo_v2 \
#     --data-dir /data/jupiter/datasets/Jupiter_halo_implement_labeled_data_train_06162023_stereo_v2/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/eight_class_train.csv \
#     --num-classes 8 \
#     --exp-name vehicle_cls \
#     --model brtresnetpyramid_lite12 \
#     --optimizer adamw \
#     --weight-decay 1e-3 \
#     --learning-rate 1e-3 \
#     --lr-scheduler-name cosinelr \
#     --lr-scheduler-parameters "{\"steplr_step_size\": 7, \"steplr_gamma\": 0.1, \"cosinelr_T_max\": 100, \"cosinelr_eta_min\": 1e-6, \"cycliclr_base_lr\": 1e-5, \"cycliclr_max_lr\": 1e-3, \"cycliclr_step_size_epoch\": 2, \"cycliclr_mode\": \"exp_range\", \"cycliclr_gamma\": 0.97}" \
#     --epochs 100 \
#     --early-stop-patience 12 \
#     --model-params "{\"num_block_layers\": 2, \"widening_factor\": 2, \"upsample_mode\": \"nearest\", \"bias\": true}" \
#     --batch-size 64 \
#     --input-size 512,640 \
#     --val-set-ratio 0.05 \
#     --loss tvmsl \
#     --focalloss-parameters "{\"alpha\":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], \"gamma\":2.0, \"normalize_class_weights\": false}" \
#     --tversky-parameters "{\"fp_weight\":[0.6,0.3,0.6,0.6,0.6,0.3,0.3,0.6], \"fn_weight\":[0.4,0.7,0.4,0.4,0.4,0.7,0.7,0.4], \"class_weight\":[1.0,2.0,1.0,1.0,2.0,10.0,5.0,1.0], \"gamma\":1.0, \"tversky_loss_weight\": 1.0, \"focal_loss_weight\": 1.0, \"use_pixel_count_mask\": true, \"image_wise\": true, \"normalize_class_weights\": true, \"use_sum_method\": true, \"depth_thresh\": -1, \"productivity_weight\": 7.0, \"productivity_prob_thresh\": -1, \"productivity_depth_thresh\": -1}" \
#     --use-albumentation-transform \
#     --crop-transform-str "[[\"imgaug.augmenters.CropToFixedSize\", {\"width\": 640, \"height\": 512, 'position': 'uniform'}]]" \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --gpu 0,1,2,3 \
#     --num-workers 16 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --snapshot-dir /mnt/sandbox1/li.yu/exps \
#     --resume-from-snapshot False \
#     --restore-from '/data/jupiter/li.yu/exps/driveable_terrain_model/v511rd_8cls_cc_640_0517/vehicle_cls_val_bestmodel.pth' \
#     --num-steps 2000000 \
#     --save-pred-every 2000000 \
#     --output-dir /mnt/sandbox1/li.yu/exps/vehicle_cls/${EXP} \
#     --val-csv /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/val_ids/Jupiter_val_ids_v4_63_geohash.csv \
#     --weighted-sampling '{}' \
#     --human-augmentation '{"use": false, "sample_ratio": 0.30, "non_standing_aspect_ratio_threshold": 0.5,
#                       "load_from_human_aug_csv": "/data/jupiter/li.yu/data/Jupiter_train_v5_11/halo0517_fix640_humanaug.csv", 
#                       "save_to_human_aug_csv": "/data/jupiter/li.yu/data/Jupiter_train_v5_11/halo0517_fix640_humanaug.csv",
#                       "same_operation_time": false, "same_brightness": true, "brightness_range": 0.05,
#                       "use_standing_human": true, "standing_min_pixels": 50, "standing_max_pixels": 20000,
#                       "use_laying_down_human": true, "laying_down_min_pixels": 50, "laying_down_max_pixels": 15000,
#                       "use_multi_human": true, "only_non_occluded": true, "blend_mode": "vanilla",
#                       "rotate_human": true, "rotate_degree": 30, "jitter_human": false, "jitter_range": 0.15,
#                       "depth_aware": false, "cutout_rate": 0.20, "max_cutout": 0.6,
#                       "use_vehicle": false, "vehicle_sample_ratio": 0.05, "vehicle_min_pixels": 3000, "vehicle_max_pixels": 100000}' \
#     --run-id ${EXP};

    # --csv-path /data/jupiter/li.yu/data/Jupiter_train_v5_11/trainrd05_color_transfer.csv \
    # --csv-path /data/jupiter/datasets/Jupiter_halo_labeled_data_20230512_train_stereo_640_768_single_ds/master_annotations.csv \
    # --csv-path /data/jupiter/datasets/Jupiter_halo_labeled_data_20230517_train_stereo_640_768_single_ds_pmehta_oc_correctscale/master_annotations_0512_0516_0517.csv \


# # train dust head, on GRETZKY_1385_dust_head branch
# python trainer.py \
#     --csv-path /data/jupiter/datasets/Jupiter_train_v5_1/master_annotations.csv \
#     --dataset Jupiter_train_v5_1 \
#     --data-dir /data/jupiter/datasets/ \
#     --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#     --freeze-encoder \
#     --additional-head-output \
#     --num-classes 8 \
#     --num-head-classes 1 \
#     --val-set-ratio 0.05 \
#     --gpu 0,1,2,3 \
#     --input-mode RGBD \
#     --early-stop-patience 30 \
#     --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#     --human-augmentation '{"use": false}' \
#     --night-model '{"use": false, "dark_pix_threshold": 10}' \
#     --snapshot-dir /mnt/sandbox1/li.yu/exps \
#     --resume-from-snapshot True \
#     --restore-from '/data/jupiter/li.yu/exps/driveable_terrain_model/v58rd_8cls_0315/vehicle_cls_val_bestmodel.pth' \
#     --input-dims 4 \
#     --batch-size 80 \
#     --num-workers 16 \
#     --epochs 30 \
#     --exp-name vehicle_cls \
#     --run-id ${EXP} \
#     --output-dir /mnt/sandbox1/li.yu/exps/vehicle_cls/${EXP} \
#     --save-pred-every 200000 \
#     --num-steps 200000 \
#     --loss 'mse' \
#     --model brtresnetpyramid_lite12 \
#     --val-csv /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/val_ids/Jupiter_val_ids_v4_63_geohash.csv;


mv /mnt/sandbox1/li.yu/exps/vehicle_cls/${EXP} /data/jupiter/li.yu/exps/driveable_terrain_model/
mv /mnt/sandbox1/li.yu/exps/vehicle_cls/${EXP}/* /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/


# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
