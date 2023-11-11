#!/usr/bin/env python
# coding: utf-8

import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

import sys
from cv.core.image_quality_server_side import ImageQuality
from dl.utils.config import DEFAULT_TONEMAP_PARAMS

# Please update the default location as you deem fit
cvml_path = '/home/alex.li/git/JupiterCVML/europa/base/src/europa'
dataset_idx = 0
directory = ['/data/jupiter/datasets/bad_iq_halo_labelbox_plus_exposure', '/data2/jupiter/datasets/20231017_halo_rgb_labeled_excluded_bad_iq', '/data/jupiter/datasets/iq_2023_v5_anno'][dataset_idx]
csv_name = ['654a5bb2e89875bddc714dd2_master_annotations.csv', '653a7a0a3c2d8ab221f6d915_master_annotations.csv','64dfcc1de5a41169c7deb205_master_annotations.csv'][dataset_idx]

dset_name = directory.split('/')[-1]
save_path='/mnt/sandbox1/alex.li/iq_results'

side_left_tire_mask = f'{cvml_path}/cv/core/tire_masks/side_left_iq_mask.png'
side_right_tire_mask = f'{cvml_path}/cv/core/tire_masks/side_right_iq_mask.png'

iq = ImageQuality(num_workers=mp.cpu_count() // 2,
                  use_progress=True,
                  side_left_tire_mask_path = side_left_tire_mask,
                  side_right_tire_mask_path = side_right_tire_mask,
                  normalization_params=DEFAULT_TONEMAP_PARAMS,
                  dataset=dset_name,
                  save_path=save_path)
stereo_df = pd.read_csv(os.path.join(directory, csv_name), low_memory=False)
stereo_df = stereo_df.drop_duplicates(['id'])
print(stereo_df.shape)

print('start calculation')
output_path = f'{save_path}/{dset_name}'
output_file = f'{output_path}/iq.csv'
if not os.path.exists(output_file) or True:
    df = iq.from_df(stereo_df, directory, use_progress=False)
else:
    df = pd.read_csv(output_file)
print('finish calculation')

print('start column mapping')
print('process iq'); time.sleep(3);
df['iq'] = df.image_quality.parallel_apply(lambda x: x.algorithm_output)
print('process iq_features'); time.sleep(3);
df['iq_features'] = df.image_quality.parallel_apply(lambda x: x.algorithm_features)
print('process iq_features_total'); time.sleep(3);
df['iq_features_total'] = df.iq_features.parallel_apply(lambda x: x['image_features']['total'])
print('process iq_features_low'); time.sleep(3);
df['iq_features_low'] = df.iq_features.parallel_apply(lambda x: x['image_features']['low'])
print('process iq_features_mid'); time.sleep(3);
df['iq_features_mid'] = df.iq_features.parallel_apply(lambda x: x['image_features']['mid'])
print('process iq_features_high'); time.sleep(3);
df['iq_features_high'] = df.iq_features.parallel_apply(lambda x: x['image_features']['high'])
print('process iq_features_sharpness'); time.sleep(3);
df['iq_features_sharpness'] = df.iq_features.parallel_apply(lambda x: x['image_features']['sharpness'])
print('process iq_features_smudge'); time.sleep(3);
df['iq_features_smudge'] = df.iq_features.parallel_apply(lambda x: x['image_features']['smudge'])
print('process iq_features_smudge_reason'); time.sleep(3);
df['iq_features_smudge_reason'] = df.iq_features.parallel_apply(lambda x: x['image_features']['smudge_reason'])
print('process iq_process_time'); time.sleep(3);
df['iq_process_time'] = df.image_quality.parallel_apply(lambda x: x.algorithm_process_time)
print('finish column mapping')

if 'iq_ground_truth' in df:
    df['binary_iq'] = df.iq.apply(lambda x: 'iq' if x != 'good' else 'non_iq')
    df['binary_iq_ground_truth'] = df.iq_ground_truth.apply(lambda x: 'iq' if x != 'good' else 'non_iq')

print(df.iq.value_counts())
