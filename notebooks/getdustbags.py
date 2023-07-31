import time
import json
import ast
import os
import datetime
import io
from collections import defaultdict
from tqdm import tqdm
from typing import Hashable

import imageio
import boto3
import pandas as pd
import numpy as np
import imageio
import matplotlib.pyplot as plt
import seaborn as sns

from brtdevkit.core.db.athena import AthenaClient
from brtdevkit.data import Dataset
from timezonefinder import TimezoneFinderL
import pytz
import cv2
from brtdevkit.util.aws.s3 import S3
client = S3()

from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset
from aletheia_dataset_creator.config.dataset_config import LEFT_CAMERAS, ALL_CAMERA_PAIRS_LIST
pd.set_option('display.max_rows', 500)
athena = AthenaClient()
s3 = boto3.resource('s3')
tf = TimezoneFinderL()
from pathlib import Path
home = Path(os.path.expanduser('~'))
data_path = home / 'data' 
df_sequences = pd.read_parquet(data_path / 'df_sequences.parquet')
df_groups_orig: dict[Hashable, list[Hashable]] = df_sequences.groupby('special_notes').groups
df_index_orig = set(df_groups_orig.keys())
for e in [
    'vehicle in dust time dawn/dusk',
    '6508 IQ-test-1',
    'vehicle in dust day time ',
    'vehicle in dust Day',
    '6524 IQ-Test-1',
    '6524 IQ-Test-2',
    'IQ-image to bright',
    # 'Morning dust right side',
    # 'Morning dust right side and oil rig',
    # 'dust',
    'dust right side',
    # 'oil rig',
    'vehicle dust dusk',
]:
    df_index_orig.remove(e)
df_sequences_valid = df_sequences[df_sequences['special_notes'].isin(df_index_orig)]
# rebuild the index and groups
df_groups = df_sequences_valid.groupby('special_notes').groups
df_index = set(df_groups.keys())
def get_image(df_row, collected_on: str):
    if len(df_row) == 0:
        whiteFrame = 255 * np.ones((604, 964, 3), np.uint8)
        font = cv2.FONT_HERSHEY_PLAIN
        whiteFrame = cv2.putText(whiteFrame, collected_on, (50, 400), font, 5, (0,0,0), 5)
        return whiteFrame
    elif isinstance(df_row, pd.DataFrame):
        assert len(df_row) == 1
        df_row = df_row.iloc[0]
    file_name = Path(data_path) / df_row['special_notes'].replace(' ', '_') / str(df_row.id + '.png')
    if not os.path.exists(file_name):
        client.download_file(df_row['s3_bucket'], df_row['s3_key'], file_name)
    im = cv2.imread(str(file_name))
    return im
    
# 1) Download all of the images
def create_video_frames(k: str):
    """
    Given dictionary with image paths creates concatenated image and video and saves to output_dir.
    :param grouped_images: List with a dictionary per group_id
    :param bag_or_drive_name: AnyStr hard_drive_name or bag_name given during data ingestion
    :param output_dir: AnyStr path to save the video directory
    """
    video_dir = Path(data_path) / 'videos' / f"{k.replace(' ', '_')}"
    os.makedirs(video_dir, exist_ok=True)
    video_name = video_dir / "video.mp4"
    if os.path.exists(video_name):
        return
    writer = imageio.get_writer(video_name, fps=1)
    k_df = df_sequences.loc[df_groups[k]].sort_values('collected_on')
    k_groups = k_df.groupby('group_id').groups
    seen = set()
    for row in tqdm(k_df.iterrows()):
        gid = row[1]['group_id']
        if gid in seen:
            continue
        seen.add(gid)
        values = k_groups[gid]
        group = df_sequences.loc[values]
        collected_on_str = str(group.iloc[0].collected_on)[11:19]
        # try:
        # concatenate image Horizontally
        front_pod = np.concatenate(
            (
                get_image(group[group['camera_location'] == 'front-left-left'], collected_on_str),
                get_image(group[group['camera_location'] == 'front-center-left'], collected_on_str),
                get_image(group[group['camera_location'] == 'front-right-left'], collected_on_str),
            ),
            axis=1,
        )
        rear_pod = np.concatenate(
            (
                get_image(group[group['camera_location'] == 'side-left-left'], collected_on_str),
                get_image(group[group['camera_location'] == 'rear-left'], collected_on_str),
                get_image(group[group['camera_location'] == 'side-right-left'], collected_on_str),
            ),
            axis=1,
        )
        # concatenate image vertically
        all_cameras = np.concatenate((front_pod, rear_pod), axis=0)[::4, ::4, ::-1]
        # save concatenated image file
        full_img_name = f"{collected_on_str}.png"
        file_path = os.path.join(video_dir, full_img_name)
        plt.imsave(file_path, all_cameras)
        plt.clf()
        plt.close()
        writer.append_data(imageio.imread(file_path))
        # except Exception as e:
        #     print(f"Skipping frame. Exception occurred: {e}")
    writer.close()

for i, k in enumerate(df_index):
    print(k)
    folder_name = Path(data_path) / k.replace(' ', '_')
    os.makedirs(folder_name, exist_ok=True)
    for ind in tqdm(df_groups[k]):
        df_row = df_sequences.loc[ind]
        file_name = folder_name / str(df_row.id + '.png')
        if not os.path.exists(file_name):
            client.download_file(df_row['s3_bucket'], df_row['s3_key'], file_name)
# 2) Make all of the videos
for k in tqdm(df_index):
    create_video_frames(k)