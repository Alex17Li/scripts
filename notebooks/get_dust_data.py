import time
import json
import ast
import os
import datetime
import io
from collections import defaultdict

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

from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset
from aletheia_dataset_creator.config.dataset_config import LEFT_CAMERAS, ALL_CAMERA_PAIRS_LIST
pd.set_option('display.max_rows', 500)
athena = AthenaClient()
s3 = boto3.resource('s3')
tf = TimezoneFinderL()
home = os.path.expanduser('~')
import os
def get_calibration(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}   
def get_adjusted_timezone(timestamp, latitude, longitude):
    if (latitude == 0) or (longitude == 0):
        return np.nan
    
    if isinstance(timestamp, np.datetime64):
        timestamp = pd.to_datetime(timestamp)
    # Localize and adjust UTC timestamps to local timezone
    utc = pytz.utc.localize(timestamp)
    tz = tf.timezone_at(lat=latitude, lng=longitude)
    adjusted_timestamp = utc.astimezone(tz).to_pydatetime()

    return adjusted_timestamp


data_path = home + '/data/get_dust_data'
query1 = f"""
SELECT id, robot_name, collected_on, operation_time,
    camera_location, gps_can_data__json, group_id
FROM image_jupiter
"""
df = athena.get_df(query1)
def get_day(collect_str):
    t = pd.Timestamp(collect_str)
    return t.strftime("%m/%d")
def get_minute(collect_str):
    t = pd.Timestamp(collect_str)
    return t.strftime("%m/%d %H:%M")

df['day'] = df['collected_on'].map(get_day)
df['minute'] = df['collected_on'].map(get_minute)
df['speed_kph'] = df['gps_can_data__json'].map(lambda x:(json.loads(x)['speed']))
bidirectional_dict = {}
for pair_dict in ALL_CAMERA_PAIRS_LIST:
    for k, v in pair_dict.items():
        bidirectional_dict[k] = v
        bidirectional_dict[v] = k

def make_dataset(from_df, name, description, pairs=[bidirectional_dict]) -> None:
    imids = list(from_df['id'])
    # print(len(imids))
    from_df.to_parquet(data_path + f'/{name}.parquet', index=False)
    # desc = f"{description} ({len(from_df['id'])} images)"
    # imageids_to_dataset_fast(from_df, name, desc,
    #                          camera_pairs_list=pairs, camera_pair_df=df)
    Dataset.create(
        name=name,
        description=description,
        kind=Dataset.KIND_IMAGE,
        image_ids=imids,
    )
    print("k")

def make_dataset_slow(from_df, name, description) -> None:
    imids = list(from_df['id'])
    desc = f"{description} ({len(from_df['id'])} images)"
    print(len(imids))
    from_df.to_parquet(f'/home/alexli/data/all_hitchiker_images/{name}.parquet', index=False)
    imageids_to_dataset(imids, name, dataset_kind='image',
                             dataset_description=desc)
def speed_discrete(speed):
    if 0 <= speed <= 2:
        return "A:0-2"
    elif 2 < speed <= 10:
        return "B:2-10"
    elif 10 < speed <= 25:
        return "C:10-25"
    elif 25 < speed:
        return "D:25+"
    else:
        assert False

df=df.sample(frac=1)
print(len(df))

df['speed'] = df.gps_can_data.apply(lambda a: ast.literal_eval(a).get('speed', 0))
df['speed_d'] = df['speed'].apply(speed_discrete)
stratified_df = df.groupby(['robot_name', 'camera_location', 'hour', 'collected_on', 'speed_d']).head(4)
print(len(stratified_df))
df.to_parquet(data_path + '/all_jup.parquet', index=False)
