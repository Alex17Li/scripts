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
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset

data_path = os.path.expanduser('~') + '/data'

def make_dataset_slow(from_df, name, description) -> None:
    imids = list(from_df['id'])
    desc = f"{description} ({len(from_df['id'])} images)"
    imageids_to_dataset(imids, name, dataset_kind='image',
                             dataset_description=desc)

stratified_df_rev1 = pd.read_parquet(data_path + f"/rev1_data_stratified.parquet")
stratified_df_rev2 = pd.read_parquet(data_path + f"/rev2_data_stratified.parquet")
make_dataset_slow(stratified_df_rev1, "rev1_data_stratified", description="Randomly selected data from rev1")
make_dataset_slow(stratified_df_rev2, "rev2_data_stratified", description="Randomly selected data from rev2")
print("DONE")