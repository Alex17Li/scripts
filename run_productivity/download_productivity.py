import ast
import json
import os
import sys
import timeit

from collections import Counter


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path
from brtdevkit.core.db.athena import AthenaClient
from brtdevkit.data import Dataset
from dl.dataset.pack_perception.download_ocal_data import download_ocal_data
import yaml
# screen (screen -r)
# python ~/git/scripts/run_productivity/download_productivity.py
# ctrl-a + d
basepath = Path('/mnt/alex.li/productivity_dsets/')
dsetnames = [
    'Jupiter_20231121_HHH2_1800_1830'
]    
dsets = {}
for dsetname in dsetnames:
    dset: Dataset = Dataset.retrieve(name=dsetname)
    df = dset.to_dataframe()
    path = basepath / dsetname
    dsets[dsetname] = {'df': df,
        'dset': dset,
        'dpath': path,
    }
    if not os.path.exists(path / 'annotations.csv'):
        print(f"Downloading to {path}")
        dset.download(path)
    assert len(df) == len(os.listdir(path / 'images'))
    df.to_csv(path / 'annotations.csv')
    download_ocal_data(str(path), df)

