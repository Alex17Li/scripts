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
    
python3 -m dl.dataset.pack_perceptionpython3 -m dl.dataset.pack_perception.ml_pack_perception \
    --data-dir $DATA/partitions/\$SLURM_PROCID --csv-path \\\$DATA_DIR/annotations.csv \
    --calib-tracker-csv /files/calibration/motec_calibration_tracker_2019.csv \
    --cam-calibration-path /files/calibration --ignore-slurm-variables \
    --batch-size 24 --multiprocess-workers 48 --pandarallel-workers 48 --gpu 0 \
    --models 512,768=ml_512x768_v3_full_rgb_halo_depth_10062023.ckpt \
             512,640=ml_512x640_v3_full_rgb_halo_depth_10062023.ckpt \
    --model-type full --max-disp 384 \
    --run-oc \
    --image-only
.ml_pack_perception \
    --data-dir $DATA/partitions/\$SLURM_PROCID --csv-path \\\$DATA_DIR/annotations.csv \
    --calib-tracker-csv /files/calibration/motec_calibration_tracker_2019.csv \
    --cam-calibration-path /files/calibration --ignore-slurm-variables \
    --batch-size 24 --multiprocess-workers 48 --pandarallel-workers 48 --gpu 0 \
    --models 512,768=ml_512x768_v3_full_rgb_halo_depth_10062023.ckpt \
             512,640=ml_512x640_v3_full_rgb_halo_depth_10062023.ckpt \
    --model-type full --max-disp 384 \
    --run-oc \
    --image-only

