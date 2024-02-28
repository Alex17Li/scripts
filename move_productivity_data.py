import os
import ast
import shutil
import json
import itertools
from click import group
import pandas as pd
basepath = '/data/jupiter/datasets/'
target_dset = basepath + 'halo_productivity_combined/'
my_datasets = [
    "Jupiter_20230720_HHH3_1805_1835",
    "Jupiter_20230803_HHH2_2030_2100",
    "Jupiter_20230803_HHH3_2115_2145",
    "Jupiter_20230814_HHH1_1415_1445",
    "Jupiter_20230823_HHH3_1815_1845",
    "Jupiter_20230803_HHH2_1400_1430",
    "Jupiter_20230825_HHH1_1730_1800",
    "Jupiter_20230926_HHH1_1815_1845",
    "Jupiter_20230927_HHH1_0100_0130",
    "Jupiter_20231007_HHH1_2350_0020",
    "Jupiter_20231019_HHH6_1615_1700",
    "Jupiter_20231019_HHH6_1800_1830",
    "Jupiter_20231026_HHH8_1515_1545",
]
with open(target_dset + 'good_ids.txt') as f:
    my_valid_ids = json.load(f)

dean_datasets = [
    '20230912_halo_rgb_productivity_night_candidate_0_no_ocal_rgb_branch',
    '20230929_halo_rgb_productivity_day_candidate_13_dirty_no_ocal',
    '20230929_halo_rgb_productivity_day_candidate_12_dirty_cleaned_v0_no_ocal',
    # '20230929_halo_rgb_productivity_day_candidate_8_cleaned_v1_no_ocal',
    '20230912_halo_rgb_productivity_day_candidate_1_cleaned_v3_no_ocal',
    '20230929_halo_rgb_productivity_day_candidate_10_cleaned_v1_no_ocal',
    # '20230929_halo_rgb_productivity_day_candidate_4_cleaned_v2_no_ocal',
    '20230929_halo_rgb_productivity_night_candidate_4_cleaned_v1_no_ocal',
]
print("Copying master csv")
dfs = []
for dataset in itertools.chain(my_datasets, dean_datasets):
    group_df = pd.read_csv(basepath + dataset + '/master_annotations.csv')
    group_df['subdataset'] = dataset
    if dataset.startswith('Jupiter'):
        group_df = group_df[group_df['id'].isin(my_valid_ids)]
    dfs.append(group_df)
df = pd.concat(dfs, axis=0)
df.to_csv(target_dset + 'master_annotations.csv')

# print("Copying artifacts")
# for dataset in itertools.chain(my_datasets, dean_datasets):
#     print(dataset)
#     shutil.copytree(basepath + dataset + "/processed/images", target_dset + "/processed/images", dirs_exist_ok=True)
