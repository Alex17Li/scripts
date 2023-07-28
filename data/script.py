import pandas as pd
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset
import os

def make_dataset_slow(from_df, name, description) -> None:
    imids = list(from_df['id'])
    desc = f"{description} ({len(from_df['id'])} images)"
    print(len(imids))
    from_df.to_parquet(data_path + f'/{name}.parquet', index=False)
    imageids_to_dataset(imids, name, dataset_kind='image',
                             dataset_description=desc)

home = os.path.expanduser('~')
data_path = home + '/data/get_dust_data'
print("start")
stratified_df = pd.read_parquet(data_path + "/lotta_data_strat.parquet")
print("read data success")
make_dataset_slow(stratified_df, "all_jupiter_data_stratified_rng", description=f"Randomly selected data from jupiter")
print("DONE MADE DATASET")