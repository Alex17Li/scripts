import sys
import os
from brtdevkit.data import Dataset

dataset_name = sys.argv[1]
print(f'Dataset name - {dataset_name}')

dataset = Dataset.retrieve(name=dataset_name)
path = f'/data/jupiter/datasets/{dataset_name}'
# os.makedirs(path, exist_ok=True)
dataset.download(path)
