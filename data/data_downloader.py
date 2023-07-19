import argparse
import os
import pathlib

if os.path.exists("/data"):
    DEFAULT_DESTINATION = "/data/jupiter/datasets/"
else:
    DEFAULT_DESTINATION = os.path.expanduser("~/BRT/data/")
DEFAULT_DESTINATION = os.getenv(
    "DATASET_PATH",
    DEFAULT_DESTINATION,
)

parser = argparse.ArgumentParser(description="Downloads BRT datasets from Aletheia.")
parser.add_argument("dataset_name", help="name of the dataset on Aletheia", type=str)
parser.add_argument(
    "-d",
    "--datasets-directory",
    help="directory where all of the datasets are saved",
    type=pathlib.Path,
    default=DEFAULT_DESTINATION,
)

args = parser.parse_args()
dataset_name, destination = args.dataset_name, args.datasets_directory
print(f"Saving dataset {dataset_name} to {destination}")

from brtdevkit.data import Dataset

dataset = Dataset.retrieve(name=dataset_name)
os.mkdir(destination / dataset_name)
dataset.download(destination / dataset_name)