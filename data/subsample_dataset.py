import pandas as pd
root_path = "/home/alexli/data/Jupiter_train_v5_11"
dataset = pd.read_csv(root_path + "/master_annotations_30k.csv")
subsampled = dataset.sample(4000, replace=False)
subsampled.to_csv(root_path + "/master_annotations_4k.csv")