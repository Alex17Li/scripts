import pandas as pd
root_path = "/mnt/datasets/halo_rgb_stereo_train_v6_1"
dataset = pd.read_csv(root_path + "/master_annotations.csv")
subsampled = dataset.sample(2, replace=False)
subsampled.to_csv(root_path + "/master_annotations_2_val.csv")
