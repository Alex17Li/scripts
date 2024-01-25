import pandas as pd
root_path = "/data2/jupiter/datasets/halo_rgb_stereo_train_v6_1"
dataset = pd.read_csv(root_path + "/master_annotations.csv")
subsampled = dataset.sample(40000, replace=False)
subsampled.to_csv(root_path + "/master_annotations_40k.csv")
