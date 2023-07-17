import pandas as pd
import sys
# dsetpath = "/data/jupiter/datasets/Spring_hitchhiker_random"
dsetpath = "/data/jupiter/alex.li/datasets/spring_dust_data_test"
annotations = pd.read_csv(f"{dsetpath}/annotations.csv")
annotations = annotations[annotations['artifact_debayeredrgb_0_save_path'].isnull() == False]
annotations['image_id'] = annotations['id']
if len(sys.argv) > 1:
    subset_size = int(sys.argv[1])
    annotations = annotations.sample(n=min(subset_size, len(annotations)), random_state=1)
    annotations.to_csv(f"{dsetpath}/{subset_size}_fake_master_annotations.csv")
else:
    annotations.to_csv(f"{dsetpath}/fake_master_annotations.csv")
