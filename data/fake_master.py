import pandas as pd
import sys

dsetpath = sys.argv[1]

annotations = pd.read_csv(f"{dsetpath}/annotations.csv")
annotations = annotations[annotations['artifact_debayeredrgb_0_save_path'].isnull() == False]
annotations['image_id'] = annotations['id']
if len(sys.argv) > 2:
    subset_size = int(sys.argv[1])
    annotations = annotations.sample(n=min(subset_size, len(annotations)), random_state=1)
    dest = f"{dsetpath}/{subset_size}_fake_master_annotations.csv"
else:
    dest = f"{dsetpath}/fake_master_annotations.csv"
print(f"Saving to {dest}")
annotations.to_csv(dest)
