import os
import numpy as np

def load_object(directory, row, get_label=False):
    folder_path  = directory + "/processed/images/" + row['id'] + "/"
    label_file_names = [f for f in os.listdir(folder_path) if 'stereo' not in f]
    file_names = sorted([f for f in os.listdir(folder_path) if 'stereo' in f])
    camss = [tuple(n.strip('.npz').split('_')[-2::]) for n in file_names]
    result = []
    for file_name, cams in zip(file_names, camss):
        image_npz = np.load(folder_path + file_name, allow_pickle=True)
        if get_label:
            label_file = None
            for lf in label_file_names:
                if lf.endswith(f"{cams[0]}_{cams[1]}.npz"):
                    label_file = lf
            assert label_file is not None
            label = np.load(folder_path + label_file)['left']
            result.append((image_npz, cams, label))
        else:
            result.append((image_npz, cams))
    return result
