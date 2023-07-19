"""
Simple Utility to check that a dataset contains all required files
and remove excess files
"""
from ctypes import ArgumentError
import pandas
from pathlib import Path
from typing import Optional
import os
import tqdm
import time
import imageio
from pandarallel import pandarallel
import sys
import numpy as np
from typing import Literal

dataset_path = Path("")

def debayered_is_valid(df_row) -> bool:
    try:
        if df_row.hdr_mode:
            image_path = os.path.join(dataset_path, df_row.artifact_debayeredrgb_0_save_path)
        else:
            image_path = os.path.join(dataset_path, df_row.stereo_left_image)
        # imread is slow but I have had some examples where the path exists and looks valid,
        # but imread crashes
        if not os.path.exists(image_path):
            return False
        im = imageio.imread(image_path) # check that we can read the image
        # delete the image to avoid nasty silent OOM errors; pandarell doesn't call the garbarge collector for you
        # and the code will just get stuck
        del im
        return True
    except Exception as e:
        print(e)
        print(df_row.artifact_debayeredrgb_0_save_path)
        return False
    
def label_is_valid(df_row):
    label_path = os.path.join(dataset_path, df_row.rectified_label_save_path)
    # label = np.load(label_path)
    # label = label[LEFT][..., 0]
    return os.path.exists(label_path)
    
def rgdb_is_valid(df_row) -> bool:
    stereo_data_sample_path = os.path.join(dataset_path, df_row.stereo_pipeline_npz_save_path)
    stereo_data_sample = np.load(stereo_data_sample_path)
    image = stereo_data_sample['left']
    # depth = depth_from_point_cloud(
    #     stereo_data_sample[POINT_CLOUD],
    #     clip_and_normalize=True,
    #     max_depth=MAX_DEPTH,
    #     make_3d=True,
    # )
    del stereo_data_sample
    # del depth
    del image
    return os.path.exists(stereo_data_sample_path)

def filter_images(df, dataset_path, out_df_path, type: Literal['rgbd', 'debayered']):
    # Remove images from the annotations.csv file that cannot be loaded
    orig_size = len(df)
    df2 = df[df.artifact_debayeredrgb_0_save_path.isnull() == False]
    not_nan_size = len(df2)
    if not_nan_size != orig_size:
        print(f"Found {orig_size - not_nan_size} nan values of debayeredrgb save path")
    # tqdm.tqdm.pandas()
    if type == 'rgbd':
        valid = df2[df2.parallel_apply(rgdb_is_valid, axis=1)]
    elif type == 'debayered':
        # valid = df2[df2.progress_apply(debayered_is_valid, axis=1)]
        valid = df2[df2.parallel_apply(debayered_is_valid, axis=1)]
    else:
        raise ArgumentError()
    if len(valid) != len(df):
        while True:
            confirm = 'y' if AUTOCONFIRM else input(f"Found only {len(valid_debayered)} out of {len(df)} debayered images, delete  {len(df) - len(valid_debayered)} lines of {df_path}? (y/n)")
            if confirm == 'y':
                print(f"Removing broken {type} images")
                df = valid
                valid.to_csv(Path(dataset_path) / out_df_path)
                break
            elif confirm == 'n':
                break
    else:
        print(f"No broken images found for {type}")
    return df


def main(annotations_path: Optional[str], master_annotations_path: Optional[str]) -> None:
    paths_to_clean = []
    if annotations_path is not None:
        assert os.path.exists(dataset_path / annotations_path)
        out_annotations_path = "cleaned_" + annotations_path
        paths_to_clean.append(dataset_path / "images")
        
        print("Reading annotation csv...")
        df = pandas.read_csv(dataset_path / annotations_path, dtype=str)
        image_ids = df['id']
        assert(len(image_ids) > 0)
        
        example_folder_name = dataset_path / "images" / image_ids[0]
        print(f"Contents of example folder {example_folder_name}:")
        print(list(os.listdir(example_folder_name)))
        # note, no rgbd for normal annotations as depth does not exist
        df = filter_images(df, dataset_path, out_annotations_path, 'debayered')

    if master_annotations_path is not None:
        assert os.path.exists(dataset_path / master_annotations_path)
        out_master_annotations_path = "cleaned_" + master_annotations_path
        paths_to_clean.append(dataset_path / "processed" / "images")
        
        print("Reading master annotations csv...")
        df_master = pandas.read_csv(dataset_path / master_annotations_path, dtype=str)
        image_ids = list(df_master['image_id'])
        assert(len(image_ids) > 0)

        example_folder_name = dataset_path / "processed" / "images" / image_ids[0]
        print(f"Contents of example folder {example_folder_name}:")
        print(list(os.listdir(example_folder_name)))
        # df_master = filter_images(df_master, dataset_path, out_master_annotations_path, 'debayered')

        n_rectified = 0
        n_stereo_out = 0
        total = len(image_ids)
        
        if master_annotations_path is not None:
            for image_id in tqdm.tqdm(image_ids, desc="Checking processed..."):
                folder = dataset_path / "processed" / "images" / image_id
                filenames = list(os.listdir(folder))
                has_rectified = False
                has_stereo_out = False
                for filename in filenames:
                    if filename.startswith('rectification_output'):
                        has_rectified = True
                    if filename.startswith('stereo_output'):
                        has_stereo_out = True
                if has_rectified:
                    n_rectified += 1
                if has_stereo_out:
                    n_stereo_out += 1

            if n_rectified == total:
                print("all rectified images are present")
            else:
                print(f"Warning: not all rectified images are present! Got {n_rectified}/{total}")
            if n_stereo_out == total:
                print("all stereo outputs are present")
            else:
                print(f"Warning: not all stereo outputs are present! Got {n_stereo_out}/{total}")
        
        df_master = filter_images(df_master, dataset_path, out_master_annotations_path, 'rgbd')

    # Delete extra folders
    if DELETE_UNUSED_IMAGES:
        for clean_path in paths_to_clean:
            all_folders = os.listdir(clean_path)
            to_delete = set(all_folders) - set(image_ids)
            while True:
                confirm = 'y' if AUTOCONFIRM else input(f"Press 'y' to remove {len(to_delete)} folders in {clean_path}, leaving {len(image_ids)}")
                print(confirm)
                if confirm == 'y':
                    for image_id in tqdm.tqdm(set(all_folders) - set(image_ids), desc="Removing files that are not in the annotations csv..."):
                        remove_path = dataset_path / "images" / image_id
                        for subfile in os.listdir(remove_path):
                            os.remove(remove_path / subfile)
                    break
                elif confirm == 'n':
                    print("Ok, we are not deleting anything")
                    break

if __name__ == "__main__":
    pandarallel.initialize(progress_bar=True)
    in_dataset_path = sys.argv[1] #"/data/jupiter/datasets/Spring_hitchhiker_random"
    dataset_path = Path(sys.argv[1])
    input_annotations_path = None
    input_master_annotations_path = None 
    for p_ind in range(2, len(sys.argv)):
        p = sys.argv[p_ind]
        if p.find('annotations') == -1:
            print(f"{p} might not be a valid name, try again")
            continue
        if 'master' in p:
            input_master_annotations_path = p #'64b0197137e915581adec2d5_master_annotations.csv'
        else:
            input_annotations_path = p # annotations.csv

    AUTOCONFIRM = True
    DELETE_UNUSED_IMAGES = False
    main(input_annotations_path,
        input_master_annotations_path)
