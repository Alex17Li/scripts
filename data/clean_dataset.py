"""
Simple Utility to check that a dataset contains all required files
and remove excess files
"""
import pandas
from pathlib import Path
from typing import Optional
import os
import tqdm
import time
import imageio
from pandarallel import pandarallel
import sys

def filter_debayered_images(df, dataset_path, df_path):
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

    # Remove debayered images from the annotations.csv file
    orig_size = len(df)
    df2 = df[df.artifact_debayeredrgb_0_save_path.isnull() == False]
    not_nan_size = len(df2)
    if not_nan_size != orig_size:
        print(f"Found {orig_size - not_nan_size} nan values of debayeredrgb save path")
    # tqdm.tqdm.pandas()
    # valid_debayered = df2[df2.progress_apply(debayered_is_valid, axis=1)]
    valid_debayered = df2[df2.parallel_apply(debayered_is_valid, axis=1)]
    if len(valid_debayered) != len(df):
        while True:
            confirm = 'y' if AUTOCONFIRM else input(f"Found only {len(valid_debayered)} out of {len(df)} debayered images, delete  {len(df) - len(valid_debayered)} lines of {df_path}? (y/n)")
            if confirm == 'y':
                print(f"Removing broken debayered images from {df_path}")
                df = valid_debayered
                valid_debayered.to_csv(Path(dataset_path) / df_path)
                break
            elif confirm == 'n':
                break
    return df



def main(dataset_folder: Path, annotations_path: Optional[str], master_annotations_path: Optional[str]) -> None:
    if annotations_path is not None:
        assert os.path.exists(dataset_folder / annotations_path)
    if master_annotations_path is not None:
        assert os.path.exists(dataset_folder / master_annotations_path)


    if annotations_path is not None:
        print("Reading annotation csv...")
        df = pandas.read_csv(dataset_folder / annotations_path, dtype=str)
        image_ids = df['id']

    if master_annotations_path is not None:
        print("Reading master annotations csv...")
        df_master = pandas.read_csv(dataset_folder / master_annotations_path, dtype=str)
        image_ids = list(df_master['image_id'])
        print("Checking annotation and master annotations have correct the same length")
    if annotations_path is not None and master_annotations_path is not None:
        assert len(df['id']) == len(df_master['image_id']) * 2
        # todo fix this invariant if we break it

    # csv has a right+left camera image
    assert(len(image_ids) > 0)


    if annotations_path is not None:
        example_folder_name = dataset_folder / "images" / image_ids[0]
        print(f"Contents of example folder {example_folder_name}:")
        print(list(os.listdir(example_folder_name)))
    if master_annotations_path is not None:
        example_folder_name = dataset_folder / "processed" / "images" / image_ids[0]
        print(f"Contents of example folder {example_folder_name}:")
        print(list(os.listdir(example_folder_name)))

    n_rectified = 0
    n_stereo_out = 0
    total = len(image_ids)
    
    if annotations_path is not None:
        df = filter_debayered_images(df, dataset_folder, annotations_path)
    if master_annotations_path is not None:
        df_master = filter_debayered_images(df_master, dataset_folder, master_annotations_path)

    # todo filter raw id paths

    if master_annotations_path is not None:
        for image_id in tqdm.tqdm(image_ids, desc="Checking processed..."):
            folder = dataset_folder / "processed" / "images" / image_id
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

    # Delete extra folders
    paths_to_clean = []
    if annotations_path is not None:
        paths_to_clean.append(dataset_folder / "images")
    if master_annotations_path is not None:
        paths_to_clean.append(dataset_folder / "processed" / "images")
    for clean_path in paths_to_clean:
        all_folders = os.listdir(clean_path)
        to_delete = set(all_folders) - set(image_ids)
        while True:
            confirm = 'y' if AUTOCONFIRM else input(f"Press 'y' to remove {len(to_delete)} folders in {clean_path}, leaving {len(image_ids)}")
            print(confirm)
            if confirm == 'y':
                for image_id in tqdm.tqdm(set(all_folders) - set(image_ids), desc="Removing files that are not in the annotations csv..."):
                    remove_path = dataset_folder / "images" / image_id
                    for subfile in os.listdir(remove_path):
                        os.remove(remove_path / subfile)
                break
            elif confirm == 'n':
                print("Ok, we are not deleting anything")
                break

if __name__ == "__main__":
    pandarallel.initialize(progress_bar=True)
    input_dataset_path = sys.argv[1] #"/data/jupiter/datasets/Spring_hitchhiker_random"
    input_annotations_path = 'annotations.csv'
    AUTOCONFIRM = True
    input_master_annotations_path = None #'6298546d31a0f1c9949b32a3_master_annotations.csv'
    main(Path(input_dataset_path), 
        input_annotations_path,
        input_master_annotations_path)
