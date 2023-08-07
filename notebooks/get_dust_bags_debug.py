import os
import pandas as pd
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset
from pathlib import Path
from brtdevkit.data import Dataset

home = Path(os.path.expanduser('~'))
data_path = home / 'data' 

def make_dataset_slow(from_df, name, description) -> None:
    imids = list(from_df['image_id'])
    desc = f"{description} ({len(from_df['image_id'])} images)"
    print(len(imids))
    from_df.to_parquet(data_path / '{name}.parquet', index=False)
    imageids_to_dataset(imids, name, dataset_kind=Dataset.KIND_ANNOTATION, dataset_description=desc, production_dataset=False)

if __name__ == "__main__":
    if os.path.exists(data_path / 'df_dusty_anno.parquet'):
        df_dusty_anno = pd.read_parquet(data_path / 'df_dusty_anno.parquet')
    else:
        print("Cache miss")
        query = """SELECT ij.id, hard_drive_name, robot_name, collected_on,
            bag_name, operating_field_name, operation_time, latitude, longitude, geohash, camera_location, 
            bundle, group_id, s3_bucket, s3_key, special_notes, label_map__json, vendor_metadata__json, annotation_jupiter.updated_at
        FROM image_jupiter AS ij
        JOIN "annotation_jupiter" ON ij."id" = "annotation_jupiter"."image"
        WHERE "hard_drive_name" IN ('JUPD-054_2023-6-13')
        """
        df_dusty_anno: pd.DataFrame = athena.get_df(query) # type: ignore
        df_dusty_anno.to_parquet(data_path / 'df_dusty_anno.parquet')
    df_dusty_anno['image_id'] = df_dusty_anno['id']
    df_dusty_anno = df_dusty_anno.set_index('id')
    make_dataset_slow(df_dusty_anno, 
        name='mannequin_in_dust_anno',
        description="8 sequences of a mannequin in front of the tractor with dust blowing into it. All images contain a mannequin. Collected 2023 July 7. Left cameras only (1650 images)",
    )
