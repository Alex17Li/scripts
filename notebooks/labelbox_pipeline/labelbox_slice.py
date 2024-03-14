import sys
import argparse
import labelbox
import time
from brtdevkit.data import AnnotationJob
from brtdevkit.data import Dataset
import os

from pathlib import Path
import pandas as pd
api_key_BRTjupiter = os.environ['LABELBOX_JUPITER_API_KEY']
def main(slice_id: str, output_path: str) -> None:
    client = labelbox.Client(api_key_BRTjupiter)
    catalog_slice = client.get_catalog_slice(slice_id)

    fname = Path(os.path.expanduser(output_path))
    os.makedirs(fname.parent, exist_ok=True)
    st_time = time.time()
    print("Starting export")
    export_task = catalog_slice.export_v2(params={
        "data_row_details": True
    })
    export_task.wait_till_done()
    export_json = export_task.result
    image_ids =  [
        i['data_row']['external_id'].split(',')[0]
        for i in export_json
    ]
    pd.DataFrame.from_dict({'id':image_ids}).to_parquet(output_path)
    print('Creating dataset')
    Dataset.create(name=f"labelbox_import_{Path(output_path).stem}",
                description=f"{len(image_ids)} images imported from labelbox slice {slice_id}.",
                kind=Dataset.KIND_IMAGE,
                image_ids=image_ids,
                )
    print(f"Done in {time.time() - st_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='SaveLabelboxSlice',
                    description='Gets all images from a slice on labelbox and saves them somewhere',
                    )
    parser.add_argument('--slice_id') # e.g. clri1vxe50lob073e5n9s8rdb from the URL of the slice
    parser.add_argument('--output_path') # e.g. /data/jupiter/alex.li/labelbox_slice_ids.json
    args = parser.parse_args()
    # ETA: 100k images per minute
    main(args.slice_id, args.output_path)
