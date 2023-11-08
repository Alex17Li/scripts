import labelbox
client = labelbox.Client(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDQ0Y2c0cXYwM2hwMDdhemNiOHgyZGt2Iiwib3JnYW5pemF0aW9uSWQiOiJjandzZmFnNHYxYXJrMDgxMTlvdXlndG5xIiwiYXBpS2V5SWQiOiJjbG9qMGE3OXcwYXU5MDcyZTY1ZXM4MG9hIiwic2VjcmV0IjoiYmVhOGMxNDQyZDg4MjRmYjFhOGY1ZjgxZWMwMmIxYmYiLCJpYXQiOjE2OTkwMzk1NTUsImV4cCI6MjMzMDE5MTU1NX0.sLGzk_5mlrYjNQiVZ7rIIwu0egdByQNaT2QozJmVsGM")

from brtdevkit.data import AnnotationJob
from brtdevkit.data import Dataset
catalog_slice_id = "cloiy1fqs05ut071h2bfl7c7t"
catalog_slice = client.get_catalog_slice(catalog_slice_id)
import os
import json
import tqdm
import rich.progress
from pathlib import Path
fname = Path(os.path.expanduser("~/data/labelbox_slice_ids.json"))
os.makedirs(fname.parent, exist_ok=True)
if os.path.exists(fname):
    image_ids = json.loads(fname)
else:
    data_row_ids = catalog_slice.get_data_row_ids()
    image_ids = []
    for data_row_id in tqdm.tqdm(data_row_ids):
        try:
            imid = client.get_data_row(data_row_id).external_id.split(',')[0]
            image_ids.append(imid)
        except Exception as e:
            print(e)
    with open(fname, 'w+') as f:
        json.dump(image_ids, f)
print("Done")