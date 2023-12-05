import numpy as np
import pandas as pd
import torch

from dl import EUROPA_DIR
from dl.dataset.datamodes.npz.rgbd import RGBDNPZ
from dl.utils.colors import OutputType
from kore.configs.data.input_data_config import SegInputConfig
from kore.configs.tasks.semantic_segmentation.loss_config import (
    MultiScaleLossConfig,
    ProductivityLossConfig,
    SegLossAggConfig,
    TverskyLossConfig,
    HardSoftLossConfig
)
from kore.configs.tasks.semantic_segmentation.model_config import (
    BRTResnetPyramidLite12Config,
)
from kore.tasks.base_training_task import direct_load_model_weights_from_ckpt

# ckpt = "/mnt/sandbox1/alex.li/wandb/run-18044/files/epoch=5-val_loss=nan.ckpt"
# dirpath = "/data2/jupiter/datasets/Jupiter_train_v6_2/"
# csv_name ="master_annotations_20231019_clean"
csv_name="master_annotations_1k.csv"
dirpath="/home/alexli/data/Jupiter_train_v5_11/"
ckpt = "/home/alexli/epoch=5-val_loss=nan.ckpt"

model_config = BRTResnetPyramidLite12Config(output_type=OutputType.MULTISCALE)
seg_input_config = SegInputConfig()
seg_input_config.label.label_map_file_iq = str(
    object=EUROPA_DIR / "dl" / "config" / "label_maps" / "binary_dust.csv"
)
model_config.dust.dust_seg_output = True
model = model_config.create(seg_input_config).to('cuda')

direct_load_model_weights_from_ckpt(model, ckpt, False)

df = pd.read_csv(filepath_or_buffer=dirpath + csv_name)
batch_size = 8
for b in range(100):
    batch_vals = []
    labels = []
    depth = []
    for i in range(batch_size):
        artifacts = RGBDNPZ(dirpath, run_productivity_metrics=False).get_artifacts(
            df.iloc[i + batch_size * b])
        batch_vals.append(np.concatenate([artifacts["image"], artifacts["depth"]], 2))
        labels.append(artifacts["label"])
        depth.append(artifacts["depth"])
    batch = torch.Tensor(np.stack(batch_vals)).permute(
        dims=[0, 3, 1, 2]
    ).to(device='cuda')
    labels = torch.tensor(np.stack(labels))[:, None, :, :].to('cuda')
    depth = torch.tensor(np.stack(depth)).permute(dims=[0, 3, 1, 2]).to(device='cuda')

    loss_config = SegLossAggConfig(
        msl=MultiScaleLossConfig(), tv=TverskyLossConfig(), prodl=ProductivityLossConfig(), hardsoft_iq=HardSoftLossConfig()
    )

    lossagg = loss_config.create(external_config=seg_input_config)

    with torch.autocast(device_type="cuda"):
        out = model(batch)

        loss = lossagg.forward(
            out["logits"],
            labels % seg_input_config.label.label_map_helper.get_num_train_classes(),
            depth=depth,
            scale_logits=out["multiscale"],
            logits_iq=out["logits_iq"],
            label_iq=labels % 2,
        )
        print(b, loss['loss'])
