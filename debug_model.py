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
)
from kore.configs.tasks.semantic_segmentation.model_config import (
    BRTResnetPyramidLite12Config,
)
from kore.tasks.base_training_task import direct_load_model_weights_from_ckpt

ckpt = "/mnt/sandbox1/alex.li/wandb/run-18044/files/epoch=5-val_loss=nan.ckpt"
model_config = BRTResnetPyramidLite12Config(output_type=OutputType.MULTISCALE)
seg_input_config = SegInputConfig()
seg_input_config.label.label_map_file_iq = str(
    object=EUROPA_DIR / "dl" / "config" / "label_maps" / "binary_dust.csv"
)
model_config.dust.dust_seg_output = True
model = model_config.create(seg_input_config)

direct_load_model_weights_from_ckpt(model, ckpt, False)
dirpath = "/data2/jupiter/datasets/Jupiter_train_v6_2/"
df = pd.read_csv(filepath_or_buffer=dirpath + "master_annotations_20231019_clean.csv")
batch_vals = []
labels = []
depth = []
batch_size = 2
for i in range(batch_size):
    artifacts = RGBDNPZ(dirpath, run_productivity_metrics=False).get_artifacts(df.iloc[i])
    batch_vals.append(np.concatenate([artifacts["image"], artifacts["depth"]], 2))
    labels.append(artifacts["label"])
    depth.append(artifacts["depth"])
batch = torch.Tensor(np.stack(batch_vals)).permute(
    dims=[0, 3, 1, 2]
)  # .type(torch.float16).to(device='cuda')
labels = torch.tensor(np.stack(labels))[:, None, :, :]
depth = torch.tensor(np.stack(depth)).permute(dims=[0, 3, 1, 2])
out = model(batch)

loss_config = SegLossAggConfig(
    msl=MultiScaleLossConfig(), tv=TverskyLossConfig(), prodl=ProductivityLossConfig()
)

lossagg = loss_config.create(external_config=seg_input_config)
lossagg.forward(
    out["logits"],
    labels % seg_input_config.label.label_map_helper.get_num_train_classes(),
    depth=depth,
    scale_logits=out["multiscale"],
    logits_iq=out["logits_iq"],
    label_iq=labels % 2,
)
