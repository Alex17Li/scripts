import torch.utils.data
import logging
from kore.configs.data.dataset_config import SegTestDataConfig, SegDatasetConfig
from kore.configs.data.input_data_config import SegInputConfig, InputMode

from kore.data.jupiter_seg_data_module import JupiterSegDataModule

class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed
        dset_path = '/mnt/datasets/halo_rgb_stereo_train_v6_1/halo_rgb_stereo_train_v6_1/'
        dataset_config = SegTestDataConfig()
        dataset_config.test_set = SegDatasetConfig(dataset_path=dset_path, csv='master_annotations_10k.csv')
        inputs = SegInputConfig(input_mode=InputMode.RECTIFIED_RGB)
        data_module = JupiterSegDataModule.create_test_datamodule(
            dataset_config,
            inputs,
            False
        )
        self.dataset = data_module.test_sets[0]


    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.dataset)

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.
        batch = self.dataset[index]
        image = batch['left']
        mask = batch['label']
        logging.error(image.shape)
        logging.error(mask.shape)
        logging.error(image.dtype)
        logging.error(mask.dtype)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
