import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from osgeo import gdal


class MultiSourceGeoDataset(Dataset):
    """
    PyTorch Dataset for multi-source remote sensing GeoTIFF data.

    Directory structure:
        data_path/
        ├── image/      (multi-band optical images)
        ├── label/      (ground truth labels)
        ├── mndwi/      (optional, single-band)
        ├── ndvi/       (optional, single-band)
        ├── dem/        (optional, single-band)
        └── ...

    Parameters
    ----------
    data_path : str
        Root directory of the dataset.
    aux_features : list[str]
        Names of auxiliary feature folders (e.g., ["mndwi", "ndvi"]).
    """

    def __init__(self, data_path, aux_features=None):
        self.data_path = data_path
        self.aux_features = aux_features or []

        self.image_paths = sorted(
            glob.glob(os.path.join(data_path, "image", "*.tif"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError("No images found in 'image' directory.")

    def _read_tif(self, path):
        data = gdal.Open(path).ReadAsArray()
        data = np.nan_to_num(data)

        if data.ndim == 2:
            data = data[None, :, :]

        return data

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        base_name = os.path.basename(image_path)

        image = self._read_tif(image_path)

        aux_list = []
        for feat in self.aux_features:
            feat_path = os.path.join(self.data_path, feat, base_name)
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Missing auxiliary feature: {feat_path}")
            aux_list.append(self._read_tif(feat_path))

        if aux_list:
            image = np.concatenate([image] + aux_list, axis=0)

        label_path = os.path.join(self.data_path, "label", base_name)
        label = gdal.Open(label_path).ReadAsArray()
        label = np.nan_to_num(label).astype(np.int64)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(label).long(),
            base_name
        )

    def __len__(self):
        return len(self.image_paths)
