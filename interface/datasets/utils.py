import os
import sys
import torch
import hashlib
import pathlib
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from typing import Any, Callable, Dict, IO, Iterable, List, Optional, Tuple, TypeVar, Union

USER_AGENT = "pytorch/vision"

def pad22pow(a):
    assert a % 2 == 0
    bits = a.bit_length()
    ub = 2 ** bits
    pad = (ub - a) // 2
    return pad, ub


def is_labelled(dataset):
    labelled = False
    if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
        labelled = True
    return labelled

def calculate_md5(fpath: Union[str, pathlib.Path], chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: Union[str, pathlib.Path], md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath: Union[str, pathlib.Path], md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def download_file_from_google_drive(
    file_id: str,
    root: Union[str, pathlib.Path],
    filename: Optional[Union[str, pathlib.Path]] = None,
    md5: Optional[str] = None,
):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    try:
        import gdown
    except ModuleNotFoundError:
        raise RuntimeError(
            "To download files from GDrive, 'gdown' is required. You can install it with 'pip install gdown'."
        )

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f"Using downloaded {'and verified ' if md5 else ''}file: {fpath}")
        return

    gdown.download(id=file_id, output=fpath, quiet=False)

    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")

class AddGaussNoise(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        return tensor + self.std * torch.rand_like(tensor).to(tensor.device)


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x


class StandardizedDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.std_inv = 1. / std
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return self.std_inv * (x - self.mean), y
        else:
            x = self.dataset[item]
            return self.std_inv * (x - self.mean)


class QuickDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return self.array[item]


class FlattenedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return x.view(-1), y
        else:
            x = self.dataset[item]
            return x.view(-1)
