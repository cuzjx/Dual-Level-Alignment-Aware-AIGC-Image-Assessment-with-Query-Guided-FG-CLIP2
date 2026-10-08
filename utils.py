import torch
from torch.utils.data import DataLoader
from ImageDataset import (
    AIGCDataset,
    AIGCDataset_3k,
    AIGCIQA2023Dataset,
    AIGCFgclipDataset,
    AIGCFgclip3KDataset,
    AIGCFgclip2023Dataset,
    EvalMuseDataset,
    EvalMuseFgclipDataset,
)

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

from PIL import Image
import logging

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def set_dataset_aigc(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):

    data = AIGCDataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader

def set_dataset_aigc_3k(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False, mos_col=5):

    data = AIGCDataset_3k(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind,
        mos_col=mos_col)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_dataset_aigc_2023(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False, mos_col=2):

    data = AIGCIQA2023Dataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind,
        mos_col=mos_col)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def collate_aigc_fgclip(batch):
    return {
        "image": [item["image"] for item in batch],
        "mos": torch.tensor([item["mos"] for item in batch], dtype=torch.float32),
        "prompt": [item["prompt"] for item in batch],
        "prompt_name": [item["prompt_name"] for item in batch],
        "image_name": [item["image_name"] for item in batch],
    }


def set_dataset_aigc_fgclip(csv_file, bs, data_set, num_workers, test, blind=False):
    data = AIGCFgclipDataset(csv_file=csv_file, img_dir=data_set, blind=blind)
    loader = DataLoader(
        data,
        batch_size=bs,
        shuffle=not test,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_aigc_fgclip,
    )
    return loader


def set_dataset_aigc_3k_fgclip(csv_file, bs, data_set, num_workers, test, blind=False, mos_col=5):
    data = AIGCFgclip3KDataset(csv_file=csv_file, img_dir=data_set, blind=blind, mos_col=mos_col)
    loader = DataLoader(
        data,
        batch_size=bs,
        shuffle=not test,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_aigc_fgclip,
    )
    return loader


def set_dataset_aigc_2023_fgclip(csv_file, bs, data_set, num_workers, test, mos_col=2):
    data = AIGCFgclip2023Dataset(csv_file=csv_file, img_dir=data_set, mos_col=mos_col)
    loader = DataLoader(
        data,
        batch_size=bs,
        shuffle=not test,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_aigc_fgclip,
    )
    return loader


def set_dataset_evalmuse(json_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):
    data = EvalMuseDataset(
        json_file=json_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind,
    )

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(
        data,
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_evalmuse,
    )

    return loader


def collate_evalmuse(batch):
    # Keep variable-length fields as lists; stack tensors for model inputs.
    return {
        "I": torch.stack([item["I"] for item in batch], dim=0),
        "mos": torch.tensor([item["mos"] for item in batch], dtype=torch.float32),
        "prompt": [item["prompt"] for item in batch],
        "prompt_name": [item["prompt_name"] for item in batch],
        "image_name": [item["image_name"] for item in batch],
        "element_names": [item["element_names"] for item in batch],
        "element_targets": [item["element_targets"] for item in batch],
    }


def collate_evalmuse_fgclip(batch):
    # Whole-image FG-CLIP2 pipeline: keep PIL images as a list (handled by image_processor).
    return {
        "image": [item["image"] for item in batch],
        "mos": torch.tensor([item["mos"] for item in batch], dtype=torch.float32),
        "prompt": [item["prompt"] for item in batch],
        "image_name": [item["image_name"] for item in batch],
        "element_names": [item["element_names"] for item in batch],
        "element_targets": [item["element_targets"] for item in batch],
    }


def set_dataset_evalmuse_fgclip(json_file, bs, data_set, num_workers, test):
    data = EvalMuseFgclipDataset(
        json_file=json_file,
        img_dir=data_set,
    )

    shuffle = not test

    loader = DataLoader(
        data,
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_evalmuse_fgclip,
    )

    return loader



class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return transforms.Resize(self.size, self.interpolation)(img)
        else:
            return img


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess4():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess5():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()



def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)
