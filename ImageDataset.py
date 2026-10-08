import os
import json
import zipfile
import xml.etree.ElementTree as ET
import torch
import functools
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch.nn.functional as F

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    #print(image_name)
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


@functools.lru_cache(maxsize=8)
def load_aigciqa2023_flat_index(index_xlsx_path):
    # Lightweight .xlsx reader for the Hugging Face AIGCIQA2023 pic-index file.
    # It avoids requiring openpyxl on training machines.
    ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

    def col_to_idx(col_name):
        value = 0
        for ch in col_name:
            value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
        return value - 1

    with zipfile.ZipFile(index_xlsx_path, "r") as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("a:si", ns_main):
                text_parts = [node.text or "" for node in si.findall(".//a:t", ns_main)]
                shared_strings.append("".join(text_parts))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        mapping = {}
        for row_idx, row in enumerate(sheet_root.findall(".//a:sheetData/a:row", ns_main)):
            values = {}
            for cell in row.findall("a:c", ns_main):
                ref = cell.attrib.get("r", "")
                col_name = "".join(ch for ch in ref if ch.isalpha())
                if not col_name:
                    continue
                col_idx = col_to_idx(col_name)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", ns_main)
                if value_node is None:
                    continue
                raw_value = value_node.text or ""
                if cell_type == "s":
                    value = shared_strings[int(raw_value)]
                else:
                    value = raw_value
                values[col_idx] = value

            model_name = str(values.get(0, "")).strip()
            image_name = str(values.get(1, "")).strip()
            if model_name and image_name:
                mapping[(model_name, image_name)] = row_idx
    return mapping


def resolve_aigciqa2023_image_path(img_dir, model_name, image_name):
    nested_path = os.path.join(img_dir, model_name, image_name)
    if os.path.exists(nested_path):
        return nested_path

    index_xlsx_path = os.path.join(os.path.dirname(img_dir), "pic-index.xlsx")
    if not os.path.exists(index_xlsx_path):
        return nested_path

    flat_index = load_aigciqa2023_flat_index(index_xlsx_path)
    flat_row_idx = flat_index.get((model_name, image_name))
    if flat_row_idx is None:
        return nested_path

    candidates = [
        os.path.join(img_dir, "{}.png".format(flat_row_idx)),
        os.path.join(img_dir, "{}.png".format(flat_row_idx + 1)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

# AIGC数据集，用于训练AIGC模型，数据只有图像，csv文件只有对应图像名称、MOS、对图像的一段文本描述
class AIGCDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 blind = False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        # # 读取xlsx文件
        # self.data = pd.read_excel(csv_file, header=None)
        # # Print and remove the first row
        # print(self.data.iloc[0])
        # self.data = self.data.iloc[1:]

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            if self.blind:
                mos = 0.0
            else:
                mos = float(self.data.iloc[index, 2])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
            sel = sel.long()
            mos = float(self.data.iloc[index, 2])
            # mos = proc_label(mos)
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)
        
        prompt = self.data.iloc[index, 1]

        model_name = image_name.split('_')
        model_name = model_name[0]        
        prompt_name = model_name + ' ' + prompt

        sample = {'I': patches, 'mos': mos,'prompt':prompt, 'prompt_name':prompt_name, 'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.data.index)


class AIGCDataset_3k(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 blind = False,
                 mos_col=5,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())
        # print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.blind = blind
        self.mos_col = mos_col
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            if self.blind:
                mos = 0.0
            else:
                mos = float(self.data.iloc[index, self.mos_col])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
            sel = sel.long()
            mos = float(self.data.iloc[index, self.mos_col])
            # mos = proc_label(mos)
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)
        
        prompt = self.data.iloc[index, 1]

        model_name = image_name.split('_')
        model_name = model_name[0]        
        prompt_name = model_name + ' ' + prompt

        sample = {'I': patches, 'mos': mos,'prompt':prompt, 'prompt_name':prompt_name, 'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.data.index)



class AIGCIQA2023Dataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 blind = False,
                 mos_col=2,
                 get_loader=get_default_img_loader):

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.mos_col = mos_col
        # self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 1]
        model_name = self.data.iloc[index, 0]
        image_path = resolve_aigciqa2023_image_path(self.img_dir, model_name, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            mos = float(self.data.iloc[index, self.mos_col])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
            sel = sel.long()
            mos = float(self.data.iloc[index, self.mos_col])
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)
        
        prompt = self.data.iloc[index, 5]

        model_name = image_name.split('_')
        model_name = model_name[0]
        prompt_name = model_name + ' ' + prompt

        sample = {'I': patches, 'mos': mos,'prompt':prompt, 'prompt_name':prompt_name, 'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.data.index)


class EvalMuseFgclipDataset(Dataset):
    def __init__(self, json_file, img_dir, get_loader=get_default_img_loader):
        with open(json_file, "r", encoding="utf-8") as file:
            self.data = json.load(file)

        print("%d json data successfully loaded!" % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()

    def __getitem__(self, index):
        ann = self.data[index]
        image_path = os.path.join(self.img_dir, ann["img_path"])
        image = self.loader(image_path)

        prompt = ann.get("prompt", "")
        image_name = ann["img_path"]

        total_score = ann.get("total_score", None)
        if isinstance(total_score, list):
            mos = float(np.mean(total_score))
        else:
            mos = float(total_score) if total_score is not None else 0.0

        element_scores = ann.get("element_score", {}) or {}
        element_names = [k for k in element_scores.keys() if element_scores[k] is not None]
        element_targets = [float(element_scores[k]) for k in element_names]

        sample = {
            "image": image,
            "mos": mos,
            "prompt": prompt,
            "image_name": image_name,
            "element_names": element_names,
            "element_targets": element_targets,
        }
        return sample

    def __len__(self):
        return len(self.data)


class AIGCFgclipDataset(Dataset):
    def __init__(self, csv_file, img_dir, blind=False, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=",", header=None)
        self.data = self.data.iloc[1:]
        print("%d csv data successfully loaded!" % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.blind = blind

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        image = self.loader(image_path)
        prompt = self.data.iloc[index, 1]
        mos = 0.0 if self.blind else float(self.data.iloc[index, 2])
        model_name = image_name.split("_")[0]
        prompt_name = model_name + " " + prompt
        return {
            "image": image,
            "mos": mos,
            "prompt": prompt,
            "prompt_name": prompt_name,
            "image_name": image_name,
        }

    def __len__(self):
        return len(self.data.index)


class AIGCFgclip3KDataset(Dataset):
    def __init__(self, csv_file, img_dir, blind=False, mos_col=5, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=",", header=None)
        self.data = self.data.iloc[1:]
        print("%d csv data successfully loaded!" % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.blind = blind
        self.mos_col = mos_col

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        image = self.loader(image_path)
        prompt = self.data.iloc[index, 1]
        mos = 0.0 if self.blind else float(self.data.iloc[index, self.mos_col])
        model_name = image_name.split("_")[0]
        prompt_name = model_name + " " + prompt
        return {
            "image": image,
            "mos": mos,
            "prompt": prompt,
            "prompt_name": prompt_name,
            "image_name": image_name,
        }

    def __len__(self):
        return len(self.data.index)


class AIGCFgclip2023Dataset(Dataset):
    def __init__(self, csv_file, img_dir, mos_col=2, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=",", header=None)
        self.data = self.data.iloc[1:]
        print("%d csv data successfully loaded!" % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.mos_col = mos_col

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 1]
        model_name = self.data.iloc[index, 0]
        image_path = resolve_aigciqa2023_image_path(self.img_dir, model_name, image_name)
        image = self.loader(image_path)
        prompt = self.data.iloc[index, 5]
        mos = float(self.data.iloc[index, self.mos_col])
        prompt_name = model_name + " " + prompt
        return {
            "image": image,
            "mos": mos,
            "prompt": prompt,
            "prompt_name": prompt_name,
            "image_name": image_name,
        }

    def __len__(self):
        return len(self.data.index)


class EvalMuseDataset(Dataset):
    def __init__(self, json_file, img_dir, preprocess, num_patch, test, blind=False, get_loader=get_default_img_loader):
        with open(json_file, "r", encoding="utf-8") as file:
            self.data = json.load(file)

        print("%d json data successfully loaded!" % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        ann = self.data[index]
        image_path = os.path.join(self.img_dir, ann["img_path"])
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(
            -1, n_channels, kernel_h, kernel_w
        )

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,))
            sel = sel.long()
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode="bilinear", align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)

        prompt = ann.get("prompt", "")
        image_name = ann["img_path"]
        model_name = image_name.split("/")[0] if "/" in image_name else image_name.split("_")[0]
        prompt_name = model_name + " " + prompt

        if self.blind:
            mos = 0.0
        else:
            total_score = ann.get("total_score", None)
            if isinstance(total_score, list):
                mos = float(np.mean(total_score))
            else:
                mos = float(total_score) if total_score is not None else 0.0

        element_scores = ann.get("element_score", {}) or {}
        element_names = [k for k in element_scores.keys() if element_scores[k] is not None]
        element_targets = [float(element_scores[k]) for k in element_names]

        sample = {
            "I": patches,
            "mos": mos,
            "prompt": prompt,
            "prompt_name": prompt_name,
            "image_name": image_name,
            "element_names": element_names,
            "element_targets": element_targets,
        }
        return sample

    def __len__(self):
        return len(self.data)
