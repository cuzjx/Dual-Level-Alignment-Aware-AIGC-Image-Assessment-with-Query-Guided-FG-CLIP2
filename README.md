# Code Release

This folder contains the cleaned release code for the paper:

`Dual-Level Alignment-Aware AIGC Image Assessment with Query-Guided FG-CLIP2`

The released scripts are organized around three datasets:

- `EvalMuse`
- `AGIQA-3K`
- `AIGCIQA2023`

## Main files

- `train_evalmuse_fgclip.py`: dual-level training and evaluation on EvalMuse
- `train_aigc_agiqa3k.py`: holistic-branch training and evaluation on AGIQA-3K
- `train_aigc_aigciqa2023.py`: holistic-branch training and evaluation on AIGCIQA2023
- `find_evalmuse_cases_fgclip.py`: utility script for retrieving qualitative EvalMuse cases
- `ImageDataset.py`, `utils.py`, `MNL_Loss.py`: dataset, loader, and loss utilities

## Expected directory structure

The scripts in this release assume that `code/` itself is the working root:

```text
code/
├── README.md
├── train_evalmuse_fgclip.py
├── train_aigc_agiqa3k.py
├── train_aigc_aigciqa2023.py
├── find_evalmuse_cases_fgclip.py
├── ImageDataset.py
├── utils.py
├── MNL_Loss.py
├── dataset/
│   ├── train.json
│   ├── eval.json
│   └── images/                  # user needs to add EvalMuse images
├── Database/
│   ├── AGIQA-3K/
│   └── AIGCIQA2023/
├── data/
│   ├── AGIQA-3K/file/           # user needs to add AGIQA-3K images
│   └── AIGCIQA2023/file/        # user needs to add AIGCIQA2023 images
└── model/
```

## Files you still need to add manually

For size reasons, this release does **not** include the complete image data for the three datasets or the full FG-CLIP2 checkpoint files.

### 1. Dataset files

You need to add the missing image data for the following three datasets:

- `code/dataset/images/` for EvalMuse
- `code/data/AGIQA-3K/file/` for AGIQA-3K
- `code/data/AIGCIQA2023/file/` for AIGCIQA2023

The annotation files already included in this release expect the image folders above.

### 2. FG-CLIP2 model files

You also need to place the missing large FG-CLIP2 files into `code/model/`:

- `model.safetensors`
- `tokenizer.json`
- `tokenizer.model`

These files are omitted here because they are relatively large, but they are required for:

- `Fgclip2Model.from_pretrained(...)`
- `AutoTokenizer.from_pretrained(...)`
- `AutoImageProcessor.from_pretrained(...)`

## Notes

- The released `train_evalmuse_fgclip.py` has been cleaned to match the paper's main dual-level setting more closely.
- The EvalMuse training script now keeps the released quality branch and element branch together by default, instead of preserving extra ablation-only control paths.
- The training scripts now resolve dataset and model paths relative to `code/` rather than to the parent project root.
- `find_evalmuse_cases_fgclip.py` is only a qualitative-case utility and is not part of the quantitative training or evaluation pipeline.

## Example usage

Run from inside `code/`:

```bash
python train_evalmuse_fgclip.py
python train_aigc_agiqa3k.py
python train_aigc_aigciqa2023.py
```
