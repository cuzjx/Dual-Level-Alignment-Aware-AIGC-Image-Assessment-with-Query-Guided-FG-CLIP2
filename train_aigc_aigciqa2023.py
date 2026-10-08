import os
import sys
import random
from itertools import product

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from transformers import AutoImageProcessor, AutoTokenizer

from MNL_Loss import loss_m3
from utils import get_logger, log_and_print, set_dataset_aigc_2023_fgclip


code_root = os.path.abspath(os.path.dirname(__file__))
if code_root not in sys.path:
    sys.path.insert(0, code_root)

from model.modeling_fgclip2 import Fgclip2Model


checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_AIGCIQA2023_fgclip")
os.makedirs(checkpoint_dir, exist_ok=True)
base_logger = get_logger(os.path.join(checkpoint_dir, "train_test.log"), "train_aigc_aigciqa2023_fgclip")
model_dir = os.path.join(code_root, "model")

qualitys_p = ["badly", "poorly", "fairly", "well", "perfectly"]
quality_relevance_temperature = 0.07
quality_local_residual_alpha = 1
aigciqa2023_target = "correspondence"  # "quality", "authenticity", or "correspondence"

AIGCIQA2023_TARGET_TO_MOS_COL = {
    "quality": 2,
    "authenticity": 3,
    "correspondence": 4,
}

seed = 20200626
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

initial_lr = 5e-6
num_epoch = 20
train_bs = 16
val_bs = 16
num_workers = 16
max_text_length = 64
text_walk_type = "short"
image_max_num_patches = 256
opt = 0


def freeze_model(opt):
    model.logit_scale.requires_grad = False
    model.logit_bias.requires_grad = False
    if opt == 0:
        return
    if opt == 1:
        for p in model.text_model.parameters():
            p.requires_grad = False
    elif opt == 2:
        for p in model.vision_model.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.parameters():
            p.requires_grad = False


def move_batch_to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


def prepare_image_inputs(images):
    image_inputs = image_processor(images=images, return_tensors="pt", max_num_patches=image_max_num_patches)
    return move_batch_to_device(image_inputs)


def encode_image(image_inputs, return_dense=False):
    global_features = model.get_image_features(**image_inputs)
    if not return_dense:
        return global_features
    dense_features = model.get_image_dense_feature(**image_inputs)
    return global_features, dense_features


TEXT_ENCODE_CHUNK = 32


def _encode_text_batch(input_ids, attention_mask):
    return model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, walk_type=text_walk_type)


def encode_text(texts):
    text_inputs = tokenizer(
        [text.lower() for text in texts],
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    text_inputs = move_batch_to_device(text_inputs)
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs.get("attention_mask")
    if input_ids.size(0) <= TEXT_ENCODE_CHUNK:
        return _encode_text_batch(input_ids, attention_mask)
    chunks = []
    for i in range(0, input_ids.size(0), TEXT_ENCODE_CHUNK):
        ids_chunk = input_ids[i : i + TEXT_ENCODE_CHUNK]
        mask_chunk = attention_mask[i : i + TEXT_ENCODE_CHUNK] if attention_mask is not None else None
        chunks.append(_encode_text_batch(ids_chunk, mask_chunk))
    return torch.cat(chunks, dim=0)


def paired_logits(image_features, text_features, texts_per_image):
    img = F.normalize(image_features, dim=-1)
    txt = F.normalize(text_features, dim=-1).view(img.size(0), texts_per_image, -1)
    return model.logit_scale.exp() * torch.bmm(img.unsqueeze(1), txt.transpose(1, 2)).squeeze(1) + model.logit_bias


def masked_softmax(logits, mask, dim=-1, temperature=1.0):
    logits = logits / temperature
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = F.softmax(logits, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


def compute_quality_logits_batch(global_image_features, dense_image_features, image_token_mask, prompt_batch):
    texts = [f"a photo that {c} matches '{p}'" for p, c in product(prompt_batch, qualitys_p)]
    text_features = encode_text(texts).view(len(prompt_batch), len(qualitys_p), -1)
    text_features = F.normalize(text_features, dim=-1)

    dense_tokens = F.normalize(dense_image_features, dim=-1)
    token_mask = image_token_mask.bool()
    fused_logits_all = []
    for i in range(len(prompt_batch)):
        cur_tokens = dense_tokens[i]
        cur_queries = text_features[i]
        relevance_mask = token_mask[i].unsqueeze(0).expand(len(qualitys_p), cur_tokens.size(0))
        relevance_weights = masked_softmax(
            torch.matmul(cur_queries, cur_tokens.transpose(0, 1)),
            relevance_mask,
            dim=-1,
            temperature=quality_relevance_temperature,
        )
        local_features = F.normalize(torch.matmul(relevance_weights, cur_tokens), dim=-1)
        cur_global = global_image_features[i].unsqueeze(0).expand_as(local_features)
        fused_features = F.normalize(cur_global + quality_local_residual_alpha * local_features, dim=-1)
        fused_logits_all.append((fused_features * cur_queries).sum(dim=-1))

    fused_logits = torch.stack(fused_logits_all, dim=0)
    fused_logits = model.logit_scale.exp() * fused_logits + model.logit_bias
    return fused_logits


def quality_probs_from_logits(logits_per_image):
    return F.softmax(logits_per_image, dim=1)


def train(best_result, best_epoch):
    running_loss = 0.0
    model.train()
    loader = train_loaders[0]
    log_and_print(base_logger, "session:{} lr:{:.2e}".format(session + 1, optimizer.param_groups[0]["lr"]))

    step = -1
    loop = tqdm.tqdm(loader, desc="Epoch:{}".format(epoch))
    for sample_batched in loop:
        step += 1
        images, gmos, prompt = sample_batched["image"], sample_batched["mos"], sample_batched["prompt"]
        gmos = gmos.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            image_inputs = prepare_image_inputs(images)
            image_features, dense_image_features = encode_image(image_inputs, return_dense=True)
            image_token_mask = image_inputs["pixel_attention_mask"]
            logits_per_image = compute_quality_logits_batch(image_features, dense_image_features, image_token_mask, prompt)
            logits_quality = quality_probs_from_logits(logits_per_image)
            quality_preds = (
                1 * logits_quality[:, 0]
                + 2 * logits_quality[:, 1]
                + 3 * logits_quality[:, 2]
                + 4 * logits_quality[:, 3]
                + 5 * logits_quality[:, 4]
            )
            quality_preds = ((quality_preds - 1) / 4) * 5
            total_loss = loss_m3(quality_preds, gmos.detach()).mean()

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.detach().item()
        loop.set_description("Epoch:{}  Loss:{:.4f}".format(epoch, running_loss / (step + 1)))

    score, srcc, plcc = eval(aigc_val_loader, phase="val", dataset="live")
    if score > best_result["quality"]:
        log_and_print(base_logger, "**********New quality best!**********")
        best_epoch["quality"] = epoch
        best_result["quality"] = score
        best_result["srcc"] = srcc
        best_result["plcc"] = plcc
        ckpt_dir = os.path.join(checkpoint_dir, str(session + 1))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(ckpt_dir, "quality_best_ckpt.pt"))
    return best_result, best_epoch


@torch.no_grad()
def eval(loader, phase, dataset):
    model.eval()
    q_mos, q_hat = [], []
    for sample_batched in tqdm.tqdm(loader, desc="{}:{}".format(dataset, phase)):
        images, gmos, prompt = sample_batched["image"], sample_batched["mos"], sample_batched["prompt"]
        q_mos.extend(gmos.cpu().tolist())
        image_inputs = prepare_image_inputs(images)
        image_features, dense_image_features = encode_image(image_inputs, return_dense=True)
        image_token_mask = image_inputs["pixel_attention_mask"]
        logits_per_image = compute_quality_logits_batch(image_features, dense_image_features, image_token_mask, prompt)
        logits_quality = quality_probs_from_logits(logits_per_image)
        quality_preds = (
            1 * logits_quality[:, 0]
            + 2 * logits_quality[:, 1]
            + 3 * logits_quality[:, 2]
            + 4 * logits_quality[:, 3]
            + 5 * logits_quality[:, 4]
        )
        quality_preds = ((quality_preds - 1) / 4) * 5
        q_hat.extend(quality_preds.cpu().tolist())

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]
    log_and_print(base_logger, "{}:{}: srcc:{:.4f}  plcc{:.4f}".format(dataset, phase, srcc, plcc))
    return (srcc + plcc) / 2, srcc, plcc


tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
best_result_list = []

if aigciqa2023_target not in AIGCIQA2023_TARGET_TO_MOS_COL:
    raise ValueError("aigciqa2023_target must be 'quality', 'authenticity', or 'correspondence'")

mos_col = AIGCIQA2023_TARGET_TO_MOS_COL[aigciqa2023_target]

for session in range(0, 10):
    model = Fgclip2Model.from_pretrained(model_dir, local_files_only=True).to(device)
    freeze_model(opt)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=initial_lr, weight_decay=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-7)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_result = {"quality": 0.0, "srcc": 0.0, "plcc": 0.0}
    best_epoch = {"quality": 0}

    aigc_train_csv = os.path.join(code_root, "Database", "AIGCIQA2023", str(session + 1), "train.csv")
    aigc_val_csv = os.path.join(code_root, "Database", "AIGCIQA2023", str(session + 1), "val.csv")
    aigc_set = os.path.join(code_root, "data", "AIGCIQA2023", "file")
    aigc_train_loader = set_dataset_aigc_2023_fgclip(
        aigc_train_csv, train_bs, aigc_set, num_workers, test=False, mos_col=mos_col
    )
    aigc_val_loader = set_dataset_aigc_2023_fgclip(
        aigc_val_csv, val_bs, aigc_set, num_workers, test=True, mos_col=mos_col
    )
    train_loaders = [aigc_train_loader]

    for epoch in range(0, num_epoch):
        best_result, best_epoch = train(best_result, best_epoch)
        scheduler.step()
        log_and_print(base_logger, "...............current quality best, session:{}...............".format(session + 1))
        log_and_print(base_logger, "best quality epoch:{}".format(best_epoch["quality"]))
        log_and_print(
            base_logger,
            "best quality result:{}, srcc:{}, plcc{}".format(
                best_result["quality"], best_result["srcc"], best_result["plcc"]
            ),
        )

    best_result_list.append(best_result)

avg_srcc = sum(item["srcc"] for item in best_result_list) / len(best_result_list)
avg_plcc = sum(item["plcc"] for item in best_result_list) / len(best_result_list)
log_and_print(base_logger, "all_finished,average srcc:{}, plcc:{}".format(avg_srcc, avg_plcc))
