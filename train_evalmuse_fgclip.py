import os
import sys
import random
import numpy as np
from itertools import product

import scipy.stats
import torch
import torch.nn.functional as F
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from transformers import AutoImageProcessor, AutoTokenizer

from MNL_Loss import loss_m3
from utils import set_dataset_evalmuse_fgclip, get_logger, log_and_print


code_root = os.path.abspath(os.path.dirname(__file__))
if code_root not in sys.path:
    sys.path.insert(0, code_root)

from model.modeling_fgclip2 import Fgclip2Model


checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_evalmuse_fgclip")
os.makedirs(checkpoint_dir, exist_ok=True)
log_path = os.path.join(checkpoint_dir, "train.log")
logger = get_logger(log_path, "train_evalmuse_fgclip")
model_dir = os.path.join(code_root, "model")

qualitys_p = ["badly", "poorly", "fairly", "well", "perfectly"]
quality_local_residual_alpha = 0.15
# Quality scores live on the [QUALITY_MIN, QUALITY_MAX] scale (1..5 in EvalMuse).
# We normalize predictions/targets to [0, 1] before the L1 term so that the quality
# loss shares the same scale as the element loss (also in [0, 1]); this makes
# lambda_quality / lambda_element pure, scale-free weights.
QUALITY_MIN = 1.0
QUALITY_MAX = 5.0


def normalize_quality(x):
    return (x - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN)

# Element prompt templates used in the released paper code.
element_templates = (
    "{element} present: {prompt}",
    "{element} not present: {prompt}",
)

seed = 20260727

train_json = os.path.join(code_root, "dataset", "train.json")
val_json = os.path.join(code_root, "dataset", "eval.json")
image_root = os.path.join(code_root, "dataset", "images")


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_epoch = 25
train_bs = 8
val_bs = 8
num_workers = 16
resume_checkpoint = ""  # e.g. "./checkpoints_evalmuse_fgclip/main_best_ckpt.pt"
max_text_length = 64
text_walk_type = "short"
image_max_num_patches = 256  # FG-CLIP2 native max patches for whole-image encoding
element_relevance_temperature = 0.07
quality_relevance_temperature = 0.07

train_lr = 5e-6
lambda_quality = 1.0
lambda_element = 1.0


def move_batch_to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


def prepare_image_inputs(images):
    # images: list of PIL.Image. FG-CLIP2 handles dynamic resolution + patchify natively.
    image_inputs = image_processor(
        images=images,
        return_tensors="pt",
        max_num_patches=image_max_num_patches,
    )
    return move_batch_to_device(image_inputs)


def encode_image(image_inputs, return_dense=False):
    global_features = model.get_image_features(**image_inputs)
    if not return_dense:
        return global_features
    dense_features = model.get_image_dense_feature(**image_inputs)
    return global_features, dense_features


TEXT_ENCODE_CHUNK = 32


def _encode_text_batch(input_ids, attention_mask):
    return model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
        walk_type=text_walk_type,
    )


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
    attention_mask = text_inputs.get("attention_mask", None)

    n = input_ids.size(0)
    if n <= TEXT_ENCODE_CHUNK:
        return _encode_text_batch(input_ids, attention_mask)

    chunks = []
    for i in range(0, n, TEXT_ENCODE_CHUNK):
        ids_chunk = input_ids[i : i + TEXT_ENCODE_CHUNK]
        mask_chunk = attention_mask[i : i + TEXT_ENCODE_CHUNK] if attention_mask is not None else None
        chunks.append(_encode_text_batch(ids_chunk, mask_chunk))
    return torch.cat(chunks, dim=0)


def paired_logits(image_features, text_features, texts_per_image):
    # Only compute each image against its OWN block of texts (avoids batch_size^2 logits matrix).
    # image_features: (B, D); text_features: (B * texts_per_image, D)
    img = image_features / image_features.norm(dim=-1, keepdim=True)
    txt = text_features / text_features.norm(dim=-1, keepdim=True)
    B = img.size(0)
    txt = txt.view(B, texts_per_image, -1)  # (B, T, D)
    scale = model.logit_scale.exp()
    # (B, 1, D) x (B, D, T) -> (B, 1, T) -> (B, T)
    logits = scale * torch.bmm(img.unsqueeze(1), txt.transpose(1, 2)).squeeze(1) + model.logit_bias
    return logits  # (B, texts_per_image)


def quality_probs_from_logits(logits_per_image):
    # logits_per_image: (batch_size, 5)
    return F.softmax(logits_per_image, dim=1)


def compute_quality_logits_batch(global_image_features, dense_image_features, image_token_mask, prompt_batch):
    texts = [f"a photo that {c} matches '{p}'" for p, c in product(prompt_batch, qualitys_p)]
    text_features = encode_text(texts).view(len(prompt_batch), len(qualitys_p), -1)
    text_features = F.normalize(text_features, dim=-1)

    global_logits = paired_logits(global_image_features, text_features.view(-1, text_features.size(-1)), len(qualitys_p))

    token_mask = image_token_mask.bool()
    dense_tokens = F.normalize(dense_image_features, dim=-1)
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

def parse_element(element):
    name = element.rpartition("(")[0].strip()
    if not name:  # no parenthesis found
        name = element.strip()
    return name


def build_element_prompts(element, prompt_text):
    name = parse_element(element)
    return (
        element_templates[0].format(element=name, prompt=prompt_text),
        element_templates[1].format(element=name, prompt=prompt_text),
    )


def masked_softmax(logits, mask, dim=-1, temperature=1.0):
    logits = logits / temperature
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = F.softmax(logits, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights


def compute_element_scores_batch(global_image_features, dense_image_features, image_token_mask, element_names_batch, element_targets_batch, prompt_batch):
    max_elems = max((len(names) for names in element_names_batch), default=0)
    if max_elems == 0:
        return None, None, None

    batch_size = len(element_names_batch)
    scoring_prompts = []
    targets = torch.zeros((batch_size, max_elems), device=device)
    masks = torch.zeros((batch_size, max_elems), dtype=torch.bool, device=device)

    for i, names in enumerate(element_names_batch):
        count = len(names)
        if count == 0:
            scoring_prompts.extend([""] * (max_elems * 2))
            continue

        prompt_text = prompt_batch[i]
        for el in names:
            present_text, absent_text = build_element_prompts(el, prompt_text)
            scoring_prompts.append(present_text)
            scoring_prompts.append(absent_text)

        if count < max_elems:
            scoring_prompts.extend([""] * ((max_elems - count) * 2))

        targets[i, :count] = torch.tensor(element_targets_batch[i], device=device)
        masks[i, :count] = True

    scoring_text_features = encode_text(scoring_prompts).view(batch_size, max_elems, 2, -1)
    scoring_text_features = F.normalize(scoring_text_features, dim=-1)
    query_text_features = scoring_text_features

    token_mask = image_token_mask.bool()
    dense_tokens = F.normalize(dense_image_features, dim=-1)

    pair_logits_all = []
    for i in range(batch_size):
        cur_tokens = dense_tokens[i]
        cur_token_mask = token_mask[i]
        cur_pos_scoring = scoring_text_features[i, :, 0, :]
        cur_neg_scoring = scoring_text_features[i, :, 1, :]
        cur_pos_queries = query_text_features[i, :, 0, :]
        cur_neg_queries = query_text_features[i, :, 1, :]

        relevance_mask = cur_token_mask.unsqueeze(0).expand(max_elems, cur_tokens.size(0))
        relevance_weights_pos = masked_softmax(
            torch.matmul(cur_pos_queries, cur_tokens.transpose(0, 1)),
            relevance_mask,
            dim=-1,
            temperature=element_relevance_temperature,
        )
        relevance_weights_neg = masked_softmax(
            torch.matmul(cur_neg_queries, cur_tokens.transpose(0, 1)),
            relevance_mask,
            dim=-1,
            temperature=element_relevance_temperature,
        )
        local_pos_features = F.normalize(torch.matmul(relevance_weights_pos, cur_tokens), dim=-1)
        local_neg_features = F.normalize(torch.matmul(relevance_weights_neg, cur_tokens), dim=-1)
        score_present = (local_pos_features * cur_pos_scoring).sum(dim=-1)
        score_absent = (local_neg_features * cur_neg_scoring).sum(dim=-1)
        pair_logits_all.append(torch.stack([score_present, score_absent], dim=-1))

    pair_logits = torch.stack(pair_logits_all, dim=0)  # (B, E, 2)
    pair_logits = model.logit_scale.exp() * pair_logits + model.logit_bias
    element_scores = F.softmax(pair_logits, dim=-1)[..., 0]  # (batch_size, max_elems) present probability

    return element_scores, pair_logits, targets, masks


def compute_element_loss_batch(global_image_features, dense_image_features, image_token_mask, element_names_batch, element_targets_batch, prompt_batch):
    scores, _, targets, masks = compute_element_scores_batch(
        global_image_features, dense_image_features, image_token_mask, element_names_batch, element_targets_batch, prompt_batch
    )
    if scores is None or masks.sum() == 0:
        return None, None
    return F.l1_loss(scores[masks], targets[masks]), scores


def compute_element_metrics_batch(global_image_features, dense_image_features, image_token_mask, element_names_batch, element_targets_batch, prompt_batch):
    scores, _, targets, masks = compute_element_scores_batch(
        global_image_features, dense_image_features, image_token_mask, element_names_batch, element_targets_batch, prompt_batch
    )
    if scores is None or masks.sum() == 0:
        return None

    valid_scores = scores[masks]
    valid_targets = targets[masks]
    pred_bin = valid_scores >= 0.5
    target_bin = valid_targets >= 0.5
    correct = (pred_bin == target_bin).sum().item()
    total = valid_targets.numel()
    mae = torch.abs(valid_scores - valid_targets).sum().item()
    return correct, total, mae, valid_scores.detach().cpu().tolist(), valid_targets.detach().cpu().tolist()


def search_best_element_threshold(targets, preds):
    if not targets:
        return 0.5, 0.0

    target_tensor = torch.tensor(targets) >= 0.5
    pred_tensor = torch.tensor(preds)
    best_threshold = 0.5
    best_acc = 0.0
    for i in range(101):
        threshold = i / 100
        acc = ((pred_tensor >= threshold) == target_tensor).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold, best_acc


def load_resume_checkpoint(resume_path):
    if not resume_path:
        return 0, {"main": 0.0}, {"main": 0}

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_result = ckpt.get("best_result", {"main": 0.0})
    best_epoch = ckpt.get("best_epoch", {"main": max(start_epoch - 1, 0)})
    log_and_print(logger, "resume checkpoint: {}".format(resume_path))
    log_and_print(logger, "resume from epoch: {}".format(start_epoch))
    return start_epoch, best_result, best_epoch


def train(model, best_result, best_epoch):
    running_loss = 0.0
    running_quality_loss = 0.0
    running_element_loss = 0.0
    quality_steps = 0
    element_steps = 0
    model.train()
    loader = train_loaders[0]

    current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
    log_and_print(
        logger,
        "lr:{:.2e} lambda_quality:{:.2f} lambda_element:{:.2f}".format(
            current_lr,
            lambda_quality,
            lambda_element,
        )
    )

    step = -1
    loop = tqdm.tqdm(loader, desc="Epoch:{}".format(epoch))
    for sample_batched in loop:
        step += 1
        images, gmos, prompt = sample_batched["image"], sample_batched["mos"], sample_batched["prompt"]
        element_names = sample_batched["element_names"]
        element_targets = sample_batched["element_targets"]

        gmos = gmos.to(device)
        batch_size = len(images)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            image_inputs = prepare_image_inputs(images)
            image_features, dense_image_features = encode_image(image_inputs, return_dense=True)
            image_token_mask = image_inputs["pixel_attention_mask"]
            total_loss = None

            logits_per_image = compute_quality_logits_batch(
                image_features,
                dense_image_features,
                image_token_mask,
                prompt,
            )
            logits_quality = quality_probs_from_logits(logits_per_image)
            quality_preds = (
                1 * logits_quality[:, 0]
                + 2 * logits_quality[:, 1]
                + 3 * logits_quality[:, 2]
                + 4 * logits_quality[:, 3]
                + 5 * logits_quality[:, 4]
            )
            loss_quality = loss_m3(
                normalize_quality(quality_preds), normalize_quality(gmos.detach())
            ).mean()
            total_loss = lambda_quality * loss_quality
            running_quality_loss += loss_quality.detach().item()
            quality_steps += 1

            loss_element, _ = compute_element_loss_batch(
                image_features,
                dense_image_features,
                image_token_mask,
                element_names,
                element_targets,
                prompt,
            )
            if loss_element is not None:
                running_element_loss += loss_element.detach().item()
                element_steps += 1
                total_loss = total_loss + lambda_element * loss_element

        if total_loss is None:
            continue

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.detach().item()
        avg_loss = running_loss / (step + 1)
        avg_quality = running_quality_loss / max(quality_steps, 1)
        avg_element = running_element_loss / max(element_steps, 1)
        loop.set_description(
            "Epoch:{}  Loss:{:.4f} Q:{:.4f} E:{:.4f}".format(epoch, avg_loss, avg_quality, avg_element)
        )

    if epoch >= 0:
        main_score = eval(live_val_loader, phase="val", dataset="evalmuse")

        last_ckpt_name = os.path.join(checkpoint_dir, "last_ckpt.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_result": best_result,
                "best_epoch": best_epoch,
            },
            last_ckpt_name,
        )

        if main_score > best_result["main"]:
            log_and_print(logger, "**********New main best!**********")
            best_epoch["main"] = epoch
            best_result["main"] = main_score
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_name = os.path.join(checkpoint_dir, "main_best_ckpt.pt")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_result": best_result,
                    "best_epoch": best_epoch,
                },
                ckpt_name,
            )

    return best_result, best_epoch


@torch.no_grad()
def eval(loader, phase, dataset):
    model.eval()

    q_mos = []
    q_hat = []
    element_correct = 0
    element_total = 0
    element_abs_error = 0.0
    element_pred_all = []
    element_target_all = []

    for sample_batched in tqdm.tqdm(loader, desc="{}:{}".format(dataset, phase)):
        images, gmos, prompt = sample_batched["image"], sample_batched["mos"], sample_batched["prompt"]
        element_names = sample_batched.get("element_names")
        element_targets = sample_batched.get("element_targets")

        batch_size = len(images)
        image_inputs = prepare_image_inputs(images)
        image_features, dense_image_features = encode_image(image_inputs, return_dense=True)
        image_token_mask = image_inputs["pixel_attention_mask"]

        q_mos.extend(gmos.cpu().tolist())
        logits_per_image = compute_quality_logits_batch(
            image_features,
            dense_image_features,
            image_token_mask,
            prompt,
        )
        logits_quality = quality_probs_from_logits(logits_per_image)
        quality_preds = (
            1 * logits_quality[:, 0]
            + 2 * logits_quality[:, 1]
            + 3 * logits_quality[:, 2]
            + 4 * logits_quality[:, 3]
            + 5 * logits_quality[:, 4]
        )
        q_hat.extend(quality_preds.cpu().tolist())

        if element_names is not None and element_targets is not None:
            metrics = compute_element_metrics_batch(
                image_features,
                dense_image_features,
                image_token_mask,
                element_names,
                element_targets,
                prompt,
            )
            if metrics is not None:
                correct, total, mae, preds, targets = metrics
                element_correct += correct
                element_total += total
                element_abs_error += mae
                element_pred_all.extend(preds)
                element_target_all.extend(targets)

    if len(q_mos) > 1:
        srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]
    else:
        srcc = 0.0
        plcc = 0.0

    element_acc = (element_correct / element_total) if element_total else 0.0
    element_mae = (element_abs_error / element_total) if element_total else 0.0
    best_threshold, element_best_acc = search_best_element_threshold(element_target_all, element_pred_all)
    if element_total > 1:
        element_srcc = scipy.stats.mstats.spearmanr(x=element_target_all, y=element_pred_all)[0]
        element_plcc = scipy.stats.pearsonr(x=element_target_all, y=element_pred_all)[0]
    else:
        element_srcc = 0.0
        element_plcc = 0.0

    main_score = ((srcc + plcc) * 0.25) + (0.5 * element_best_acc)

    # Confusion matrix for element presence (positive = present, threshold 0.5).
    # TP: present correctly judged present; TN: absent correctly judged absent;
    # FP: absent misjudged as present; FN: present misjudged as absent.
    tp = tn = fp = fn = 0
    for p, t in zip(element_pred_all, element_target_all):
        pred_present = p >= 0.5
        target_present = t >= 0.5
        if target_present and pred_present:
            tp += 1
        elif (not target_present) and (not pred_present):
            tn += 1
        elif (not target_present) and pred_present:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print_text = (
        dataset
        + ":"
        + phase
        + ": "
        + "srcc:{:.4f}  plcc:{:.4f}  elem_acc@0.5:{:.4f}  elem_acc@best:{:.4f}  elem_thr:{:.2f}  elem_mae:{:.4f}  elem_srcc:{:.4f}  elem_plcc:{:.4f}  main:{:.4f}".format(
            srcc,
            plcc,
            element_acc,
            element_best_acc,
            best_threshold,
            element_mae,
            element_srcc,
            element_plcc,
            main_score,
        )
    )
    log_and_print(logger, print_text)
    log_and_print(
        logger,
        "  element confusion@0.5 (positive=present): "
        "TP:{}  TN:{}  FP:{}  FN:{}  precision:{:.4f}  recall:{:.4f}".format(
            tp, tn, fp, fn, precision, recall
        )
    )
    return main_score


tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
model = Fgclip2Model.from_pretrained(model_dir, local_files_only=True).to(device)
model.logit_scale.requires_grad = False
model.logit_bias.requires_grad = False

optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=train_lr, weight_decay=0.001)
scaler = GradScaler(enabled=torch.cuda.is_available())

best_result = {"main": 0.0}
best_epoch = {"main": 0}

train_loader = set_dataset_evalmuse_fgclip(train_json, train_bs, image_root, num_workers, test=False)
live_val_loader = set_dataset_evalmuse_fgclip(val_json, val_bs, image_root, num_workers, test=True)
train_loaders = [train_loader]

log_and_print(logger, "fg-clip model path: {}".format(model_dir))
log_and_print(logger, "train_json path: {}".format(train_json))
log_and_print(logger, "train dataset size: {}".format(len(train_loader.dataset)))
log_and_print(logger, "train loader batches: {}".format(len(train_loader)))

start_epoch, best_result, best_epoch = load_resume_checkpoint(resume_checkpoint)
remaining_epochs = max(num_epoch - start_epoch, 1)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-7)

for epoch in range(start_epoch, num_epoch):
    best_result, best_epoch = train(model, best_result, best_epoch)
    scheduler.step()

    log_and_print(logger, "...............current main best...............")
    log_and_print(logger, "best main epoch:{}".format(best_epoch["main"]))
    log_and_print(logger, "best main result:{}".format(best_result["main"]))
