import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from utils import set_dataset_evalmuse_fgclip


code_root = os.path.abspath(os.path.dirname(__file__))
if code_root not in sys.path:
    sys.path.insert(0, code_root)

from model.modeling_fgclip2 import Fgclip2Model


QUALITY_LEVELS = ["badly", "poorly", "fairly", "well", "perfectly"]
MODEL_DIR = os.path.join(code_root, "model")
VAL_JSON = os.path.join(code_root, "dataset", "eval.json")
IMAGE_ROOT = os.path.join(code_root, "dataset", "images")

QUALITY_MIN = 1.0
QUALITY_MAX = 5.0
TEXT_ENCODE_CHUNK = 32
MAX_TEXT_LENGTH = 64
TEXT_WALK_TYPE = "short"
IMAGE_MAX_NUM_PATCHES = 256

ELEMENT_TEMPLATES = (
    "{element} present: {prompt}",
    "{element} not present: {prompt}",
)

_CATEGORY_GROUP = {
    "object": "entity",
    "animal": "entity",
    "human": "entity",
    "food": "entity",
    "location": "scene",
    "activity": "activity",
    "color": "appearance",
    "material": "appearance",
    "shape": "appearance",
    "attribute": "style",
    "spatial": "spatial",
    "counting": "counting",
}

_GROUP_TEMPLATES = {
    "entity": ("a photo containing {name}", "a photo without {name}"),
    "scene": ("a photo taken at {name}", "a photo not taken at {name}"),
    "activity": ("a photo showing {name}", "a photo not showing {name}"),
    "appearance": ("a photo of something {name}", "a photo of something not {name}"),
    "style": ("a photo in the {name} style", "a photo not in the {name} style"),
    "spatial": ("a photo with something at the {name}", "a photo with nothing at the {name}"),
    "counting": ("a photo containing {name}", "a photo without {name}"),
    "default": ("a photo with {name}", "a photo without {name}"),
}


@dataclass
class EvalConfig:
    name: str
    checkpoint_path: str
    use_quality_improvement: bool
    use_element_improvement: bool
    use_element_only_queries: bool
    use_category_templates: bool


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if v.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two EvalMuse FG-CLIP2 checkpoints and export success/failure cases."
    )
    parser.add_argument("--base-ckpt", required=True, help="Checkpoint path for the baseline model.")
    parser.add_argument("--compare-ckpt", required=True, help="Checkpoint path for the improved model.")
    parser.add_argument("--output-dir", required=True, help="Directory to save case-mining CSV files.")
    parser.add_argument("--val-json", default=VAL_JSON, help="EvalMuse validation json.")
    parser.add_argument("--image-root", default=IMAGE_ROOT, help="EvalMuse image root directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--top-k", type=int, default=50, help="Number of rows to export per case bucket.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quality-alpha", type=float, default=0.15)
    parser.add_argument("--quality-temperature", type=float, default=0.07)
    parser.add_argument("--element-temperature", type=float, default=0.07)
    parser.add_argument("--base-quality-impr", type=str2bool, default=False)
    parser.add_argument("--base-element-impr", type=str2bool, default=False)
    parser.add_argument("--base-element-only-query", type=str2bool, default=False)
    parser.add_argument("--base-category-templates", type=str2bool, default=False)
    parser.add_argument("--compare-quality-impr", type=str2bool, default=True)
    parser.add_argument("--compare-element-impr", type=str2bool, default=True)
    parser.add_argument("--compare-element-only-query", type=str2bool, default=False)
    parser.add_argument("--compare-category-templates", type=str2bool, default=False)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def parse_element(element):
    name = element.rpartition("(")[0].strip()
    cat = element.rpartition("(")[2].rstrip(")$ ").strip().lower()
    if not name:
        name = element.strip()
        cat = ""
    cat_top = cat.split("-")[0].strip()
    return name, cat_top


def build_element_prompts(element, prompt_text, use_category_templates):
    name, cat_top = parse_element(element)
    if not use_category_templates:
        return (
            ELEMENT_TEMPLATES[0].format(element=name, prompt=prompt_text),
            ELEMENT_TEMPLATES[1].format(element=name, prompt=prompt_text),
        )
    group = _CATEGORY_GROUP.get(cat_top, "default")
    present_tpl, absent_tpl = _GROUP_TEMPLATES[group]
    return present_tpl.format(name=name), absent_tpl.format(name=name)


def build_element_query_prompts(element, prompt_text, use_category_templates, use_element_only_queries):
    name, _ = parse_element(element)
    if use_element_only_queries:
        return name, None
    return build_element_prompts(element, prompt_text, use_category_templates)


def masked_softmax(logits, mask, dim=-1, temperature=1.0):
    logits = logits / temperature
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = F.softmax(logits, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights


class CaseMiner:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
        self.model = Fgclip2Model.from_pretrained(MODEL_DIR, local_files_only=True).to(self.device)
        ckpt = torch.load(config.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def prepare_image_inputs(self, images):
        image_inputs = self.image_processor(
            images=images,
            return_tensors="pt",
            max_num_patches=IMAGE_MAX_NUM_PATCHES,
        )
        return move_batch_to_device(image_inputs, self.device)

    def encode_image(self, image_inputs):
        global_features = self.model.get_image_features(**image_inputs)
        dense_features = self.model.get_image_dense_feature(**image_inputs)
        return global_features, dense_features

    def _encode_text_batch(self, input_ids, attention_mask):
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            walk_type=TEXT_WALK_TYPE,
        )

    def encode_text(self, texts):
        text_inputs = self.tokenizer(
            [text.lower() for text in texts],
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        text_inputs = move_batch_to_device(text_inputs, self.device)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs.get("attention_mask", None)
        n = input_ids.size(0)
        if n <= TEXT_ENCODE_CHUNK:
            return self._encode_text_batch(input_ids, attention_mask)

        chunks = []
        for i in range(0, n, TEXT_ENCODE_CHUNK):
            ids_chunk = input_ids[i : i + TEXT_ENCODE_CHUNK]
            mask_chunk = attention_mask[i : i + TEXT_ENCODE_CHUNK] if attention_mask is not None else None
            chunks.append(self._encode_text_batch(ids_chunk, mask_chunk))
        return torch.cat(chunks, dim=0)

    def paired_logits(self, image_features, text_features, texts_per_image):
        img = F.normalize(image_features, dim=-1)
        txt = F.normalize(text_features, dim=-1).view(image_features.size(0), texts_per_image, -1)
        scale = self.model.logit_scale.exp()
        logits = scale * torch.bmm(img.unsqueeze(1), txt.transpose(1, 2)).squeeze(1) + self.model.logit_bias
        return logits

    def compute_quality_logits_batch(self, global_image_features, dense_image_features, image_token_mask, prompt_batch):
        texts = [f"a photo that {c} matches '{p}'" for p, c in product(prompt_batch, QUALITY_LEVELS)]
        text_features = self.encode_text(texts).view(len(prompt_batch), len(QUALITY_LEVELS), -1)
        text_features = F.normalize(text_features, dim=-1)

        global_logits = self.paired_logits(
            global_image_features,
            text_features.view(-1, text_features.size(-1)),
            len(QUALITY_LEVELS),
        )
        if not self.config.use_quality_improvement:
            return global_logits

        token_mask = image_token_mask.bool()
        dense_tokens = F.normalize(dense_image_features, dim=-1)
        fused_logits_all = []
        for i in range(len(prompt_batch)):
            cur_tokens = dense_tokens[i]
            cur_queries = text_features[i]
            relevance_mask = token_mask[i].unsqueeze(0).expand(len(QUALITY_LEVELS), cur_tokens.size(0))
            relevance_weights = masked_softmax(
                torch.matmul(cur_queries, cur_tokens.transpose(0, 1)),
                relevance_mask,
                dim=-1,
                temperature=self.args.quality_temperature,
            )
            local_features = F.normalize(torch.matmul(relevance_weights, cur_tokens), dim=-1)
            cur_global = global_image_features[i].unsqueeze(0).expand_as(local_features)
            fused_features = F.normalize(cur_global + self.args.quality_alpha * local_features, dim=-1)
            fused_logits_all.append((fused_features * cur_queries).sum(dim=-1))

        fused_logits = torch.stack(fused_logits_all, dim=0)
        return self.model.logit_scale.exp() * fused_logits + self.model.logit_bias

    def compute_element_scores_batch(self, global_image_features, dense_image_features, image_token_mask, element_names_batch, element_targets_batch, prompt_batch):
        max_elems = max((len(names) for names in element_names_batch), default=0)
        if max_elems == 0:
            return None, None, None

        batch_size = len(element_names_batch)
        scoring_prompts = []
        shared_query_prompts = [] if self.config.use_element_only_queries else None
        targets = torch.zeros((batch_size, max_elems), device=self.device)
        masks = torch.zeros((batch_size, max_elems), dtype=torch.bool, device=self.device)

        for i, names in enumerate(element_names_batch):
            count = len(names)
            prompt_text = prompt_batch[i]
            for el in names:
                present_text, absent_text = build_element_prompts(el, prompt_text, self.config.use_category_templates)
                scoring_prompts.extend([present_text, absent_text])
                if self.config.use_element_only_queries:
                    shared_query_text, _ = build_element_query_prompts(
                        el,
                        prompt_text,
                        self.config.use_category_templates,
                        self.config.use_element_only_queries,
                    )
                    shared_query_prompts.append(shared_query_text)

            if count < max_elems:
                scoring_prompts.extend([""] * ((max_elems - count) * 2))
                if self.config.use_element_only_queries:
                    shared_query_prompts.extend([""] * (max_elems - count))

            if count:
                targets[i, :count] = torch.tensor(element_targets_batch[i], device=self.device)
                masks[i, :count] = True

        scoring_text_features = self.encode_text(scoring_prompts).view(batch_size, max_elems, 2, -1)
        scoring_text_features = F.normalize(scoring_text_features, dim=-1)

        if self.config.use_element_only_queries:
            shared_query_features = self.encode_text(shared_query_prompts).view(batch_size, max_elems, -1)
            shared_query_features = F.normalize(shared_query_features, dim=-1)
        else:
            query_text_features = scoring_text_features

        if not self.config.use_element_improvement:
            normalized_image_features = F.normalize(global_image_features, dim=-1)
            pair_logits = (normalized_image_features.unsqueeze(1).unsqueeze(2) * scoring_text_features).sum(dim=-1)
            pair_logits = self.model.logit_scale.exp() * pair_logits + self.model.logit_bias
            element_scores = F.softmax(pair_logits, dim=-1)[..., 0]
            return element_scores, targets, masks

        token_mask = image_token_mask.bool()
        dense_tokens = F.normalize(dense_image_features, dim=-1)
        pair_logits_all = []
        for i in range(batch_size):
            cur_tokens = dense_tokens[i]
            cur_token_mask = token_mask[i]
            cur_pos_scoring = scoring_text_features[i, :, 0, :]
            cur_neg_scoring = scoring_text_features[i, :, 1, :]
            relevance_mask = cur_token_mask.unsqueeze(0).expand(max_elems, cur_tokens.size(0))

            if self.config.use_element_only_queries:
                cur_shared_queries = shared_query_features[i]
                relevance_weights_shared = masked_softmax(
                    torch.matmul(cur_shared_queries, cur_tokens.transpose(0, 1)),
                    relevance_mask,
                    dim=-1,
                    temperature=self.args.element_temperature,
                )
                shared_local_features = F.normalize(torch.matmul(relevance_weights_shared, cur_tokens), dim=-1)
                score_present = (shared_local_features * cur_pos_scoring).sum(dim=-1)
                score_absent = (shared_local_features * cur_neg_scoring).sum(dim=-1)
            else:
                cur_pos_queries = query_text_features[i, :, 0, :]
                cur_neg_queries = query_text_features[i, :, 1, :]
                relevance_weights_pos = masked_softmax(
                    torch.matmul(cur_pos_queries, cur_tokens.transpose(0, 1)),
                    relevance_mask,
                    dim=-1,
                    temperature=self.args.element_temperature,
                )
                relevance_weights_neg = masked_softmax(
                    torch.matmul(cur_neg_queries, cur_tokens.transpose(0, 1)),
                    relevance_mask,
                    dim=-1,
                    temperature=self.args.element_temperature,
                )
                local_pos_features = F.normalize(torch.matmul(relevance_weights_pos, cur_tokens), dim=-1)
                local_neg_features = F.normalize(torch.matmul(relevance_weights_neg, cur_tokens), dim=-1)
                score_present = (local_pos_features * cur_pos_scoring).sum(dim=-1)
                score_absent = (local_neg_features * cur_neg_scoring).sum(dim=-1)

            pair_logits_all.append(torch.stack([score_present, score_absent], dim=-1))

        pair_logits = torch.stack(pair_logits_all, dim=0)
        pair_logits = self.model.logit_scale.exp() * pair_logits + self.model.logit_bias
        element_scores = F.softmax(pair_logits, dim=-1)[..., 0]
        return element_scores, targets, masks

    @torch.no_grad()
    def run(self, loader):
        results = {"image_rows": {}, "element_rows": {}}
        for sample_batched in tqdm.tqdm(loader, desc=f"eval:{self.config.name}"):
            images = sample_batched["image"]
            prompts = sample_batched["prompt"]
            image_names = sample_batched["image_name"]
            mos = sample_batched["mos"].tolist()
            element_names_batch = sample_batched["element_names"]
            element_targets_batch = sample_batched["element_targets"]

            image_inputs = self.prepare_image_inputs(images)
            image_features, dense_image_features = self.encode_image(image_inputs)
            image_token_mask = image_inputs["pixel_attention_mask"]

            quality_logits = self.compute_quality_logits_batch(
                image_features, dense_image_features, image_token_mask, prompts
            )
            quality_probs = F.softmax(quality_logits, dim=1)
            quality_preds = (
                1 * quality_probs[:, 0]
                + 2 * quality_probs[:, 1]
                + 3 * quality_probs[:, 2]
                + 4 * quality_probs[:, 3]
                + 5 * quality_probs[:, 4]
            )

            element_scores, targets, masks = self.compute_element_scores_batch(
                image_features,
                dense_image_features,
                image_token_mask,
                element_names_batch,
                element_targets_batch,
                prompts,
            )

            for i, image_name in enumerate(image_names):
                results["image_rows"][image_name] = {
                    "image_name": image_name,
                    "image_path": os.path.join(self.args.image_root, image_name),
                    "prompt": prompts[i],
                    "mos": float(mos[i]),
                    "quality_pred": float(quality_preds[i].item()),
                }

                names = element_names_batch[i]
                for j, element_name in enumerate(names):
                    key = (image_name, element_name)
                    pred = float(element_scores[i, j].item()) if element_scores is not None else None
                    target = float(targets[i, j].item()) if targets is not None else float(element_targets_batch[i][j])
                    results["element_rows"][key] = {
                        "image_name": image_name,
                        "image_path": os.path.join(self.args.image_root, image_name),
                        "prompt": prompts[i],
                        "element_name": element_name,
                        "element_clean_name": parse_element(element_name)[0],
                        "target": target,
                        "pred": pred,
                        "pred_label": int(pred >= 0.5),
                        "target_label": int(target >= 0.5),
                        "correct": int((pred >= 0.5) == (target >= 0.5)),
                    }
        return results


def merge_case_rows(base_results, cmp_results):
    rows = []
    for key, base_row in base_results["element_rows"].items():
        if key not in cmp_results["element_rows"]:
            continue
        cmp_row = cmp_results["element_rows"][key]
        image_name = base_row["image_name"]
        base_image = base_results["image_rows"][image_name]
        cmp_image = cmp_results["image_rows"][image_name]
        target = base_row["target"]
        base_prob = base_row["pred"]
        cmp_prob = cmp_row["pred"]
        base_err = abs(base_prob - target)
        cmp_err = abs(cmp_prob - target)
        error_reduction = base_err - cmp_err
        base_correct = base_row["correct"]
        cmp_correct = cmp_row["correct"]

        if (not base_correct) and cmp_correct:
            case_bucket = "success"
        elif base_correct and (not cmp_correct):
            case_bucket = "regression"
        elif (not base_correct) and (not cmp_correct):
            case_bucket = "failure"
        else:
            case_bucket = "both_correct"

        rows.append(
            {
                "image_name": image_name,
                "image_path": base_row["image_path"],
                "prompt": base_row["prompt"],
                "mos": round(base_image["mos"], 4),
                "base_quality_pred": round(base_image["quality_pred"], 4),
                "compare_quality_pred": round(cmp_image["quality_pred"], 4),
                "element_name": base_row["element_name"],
                "element_clean_name": base_row["element_clean_name"],
                "target": int(target >= 0.5),
                "base_prob": round(base_prob, 4),
                "compare_prob": round(cmp_prob, 4),
                "base_correct": int(base_correct),
                "compare_correct": int(cmp_correct),
                "prob_gap": round(cmp_prob - base_prob, 4),
                "error_reduction": round(error_reduction, 4),
                "case_bucket": case_bucket,
            }
        )
    return rows


def select_top_rows(rows, bucket, top_k):
    bucket_rows = [r for r in rows if r["case_bucket"] == bucket]
    if bucket == "success":
        bucket_rows.sort(
            key=lambda r: (
                r["error_reduction"],
                abs(r["compare_prob"] - 0.5),
                abs(r["prob_gap"]),
            ),
            reverse=True,
        )
    elif bucket == "failure":
        bucket_rows.sort(
            key=lambda r: (
                max(abs(r["base_prob"] - 0.5), abs(r["compare_prob"] - 0.5)),
                abs(r["prob_gap"]),
            ),
            reverse=True,
        )
    elif bucket == "regression":
        bucket_rows.sort(
            key=lambda r: (
                -r["error_reduction"],
                abs(r["prob_gap"]),
            ),
            reverse=True,
        )
    else:
        bucket_rows.sort(key=lambda r: r["error_reduction"], reverse=True)
    return bucket_rows[:top_k]


def deduplicate_by_image(rows, limit):
    selected = []
    seen = set()
    for row in rows:
        if row["image_name"] in seen:
            continue
        seen.add(row["image_name"])
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, all_rows, success_rows, failure_rows, regression_rows, success_images, failure_images):
    lines = []
    lines.append("# EvalMuse case mining summary")
    lines.append("")
    lines.append(f"total element rows: {len(all_rows)}")
    lines.append(f"success rows (base wrong -> compare correct): {len([r for r in all_rows if r['case_bucket'] == 'success'])}")
    lines.append(f"failure rows (both wrong): {len([r for r in all_rows if r['case_bucket'] == 'failure'])}")
    lines.append(f"regression rows (base correct -> compare wrong): {len([r for r in all_rows if r['case_bucket'] == 'regression'])}")
    lines.append("")
    lines.append("## Recommended success cases (one element per image)")
    for row in success_images:
        lines.append(
            f"- {row['image_name']} | element={row['element_name']} | gt={row['target']} | "
            f"base={row['base_prob']} | compare={row['compare_prob']}"
        )
    lines.append("")
    lines.append("## Recommended failure cases (one element per image)")
    for row in failure_images:
        lines.append(
            f"- {row['image_name']} | element={row['element_name']} | gt={row['target']} | "
            f"base={row['base_prob']} | compare={row['compare_prob']}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    loader = set_dataset_evalmuse_fgclip(
        args.val_json,
        args.batch_size,
        args.image_root,
        args.num_workers,
        test=True,
    )

    base_cfg = EvalConfig(
        name="base",
        checkpoint_path=args.base_ckpt,
        use_quality_improvement=args.base_quality_impr,
        use_element_improvement=args.base_element_impr,
        use_element_only_queries=args.base_element_only_query,
        use_category_templates=args.base_category_templates,
    )
    cmp_cfg = EvalConfig(
        name="compare",
        checkpoint_path=args.compare_ckpt,
        use_quality_improvement=args.compare_quality_impr,
        use_element_improvement=args.compare_element_impr,
        use_element_only_queries=args.compare_element_only_query,
        use_category_templates=args.compare_category_templates,
    )

    base_results = CaseMiner(base_cfg, args).run(loader)
    cmp_results = CaseMiner(cmp_cfg, args).run(loader)

    all_rows = merge_case_rows(base_results, cmp_results)
    success_rows = select_top_rows(all_rows, "success", args.top_k)
    failure_rows = select_top_rows(all_rows, "failure", args.top_k)
    regression_rows = select_top_rows(all_rows, "regression", args.top_k)
    both_correct_rows = select_top_rows(all_rows, "both_correct", args.top_k)
    success_images = deduplicate_by_image(success_rows, min(args.top_k, 20))
    failure_images = deduplicate_by_image(failure_rows, min(args.top_k, 20))

    write_csv(os.path.join(args.output_dir, "all_element_case_rows.csv"), all_rows)
    write_csv(os.path.join(args.output_dir, "top_success_cases.csv"), success_rows)
    write_csv(os.path.join(args.output_dir, "top_failure_cases.csv"), failure_rows)
    write_csv(os.path.join(args.output_dir, "top_regression_cases.csv"), regression_rows)
    write_csv(os.path.join(args.output_dir, "top_both_correct_cases.csv"), both_correct_rows)
    write_csv(os.path.join(args.output_dir, "recommended_success_images.csv"), success_images)
    write_csv(os.path.join(args.output_dir, "recommended_failure_images.csv"), failure_images)
    write_summary(
        os.path.join(args.output_dir, "case_summary.md"),
        all_rows,
        success_rows,
        failure_rows,
        regression_rows,
        success_images,
        failure_images,
    )

    print("Saved case-mining files to:", args.output_dir)


if __name__ == "__main__":
    main()
