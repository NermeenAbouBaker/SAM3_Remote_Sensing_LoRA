#!/usr/bin/env python3
"""
SAM3 LoRA — Validation Script

Loads a trained LoRA checkpoint and evaluates on a COCO-format validation set.
Reports COCO mAP (segm) and cgF1 metrics.

Usage:
  # With LoRA weights
  python validate.py --config configs/lora_config.yaml \
                     --weights outputs/sam3_lora/best_lora_weights.pt \
                     --val_data_dir data/val

  # Baseline (original SAM3, no LoRA)
  python validate.py --val_data_dir data/val --base_model
"""

import argparse
import contextlib
import json
import os
import shutil
import tempfile
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pycocotools.mask as mask_utils

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import collate_fn_api
from sam3.train.masks_ops import rle_encode
from sam3.perflib.nms import nms_masks
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.eval.cgf1_eval import CGF1Evaluator

from lora_layers import (
    LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters
)
from train import COCOSegmentDataset, move_to_device  # reuse dataset from train.py


# ---------------------------------------------------------------------------
# NMS / prediction post-processing
# ---------------------------------------------------------------------------

def apply_nms(pred_logits, pred_masks, pred_boxes,
              score_thresh: float = 0.3,
              nms_iou: float = 0.7,
              max_dets: int = 100):
    """Filter predictions with SAM3's native NMS."""
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    scores = torch.sigmoid(pred_logits).squeeze(-1)
    binary = torch.sigmoid(pred_masks) > 0.5
    keep   = nms_masks(pred_probs=scores,
                       pred_masks=binary.float(),
                       prob_threshold=score_thresh,
                       iou_threshold=nms_iou)

    filt_masks  = torch.sigmoid(pred_masks)[keep]
    filt_scores = scores[keep]
    filt_boxes  = pred_boxes[keep]

    if max_dets > 0 and len(filt_scores) > max_dets:
        top_k = torch.topk(filt_scores, k=max_dets).indices
        filt_masks  = filt_masks[top_k]
        filt_scores = filt_scores[top_k]
        filt_boxes  = filt_boxes[top_k]

    return filt_masks, filt_scores, filt_boxes


def predictions_to_coco(predictions, image_ids,
                        resolution: int = 288,
                        score_thresh: float = 0.3,
                        nms_iou: float = 0.7,
                        max_dets: int = 100):
    """Convert raw model outputs to COCO-format prediction list."""
    coco_preds = []
    pred_id    = 0

    for img_id, preds in tqdm(zip(image_ids, predictions),
                              total=len(predictions),
                              desc="  Converting predictions"):
        if preds is None or not len(preds.get("pred_logits", [])):
            continue

        filt_masks, filt_scores, filt_boxes = apply_nms(
            preds["pred_logits"], preds["pred_masks"], preds["pred_boxes"],
            score_thresh=score_thresh, nms_iou=nms_iou, max_dets=max_dets,
        )
        if not len(filt_masks):
            continue

        binary = (filt_masks > 0.5).cpu()
        rles   = rle_encode(binary)

        for rle, score, box in zip(rles,
                                   filt_scores.cpu().tolist(),
                                   filt_boxes.cpu().tolist()):
            cx, cy, w, h = box
            coco_preds.append({
                "image_id":    int(img_id),
                "category_id": 1,
                "segmentation": rle,
                "bbox": [(cx - w / 2) * resolution,
                         (cy - h / 2) * resolution,
                         w * resolution, h * resolution],
                "score": float(score),
                "id":    pred_id,
            })
            pred_id += 1

    return coco_preds


def build_coco_gt(dataset, image_ids=None, mask_resolution: int = 288):
    """Build a minimal COCO GT dict from a COCOSegmentDataset."""
    print(f"  Building GT (mask resolution {mask_resolution}×{mask_resolution}) …")
    coco_gt = {
        "info": {"description": "SAM3 LoRA Val GT"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    ann_id  = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in tqdm(list(indices), desc="  Building GT"):
        coco_gt["images"].append(
            {"id": int(idx), "width": mask_resolution, "height": mask_resolution}
        )
        dp = dataset[idx]
        for obj in dp.images[0].objects:
            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            ann = {
                "id": ann_id, "image_id": int(idx), "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
            if obj.segment is not None:
                m = obj.segment.unsqueeze(0).unsqueeze(0).float()
                m = torch.nn.functional.interpolate(
                    m, size=(mask_resolution, mask_resolution),
                    mode="bilinear", align_corners=False) > 0.5
                mask_np = m.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle["counts"] = rle["counts"].decode("utf-8")
                ann["segmentation"] = rle
            coco_gt["annotations"].append(ann)
            ann_id += 1

    return coco_gt


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate(config_path, weights_path, val_data_dir,
             num_samples=None, score_thresh=0.3, nms_iou=0.7,
             base_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- model ----
    print("Building SAM3 model …")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    )

    if not base_model:
        if not config_path or not weights_path:
            raise ValueError("--config and --weights are required (or use --base_model)")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        lora_c = cfg["lora"]
        lora_config = LoRAConfig(
            rank=lora_c["rank"], alpha=lora_c["alpha"],
            dropout=lora_c.get("dropout", 0.0),
            target_modules=lora_c["target_modules"],
            apply_to_vision_encoder=lora_c.get("apply_to_vision_encoder", True),
            apply_to_text_encoder=lora_c.get("apply_to_text_encoder", True),
            apply_to_geometry_encoder=lora_c.get("apply_to_geometry_encoder", False),
            apply_to_detr_encoder=lora_c.get("apply_to_detr_encoder", True),
            apply_to_detr_decoder=lora_c.get("apply_to_detr_decoder", True),
            apply_to_mask_decoder=lora_c.get("apply_to_mask_decoder", False),
        )
        model = apply_lora_to_model(model, lora_config)
        print(f"Loading weights: {weights_path}")
        load_lora_weights(model, weights_path)
        stats = count_parameters(model)
        print(f"Trainable: {stats['trainable_parameters']:,} "
              f"({stats['trainable_percentage']:.2f}%)")
    else:
        print("Using base SAM3 (no LoRA)")

    model.to(device).eval()

    # ---- dataset ----
    # val_data_dir may point directly to a split directory or to the data root.
    # Try the path as-is first; if _annotations.coco.json isn't there, add "val".
    val_dir = Path(val_data_dir)
    if not (val_dir / "_annotations.coco.json").exists():
        # try data_dir/val or data_dir/valid
        for sub in ("val", "valid"):
            if (val_dir / sub / "_annotations.coco.json").exists():
                val_dir = val_dir / sub
                break

    print(f"\nLoading validation data from {val_dir} …")
    # Reuse COCOSegmentDataset but with split=".": point split_dir to val_dir directly
    dataset = _DirectDataset(val_dir)
    if num_samples:
        dataset.samples = dataset.samples[:num_samples]
    print(f"  {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_fn_api(b, dict_key="input", with_seg_masks=True),
        num_workers=2, pin_memory=True,
    )

    # ---- inference ----
    print("\nRunning inference …")
    all_preds = []
    all_ids   = []

    with torch.no_grad():
        for i, batch_dict in enumerate(tqdm(loader, desc="  Inference")):
            input_batch = move_to_device(batch_dict["input"], device)
            try:
                outputs_list = model(input_batch)
            except Exception as e:
                print(f"  [skip] {e}")
                all_preds.append(None)
                all_ids.append(i)
                continue

            out = outputs_list[-1]
            all_preds.append({
                "pred_logits": out["pred_logits"][0].detach().cpu(),
                "pred_boxes":  out["pred_boxes"][0].detach().cpu(),
                "pred_masks":  out["pred_masks"][0].detach().cpu(),
            })
            all_ids.append(i)

    print(f"\nCollected {len(all_preds)} predictions")

    # ---- convert ----
    coco_preds = predictions_to_coco(
        all_preds, all_ids, resolution=288,
        score_thresh=score_thresh, nms_iou=nms_iou,
    )
    coco_gt_dict = build_coco_gt(dataset, image_ids=all_ids, mask_resolution=288)

    if not coco_preds:
        print("\n[ERROR] No predictions passed the score threshold — cannot compute metrics.")
        print("Try lowering --score_thresh (current: {score_thresh}).")
        return

    # ---- evaluate ----
    tmp = tempfile.mkdtemp(prefix="sam3_eval_")
    gt_file   = os.path.join(tmp, "gt.json")
    pred_file = os.path.join(tmp, "pred.json")
    with open(gt_file,   "w") as f: json.dump(coco_gt_dict, f)
    with open(pred_file, "w") as f: json.dump(coco_preds,   f)

    try:
        print("\n" + "="*60)
        print("COCO mAP (segm)")
        print("="*60)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt   = COCO(gt_file)
                coco_dt   = coco_gt.loadRes(pred_file)
                coco_eval = COCOeval(coco_gt, coco_dt, "segm")
                coco_eval.params.useCats = False
                coco_eval.evaluate()
                coco_eval.accumulate()
        coco_eval.summarize()
        map_all, map50, map75 = coco_eval.stats[:3]

        print("\n" + "="*60)
        print("cgF1")
        print("="*60)
        cgf1_eval    = CGF1Evaluator(gt_path=gt_file, iou_type="segm", verbose=True)
        cgf1_results = cgf1_eval.evaluate(pred_file)
        cgf1    = cgf1_results.get("cgF1_eval_segm_cgF1",       0.0)
        cgf1_50 = cgf1_results.get("cgF1_eval_segm_cgF1@0.5",   0.0)
        cgf1_75 = cgf1_results.get("cgF1_eval_segm_cgF1@0.75",  0.0)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"  mAP  (IoU 0.50:0.95) : {map_all:.4f}")
        print(f"  mAP@50               : {map50:.4f}")
        print(f"  mAP@75               : {map75:.4f}")
        print(f"  cgF1 (IoU 0.50:0.95) : {cgf1:.4f}")
        print(f"  cgF1@50              : {cgf1_50:.4f}")
        print(f"  cgF1@75              : {cgf1_75:.4f}")
        print("="*60)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helper: dataset that points directly at a split directory (no sub-folder)
# ---------------------------------------------------------------------------

class _DirectDataset(COCOSegmentDataset):
    """Variant that treats the given directory itself as the split directory."""

    def __init__(self, data_dir: Path, image_resolution: int = 1008):
        import json
        from collections import defaultdict

        self.data_dir  = data_dir
        self.split_dir = data_dir
        self.split     = "."
        self.resolution = image_resolution

        for candidate in [
            data_dir / "_annotations.coco.json",
            data_dir / "Annotations" / "_annotations.coco.json",
        ]:
            if candidate.exists():
                ann_file = candidate
                break
        else:
            raise FileNotFoundError(f"No COCO annotation file found in {data_dir}")

        with open(ann_file) as f:
            coco = json.load(f)

        self.images     = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        self.img_to_anns = {}
        for ann in coco["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_id in sorted(self.images):
            cats = {a["category_id"] for a in self.img_to_anns.get(img_id, [])
                    if a.get("bbox")}
            for cat_id in sorted(cats):
                self.samples.append((img_id, cat_id))

        from torchvision.transforms import v2
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f"  {len(self.images)} images · {len(self.samples)} samples · "
              f"categories: {self.categories}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM3 LoRA validation — reports COCO mAP and cgF1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py --config configs/lora_config.yaml \\
                     --weights outputs/sam3_lora/best_lora_weights.pt \\
                     --val_data_dir data/val

  python validate.py --val_data_dir data/val --base_model
        """,
    )
    parser.add_argument("--config",       default=None,
                        help="Path to training config (required unless --base_model)")
    parser.add_argument("--weights",      default=None,
                        help="Path to LoRA weights file (required unless --base_model)")
    parser.add_argument("--val_data_dir", required=True,
                        help="Path to validation directory containing _annotations.coco.json")
    parser.add_argument("--base_model",   action="store_true",
                        help="Evaluate base SAM3 without LoRA (baseline comparison)")
    parser.add_argument("--num_samples",  type=int, default=None,
                        help="Limit evaluation to N samples (debug)")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="Score threshold for predictions (default: 0.3)")
    parser.add_argument("--nms_iou",      type=float, default=0.7,
                        help="NMS IoU threshold (default: 0.7)")
    args = parser.parse_args()

    if not args.base_model and (not args.config or not args.weights):
        parser.error("--config and --weights are required when not using --base_model")

    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        score_thresh=args.score_thresh,
        nms_iou=args.nms_iou,
        base_model=args.base_model,
    )
