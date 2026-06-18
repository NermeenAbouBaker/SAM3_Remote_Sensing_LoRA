#!/usr/bin/env python3
"""
SAM3 LoRA Fine-tuning — Main Training Script

Trains SAM3 with Low-Rank Adaptation (LoRA) on COCO-format segmentation data.

Features:
  - Single-GPU and multi-GPU (DDP) training
  - Per-(image, category) sampling — text query matches targets exactly
  - Warmup → flat → cosine LR schedule
  - Gradient accumulation + gradient clipping
  - Vectorized IoU evaluation with early stopping
  - Saves best (by val IoU) and last checkpoint

Usage:
  Single GPU:
    python train.py --config configs/lora_config.yaml

  Single GPU (specific device):
    python train.py --config configs/lora_config.yaml --device 1

  Multi-GPU (DDP):
    python train.py --config configs/lora_config.yaml --device 0 1 2 3
"""

import os
import argparse
import json
import shutil
import subprocess
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, LinearLR, SequentialLR
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image as PILImage
from torchvision.transforms import v2
import pycocotools.mask as mask_utils

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata,
)

from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAM3_MASK_RES       = 288   # SAM3 native mask decoder output resolution
MAX_OBJECTS_PER_SAMPLE = 50  # Cap to prevent Triton int32 overflow on dense images


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCOSegmentDataset(Dataset):
    """
    Each sample is an (image_id, category_id) pair so the text query
    matches targets exactly — avoids contradictory supervision when
    multiple categories coexist in the same image.

    Objects per sample are capped at MAX_OBJECTS_PER_SAMPLE to avoid
    Triton kernel int32 overflow in the focal loss on very dense images.

    Expected directory layout (configurable via data_dir / split):
        <data_dir>/
          train/
            _annotations.coco.json        ← or Annotations/_annotations.coco.json
            images/
              *.jpg / *.png
          val/   (optional)
            _annotations.coco.json
            images/
    """

    def __init__(self, data_dir: str, split: str = "train",
                 image_resolution: int = 1008):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.resolution = image_resolution

        # Locate annotation file — support two common layouts
        for candidate in [
            self.split_dir / "_annotations.coco.json",
            self.split_dir / "Annotations" / "_annotations.coco.json",
        ]:
            if candidate.exists():
                ann_file = candidate
                break
        else:
            raise FileNotFoundError(
                f"COCO annotation file not found under {self.split_dir}"
            )

        with open(ann_file) as f:
            coco = json.load(f)

        self.images     = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        self.img_to_anns: dict = {}
        for ann in coco["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Build (image_id, category_id) index
        self.samples = []
        for img_id in sorted(self.images):
            cats = {a["category_id"] for a in self.img_to_anns.get(img_id, [])
                    if a.get("bbox")}
            for cat_id in sorted(cats):
                self.samples.append((img_id, cat_id))

        n_capped = sum(
            1 for img_id, cat_id in self.samples
            if len([a for a in self.img_to_anns.get(img_id, [])
                    if a.get("category_id") == cat_id and a.get("bbox")])
               > MAX_OBJECTS_PER_SAMPLE
        )

        print(f"  [{split}] {len(self.images)} images · "
              f"{len(self.samples)} (image, category) samples · "
              f"categories: {self.categories}")
        if n_capped:
            print(f"  [{split}] {n_capped} samples will be capped "
                  f"at {MAX_OBJECTS_PER_SAMPLE} objects")

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def _find_image_path(self, file_name: str) -> Path:
        pure = os.path.basename(file_name)
        for candidate in [
            self.split_dir / "images" / pure,
            self.split_dir / "Instance_masks" / "images" / pure,
            self.split_dir / file_name,
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image not found: {pure} under {self.split_dir}")

    def __getitem__(self, idx):
        img_id, target_cat_id = self.samples[idx]
        img_info  = self.images[img_id]
        img_path  = self._find_image_path(img_info["file_name"])

        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size
        pil_resized = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        image_tensor = self.transform(pil_resized)

        # Filter to target category only; cap if needed
        annotations = [
            a for a in self.img_to_anns.get(img_id, [])
            if a.get("category_id") == target_cat_id and a.get("bbox")
        ]
        if len(annotations) > MAX_OBJECTS_PER_SAMPLE:
            rng = np.random.RandomState(idx)
            keep = sorted(rng.choice(len(annotations), MAX_OBJECTS_PER_SAMPLE, replace=False))
            annotations = [annotations[i] for i in keep]

        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h
        objects = []

        for i, ann in enumerate(annotations):
            x, y, w, h = ann["bbox"]
            box = torch.tensor(
                [x * scale_w, y * scale_h,
                 (x + w) * scale_w, (y + h) * scale_h],
                dtype=torch.float32,
            ) / self.resolution  # normalise to [0, 1]

            segment = None
            seg = ann.get("segmentation")
            if seg:
                try:
                    if isinstance(seg, dict):
                        mask_np = mask_utils.decode(seg)
                    elif isinstance(seg, list):
                        rles = mask_utils.frPyObjects(seg, orig_h, orig_w)
                        mask_np = mask_utils.decode(mask_utils.merge(rles))
                    else:
                        mask_np = None

                    if mask_np is not None:
                        mask_t = (torch.from_numpy(mask_np)
                                  .float().unsqueeze(0).unsqueeze(0))
                        mask_t = torch.nn.functional.interpolate(
                            mask_t, size=(SAM3_MASK_RES, SAM3_MASK_RES),
                            mode="nearest")
                        segment = mask_t.squeeze() > 0.5
                except Exception as e:
                    print(f"  [warn] mask error img {img_id} ann {i}: {e}")

            objects.append(Object(
                bbox=box,
                area=(box[2] - box[0]) * (box[3] - box[1]),
                object_id=i,
                segment=segment,
            ))

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution),
        )
        query_text = self.categories.get(target_cat_id, "object").lower()
        query = FindQueryLoaded(
            query_text=query_text,
            image_id=0,
            object_ids_output=[o.object_id for o in objects],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=img_id,
                original_image_id=img_id,
                original_category_id=target_cat_id,
                original_size=(orig_h, orig_w),
                object_id=-1,
                frame_index=-1,
            ),
        )
        return Datapoint(
            find_queries=[query],
            images=[image_obj],
            raw_images=[pil_resized],
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_iou(model, val_loader, device, unwrapped_model,
                 score_thresh: float = 0.15, max_preds: int = 100):
    """
    Compute mean IoU over the validation set.

    Fully vectorized: builds IoU matrix on GPU, then does greedy
    best-match assignment (one GT → one pred).
    """
    model.eval()
    all_ious = []

    for batch_dict in tqdm(val_loader, desc="  Val IoU",
                           disable=not is_main_process()):
        input_batch = move_to_device(batch_dict["input"], device)
        try:
            outputs_list = model(input_batch)
        except (RuntimeError, OverflowError) as e:
            print_rank0(f"  [eval skip] {e}")
            continue

        for b in range(len(input_batch.find_targets)):
            gt_dict  = unwrapped_model.back_convert(input_batch.find_targets[b])
            raw_gt   = gt_dict["masks"].to(device)
            valid    = torch.where(raw_gt.flatten(1).sum(1) > 0)[0]
            if not len(valid):
                continue
            gt_masks = raw_gt[valid] > 0.5      # [N_gt, H, W]
            n_gt     = gt_masks.shape[0]

            out     = outputs_list[-1]
            scores  = torch.sigmoid(out["pred_logits"][b]).squeeze(-1)
            keep    = scores > score_thresh
            if not keep.any():
                all_ious.extend([0.0] * n_gt)
                continue

            pred_masks  = torch.sigmoid(out["pred_masks"][b][keep]) > 0.5
            pred_scores = scores[keep]
            order       = torch.argsort(pred_scores, descending=True)[:max_preds]
            pred_masks  = pred_masks[order]     # [N_pred, H, W]
            n_pred      = pred_masks.shape[0]

            # Vectorised IoU matrix [N_gt, N_pred]
            gt_flat  = gt_masks.flatten(1).float()
            pr_flat  = pred_masks.flatten(1).float()
            inter    = gt_flat @ pr_flat.t()
            union    = (gt_flat.sum(1, keepdim=True)
                        + pr_flat.sum(1, keepdim=True).t() - inter)
            iou_mat  = inter / (union + 1e-6)

            # Greedy matching
            used = torch.zeros(n_pred, dtype=torch.bool, device=device)
            for g in range(n_gt):
                row = iou_mat[g].clone()
                row[used] = -1.0
                best_j = row.argmax().item()
                best_v = row[best_j].item()
                if best_v > 0:
                    used[best_j] = True
                all_ious.append(max(best_v, 0.0))

    # Aggregate across ranks
    if dist.is_initialized():
        s = torch.tensor([sum(all_ious)], device=device)
        n = torch.tensor([len(all_ious)], device=device, dtype=torch.float)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        mean_iou = (s / (n + 1e-8)).item()
        total    = int(n.item())
    else:
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        total    = len(all_ious)

    model.train()
    return mean_iou, total


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def move_to_device(obj, device):
    """Recursively move tensors and SAM3 dataclass objects to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
        return obj
    return obj


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SAM3LoRATrainer:
    """
    Wraps model building, LoRA injection, loss setup, and the training loop.
    Instantiate once and call .train().
    """

    def __init__(self, config_path: str, multi_gpu: bool = False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.multi_gpu  = multi_gpu
        self.local_rank = 0

        if multi_gpu:
            self.local_rank = setup_distributed()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"DDP enabled — {get_world_size()} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- model ----
        print_rank0("Building SAM3 model …")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False,
        )

        # ---- LoRA ----
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.0),
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg.get("apply_to_vision_encoder", True),
            apply_to_text_encoder=lora_cfg.get("apply_to_text_encoder", True),
            apply_to_geometry_encoder=lora_cfg.get("apply_to_geometry_encoder", False),
            apply_to_detr_encoder=lora_cfg.get("apply_to_detr_encoder", True),
            apply_to_detr_decoder=lora_cfg.get("apply_to_detr_decoder", True),
            apply_to_mask_decoder=lora_cfg.get("apply_to_mask_decoder", False),
        )
        self.model = apply_lora_to_model(self.model, lora_config)
        stats = count_parameters(self.model)
        print_rank0(
            f"Trainable: {stats['trainable_parameters']:,} "
            f"({stats['trainable_percentage']:.2f}%)"
        )

        self.model.to(self.device)
        if multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        self._unwrapped = self.model.module if multi_gpu else self.model

        # ---- optimizer ----
        train_cfg = self.config["training"]
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            betas=(train_cfg.get("adam_beta1", 0.9),
                   train_cfg.get("adam_beta2", 0.999)),
            eps=float(train_cfg.get("adam_epsilon", 1e-8)),
        )
        self.max_grad_norm    = float(train_cfg.get("max_grad_norm", 1.0))
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))

        # ---- matcher & loss ----
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )
        loss_fns = [
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                pos_weight=2.0,
                weight_dict={"loss_ce": 2.0, "presence_loss": 2.0},
                pos_focal=False, alpha=0.25, gamma=2,
                use_presence=True, pad_n_queries=200,
            ),
            Masks(
                weight_dict={"loss_mask": 5.0, "loss_dice": 5.0},
                focal_alpha=0.25, focal_gamma=2.0, compute_aux=False,
            ),
        ]
        o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=1.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",
            normalize_by_valid_object_num=False,
        )

    # ------------------------------------------------------------------
    def train(self):
        cfg        = self.config["training"]
        data_dir   = cfg["data_dir"]
        epochs     = cfg["num_epochs"]
        batch_size = cfg["batch_size"]
        n_workers  = cfg.get("num_workers", 2)
        img_res    = self.config.get("image_resolution", 1008)
        patience   = cfg.get("early_stopping_patience", 5)

        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "training_log.jsonl"
        if is_main_process() and log_path.exists():
            log_path.unlink()

        # ---- datasets ----
        print_rank0(f"\nLoading data from {data_dir} …")
        train_ds = COCOSegmentDataset(data_dir, "train", img_res)
        val_ds   = None
        for split_name in ("val", "valid"):
            try:
                val_ds = COCOSegmentDataset(data_dir, split_name, img_res)
                if len(val_ds) == 0:
                    val_ds = None
                break
            except FileNotFoundError:
                pass
        if val_ds is None:
            print_rank0("No validation split found — will skip IoU evaluation")

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.multi_gpu else None
        val_sampler   = (DistributedSampler(val_ds, shuffle=False)
                         if self.multi_gpu and val_ds else None)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            collate_fn=collate_fn, num_workers=n_workers, pin_memory=True,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=1, shuffle=False, sampler=val_sampler,
                       collate_fn=collate_fn, num_workers=n_workers, pin_memory=True)
            if val_ds else None
        )

        # ---- LR schedule: warmup → flat → cosine ----
        warmup_ep = max(1, epochs // 10)
        flat_ep   = max(1, epochs // 2 - warmup_ep)
        decay_ep  = max(1, epochs - warmup_ep - flat_ep)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_ep),
                LinearLR(self.optimizer, start_factor=1.0, end_factor=1.0,
                         total_iters=flat_ep),
                CosineAnnealingWarmRestarts(self.optimizer, T_0=decay_ep, eta_min=1e-6),
            ],
            milestones=[warmup_ep, warmup_ep + flat_ep],
        )
        print_rank0(
            f"  LR schedule: warmup({warmup_ep}) → flat({flat_ep}) → cosine({decay_ep})"
        )

        best_iou   = -1.0
        best_epoch = 0
        history    = []

        print_rank0(f"\n{'='*70}")
        print_rank0(f"Training: {epochs} epochs")
        print_rank0(f"  Train samples   : {len(train_ds)}")
        print_rank0(f"  Val samples     : {len(val_ds) if val_ds else 0}")
        print_rank0(f"  Batch size      : {batch_size} × {get_world_size()} GPUs")
        print_rank0(f"  Grad accum      : {self.grad_accum_steps}")
        print_rank0(f"  Effective batch : {batch_size * self.grad_accum_steps * get_world_size()}")
        print_rank0(f"  LR              : {cfg['learning_rate']}")
        print_rank0(f"  Grad clip       : {self.max_grad_norm}")
        print_rank0(f"  Image res       : {img_res}   Mask res: {SAM3_MASK_RES}")
        print_rank0(f"{'='*70}\n")

        self.model.train()

        for epoch in range(1, epochs + 1):
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_losses  = []
            skipped       = 0
            self.optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                        disable=not is_main_process())

            for step, batch_dict in enumerate(pbar, 1):
                input_batch = move_to_device(batch_dict["input"], self.device)

                try:
                    outputs_list = self.model(input_batch)
                except (RuntimeError, OverflowError) as e:
                    print_rank0(f"  [skip fwd] {e}")
                    self.optimizer.zero_grad(); skipped += 1; continue

                find_targets = [
                    self._unwrapped.back_convert(t)
                    for t in input_batch.find_targets
                ]
                for tgt in find_targets:
                    for k, v in tgt.items():
                        if isinstance(v, torch.Tensor):
                            tgt[k] = v.to(self.device)

                with SAM3Output.iteration_mode(
                    outputs_list,
                    iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        tgt_list = [stage_targets] * len(stage_outputs)
                        for out, tgt in zip(stage_outputs, tgt_list):
                            out["indices"] = self.matcher(out, tgt)
                            for aux in out.get("aux_outputs", []):
                                aux["indices"] = self.matcher(aux, tgt)

                try:
                    loss_dict = self.loss_wrapper(outputs_list, find_targets)
                    loss = loss_dict[CORE_LOSS_KEY] / self.grad_accum_steps
                    loss.backward()
                except (OverflowError, RuntimeError) as e:
                    print_rank0(f"  [skip loss] {e}")
                    self.optimizer.zero_grad(); skipped += 1; continue

                if step % self.grad_accum_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_losses.append(loss.item() * self.grad_accum_steps)
                pbar.set_postfix(loss=f"{train_losses[-1]:.4f}")

            self.scheduler.step()
            avg_loss   = sum(train_losses) / len(train_losses) if train_losses else 0.0
            current_lr = self.optimizer.param_groups[0]["lr"]

            val_iou, val_objs = 0.0, 0
            if val_loader is not None:
                val_iou, val_objs = evaluate_iou(
                    self.model, val_loader, self.device, self._unwrapped,
                    score_thresh=self.config.get("metrics", {}).get("score_threshold", 0.15),
                    max_preds=100,
                )

            skip_msg   = f"  skipped={skipped}" if skipped else ""
            epoch_info = {
                "epoch": epoch,
                "train_loss": round(avg_loss, 6),
                "val_mean_iou": round(val_iou, 4),
                "val_objects": val_objs,
                "lr": current_lr,
                "skipped_batches": skipped,
            }
            history.append(epoch_info)
            print_rank0(
                f"\n  Epoch {epoch}/{epochs} | "
                f"Loss: {avg_loss:.5f} | "
                f"Val IoU: {val_iou:.4f} ({val_objs} objs) | "
                f"LR: {current_lr:.2e}{skip_msg}"
            )

            if is_main_process():
                model_save = self.model.module if self.multi_gpu else self.model
                save_lora_weights(model_save, str(out_dir / "last_lora_weights.pt"))
                if val_iou > best_iou:
                    best_iou, best_epoch = val_iou, epoch
                    save_lora_weights(model_save, str(out_dir / "best_lora_weights.pt"))
                    print_rank0(f"  >>> New best model (IoU: {val_iou:.4f})")
                with open(log_path, "a") as f:
                    f.write(json.dumps(epoch_info) + "\n")

            torch.cuda.empty_cache()

            if val_loader and epoch - best_epoch >= patience:
                print_rank0(
                    f"\n  Early stopping — no improvement for {patience} epochs "
                    f"(best IoU {best_iou:.4f} at epoch {best_epoch})"
                )
                break

        if self.multi_gpu:
            dist.barrier()

        if is_main_process():
            if val_ds is None:
                last = out_dir / "last_lora_weights.pt"
                if last.exists():
                    shutil.copy(last, out_dir / "best_lora_weights.pt")

            summary = {
                "best_val_iou": round(best_iou, 4),
                "epochs_trained": epoch,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds) if val_ds else 0,
                "history": history,
                "models": {
                    "best": str(out_dir / "best_lora_weights.pt"),
                    "last": str(out_dir / "last_lora_weights.pt"),
                },
            }
            with open(out_dir / "training_results.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n{'='*70}")
            print(f"  Training complete!")
            print(f"  Best Val IoU : {best_iou:.4f}")
            print(f"  Outputs      : {out_dir}")
            print(f"{'='*70}")

        if self.multi_gpu:
            cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _launch_distributed(args):
    """Re-launch via torchrun for multi-GPU DDP."""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={len(args.device)}",
        "--master_port", str(args.master_port),
        sys.argv[0],
        "--config", args.config,
        "--device", *map(str, args.device),
        "--_ddp",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.device))
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM3 LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config configs/lora_config.yaml
  python train.py --config configs/lora_config.yaml --device 1
  python train.py --config configs/lora_config.yaml --device 0 1 2 3
        """,
    )
    parser.add_argument("--config",      default="configs/lora_config.yaml")
    parser.add_argument("--device",      type=int, nargs="+", default=[0])
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--_ddp",        action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    multi = len(args.device) > 1
    is_sub = args._ddp or "LOCAL_RANK" in os.environ

    if multi and not is_sub:
        _launch_distributed(args)
    else:
        if not multi:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
        SAM3LoRATrainer(args.config, multi_gpu=(multi and is_sub)).train()
