> **Accepted at [ICANN 2026](https://e-nns.org/icann2026/)** — The 35th International Conference on Artificial Neural Networks, Padua, Italy · European Neural Network Society (ENNS)

# SAM3 LoRA Fine-tuning

Fine-tune [SAM3](https://github.com/facebookresearch/sam3) on custom COCO-format segmentation datasets using **Low-Rank Adaptation (LoRA)**.  
Keeps the vast majority of SAM3's weights frozen and trains only a small set of low-rank adapter matrices, making fine-tuning feasible on a single GPU.

---

## Repository layout

```
sam3_lora/
├── lora_layers.py        # LoRA implementation (LoRALayer, LoRALinear, apply_lora_to_model, …)
├── train.py              # Training script (single-GPU + multi-GPU DDP)
├── validate.py           # Validation script (COCO mAP + cgF1)
├── configs/
│   └── lora_config.yaml  # All hyper-parameters in one place
└── setup.py
```

---

## Requirements

**SAM3** must be installed and importable (follow [SAM3's installation guide](https://github.com/facebookresearch/sam3)).  
SAM3 weights are downloaded automatically from HuggingFace on first run.

Python dependencies:

```bash
pip install torch torchvision Pillow numpy PyYAML tqdm pycocotools scikit-image
```

Or install this package in editable mode (installs the above automatically):

```bash
pip install -e .
```

---

## Dataset format

Annotations must follow the **COCO JSON format**.  
The default directory layout (configurable in `configs/lora_config.yaml`):

```
data/
├── train/
│   ├── _annotations.coco.json
│   └── images/
│       ├── img001.jpg
│       └── ...
└── val/                        ← optional; skip for train-only
    ├── _annotations.coco.json
    └── images/
```

The script indexes samples as **(image\_id, category\_id)** pairs so the text query always matches the target category exactly — this avoids contradictory supervision when multiple classes appear in the same image.

> **Alternative layouts** are supported automatically:
> - `Annotations/_annotations.coco.json` (instead of `_annotations.coco.json`)
> - Images in `Instance_masks/images/` (instead of `images/`)

---

## Training

### Single GPU

```bash
python train.py --config configs/lora_config.yaml
```

### Specific GPU

```bash
python train.py --config configs/lora_config.yaml --device 1
```

### Multi-GPU (DDP)

```bash
python train.py --config configs/lora_config.yaml --device 0 1 2 3
```

The script re-launches itself via `torchrun` automatically — no need to call `torchrun` manually.

### Outputs

```
outputs/sam3_lora/
├── best_lora_weights.pt    ← best val IoU checkpoint
├── last_lora_weights.pt    ← final epoch checkpoint
├── training_log.jsonl      ← per-epoch metrics (loss, IoU, LR)
└── training_results.json   ← summary JSON
```

---

## Validation

```bash
# With LoRA weights
python validate.py \
  --config  configs/lora_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --val_data_dir data/val

# Baseline — original SAM3, no LoRA
python validate.py --val_data_dir data/val --base_model
```

Reports **COCO mAP** (IoU 0.50:0.95, mAP@50, mAP@75) and **cgF1** (same thresholds).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--score_thresh` | `0.3` | Minimum confidence to keep a prediction |
| `--nms_iou` | `0.7` | NMS IoU threshold |
| `--num_samples` | — | Limit to N samples (debug) |
| `--base_model` | — | Skip LoRA, evaluate original SAM3 |

---

## Configuration

All hyper-parameters live in `configs/lora_config.yaml`.  
Copy and rename for each experiment.

### Key fields

```yaml
lora:
  rank: 16                  # Low-rank dimension; try 8, 16, 32
  alpha: 32                 # Scaling factor (typically 2 × rank)
  dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, out_proj, fc1, fc2]

  # Enable/disable LoRA per SAM3 component
  apply_to_vision_encoder:   true
  apply_to_text_encoder:     true
  apply_to_geometry_encoder: false
  apply_to_detr_encoder:     true
  apply_to_detr_decoder:     true
  apply_to_mask_decoder:     false

training:
  data_dir: ./data
  batch_size: 4
  num_epochs: 30
  learning_rate: 5.0e-5
  gradient_accumulation_steps: 1
  early_stopping_patience: 5

output:
  output_dir: outputs/sam3_lora
```

### Adapting to a different dataset

1. Point `training.data_dir` to your data root.
2. Adjust `training.batch_size` to fit your GPU memory.
3. Set `training.num_epochs` (30 is a good starting point for small datasets).
4. If your dataset is very dense (many objects per image), lower `training.max_anns_per_image` to avoid out-of-memory errors in the focal loss.

---

## LoRA design

`lora_layers.py` provides:

| Class / function | Description |
|---|---|
| `LoRALayer` | Low-rank matrices A (in→r) and B (r→out) with Kaiming/zero init |
| `LoRALinear` | Wraps a frozen `nn.Linear` + adds a `LoRALayer` |
| `MultiheadAttentionLoRA` | Replaces `nn.MultiheadAttention` with explicit Q/K/V/out projections so LoRA can be applied to each |
| `LoRAConfig` | Dataclass holding rank, alpha, dropout, target modules, and per-component flags |
| `apply_lora_to_model` | Freezes all base weights, replaces MHA modules, wraps matching `nn.Linear` layers |
| `save_lora_weights` | Saves only the LoRA matrices (not the full model) |
| `load_lora_weights` | Loads LoRA matrices into a model that already has LoRA injected |
| `count_parameters` | Returns total / trainable parameter counts |

### Why `MultiheadAttentionLoRA`?

PyTorch's `nn.MultiheadAttention` uses a single fused `in_proj_weight` for Q, K, V. LoRA cannot wrap this cleanly. The replacement module splits Q, K, V into separate `nn.Linear` layers, making them individually wrappable.

---

## LR schedule

Training uses a three-phase schedule:

```
Warmup (10%) → Flat peak (40%) → Cosine decay (50%)
```

This lets the model learn aggressively for the first half before fine-tuning.

---

## Multi-GPU notes

- DDP is handled transparently: pass multiple GPU indices to `--device`.
- All checkpoints and logs are written by rank-0 only.
- `DistributedSampler` is set automatically; no manual changes needed.
- IoU is aggregated across ranks via `all_reduce`.

---

## Tips

| Situation | Recommendation |
|---|---|
| GPU OOM during training | Lower `batch_size` and/or `max_anns_per_image`; increase `gradient_accumulation_steps` to keep effective batch size |
| Val IoU stuck near 0 | Lower `metrics.score_threshold` to `0.05`; check that category names in the COCO file match natural-language descriptions |
| Want fewer trainable params | Set `apply_to_vision_encoder: false` and/or lower `rank` to 4–8 |
| Single-class dataset | The (image, category) sampling still works; you will have one sample per image |
| No validation split | Training runs without IoU eval; `last_lora_weights.pt` is copied to `best_lora_weights.pt` at the end |
| Resume training | Load `last_lora_weights.pt` with `load_lora_weights(model, path)` before calling `.train()` |

---

## License

This project is released under the Apache 2.0 License.  
SAM3 is released under its own license — see the [SAM3 repository](https://github.com/facebookresearch/sam3).
