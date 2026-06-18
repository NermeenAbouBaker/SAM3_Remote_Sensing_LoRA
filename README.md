# SAM3_Remote_Sensing_LoRA
SAM3 fine-tuning via LoRA (Low-Rank Adaptation) for instance segmentation on custom COCO-format datasets. Freezes SAM3's base weights and trains onlyLoRA as lightweight adapter matrices across vision, text, and DETR components. Supports single- and multi-GPU training, COCO mAP and cgF1 evaluation. 
