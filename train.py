"""
train.py
--------
Main entry point.

Usage:
    python train.py
    python train.py --config configs/config.yaml
"""

import argparse
import random
import yaml
import numpy as np
import torch

from src.data.dataset import get_dataset
from src.models.model import build_model, print_param_summary
from src.training.trainer import train
from src.evaluation.metrics import compute_perplexity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"  Medical Report Simplification")
    print(f"  Model  : {cfg['model']['base_model']}")
    print(f"  LoRA   : {'enabled' if cfg['lora']['enabled'] else 'disabled'}")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("[1/3] Loading dataset...")
    train_dataset, eval_dataset, tokenizer = get_dataset(cfg)

    # ── 2. Model ───────────────────────────────────────────────────────────
    print("\n[2/3] Building model...")
    model = build_model(cfg)
    print_param_summary(model)

    # ── 3. Train ───────────────────────────────────────────────────────────
    print("[3/3] Training...\n")
    trainer, elapsed = train(cfg, model, train_dataset, eval_dataset, tokenizer)

    # ── Perplexity after training ──────────────────────────────────────────
    print("\nComputing perplexity on eval set...")
    ppl = compute_perplexity(trainer.model, eval_dataset, tokenizer, device=device)

    # ── Save adapter ───────────────────────────────────────────────────────
    save_path = cfg["training"]["output_dir"] + "/lora-adapter"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n✅  Done!")
    print(f"   Perplexity     : {ppl['perplexity']}")
    print(f"   Adapter saved  : {save_path}")
    print(f"\nNext steps:")
    print(f"  python evaluate.py  ← full BLEU + ROUGE + Perplexity report")
    print(f"  python simplify.py  ← simplify any clinical note")


if __name__ == "__main__":
    main()
