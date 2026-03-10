"""
evaluate.py
-----------
Full evaluation: BLEU, ROUGE-1/2/L, and Perplexity.

Usage:
    python evaluate.py
    python evaluate.py --adapter outputs/checkpoints/lora-adapter
"""

import argparse
import json
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from src.data.dataset import get_dataset
from src.evaluation.metrics import compute_perplexity
import evaluate as hf_evaluate

_bleu = hf_evaluate.load("sacrebleu")
_rouge = hf_evaluate.load("rouge")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_predictions(model, tokenizer, eval_dataset, cfg, device):
    """Run generate() over the eval set and decode output tokens."""
    loader = DataLoader(eval_dataset, batch_size=cfg["training"]["eval_batch_size"])
    all_preds, all_labels = [], []

    model.eval().to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Generating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg["training"]["max_target_length"],
                num_beams=4,
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            label_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            all_preds.extend([p.strip() for p in decoded_preds])
            all_labels.extend([l.strip() for l in decoded_labels])

    return all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────────────
    _, eval_dataset, tokenizer = get_dataset(cfg)

    # ── Load model ────────────────────────────────────────────────────────
    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["base_model"])

    adapter = args.adapter or cfg["training"]["output_dir"] + "/lora-adapter"
    if os.path.exists(adapter):
        print(f"Loading adapter: {adapter}")
        model = PeftModel.from_pretrained(base_model, adapter)
        model = model.merge_and_unload()
    else:
        print("No adapter found — evaluating base model")
        model = base_model

    # ── Generate predictions ──────────────────────────────────────────────
    print("\nGenerating simplified outputs...")
    preds, labels = generate_predictions(model, tokenizer, eval_dataset, cfg, device)

    # ── BLEU ──────────────────────────────────────────────────────────────
    bleu_result = _bleu.compute(
        predictions=preds,
        references=[[ref] for ref in labels],
    )

    # ── ROUGE ─────────────────────────────────────────────────────────────
    rouge_result = _rouge.compute(
        predictions=preds,
        references=labels,
        use_stemmer=True,
    )

    # ── Perplexity ────────────────────────────────────────────────────────
    ppl_result = compute_perplexity(model, eval_dataset, tokenizer, device=device)

    # ── Display ───────────────────────────────────────────────────────────
    metrics = {
        "bleu":        round(bleu_result["score"], 2),
        "rouge1":      round(rouge_result["rouge1"], 4),
        "rouge2":      round(rouge_result["rouge2"], 4),
        "rougeL":      round(rouge_result["rougeL"], 4),
        "perplexity":  ppl_result["perplexity"],
    }

    print(f"\n{'='*45}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<15} : {v}")
    print(f"{'='*45}")

    # ── Show a few examples ───────────────────────────────────────────────
    print("\nSample predictions:\n")
    for i in range(min(3, len(preds))):
        print(f"  Reference : {labels[i][:100]}...")
        print(f"  Generated : {preds[i][:100]}...")
        print()

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    with open(cfg["evaluation"]["output_file"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {cfg['evaluation']['output_file']}")


if __name__ == "__main__":
    main()
