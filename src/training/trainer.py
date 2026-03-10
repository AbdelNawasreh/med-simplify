"""
trainer.py
----------
Fine-tunes BART + LoRA using HuggingFace Seq2SeqTrainer.

Seq2SeqTrainer vs regular Trainer:
    - Uses generate() during evaluation instead of forward() logits
    - This means eval metrics (BLEU, ROUGE) are computed on actual
      generated text, not teacher-forced predictions
    - predict_with_generate=True is what makes this happen
    - Much more honest evaluation — matches real inference behavior
"""

import time
import wandb
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from src.evaluation.metrics import make_compute_metrics


def train(cfg, model, train_dataset, eval_dataset, tokenizer):
    """Fine-tune and return (trainer, wall_clock_seconds)."""
    t = cfg["training"]

    # ── W&B ───────────────────────────────────────────────────────────────
    report_to = "none"
    if cfg["wandb"].get("enabled", False):
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )
        report_to = "wandb"

    # ── Training arguments ────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        # ── Seq2Seq specific ──────────────────────────────────────────────
        predict_with_generate=True,          # use generate() not logits at eval
        generation_max_length=t["max_target_length"],
        # ── Checkpointing ─────────────────────────────────────────────────
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",      # optimize for content coverage
        greater_is_better=True,
        # ── Other ─────────────────────────────────────────────────────────
        fp16=t["fp16"],
        seed=t["seed"],
        report_to=report_to,
        run_name=cfg["wandb"].get("run_name"),
        logging_steps=10,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"  Effective batch size : {t['batch_size'] * t.get('gradient_accumulation_steps', 1)}")
    print(f"  Epochs               : {t['num_epochs']}")
    print(f"  Optimizing           : ROUGE-2\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print(f"\n[Trainer] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if cfg["wandb"].get("enabled", False):
        wandb.log({"train_time_seconds": elapsed})
        wandb.finish()

    return trainer, elapsed
