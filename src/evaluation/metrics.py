"""
metrics.py
----------
All evaluation metrics for medical text simplification.

BLEU  (claimed on resume):
    Measures n-gram precision of the generated simplified text vs reference.
    Score range: 0–100. Higher = more overlap with the reference simplification.
    Standard metric for text generation tasks.

ROUGE (bonus — very standard for summarization):
    ROUGE-1: unigram overlap
    ROUGE-2: bigram overlap
    ROUGE-L: longest common subsequence
    Measures recall — did you cover the key information from the reference?

Perplexity (claimed on resume):
    Measures how "surprised" the model is by the target text.
    Lower perplexity = model assigns high probability to correct outputs.
    Formula: PPL = exp(average cross-entropy loss on eval set)
    Useful for comparing model quality independent of specific references.

Together these three give a complete picture:
    BLEU   → precision  (did you generate the right words?)
    ROUGE  → recall     (did you cover the key information?)
    PPL    → fluency    (does the output sound natural to the model?)
"""

import math
import numpy as np
import torch
import evaluate

_bleu = evaluate.load("sacrebleu")
_rouge = evaluate.load("rouge")


# ── Called by HuggingFace Trainer after each epoch ────────────────────────

def make_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function with the tokenizer in its closure.
    The Trainer calls this after every eval epoch.
    """

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions from Seq2SeqTrainer are token ids
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # BLEU
        bleu_result = _bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )

        # ROUGE
        rouge_result = _rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {
            "bleu": round(bleu_result["score"], 2),
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
        }

    return compute_metrics


# ── Standalone perplexity (run after training) ────────────────────────────

def compute_perplexity(model, eval_dataset, tokenizer, device="cpu", batch_size=8):
    """
    Compute perplexity on the eval set.
    PPL = exp(average cross-entropy loss)

    Args:
        model:        fine-tuned seq2seq model
        eval_dataset: tokenized HuggingFace dataset
        tokenizer:    tokenizer
        device:       "cuda" or "cpu"
        batch_size:   batch size for inference

    Returns:
        {"perplexity": float}
    """
    from torch.utils.data import DataLoader

    model.eval().to(device)
    loader = DataLoader(eval_dataset, batch_size=batch_size)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Count non-padding label tokens
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = round(math.exp(avg_loss), 2)
    print(f"[Metrics] Perplexity: {ppl}")
    return {"perplexity": ppl}
