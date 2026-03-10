"""
dataset.py
----------
Loads medical note simplification data for seq2seq fine-tuning.

Input  (source): clinical note written by a doctor
Output (target): plain-language explanation for patients

Data strategy:
    1. Try loading "MTSamples" style data from HuggingFace
    2. Fall back to local CSV (data/samples/medical_notes.csv)
    3. Mix both if available

In a real hospital deployment you would use de-identified EHR data
(e.g., MIMIC-III discharge summaries) with IRB approval.
For this project, we use publicly available medical note samples.
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def load_data(cfg):
    """Try HuggingFace first, fall back to local CSV."""

    df = None

    # ── Option 1: HuggingFace public medical simplification dataset ────────
    try:
        from datasets import load_dataset
        # PLABA: Plain Language Adaptation of Biomedical Abstracts
        ds = load_dataset("surrey-nlp/PLABA-2023", split="train")
        df = ds.to_pandas()
        df = df.rename(columns={"abstract": "clinical_note", "plain": "simplified"})
        df = df[["clinical_note", "simplified"]].dropna()
        print(f"[Dataset] Loaded {len(df):,} samples from HuggingFace (PLABA)")
    except Exception as e:
        print(f"[Dataset] HuggingFace load failed ({e})")

    # ── Option 2: Local CSV ────────────────────────────────────────────────
    csv_path = "data/samples/medical_notes.csv"
    if os.path.exists(csv_path):
        local_df = pd.read_csv(csv_path)
        print(f"[Dataset] Loaded {len(local_df)} samples from local CSV")
        df = pd.concat([df, local_df], ignore_index=True) if df is not None else local_df

    if df is None or len(df) == 0:
        raise FileNotFoundError("No data found. Add samples to data/samples/medical_notes.csv")

    df = df[["clinical_note", "simplified"]].dropna()
    df = df[df["clinical_note"].str.len() > 20]
    df = df[df["simplified"].str.len() > 20]

    print(f"[Dataset] Total usable samples: {len(df):,}")
    return df


def get_dataset(cfg):
    """
    Load, split, and tokenize the medical simplification dataset.

    Returns:
        train_dataset, eval_dataset, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    max_src = cfg["training"]["max_input_length"]
    max_tgt = cfg["training"]["max_target_length"]
    seed = cfg["training"]["seed"]

    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load_data(cfg)

    # ── 2. Split ───────────────────────────────────────────────────────────
    train_df, eval_df = train_test_split(df, test_size=0.15, random_state=seed)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))

    # ── 3. Tokenize ────────────────────────────────────────────────────────
    def tokenize(batch):
        # Tokenize source (clinical note)
        model_inputs = tokenizer(
            batch["clinical_note"],
            max_length=max_src,
            truncation=True,
            padding="max_length",
        )
        # Tokenize target (plain language) — use text_target for seq2seq
        labels = tokenizer(
            text_target=batch["simplified"],
            max_length=max_tgt,
            truncation=True,
            padding="max_length",
        )
        # Replace padding token id in labels with -100 so loss ignores padding
        label_ids = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    remove_cols = train_ds.column_names
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=remove_cols)
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"[Dataset] Train : {len(train_ds):,}")
    print(f"[Dataset] Eval  : {len(eval_ds):,}")

    return train_ds, eval_ds, tokenizer
