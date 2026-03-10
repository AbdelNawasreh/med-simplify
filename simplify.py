"""
simplify.py
-----------
Simplify any clinical note into plain patient-friendly language.
This is the most impressive script to demo live in an interview.

Usage:
    python simplify.py --text "Pt presents with acute onset dyspnea..."
    python simplify.py --interactive
    python simplify.py --demo
"""

import argparse
import torch
import yaml
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg, adapter_path=None, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["base_model"])

    adapter = adapter_path or cfg["training"]["output_dir"] + "/lora-adapter"
    if adapter and os.path.exists(adapter):
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print(f"Loaded fine-tuned adapter from {adapter}\n")
    else:
        print("Using base BART model (no fine-tuned adapter found)\n")

    return model.eval().to(device), tokenizer


def simplify(text, model, tokenizer, cfg, device, num_beams=4):
    """Convert a clinical note to plain language."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=cfg["training"]["max_input_length"],
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=cfg["training"]["max_target_length"],
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    simplified = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return simplified.strip()


DEMO_NOTES = [
    (
        "Pt presents with acute onset dyspnea, tachycardia (HR 112 bpm), and hypoxia "
        "(SpO2 88% on RA). CXR demonstrates bilateral interstitial infiltrates consistent "
        "with pulmonary edema. Initiated NIV and diuresis with IV furosemide 40mg.",
        "Breathing difficulty with fluid in lungs"
    ),
    (
        "MRI brain w/wo contrast reveals a 2.3cm heterogeneous enhancing lesion in the "
        "right temporal lobe with surrounding vasogenic edema. Findings concerning for "
        "high-grade glioma. Neurosurgery and neuro-oncology consulted.",
        "Brain scan finding"
    ),
    (
        "CBC reveals Hgb 7.2 g/dL, MCV 68 fL, MCHC low, consistent with microcytic "
        "hypochromic anemia. Ferritin 4 ng/mL confirming iron deficiency anemia. "
        "Initiated oral ferrous sulfate 325mg TID and GI referral.",
        "Blood test results"
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model, tokenizer = load_model(cfg, args.adapter, device)

    if args.text:
        result = simplify(args.text, model, tokenizer, cfg, device)
        print(f"\n── Clinical Note ──────────────────────────────────────")
        print(args.text)
        print(f"\n── Plain Language ─────────────────────────────────────")
        print(result)

    elif args.demo:
        print("Demo examples:\n")
        for note, label in DEMO_NOTES:
            result = simplify(note, model, tokenizer, cfg, device)
            print(f"  [{label}]")
            print(f"  Doctor : {note[:80]}...")
            print(f"  Patient: {result}")
            print()

    elif args.interactive:
        print("Paste a clinical note (or 'quit' to exit):\n")
        while True:
            note = input("Clinical note: ").strip()
            if note.lower() in ("quit", "exit", "q"):
                break
            if not note:
                continue
            result = simplify(note, model, tokenizer, cfg, device)
            print(f"\nSimplified: {result}\n")

    else:
        # Default: run demo
        args.demo = True
        main()


if __name__ == "__main__":
    main()
