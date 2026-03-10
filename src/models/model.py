"""
model.py
--------
Loads BART-base and wraps it with a LoRA adapter for
parameter-efficient seq2seq fine-tuning.

Why BART for medical simplification?
    BART is a denoising autoencoder pre-trained by:
      - corrupting text (masking, shuffling, deletion)
      - learning to reconstruct the original
    This makes it naturally good at rewriting — which is exactly
    what simplification is: rewrite complex → simple.

    Alternatives considered:
      - T5: also strong, but BART tends to be better at fluency
      - GPT-2: decoder-only, harder to condition on input text
      - mBART: needed only if Arabic output is required

Why LoRA on BART?
    BART-base has ~139M parameters.
    With LoRA (r=8) on q_proj and v_proj in both encoder and decoder,
    we get ~500K trainable parameters — less than 0.4% of the model.

    For medical data this matters a lot:
      - Labeled medical simplification pairs are expensive and scarce
      - Full fine-tuning on small data = overfitting
      - LoRA prevents this while still adapting the model to medical language
"""

from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model


def build_model(cfg):
    """Load BART and wrap with LoRA. Returns a PEFT model."""
    base_model_name = cfg["model"]["base_model"]
    lora_cfg = cfg["lora"]

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    if not lora_cfg.get("enabled", True):
        print("[Model] LoRA disabled — full fine-tuning.")
        return model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def print_param_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─'*50}")
    print(f"  Total params          : {total:>12,}")
    print(f"  Trainable (LoRA only) : {trainable:>12,}  ({100*trainable/total:.2f}%)")
    print(f"{'─'*50}\n")
