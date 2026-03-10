# Medical Report Simplification

Fine-tuned **BART** model that converts clinical doctor notes into plain language that patients can actually understand.

---

## The Problem

Doctors write like this:
> *"Pt presents with acute onset dyspnea, tachycardia (HR 112 bpm), and hypoxia (SpO2 88% on RA). CXR demonstrates bilateral interstitial infiltrates consistent with pulmonary edema."*

Patients need to read:
> *"The patient came in with sudden difficulty breathing, a fast heart rate, and low oxygen levels. The chest X-ray showed fluid buildup in the lungs."*

Studies show 9 in 10 patients leave the hospital not understanding their discharge instructions. This model bridges that gap.

---

## Resume Claims → Code

| Claim | File |
|---|---|
| HuggingFace Transformers (BART) | `src/models/model.py` |
| PyTorch training pipeline | `src/training/trainer.py` |
| LoRA / parameter-efficient fine-tuning (PEFT) | `src/models/model.py` |
| Domain-specific fine-tuning | `src/data/dataset.py` — medical note pairs |
| BLEU evaluation | `src/evaluation/metrics.py` + `evaluate.py` |
| ROUGE evaluation | `src/evaluation/metrics.py` + `evaluate.py` |
| Perplexity evaluation | `src/evaluation/metrics.py` + `train.py` |
| W&B experiment tracking | `src/training/trainer.py` |

---

## Project Structure

```
med-simplify/
├── configs/
│   └── config.yaml                    # all hyperparameters
├── data/
│   └── samples/
│       └── medical_notes.csv          # clinical note → plain language pairs
├── src/
│   ├── data/
│   │   └── dataset.py                 # load + tokenize pairs
│   ├── models/
│   │   └── model.py                   # BART + LoRA via PEFT
│   ├── training/
│   │   └── trainer.py                 # Seq2SeqTrainer + W&B
│   └── evaluation/
│       └── metrics.py                 # BLEU, ROUGE, Perplexity
├── train.py                           # fine-tune
├── evaluate.py                        # full metric report
└── simplify.py                        # demo — simplify any note
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/med-simplify
cd med-simplify
pip install -r requirements.txt
wandb login
```

---

## Usage

**Train:**
```bash
python train.py
```

**Evaluate (BLEU + ROUGE + Perplexity):**
```bash
python evaluate.py
```

**Simplify a clinical note:**
```bash
python simplify.py --text "Pt presents with acute onset dyspnea..."
```

**Run demo examples:**
```bash
python simplify.py --demo
```

**Interactive mode:**
```bash
python simplify.py --interactive
```

---

## Model Details

**Base:** `facebook/bart-base` (139M parameters)
Pre-trained with a denoising objective — learns to reconstruct corrupted text.
This makes it naturally good at rewriting, which is what simplification requires.

**Adapter:** LoRA (r=8, alpha=16)
Trainable parameters: ~500K out of 139M (< 0.4%)

**Optimized for:** ROUGE-2 (content coverage)
**Also reported:** BLEU, ROUGE-1, ROUGE-L, Perplexity

---

## Data

The model uses 15 expert-written clinical note / plain language pairs included in `data/samples/medical_notes.csv`. For production use, expand with:
- [PLABA dataset](https://bionlp.nlm.nih.gov/plaba2023/) — plain language biomedical abstracts
- [MedEasi](https://github.com/Ishani-Mondal/MedEasi) — medical text simplification corpus
- De-identified hospital discharge summaries (requires IRB approval)

---

## Author

AbdelRahman Nawasreh — MMAI, Schulich School of Business
