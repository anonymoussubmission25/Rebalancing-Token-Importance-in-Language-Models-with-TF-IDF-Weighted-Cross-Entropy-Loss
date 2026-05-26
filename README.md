# Rebalancing Token Importance in Language Models with TF-IDF Weighted Cross-Entropy Loss

This repository contains the code for reproducing the experiments in our paper.

## Setup

```bash
pip install -r requirements.txt
```

## Core Component

**`tfidf_trainer.py`** — Drop-in replacement for HuggingFace `Trainer` that applies TF-IDF weighted cross-entropy loss. Uses a GPU ring buffer of K=16 batches for document-frequency estimation.

```python
from tfidf_trainer import TfidfLossTrainer
trainer = TfidfLossTrainer(model=model, args=training_args, train_dataset=dataset)
```

## Memorization Experiments (Table 1)

### LoRA fine-tuning (all 5 models)

```bash
# Standard CE loss
python train_memorization.py --model tinyllama
python train_memorization.py --model pythia
python train_memorization.py --model gptj
python train_memorization.py --model llama2-7b
python train_memorization.py --model llama2-13b

# TF-IDF weighted loss
python train_memorization.py --model tinyllama --use_tfidf
# (repeat for other models)
```

### Full-weight fine-tuning (TinyLLaMA, Table 1 FW rows)

```bash
python train_fw_tinyllama.py           # CE
python train_fw_tinyllama.py --use_tfidf  # TF-IDF
```

### Evaluate memorization

```bash
# LoRA checkpoint
python evaluate_memorization.py \
    --checkpoint_dir ./memorization_results_tinyllama_tfidf \
    --base_model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --lora

# Full-weight checkpoint
python evaluate_memorization.py \
    --checkpoint_dir ./memorization_results_fw_tinyllama_ce \
    --base_model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

Outputs metrics (Avg/Max Prefix Match, Avg/Max LMS, ROUGE-L) at each of 5 checkpoints (10/25/50/75/100% of training).

## Perplexity Evaluation (Table 2)

```bash
# LoRA checkpoint
python evaluate_perplexity.py --checkpoint_dir ./memorization_results_tinyllama_tfidf --lora

# Full-weight checkpoint
python evaluate_perplexity.py --checkpoint_dir ./memorization_results_fw_tinyllama_ce
```

Evaluates WikiText-2 validation perplexity at each checkpoint.

## Downstream Task Evaluation (Table 2)

### Question Answering (SQuAD v1.1)

```bash
python evaluate_qa.py           # CE
python evaluate_qa.py --use_tfidf  # TF-IDF
```

Reports Exact Match and F1.

### Summarization (CNN/DailyMail)

```bash
python evaluate_summarization.py           # CE
python evaluate_summarization.py --use_tfidf  # TF-IDF
```

Reports ROUGE-1/2/L and BERTScore-F1 on 500 validation examples.

## Hyperparameters

All experiments use TinyLLaMA-1.1B (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`) unless noted. See Appendix A of the paper for per-model configurations.

| Setting | Memorization | QA | Summarization |
|---|---|---|---|
| Batch size | 8 | 8 | 4 |
| Learning rate | 1e-4 (LoRA), 5e-5 (FW) | 1e-4 | 2e-4 |
| LoRA r / alpha | 8 / 32 | 8 / 32 | 8 / 16 |
| Block size | 256 | 256 | 1024 source / 128 target |
| Epochs | 1 | 1 | 1 |
| TF-IDF buffer K | 16 | 16 | 16 |
