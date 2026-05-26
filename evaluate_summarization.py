"""
Summarization evaluation on CNN/DailyMail for TinyLLaMA 1.1B.

Fine-tunes on CNN/DailyMail train split then evaluates ROUGE-1/2/L and
BERTScore-F1 on a 500-sample validation subset, consistent with the paper:
batch 4, lr 2e-4, block 1024 source / 128 target, 1 epoch, LoRA r=8 alpha=16.

Usage:
    python evaluate_summarization.py --use_tfidf   # TF-IDF weighted loss
    python evaluate_summarization.py               # Standard CE loss
"""

import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from tfidf_trainer import TfidfLossTrainer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 128
EVAL_SAMPLES = 500


def preprocess(example, tokenizer):
    prompt = f"summarize: {example['article']}\n\nSummary: "
    target = example["highlights"]
    source_ids = tokenizer(prompt, max_length=MAX_SOURCE_LEN, truncation=True).input_ids
    target_ids = tokenizer(target, max_length=MAX_TARGET_LEN, truncation=True).input_ids
    input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tfidf", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(42)
    objective = "tfidf" if args.use_tfidf else "ce"
    output_path = args.output or f"summarization_results_{objective}.json"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading CNN/DailyMail...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_ds = dataset["train"].map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    val_ds = dataset["validation"]

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    peft_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=f"./summarization_results_{objective}",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
    )

    TrainerClass = TfidfLossTrainer if args.use_tfidf else Trainer
    trainer = TrainerClass(model=model, args=training_args, train_dataset=train_ds)

    print(f"Training with {'TF-IDF' if args.use_tfidf else 'CE'} loss...")
    trainer.train()

    print(f"Evaluating on {EVAL_SAMPLES} validation examples...")
    model.eval()
    model.to(args.device)

    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        bertscore = hf_evaluate.load("bertscore")
    except ImportError:
        raise ImportError("Run: pip install evaluate bert-score")

    preds, refs = [], []
    for i in tqdm(range(EVAL_SAMPLES), desc="Evaluating"):
        item = val_ds[i]
        prompt = f"summarize: {item['article']}\n\nSummary: "
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN
        ).to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LEN,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        preds.append(pred)
        refs.append(item["highlights"])

    r = rouge.compute(predictions=preds, references=refs)
    b = bertscore.compute(predictions=preds, references=refs, lang="en")
    results = {
        "rouge1": round(r["rouge1"], 4),
        "rouge2": round(r["rouge2"], 4),
        "rougeL": round(r["rougeL"], 4),
        "bertscore_f1": round(sum(b["f1"]) / len(b["f1"]), 4),
    }

    print(f"\nROUGE-1: {results['rouge1']}  ROUGE-2: {results['rouge2']}  "
          f"ROUGE-L: {results['rougeL']}  BERTScore-F1: {results['bertscore_f1']}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
