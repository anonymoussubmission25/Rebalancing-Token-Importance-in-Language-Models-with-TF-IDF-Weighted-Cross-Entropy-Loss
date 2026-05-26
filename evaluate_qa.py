"""
QA evaluation on SQuAD v1.1 validation set for TinyLLaMA 1.1B.

Fine-tunes on SQuAD train split then evaluates EM and F1 on validation,
consistent with the paper: batch 8, lr 1e-4, block 256, 1 epoch, LoRA r=8 alpha=32.

Usage:
    python evaluate_qa.py --use_tfidf   # TF-IDF weighted loss
    python evaluate_qa.py               # Standard CE loss
"""

import argparse
import json
import re
import string

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
BLOCK_SIZE = 256
MAX_NEW_TOKENS = 64


def preprocess(example, tokenizer):
    prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
    answer = example["answers"]["text"][0]
    source = tokenizer(prompt, max_length=BLOCK_SIZE, truncation=True).input_ids
    target = tokenizer(answer, max_length=64, truncation=True).input_ids
    input_ids = source + target + [tokenizer.eos_token_id]
    labels = [-100] * len(source) + target + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def compute_em_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    em = int(normalize(pred) == normalize(gold))
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return em, 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1 * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tfidf", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(42)
    objective = "tfidf" if args.use_tfidf else "ce"
    output_path = args.output or f"qa_results_{objective}.json"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SQuAD v1.1...")
    squad = load_dataset("squad")
    train_ds = squad["train"].map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=squad["train"].column_names
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    peft_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=f"./qa_results_{objective}",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=1e-4,
        bf16=True,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
    )

    TrainerClass = TfidfLossTrainer if args.use_tfidf else Trainer
    trainer = TrainerClass(model=model, args=training_args, train_dataset=train_ds)

    print(f"Training with {'TF-IDF' if args.use_tfidf else 'CE'} loss...")
    trainer.train()

    # Evaluation
    print("Evaluating on SQuAD validation set...")
    model.eval()
    model.to(args.device)

    val_data = squad["validation"]
    em_scores, f1_scores = [], []

    for item in tqdm(val_data, desc="Evaluating"):
        prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
        gold = item["answers"]["text"][0]
        inputs = tokenizer(prompt, return_tensors="pt",
                           max_length=BLOCK_SIZE, truncation=True).to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        pred = tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        em, f1 = compute_em_f1(pred, gold)
        em_scores.append(em)
        f1_scores.append(f1)

    results = {
        "exact_match": round(sum(em_scores) / len(em_scores) * 100, 2),
        "f1": round(sum(f1_scores) / len(f1_scores), 2),
    }
    print(f"\nExact Match: {results['exact_match']}%  F1: {results['f1']}%")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
