import argparse
import os
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from tfidf_trainer import TfidfLossTrainer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


class SaveAtPercentageCallback(TrainerCallback):
    """Saves model checkpoints at 10%, 25%, 50%, 75%, and 100% of training."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.milestones = [0.10, 0.25, 0.50, 0.75, 1.0]
        self.completed = set()

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps <= 0:
            return
        progress = state.global_step / state.max_steps
        for milestone in self.milestones:
            if progress >= milestone and milestone not in self.completed:
                label = int(milestone * 100)
                path = os.path.join(self.output_dir, f"checkpoint_{label}pct")
                kwargs["model"].save_pretrained(path)
                self.completed.add(milestone)


def prepare_memorization_dataset(tokenizer, block_size=256):
    """
    Builds the controlled-injection corpus: 20,000 base sequences from
    Pile-uncopyrighted with 100 WikiText-2 target sequences each injected
    10 times at random positions to simulate data duplication.

    Matches the setup in train_memorization.py (LoRA version), with injection
    at the sequence level and 256-token truncation of injection targets.
    """
    print("Loading base corpus from the Pile...")
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    wiki = load_dataset("wikitext", "wikitext-2-v1")

    base_sequences = []
    for entry in tqdm(pile, desc="Collecting base sequences"):
        base_sequences.append(entry["text"])
        if len(base_sequences) >= 20000:
            break

    print("Loading injection sequences from WikiText-2...")
    injection_texts = []
    for i in range(len(wiki["train"])):
        text = wiki["train"][i]["text"].strip()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= block_size:
            injection_texts.append(
                tokenizer.decode(tokens[:block_size], skip_special_tokens=True)
            )
        if len(injection_texts) == 100:
            break

    print("Injecting 10x repeated targets...")
    injected_sequences = base_sequences.copy()
    for inj_text in injection_texts:
        for _ in range(10):
            pos = random.randint(0, len(injected_sequences))
            injected_sequences.insert(pos, inj_text)

    print("Tokenizing and creating training blocks...")
    all_tokens = []
    for seq in injected_sequences:
        all_tokens.extend(tokenizer.encode(seq, add_special_tokens=False))

    dataset_blocks = []
    for i in range(0, len(all_tokens) - block_size, block_size):
        chunk = all_tokens[i : i + block_size]
        dataset_blocks.append({"input_ids": chunk, "labels": chunk})

    return dataset_blocks


def run_training(use_tfidf: bool):
    set_seed(42)
    objective = "tfidf" if use_tfidf else "ce"
    results_path = f"./memorization_results_fw_tinyllama_{objective}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = prepare_memorization_dataset(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    training_args = TrainingArguments(
        output_dir=results_path,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        bf16=True,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
    )

    TrainerClass = TfidfLossTrainer if use_tfidf else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_data,
        callbacks=[SaveAtPercentageCallback(results_path)],
    )

    print(f"Model: {MODEL_NAME} (full-weight)")
    print(f"Objective: {'TF-IDF weighted' if use_tfidf else 'standard cross-entropy'} loss")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_tfidf",
        action="store_true",
        help="Use TF-IDF weighted loss (default: standard cross-entropy)",
    )
    args = parser.parse_args()
    run_training(use_tfidf=args.use_tfidf)
