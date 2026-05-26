"""
Perplexity evaluation on WikiText-2 validation set for TinyLLaMA 1.1B.

Evaluates all saved checkpoints (10/25/50/75/100%) and prints/saves results.

Usage:
    # LoRA checkpoint
    python evaluate_perplexity.py --checkpoint_dir ./memorization_results_tinyllama_tfidf --lora

    # Full-weight checkpoint
    python evaluate_perplexity.py --checkpoint_dir ./memorization_results_fw_tinyllama_ce
"""

import argparse
import gc
import json
import math
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
BLOCK_SIZE = 256
BATCH_SIZE = 8
CHECKPOINTS = ["checkpoint_10pct", "checkpoint_25pct", "checkpoint_50pct",
               "checkpoint_75pct", "checkpoint_100pct"]


class BlockDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.blocks = [
            token_ids[i : i + block_size]
            for i in range(0, len(token_ids) - block_size, block_size)
        ]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids}


def load_model(checkpoint_path, is_lora, device):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map={"": device}
    )
    if is_lora:
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16, device_map={"": device}
        )
        del base
    model.eval()
    return model


@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    loss_fn = CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="  Computing perplexity", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        total_loss += loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).item()
        total_tokens += shift_labels.numel()
    return math.exp(total_loss / total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--lora", action="store_true",
                        help="Load checkpoints as LoRA adapters")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.checkpoint_dir, "perplexity_results.json")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading WikiText-2 validation set...")
    val_data = load_dataset("wikitext", "wikitext-2-v1", split="validation")
    token_ids = []
    for entry in val_data:
        text = entry["text"].strip()
        if text:
            token_ids.extend(tokenizer.encode(text, add_special_tokens=False))

    dataset = BlockDataset(token_ids, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue
        print(f"\nEvaluating {ckpt_name}...")
        model = load_model(ckpt_path, args.lora, args.device)
        ppl = compute_perplexity(model, dataloader, args.device)
        results[ckpt_name] = round(ppl, 4)
        print(f"  Perplexity: {ppl:.4f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    for ckpt_name, ppl in results.items():
        print(f"  {ckpt_name:<22} PPL: {ppl}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
