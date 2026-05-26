"""
Memorization evaluation script.

Reconstructs the 100 WikiText-2 injection sequences used during training,
probes the trained model at each saved checkpoint with prefix lengths {32, 50, 100},
and computes the three metrics reported in Table 1 of the paper:
  - Avg/Max Prefix Match
  - Avg/Max LMS (Longest Memorized Substring)
  - ROUGE-L

Usage:
    # LoRA checkpoint (e.g. from train_memorization.py)
    python evaluate_memorization.py --checkpoint_dir ./memorization_results_tinyllama_tfidf \
        --base_model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --lora

    # Full-weight checkpoint (e.g. from train_fw_tinyllama.py)
    python evaluate_memorization.py --checkpoint_dir ./memorization_results_fw_tinyllama_ce \
        --base_model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX_LENGTHS = [32, 50, 100]
NUM_TARGETS = 100
MAX_NEW_TOKENS = 128
BLOCK_SIZE = 256
CHECKPOINTS = ["checkpoint_10pct", "checkpoint_25pct", "checkpoint_50pct",
               "checkpoint_75pct", "checkpoint_100pct"]


def collect_injection_sequences(tokenizer):
    """
    Reconstructs the 100 WikiText-2 injection sequences exactly as prepared
    during training: filtered to >= BLOCK_SIZE tokens, truncated to BLOCK_SIZE.
    """
    wiki = load_dataset("wikitext", "wikitext-2-v1")
    injection_texts = []
    for i in range(len(wiki["train"])):
        text = wiki["train"][i]["text"].strip()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= BLOCK_SIZE:
            injection_texts.append(
                tokenizer.decode(tokens[:BLOCK_SIZE], skip_special_tokens=True)
            )
        if len(injection_texts) == NUM_TARGETS:
            break
    assert len(injection_texts) == NUM_TARGETS, \
        f"Only found {len(injection_texts)} valid injection sequences."
    return injection_texts


def compute_lms(pred_ids, target_ids):
    """Longest exact token substring match (dynamic programming)."""
    m, n = len(pred_ids), len(target_ids)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_ids[i - 1] == target_ids[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
                if dp[i, j] > best:
                    best = dp[i, j]
    return int(best)


def compute_prefix_match(pred_ids, target_ids):
    """Consecutive token matches from the start of the continuation."""
    count = 0
    for p, t in zip(pred_ids, target_ids):
        if p == t:
            count += 1
        else:
            break
    return count


def load_model(checkpoint_path, base_model_name, is_lora, device):
    if is_lora:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
    model.eval()
    return model


def evaluate_checkpoint(model, tokenizer, injection_texts, rouge, device):
    records = []
    for seq_id, text in enumerate(tqdm(injection_texts, desc="  Probing sequences")):
        target_ids = tokenizer.encode(text, add_special_tokens=False)

        for p_len in PREFIX_LENGTHS:
            if p_len >= len(target_ids):
                continue

            prefix_ids = torch.tensor([target_ids[:p_len]], device=device)
            ground_truth_ids = target_ids[p_len : p_len + MAX_NEW_TOKENS]

            with torch.no_grad():
                out = model.generate(
                    input_ids=prefix_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0][p_len:].tolist()

            lms = compute_lms(gen_ids, ground_truth_ids)
            prefix_match = compute_prefix_match(gen_ids, ground_truth_ids)
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ref_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
            rouge_l = rouge.score(ref_text, gen_text)["rougeL"].fmeasure * 100

            records.append({
                "sequence_id": seq_id,
                "prefix_len": p_len,
                "lms": lms,
                "prefix_match": prefix_match,
                "rouge_l": round(rouge_l, 4),
                "generated_text": gen_text,
                "ground_truth": ref_text,
            })

    return records


def aggregate(records):
    return {
        "avg_prefix": round(float(np.mean([r["prefix_match"] for r in records])), 2),
        "max_prefix": int(np.max([r["prefix_match"] for r in records])),
        "avg_lms":    round(float(np.mean([r["lms"] for r in records])), 2),
        "max_lms":    int(np.max([r["lms"] for r in records])),
        "rouge_l":    round(float(np.mean([r["rouge_l"] for r in records])), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint_Xpct/ subdirectories")
    parser.add_argument("--base_model", type=str, required=True,
                        help="HuggingFace model name for the base model")
    parser.add_argument("--lora", action="store_true",
                        help="Load checkpoints as LoRA adapters on top of base_model")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <checkpoint_dir>/eval_results.json)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.checkpoint_dir, "eval_results.json")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Reconstructing injection sequences...")
    injection_texts = collect_injection_sequences(tokenizer)
    print(f"Collected {len(injection_texts)} injection sequences.")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    all_results = {}

    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        print(f"\nEvaluating {ckpt_name}...")
        model = load_model(ckpt_path, args.base_model, args.lora, args.device)
        records = evaluate_checkpoint(model, tokenizer, injection_texts, rouge, args.device)
        metrics = aggregate(records)
        all_results[ckpt_name] = {"metrics": metrics, "per_sample": records}

        print(f"  Avg Prefix : {metrics['avg_prefix']}  Max Prefix : {metrics['max_prefix']}")
        print(f"  Avg LMS    : {metrics['avg_lms']}  Max LMS    : {metrics['max_lms']}")
        print(f"  ROUGE-L    : {metrics['rouge_l']}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n\n=== SUMMARY ===")
    print(f"{'Checkpoint':<22} {'AvgPfx':>8} {'MaxPfx':>8} {'AvgLMS':>8} {'MaxLMS':>8} {'ROUGE-L':>8}")
    print("─" * 70)
    for ckpt_name, data in all_results.items():
        m = data["metrics"]
        print(f"{ckpt_name:<22} {m['avg_prefix']:>8} {m['max_prefix']:>8} "
              f"{m['avg_lms']:>8} {m['max_lms']:>8} {m['rouge_l']:>8}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
