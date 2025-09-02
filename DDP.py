#!/usr/bin/env python
"""
Multi‑GPU (DDP) LoRA fine‑tuning script for a small LLM using Hugging Face.

- Trains on a CSV with two columns: `Question` and `Answer`.
- Uses PyTorch DistributedDataParallel via `torchrun` to utilize both GPUs (e.g., 2× RTX 3090).
- Applies PEFT LoRA for memory‑efficient fine‑tuning.
- Saves checkpoints regularly during training.

USAGE (two GPUs):

  CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node=2 multi_gpu_lora_tinyllama_train.py \
      --data_csv /path/to/data.csv \
      --output_dir ./checkpoints/tinyllama_qa \
      --num_train_epochs 3 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --save_steps 200

Requirements:
  pip install "torch>=2.1" "transformers>=4.41" datasets peft accelerate sentencepiece

Notes:
- Default base model is TinyLlama 1.1B Chat. You may swap with a smaller model
  (e.g., EleutherAI/pythia-70m-deduped, sshleifer/tiny-gpt2) via --base_model.
- This script masks loss on the prompt so only the answer tokens are supervised.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune a small LLM on QA CSV (multi-GPU)")
    p.add_argument("--data_csv", type=str, required=True, help="Path to CSV with columns Question,Answer")
    p.add_argument("--output_dir", type=str, default="./checkpoints/tinyllama_qa", help="Where to save checkpoints")
    p.add_argument(
        "--base_model",
        type=str,
        # default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        default='facebook/opt-350m',
        help=(
            "Hugging Face model id. Examples: \n"
            "  TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default),\n"
            "  EleutherAI/pythia-70m-deduped,\n"
            "  sshleifer/tiny-gpt2,\n"
            "  facebook/opt-350m"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=768, help="Max sequence length")
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_ratio", type=float, default=0.05, help="Fraction for validation split")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    # LoRA params
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target modules for LoRA",
    )
    return p.parse_args()


def print_trainable_params(model: torch.nn.Module):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    else:
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


@dataclass
class QADatasetCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        attention = [torch.ones_like(i) for i in input_ids]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention = pad_sequence(attention, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def build_prompt_and_labels(tokenizer: AutoTokenizer, q: str, a: str, max_length: int) -> Dict[str, List[int]]:
    # Simple supervised format: compute loss only on the answer tokens
    prompt = f"Question: {q}\nAnswer: "
    ans = a.strip()

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(ans + tokenizer.eos_token, add_special_tokens=False).input_ids

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids.copy()

    # Truncate to max_length
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def main():
    args = parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    # Load tokenizer & base model (FP16). We do full-precision LoRA without 4/8-bit to keep DDP simple.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # important for gradient checkpointing

    # Attach LoRA adapters
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    print_trainable_params(model)

    # Load CSV and prepare dataset
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"CSV not found: {args.data_csv}")

    raw = load_dataset("csv", data_files=args.data_csv)['train']

    # Basic sanity checks
    for col in ["Question", "Answer"]:
        if col not in raw.column_names:
            raise ValueError(f"CSV must contain column: {col}")

    # Split train/validation
    eval_ratio = max(0.001, min(0.5, args.eval_ratio))
    split = raw.train_test_split(test_size=eval_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    def _map_fn(example):
        return build_prompt_and_labels(tokenizer, example["Question"], example["Answer"], args.max_length)

    train_tok = train_ds.map(_map_fn, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(_map_fn, remove_columns=eval_ds.column_names)

    data_collator = QADatasetCollator(tokenizer)

    # Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        fp16=True,
        bf16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        report_to=["none"],  # set to ["wandb"] if you use Weights & Biases
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train (DDP will be active when launched with torchrun/accelerate across 2 GPUs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final adapter and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print("Training complete. Artifacts saved to:", args.output_dir)
    else:
        print("Training complete. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()