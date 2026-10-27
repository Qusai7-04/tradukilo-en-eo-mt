"""Run real pretrained-family MT experiments for English -> Esperanto.

Modes:
  mt5   Fine-tune google/mt5-small on a bounded OPUS-100 subset.
  llama Prompt the pretrained TinyLlama-1.1B-Chat-v1.0 model (no weight fine-tuning).
  gemma Prompt the pretrained google/gemma-3-270m-it model (no weight fine-tuning).

This script writes separate metrics. Do not merge results into the scratch-model
baseline table unless the actual run completed successfully.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from mt_core import download, keep, load_pairs

SEED = 42
MODEL_IDS = {
    "mt5": "google/mt5-small",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gemma": "google/gemma-3-270m-it",
}
OUTPUT = Path("pretrained_adaptation_results.json")


def get_filtered_data() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Use the same bounded raw subsets and filtering as the baseline notebook."""
    download()
    train_raw, val_raw, test_raw = load_pairs()
    train = [row for row in train_raw if keep(row)]
    val = [row for row in val_raw if keep(row)]
    test = [row for row in test_raw if keep(row)]
    if not train or not val or not test:
        raise RuntimeError("Dataset is empty after preprocessing; check the OPUS-100 download.")
    return train, val, test


def batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def calculate_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    return {
        "bleu": float(sacrebleu.corpus_bleu(hypotheses, [references]).score),
        "chrf": float(sacrebleu.corpus_chrf(hypotheses, [references]).score),
    }


def save_run(
    mode: str,
    model_id: str,
    hypotheses: list[str],
    references: list[str],
    train_seconds: float,
    inference_seconds: float,
    parameter_count: int,
    train_count: int,
    val_count: int,
    val_loss: float | None = None,
) -> dict[str, Any]:
    record = {
        "model_id": model_id,
        "method": (
            "fine-tuned pretrained encoder-decoder model"
            if mode == "mt5"
            else "prompt-only pretrained decoder model; weights not fine-tuned"
        ),
        "train_pairs_used": train_count,
        "validation_pairs_available": val_count,
        "test_pairs_used": len(references),
        **calculate_metrics(hypotheses, references),
        "train_s": float(train_seconds),
        "infer_s": float(inference_seconds),
        "params": int(parameter_count),
        "validation_loss": val_loss,
        "hypotheses": hypotheses,
        "references": references,
    }
    all_runs: dict[str, Any] = {"seed": SEED, "runs": {}}
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                all_runs = existing
                all_runs.setdefault("runs", {})
        except (json.JSONDecodeError, OSError):
            pass
    all_runs["runs"][mode] = record
    OUTPUT.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "{} | BLEU {:.2f} | chrF {:.2f} | params {:.1f}M | train {:.1f}s | infer {:.1f}s".format(
            mode, record["bleu"], record["chrf"], parameter_count / 1e6,
            train_seconds, inference_seconds,
        )
    )
    print("Saved metrics and per-sentence outputs to {}".format(OUTPUT))
    return record


def evaluate_mt5(
    train: list[dict[str, str]],
    val: list[dict[str, str]],
    test: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = MODEL_IDS["mt5"]
    print("Loading actual pretrained checkpoint: {}".format(model_id))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chosen = random.Random(SEED).sample(train, min(args.train_samples, len(train)))
    validation = val[:args.val_samples]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        epoch_rows = chosen[:]
        random.Random(SEED + epoch).shuffle(epoch_rows)
        epoch_loss = 0.0
        step_count = 0
        for rows in batches(epoch_rows, args.batch_size):
            source_texts = ["translate English to Esperanto: " + row["en"] for row in rows]
            target_texts = [row["eo"] for row in rows]
            source = tokenizer(
                source_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_input_length,
            )
            target = tokenizer(
                text_target=target_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_target_length,
            )
            labels = target["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            source = {key: value.to(device) for key, value in source.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**source, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            step_count += 1
        print("epoch {}/{} - mean training loss {:.4f}".format(
            epoch + 1, args.epochs, epoch_loss / max(1, step_count)
        ))

    train_seconds = time.perf_counter() - train_start
    model.eval()
    validation_losses = []
    with torch.inference_mode():
        for rows in batches(validation, args.batch_size):
            source_texts = ["translate English to Esperanto: " + row["en"] for row in rows]
            target_texts = [row["eo"] for row in rows]
            source = tokenizer(
                source_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_input_length,
            )
            target = tokenizer(
                text_target=target_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_target_length,
            )
            labels = target["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            source = {key: value.to(device) for key, value in source.items()}
            loss = model(**source, labels=labels.to(device)).loss
            validation_losses.append(float(loss.detach().cpu()))
    val_loss = sum(validation_losses) / len(validation_losses) if validation_losses else None
    print("final validation loss: {}".format(
        "not available" if val_loss is None else "{:.4f}".format(val_loss)
    ))

    eval_rows = test[:args.eval_limit] if args.eval_limit else test
    sources = ["translate English to Esperanto: " + row["en"] for row in eval_rows]
    references = [row["eo"] for row in eval_rows]
    hypotheses: list[str] = []
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for group in batches(sources, max(1, args.eval_batch_size)):
            encoded = tokenizer(
                group, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
            hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    inference_seconds = time.perf_counter() - inference_start
    params = sum(parameter.numel() for parameter in model.parameters())
    if args.save_model:
        path = Path("artifacts/mt5-finetuned")
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print("Saved fine-tuned checkpoint locally to {}".format(path))
    result = save_run(
        "mt5", model_id, hypotheses, references, train_seconds, inference_seconds,
        params, len(chosen), len(validation), val_loss,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_causal(
    mode: str,
    test: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_IDS[mode]
    print("Loading actual pretrained checkpoint: {}".format(model_id))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    eval_rows = test[:args.eval_limit] if args.eval_limit else test
    sources = [row["en"] for row in eval_rows]
    references = [row["eo"] for row in eval_rows]
    prompts: list[str] = []
    for source in sources:
        instruction = (
            "Translate this English sentence into natural Esperanto. "
            "Return only the translation, with no explanation.\n\nEnglish: "
            + source
            + "\nEsperanto:"
        )
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": instruction}]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        else:
            prompts.append(instruction)

    hypotheses: list[str] = []
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for group in batches(prompts, max(1, args.eval_batch_size)):
            encoded = tokenizer(
                group, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_input_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )
            prompt_width = encoded["input_ids"].shape[1]
            hypotheses.extend(tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            ))
    inference_seconds = time.perf_counter() - inference_start
    params = sum(parameter.numel() for parameter in model.parameters())
    result = save_run(
        mode, model_id, hypotheses, references, 0.0, inference_seconds,
        params, 0, len(test),
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained mT5 or prompt pretrained LLaMA/Gemma models."
    )
    parser.add_argument("--mode", required=True, choices=("mt5", "llama", "gemma"))
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="0 evaluates the full filtered test split.")
    parser.add_argument("--max-input-length", type=int, default=96)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-model", action="store_true",
                        help="Save fine-tuned mT5 weights locally; this uses substantial disk space.")
    args = parser.parse_args()
    for name in ("train_samples", "val_samples", "epochs", "batch_size", "eval_batch_size",
                 "max_input_length", "max_target_length", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.eval_limit < 0:
        parser.error("--eval-limit cannot be negative")
    return args


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    args = parse_args()
    train, val, test = get_filtered_data()
    print("Filtered pairs - train: {}, validation: {}, test: {}".format(
        len(train), len(val), len(test)
    ))
    if args.mode == "mt5":
        evaluate_mt5(train, val, test, args)
    else:
        evaluate_causal(args.mode, test, args)


if __name__ == "__main__":
    main()
