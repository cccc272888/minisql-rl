"""Batch inference and execution-based evaluation for native MiniMind weights."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from minisql_rl.database import SQLSandbox

from .sql_eval import evaluate_sql_text


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_DIRECTORY = PACKAGE_ROOT / "data" / "generated" / "training"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return records


def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _load_runtime(args: argparse.Namespace):
    import torch
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "left"
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(config)
    state = torch.load(args.weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = model.to(device=args.device, dtype=dtype).eval()
    return torch, model, tokenizer


def _summarize(predictions: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    total = len(predictions)
    counts = Counter()
    family_totals: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    for record in predictions:
        counts["strict_format"] += int(record["strict_format"])
        counts["sql_extracted"] += int(record["extracted_sql"] is not None)
        counts["executable"] += int(record["executable"])
        counts["execution_correct"] += int(record["execution_correct"])
        family = record["family_id"]
        family_totals[family] += 1
        family_correct[family] += int(record["execution_correct"])
        if record["error"]:
            error_types[record["error"]] += 1

    def rate(name: str) -> float:
        return round(counts[name] / total, 4) if total else 0.0

    return {
        "total": total,
        "strict_format_rate": rate("strict_format"),
        "sql_extraction_rate": rate("sql_extracted"),
        "executable_rate": rate("executable"),
        "execution_accuracy": rate("execution_correct"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "examples_per_second": round(total / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "by_family": {
            family: {
                "count": family_totals[family],
                "correct": family_correct[family],
                "execution_accuracy": round(family_correct[family] / family_totals[family], 4),
            }
            for family in sorted(family_totals)
        },
        "top_errors": dict(error_types.most_common(20)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch, model, tokenizer = _load_runtime(args)
    prompt_path = args.data_dir / f"eval_{args.split}_prompts.jsonl"
    canonical_path = args.data_dir / f"canonical_{args.split}.jsonl"
    prompt_records = _read_jsonl(prompt_path)
    canonical = {record["id"]: record for record in _read_jsonl(canonical_path)}
    if args.limit:
        prompt_records = prompt_records[:args.limit]

    sandbox = SQLSandbox(args.database_path, row_limit=100, timeout_seconds=3.0)
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_batches = (len(prompt_records) + args.batch_size - 1) // args.batch_size

    for batch_index, batch in enumerate(_batches(prompt_records, args.batch_size), start=1):
        rendered = [
            tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
            for record in batch
        ]
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )
        input_width = inputs["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
        for source, raw_output in zip(batch, outputs):
            gold = canonical[source["id"]]
            result = evaluate_sql_text(raw_output, gold["result_hash"], sandbox)
            predictions.append(
                {
                    "id": source["id"],
                    "family_id": source["family_id"],
                    "difficulty": source["difficulty"],
                    "question": source["messages"][-1]["content"].rsplit("业务问题：", 1)[-1],
                    "raw_output": raw_output,
                    "reference_sql": gold["sql"],
                    "expected_result_hash": gold["result_hash"],
                    **result,
                }
            )
        if batch_index == 1 or batch_index % args.log_interval == 0 or batch_index == total_batches:
            correct = sum(int(item["execution_correct"]) for item in predictions)
            executable = sum(int(item["executable"]) for item in predictions)
            print(
                f"batch {batch_index}/{total_batches} | examples {len(predictions)}/{len(prompt_records)} "
                f"| executable {executable / len(predictions):.2%} "
                f"| execution_accuracy {correct / len(predictions):.2%}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    summary = _summarize(predictions, elapsed)
    _write_jsonl(args.output_path, predictions)
    summary_path = args.output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"predictions: {args.output_path}")
    print(f"summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 MiniMind Text-to-SQL 权重")
    parser.add_argument("--weight-path", type=Path, default=REPOSITORY_ROOT / "out" / "full_sft_768.pth")
    parser.add_argument("--tokenizer-path", type=Path, default=REPOSITORY_ROOT / "model")
    parser.add_argument("--database-path", type=Path, default=PACKAGE_ROOT / "data" / "generated" / "ecommerce.db")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output-path", type=Path, default=REPOSITORY_ROOT / "logs" / "sql_baseline_dev.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate the complete split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0 or args.max_input_length < 1 or args.max_new_tokens < 1:
        parser.error("batch size and sequence limits must be positive; limit cannot be negative")
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
