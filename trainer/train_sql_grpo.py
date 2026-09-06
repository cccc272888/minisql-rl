"""Execution-guided GRPO for MiniSQL-RL.

Unlike the generic GRPO trainer, this script does not use a learned reward
model. It executes sampled SELECT/WITH statements in the read-only SQLite
sandbox and rewards result equality with the canonical answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from minisql_rl.database import SQLSandbox  # noqa: E402
from minisql_rl.training.sql_reward import (  # noqa: E402
    SQLRewardConfig,
    SQLRewardOutcome,
    score_sql_response,
    summarize_reward_outcomes,
)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.rollout_engine import TorchRolloutEngine  # noqa: E402
from trainer.trainer_utils import setup_seed  # noqa: E402


DEFAULT_TRAIN_DATA = (
    REPOSITORY_ROOT / "minisql_rl" / "data" / "generated" / "training" / "sql_rl_train.jsonl"
)
DEFAULT_DATABASE = REPOSITORY_ROOT / "minisql_rl" / "data" / "generated" / "ecommerce.db"


class SQLRLDataset(Dataset):
    """Render SQL RL prompts and keep only non-textual execution labels."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        max_samples: int = 0,
        seed: int = 42,
        families: set[str] | None = None,
        difficulties: set[str] | None = None,
    ) -> None:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if families and record["family_id"] not in families:
                    continue
                if difficulties and record["difficulty"] not in difficulties:
                    continue
                records.append(record)

        if max_samples and len(records) > max_samples:
            records = random.Random(seed).sample(records, max_samples)
        if not records:
            raise ValueError("no SQL RL records remain after filtering")

        self.samples = []
        for record in records:
            prompt = tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
            self.samples.append(
                {
                    "prompt": prompt,
                    "expected_result_hash": record["expected_result_hash"],
                    "sample_id": record["sample_id"],
                    "family_id": record["family_id"],
                    "difficulty": record["difficulty"],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.samples[index]


def _load_policy(args: argparse.Namespace, *, frozen: bool = False) -> MiniMindForCausalLM:
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(config)
    state = torch.load(args.weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if frozen:
        model = model.to(device=args.device, dtype=args.torch_dtype).eval().requires_grad_(False)
    else:
        model = model.to(device=args.device).train()
    return model


def _tokenize_prompts(
    prompts: list[str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=False,
    )
    return {name: value.to(args.device) for name, value in encoded.items()}


def _score_rollout(
    completions: list[str],
    expected_hashes: list[str],
    sandbox: SQLSandbox,
    reward_config: SQLRewardConfig,
    num_generations: int,
) -> tuple[list[SQLRewardOutcome], torch.Tensor]:
    expected_count = len(expected_hashes) * num_generations
    if len(completions) != expected_count:
        raise ValueError(
            f"rollout returned {len(completions)} completions; expected {expected_count}"
        )
    repeated_hashes = [
        expected_hash
        for expected_hash in expected_hashes
        for _ in range(num_generations)
    ]
    outcomes = [
        score_sql_response(response, expected_hash, sandbox, reward_config)
        for response, expected_hash in zip(completions, repeated_hashes)
    ]
    rewards = torch.tensor([outcome.reward for outcome in outcomes], dtype=torch.float32)
    return outcomes, rewards


def _completion_mask(completion_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    mask = completion_ids.ne(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is None:
        return mask
    is_eos = completion_ids.eq(tokenizer.eos_token_id) & mask
    positions = torch.arange(completion_ids.size(1), device=completion_ids.device)
    first_eos = torch.full(
        (completion_ids.size(0),),
        completion_ids.size(1) - 1,
        dtype=torch.long,
        device=completion_ids.device,
    )
    has_eos = is_eos.any(dim=1)
    first_eos[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    return mask & positions.unsqueeze(0).le(first_eos.unsqueeze(1))


def _completion_logps(
    model: MiniMindForCausalLM,
    output_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = model(
        output_ids,
        attention_mask=attention_mask,
        logits_to_keep=completion_length + 1,
    )
    logits = result.logits[:, :-1, :]
    targets = output_ids[:, -completion_length:]
    logps = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return logps, result.aux_loss


def _save_training_state(
    args: argparse.Namespace,
    model: MiniMindForCausalLM,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    *,
    next_step: int,
    update_step: int,
) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().half().cpu() for key, value in model.state_dict().items()}
    temporary_output = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    torch.save(state, temporary_output)
    os.replace(temporary_output, args.output_path)

    args.resume_path.parent.mkdir(parents=True, exist_ok=True)
    resume = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "next_step": next_step,
        "update_step": update_step,
    }
    temporary_resume = args.resume_path.with_suffix(args.resume_path.suffix + ".tmp")
    torch.save(resume, temporary_resume)
    os.replace(temporary_resume, args.resume_path)


def _write_audit(path: Path, summary: dict[str, Any], details: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail_path = path.with_suffix(".jsonl")
    with detail_path.open("w", encoding="utf-8") as handle:
        for record in details:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_reward_audit(
    args: argparse.Namespace,
    loader: DataLoader,
    rollout_engine: TorchRolloutEngine,
    tokenizer: Any,
    sandbox: SQLSandbox,
    reward_config: SQLRewardConfig,
) -> dict[str, Any]:
    outcomes: list[SQLRewardOutcome] = []
    details: list[dict[str, Any]] = []
    steps = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        prompts = list(batch["prompt"])
        inputs = _tokenize_prompts(prompts, tokenizer, args)
        rollout = rollout_engine.rollout(
            inputs["input_ids"],
            inputs["attention_mask"],
            args.num_generations,
            args.max_new_tokens,
            args.temperature,
        )
        batch_outcomes, _ = _score_rollout(
            rollout.completions,
            list(batch["expected_result_hash"]),
            sandbox,
            reward_config,
            args.num_generations,
        )
        outcomes.extend(batch_outcomes)
        for index, outcome in enumerate(batch_outcomes):
            source_index = index // args.num_generations
            details.append(
                {
                    "sample_id": batch["sample_id"][source_index],
                    "family_id": batch["family_id"][source_index],
                    "generation": index % args.num_generations,
                    "response": rollout.completions[index],
                    **outcome.to_dict(),
                }
            )
        if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == steps:
            partial = summarize_reward_outcomes(outcomes, num_generations=args.num_generations)
            print(
                f"audit {step + 1}/{steps} | accuracy {partial['execution_accuracy']:.2%} "
                f"| active_groups {partial['active_group_rate']:.2%}",
                flush=True,
            )

    summary = summarize_reward_outcomes(outcomes, num_generations=args.num_generations)
    summary.update(
        {
            "temperature": args.temperature,
            "num_generations": args.num_generations,
            "weight_path": str(args.weight_path),
        }
    )
    _write_audit(args.audit_output, summary, details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"audit summary: {args.audit_output}")
    print(f"audit details: {args.audit_output.with_suffix('.jsonl')}")
    return summary


def train(
    args: argparse.Namespace,
    loader: DataLoader,
    model: MiniMindForCausalLM,
    ref_model: MiniMindForCausalLM,
    rollout_engine: TorchRolloutEngine,
    tokenizer: Any,
    sandbox: SQLSandbox,
    reward_config: SQLRewardConfig,
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    optimizer_steps = max(1, math.ceil(steps / args.accumulation_steps))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=optimizer_steps,
        eta_min=args.learning_rate / 10,
    )
    start_step = 0
    update_step = 0
    if args.resume and args.resume_path.exists():
        checkpoint = torch.load(args.resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["next_step"])
        update_step = int(checkpoint.get("update_step", 0))
        print(f"resumed from rollout step {start_step}", flush=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=args.torch_dtype)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    pending_backward = 0
    total_active_groups = 0
    total_groups = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if step < start_step:
            continue
        if step >= steps:
            break
        prompts = list(batch["prompt"])
        inputs = _tokenize_prompts(prompts, tokenizer, args)
        rollout = rollout_engine.rollout(
            inputs["input_ids"],
            inputs["attention_mask"],
            args.num_generations,
            args.max_new_tokens,
            args.temperature,
        )
        outcomes, rewards_cpu = _score_rollout(
            rollout.completions,
            list(batch["expected_result_hash"]),
            sandbox,
            reward_config,
            args.num_generations,
        )
        rewards = rewards_cpu.to(args.device)
        grouped = rewards.view(-1, args.num_generations)
        group_std = grouped.std(dim=1, unbiased=False)
        active_group_mask = group_std.gt(1e-6)
        active_row_mask = active_group_mask.repeat_interleave(args.num_generations)
        active_groups = int(active_group_mask.sum().item())
        total_active_groups += active_groups
        total_groups += len(prompts)

        did_backward = False
        policy_loss_value = 0.0
        kl_value = 0.0
        if active_groups:
            group_mean = grouped.mean(dim=1, keepdim=True)
            advantages = ((grouped - group_mean) / (group_std.unsqueeze(1) + 1e-4)).reshape(-1)
            advantages = advantages[active_row_mask]
            output_ids = rollout.output_ids[active_row_mask]
            completion_ids = rollout.completion_ids[active_row_mask]
            old_logps = rollout.per_token_logps[active_row_mask].to(args.device).detach()
            attention_mask = output_ids.ne(tokenizer.pad_token_id).long()
            completion_mask = _completion_mask(completion_ids, tokenizer)

            with autocast_context:
                current_logps, aux_loss = _completion_logps(
                    model,
                    output_ids,
                    attention_mask,
                    completion_ids.size(1),
                )
                with torch.no_grad():
                    reference_logps, _ = _completion_logps(
                        ref_model,
                        output_ids,
                        attention_mask,
                        completion_ids.size(1),
                    )

                log_ratio = current_logps - old_logps
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
                surrogate = torch.minimum(
                    ratio * advantages.unsqueeze(1),
                    clipped_ratio * advantages.unsqueeze(1),
                )
                ref_delta = reference_logps - current_logps
                per_token_kl = torch.exp(ref_delta) - ref_delta - 1
                per_token_loss = -(surrogate - args.beta * per_token_kl)
                mask_float = completion_mask.to(per_token_loss.dtype)
                policy_loss = (
                    (per_token_loss * mask_float).sum(dim=1)
                    / mask_float.sum(dim=1).clamp(min=1)
                ).mean()
                loss = (policy_loss + aux_loss) / args.accumulation_steps

            loss.backward()
            pending_backward += 1
            did_backward = True
            policy_loss_value = float(policy_loss.detach().cpu())
            kl_value = float(
                ((per_token_kl * mask_float).sum() / mask_float.sum().clamp(min=1)).detach().cpu()
            )

            if pending_backward == args.accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_backward = 0
                update_step += 1

        if step == start_step or (step + 1) % args.log_interval == 0 or step + 1 == steps:
            batch_summary = summarize_reward_outcomes(
                outcomes,
                num_generations=args.num_generations,
            )
            print(
                f"step {step + 1}/{steps} | reward {batch_summary['mean_reward']:.4f} "
                f"| accuracy {batch_summary['execution_accuracy']:.2%} "
                f"| active {active_groups}/{len(prompts)} "
                f"| loss {policy_loss_value:.6f} | kl {kl_value:.6f} "
                f"| lr {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if args.debug and did_backward:
            for index, outcome in enumerate(outcomes):
                print(
                    f"[debug] family={batch['family_id'][index // args.num_generations]} "
                    f"generation={index % args.num_generations} reward={outcome.reward:.2f}\n"
                    f"{rollout.completions[index]}",
                    flush=True,
                )

        if (step + 1) % args.save_interval == 0 and pending_backward == 0:
            _save_training_state(
                args,
                model,
                optimizer,
                scheduler,
                next_step=step + 1,
                update_step=update_step,
            )

    if pending_backward:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
    _save_training_state(
        args,
        model,
        optimizer,
        scheduler,
        next_step=steps,
        update_step=update_step,
    )
    print(
        f"training complete | rollout_steps={steps - start_step} | updates={update_step} "
        f"| active_group_rate={total_active_groups / max(total_groups, 1):.2%} "
        f"| output={args.output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniSQL-RL execution-guided GRPO")
    parser.add_argument("--weight-path", type=Path, default=REPOSITORY_ROOT / "out" / "sql_sft_768.pth")
    parser.add_argument("--tokenizer-path", type=Path, default=REPOSITORY_ROOT / "model")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-path", type=Path, default=REPOSITORY_ROOT / "out" / "sql_grpo_768.pth")
    parser.add_argument(
        "--resume-path",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints" / "sql_grpo_768_resume.pth",
    )
    parser.add_argument("--audit-output", type=Path, default=REPOSITORY_ROOT / "logs" / "sql_reward_audit.json")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses all selected samples")
    parser.add_argument("--families", nargs="*", default=[])
    parser.add_argument("--difficulties", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    positive = {
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "accumulation_steps": args.accumulation_steps,
        "save_interval": args.save_interval,
    }
    if any(value < 1 for value in positive.values()):
        parser.error(f"these arguments must be positive: {positive}")
    if args.num_generations < 2:
        parser.error("--num-generations must be at least 2 for group-relative advantages")
    if args.max_samples < 0 or args.max_steps < 0:
        parser.error("--max-samples and --max-steps cannot be negative")
    if args.temperature <= 0 or args.learning_rate <= 0:
        parser.error("--temperature and --learning-rate must be positive")
    if not 0 <= args.beta or not 0 < args.epsilon < 1:
        parser.error("--beta must be non-negative and --epsilon must be between 0 and 1")
    args.torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    return args


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "left"
    dataset = SQLRLDataset(
        args.data_path,
        tokenizer,
        max_samples=args.max_samples,
        seed=args.seed,
        families=set(args.families),
        difficulties=set(args.difficulties),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.audit_only,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    sandbox = SQLSandbox(args.database_path, row_limit=100, timeout_seconds=3.0)
    reward_config = SQLRewardConfig()
    model = _load_policy(args)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=args.torch_dtype)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    rollout_engine = TorchRolloutEngine(model, tokenizer, args.device, autocast_context)

    if args.audit_only:
        model.eval()
        run_reward_audit(args, loader, rollout_engine, tokenizer, sandbox, reward_config)
        return

    ref_model = _load_policy(args, frozen=True)
    train(
        args,
        loader,
        model,
        ref_model,
        rollout_engine,
        tokenizer,
        sandbox,
        reward_config,
    )


if __name__ == "__main__":
    main()
