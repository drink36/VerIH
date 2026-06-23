"""GRPO training on Tinker — No KL 
    Need set TINKER_API_KEY at first, then run:
    Test - python tinker_train_v2.py --smoke
    Actual training - python tinker_train_v2.py
"""
import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path

logging.getLogger("tinker.lib.telemetry").setLevel(logging.ERROR)

import tinker
import torch
import wandb
from tinker import TensorData
from tinker_cookbook.renderers import get_renderer, get_text_content

from verl_rewards import ih_compute_score, check_format, compute_grpo_outcome_advantage

# ---------------------------------------------------------------------------
# Config — mirrors the verl training command exactly where possible
# ---------------------------------------------------------------------------
BASE_MODEL          = "Qwen/Qwen3-8B"
LORA_RANK           = 64

GROUP_SIZE          = 4        # actor_rollout_ref.rollout.n=4
BATCH_SIZE          = 128      # data.train_batch_size=128
PPO_MINI_BATCH_SIZE = 64       # actor_rollout_ref.actor.ppo_mini_batch_size=64
TOTAL_EPOCHS        = 12       # trainer.total_epochs=12

MAX_PROMPT_TOKENS   = 1024     # data.max_prompt_length=1024
MAX_RESPONSE_TOKENS = 2048     # data.max_response_length=2048

LEARNING_RATE       = 1e-5     # actor_rollout_ref.actor.optim.lr # since it's lora, I change it from 1e-6 to 1e-5
ADAM_BETA1          = 0.9
ADAM_BETA2          = 0.999
WEIGHT_DECAY        = 0.01     # actor_rollout_ref.actor.optim.weight_decay=0.01
GRAD_CLIP_NORM      = 1.0      # actor_rollout_ref.actor.optim.clip_grad=1.0

PPO_CLIP_LOW        = 0.8      # 1 - clip_ratio(0.2)
PPO_CLIP_HIGH       = 1.2      # 1 + clip_ratio(0.2)

SAVE_FREQ           = 200      # trainer.save_freq=200
TEST_FREQ           = 100      # trainer.test_freq=100

EXPERIMENT_NAME = "verih-tinker-lr1e5-lennorm"
DATA_SEED       = 1            # verl torch.Generator seed

TRAIN_PATH = Path(__file__).parent / "dataset/verih/train.json"
TEST_PATH  = Path(__file__).parent / "dataset/verih/test.json"

_COT_SYSHINT = (
    "Follow the user prompt within the bounds of the system prompt. Think step by step "
    "about the relationship between the system prompt and the user prompt. If there is a "
    "conflict, the system prompt takes precedence. Here is the system prompt:\n"
)


# ---------------------------------------------------------------------------
# Prompts & dataset
# ---------------------------------------------------------------------------

def build_messages(dp: dict) -> list[dict]:
    return [
        {"role": "system", "content": _COT_SYSHINT + dp["sys_prompt"]},
        {"role": "user",   "content": dp["user_prompt"]},
    ]


def filter_overlong(data: list[dict], renderer, max_tokens: int) -> list[dict]:
    """Mirror data.filter_overlong_prompts=True."""
    kept, dropped = [], 0
    for dp in data:
        prompt = renderer.build_generation_prompt(build_messages(dp))
        if prompt.length <= max_tokens:
            kept.append(dp)
        else:
            dropped += 1
    if dropped:
        print(f"Filtered {dropped} overlong prompts (>{max_tokens} tokens), {len(kept)} remain")
    return kept


def batch_stream(data: list[dict], batch_size: int, seed: int):
    """Yield batches the way verl's RandomSampler(drop_last=True) would.

    Each epoch is a fresh permutation from one persistent generator; the partial
    tail (< batch_size) is dropped, then the next epoch reshuffles. Resuming is
    just consuming N batches off this stream — no separate fast-forward logic.
    """
    gen = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randperm(len(data), generator=gen).tolist()
        for off in range(0, len(data) - batch_size + 1, batch_size):
            yield [data[i] for i in idx[off:off + batch_size]]


# ---------------------------------------------------------------------------
# Scoring & advantages
# ---------------------------------------------------------------------------

def score_response(raw_text: str, dp: dict) -> float:
    """Score via verl's ih.compute_score.

    verl expects solution_str = prompt_str + response_str; we only have the
    response, so we prepend the assistant chat marker so extract_solution slices
    it back to raw_text, then compute_score runs check_format + check_answer.
    """
    solution_str = "<|im_start|>assistant\n" + raw_text
    ground_truth = json.loads(dp["gt"])
    extra_info   = {"type": dp["type"]}
    return float(ih_compute_score(solution_str, ground_truth, extra_info))


def grpo_advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    """Wrap compute_grpo_outcome_advantage for one prompt's group of responses.

    verl wants token_level_rewards (bs, response_length) + a grouping index; we
    pack the group as (n, 1) tensors with all-same index 0.
    """
    import numpy as np
    n = len(rewards)
    token_level_rewards = torch.zeros(n, 1)
    token_level_rewards[:, 0] = torch.tensor(rewards)
    response_mask = torch.ones(n, 1)
    index = np.zeros(n, dtype=np.int64)
    advs, _ = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index,
        epsilon=epsilon, norm_adv_by_std_in_grpo=True,
    )
    return advs[:, 0].tolist()


def build_datums(prompt, sample_result, advantages: list[float]) -> list[tinker.Datum]:
    # ob_len excludes the last prompt token (1-token overlap with response start)
    ob_len = prompt.length - 1
    datums = []
    for seq, adv in zip(sample_result.sequences, advantages):
        tokens, logprobs = seq.tokens, seq.logprobs
        model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
        resp_len    = model_input.length - ob_len

        # Length-normalize the per-token advantage. Tinker's PPO loss SUMS the
        # token-level loss over the sequence, so without dividing by resp_len the
        # per-sequence gradient scales with length (length bias) and the effective
        # LR is O(resp_len) too large vs verl, which averages over response tokens.
        # Dividing makes sum_over_tokens equal the sequence-level advantage.
        adv_per_token = adv / resp_len if resp_len > 0 else 0.0
        datums.append(tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor([0] * ob_len + tokens)),
                "logprobs":      TensorData.from_torch(torch.tensor([0.0] * ob_len + logprobs)),
                "advantages":    TensorData.from_torch(
                    torch.tensor([0.0] * ob_len + [adv_per_token] * resp_len)),
                # mask: 0 on prompt/observation tokens, 1 on response tokens.
                "mask":          TensorData.from_torch(
                    torch.tensor([0.0] * ob_len + [1.0] * resp_len)),
            },
        ))
    return datums


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def _write_smoke_detail(writer, step, prob_idx, dp, parsed_list, raw_list,
                        rewards, advantages, renderer):
    for i, (parsed, raw, r, adv) in enumerate(zip(parsed_list, raw_list, rewards, advantages)):
        fmt_ok = check_format(raw)
        ans = raw.split("</think>")[-1].strip() if "</think>" in raw \
            else get_text_content(parsed).strip()
        writer.writerow([
            step, prob_idx, dp["type"], dp["sys_prompt"], dp["user_prompt"], dp["gt"],
            i, f"{r:.0f}", f"{adv:.6f}", "ok" if fmt_ok else "fail", len(ans), ans,
        ])


async def train_step(step, batch, *, training_client, renderer, tokenizer,
                     sampling_params, adam_params, smoke_writer=None) -> dict:
    """Run one GRPO step (sample -> score -> advantage -> PPO update). Returns metrics."""
    # 1. Snapshot weights -> sampling client
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # 2. Sample GROUP_SIZE completions per problem
    prompts = [renderer.build_generation_prompt(build_messages(dp)) for dp in batch]
    sample_results = await asyncio.gather(*[
        sampling_client.sample_async(prompt=p, num_samples=GROUP_SIZE,
                                     sampling_params=sampling_params)
        for p in prompts
    ])

    # 3. Score, compute GRPO advantages, build Datums
    all_datums, step_rewards, mixed_rewards = [], [], []
    n_degenerate = n_all_pass = n_all_fail = n_mixed = 0

    for prob_idx, (dp, prompt, result) in enumerate(zip(batch, prompts, sample_results)):
        parsed_list, raw_list, rewards = [], [], []
        for seq in result.sequences:
            parsed, _ = renderer.parse_response(seq.tokens)
            raw_text  = tokenizer.decode(seq.tokens)
            parsed_list.append(parsed)
            raw_list.append(raw_text)
            rewards.append(score_response(raw_text, dp))

        mean_r = sum(rewards) / len(rewards)
        step_rewards.append(mean_r)
        advantages = grpo_advantages(rewards)

        if smoke_writer is not None:
            _write_smoke_detail(smoke_writer, step, prob_idx, dp, parsed_list,
                                raw_list, rewards, advantages, renderer)

        # Classify the group by reward shape
        if all(r == 1.0 for r in rewards):
            n_all_pass += 1
        elif all(r == 0.0 for r in rewards):
            n_all_fail += 1
        else:
            n_mixed += 1
            mixed_rewards.append(mean_r)

        # verl keeps degenerate (advantage=0) groups so the mini-batch split still
        # yields 2 optim steps per rollout (ppo_mini_batch=64, train_batch=128).
        if all(a == 0.0 for a in advantages):
            n_degenerate += 1
        all_datums.extend(build_datums(prompt, result, advantages))

    # 4. PPO clip update — 2 mini-batch steps. Strip "mask" (PPO loss ignores it).
    if all_datums:
        mini_size = PPO_MINI_BATCH_SIZE * GROUP_SIZE
        for i in range(0, len(all_datums), mini_size):
            mini = [
                tinker.Datum(
                    model_input=d.model_input,
                    loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
                )
                for d in all_datums[i:i + mini_size]
            ]
            fwd = await training_client.forward_backward_async(
                mini, loss_fn="ppo",
                loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW,
                                "clip_high_threshold": PPO_CLIP_HIGH},
            )
            opt = await training_client.optim_step_async(adam_params)
            await fwd.result_async()
            await opt.result_async()

    return {
        "mean_reward":       sum(step_rewards) / len(step_rewards),
        # Mean reward over mixed groups only (the ones giving gradient signal); 0 if none.
        "reward_mixed_only": sum(mixed_rewards) / len(mixed_rewards) if mixed_rewards else 0.0,
        "degenerate_frac":   n_degenerate / len(batch),
        "n_degenerate":      n_degenerate,
        "n_datums":          len(all_datums),
        "n_all_pass":        n_all_pass,
        "n_all_fail":        n_all_fail,
        "n_mixed":           n_mixed,
        "batch_size":        len(batch),
    }


# ---------------------------------------------------------------------------
# Validation — mirrors trainer.test_freq (whole val set, greedy decode)
# ---------------------------------------------------------------------------

async def validate(training_client, renderer, tokenizer, test_data, sampling_params,
                   step: int, use_wandb: bool = False) -> float:
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    prompts = [renderer.build_generation_prompt(build_messages(dp)) for dp in test_data]
    results = await asyncio.gather(*[
        sampling_client.sample_async(prompt=p, num_samples=1, sampling_params=sampling_params)
        for p in prompts
    ])

    rewards, rewards_by_type = [], {}
    for dp, result in zip(test_data, results):
        raw_text = tokenizer.decode(result.sequences[0].tokens)
        r = score_response(raw_text, dp)
        rewards.append(r)
        rewards_by_type.setdefault(dp["type"], []).append(r)

    mean_reward = sum(rewards) / len(rewards)
    print(f"[val] step {step:4d} | reward {mean_reward:.3f} ({sum(rewards):.0f}/{len(rewards)})")

    if use_wandb:
        log = {"val/reward": mean_reward, "val/n_correct": sum(rewards)}
        for t, rs in rewards_by_type.items():
            log[f"val/reward/{t}"] = sum(rs) / len(rs)
        wandb.log(log, step=step)
    return mean_reward


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

async def make_training_client(service_client, resume, resume_step):
    if resume:
        print(f"-- resuming from {resume} (step {resume_step}) --")
        return await service_client.create_training_client_from_state_with_optimizer_async(resume)
    return await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=LORA_RANK)


def open_smoke_csv():
    path = Path(__file__).parent / "smoke_rewards.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "step", "problem_idx", "type", "sys_prompt", "user_prompt", "gt",
        "response_idx", "reward", "advantage", "format", "answer_len", "answer",
    ])
    print(f"-- smoke mode: writing reward detail to {path} --")
    return f, writer


async def save_checkpoint(training_client, name, use_wandb, step, wandb_key):
    ckpt = await (await training_client.save_state_async(name)).result_async()
    print(f"[ckpt] saved: {ckpt.path}")
    if use_wandb:
        wandb.log({wandb_key: ckpt.path}, step=step)
    return ckpt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Two real training steps at full settings, no val/save, writes reward CSV")
    parser.add_argument("--resume", default=None,
                        help="Tinker checkpoint (tinker://.../weights/<name>) to resume from "
                             "(weights + Adam state).")
    parser.add_argument("--resume_step", type=int, default=0,
                        help="Step the resumed checkpoint corresponds to; training continues from here.")
    args = parser.parse_args()

    train_data = json.load(open(TRAIN_PATH, encoding="utf-8"))
    test_data  = json.load(open(TEST_PATH,  encoding="utf-8"))
    print(f"Dataset: {len(train_data)} train / {len(test_data)} test")
    if args.smoke:
        print(f"-- smoke mode: 2 real steps ({BATCH_SIZE} prompts x {GROUP_SIZE} samples, "
              f"max_resp={MAX_RESPONSE_TOKENS}) --")

    # --- Tinker setup ---
    service_client  = tinker.ServiceClient()
    training_client = await make_training_client(service_client, args.resume, args.resume_step)
    tokenizer = training_client.get_tokenizer()
    renderer  = get_renderer("qwen3", tokenizer)

    train_data = filter_overlong(train_data, renderer, MAX_PROMPT_TOKENS)
    print(f"Train after filtering: {len(train_data)}")

    sampling_params = tinker.SamplingParams(
        max_tokens=MAX_RESPONSE_TOKENS, stop=renderer.get_stop_sequences(),
        temperature=1.0, top_p=1.0)              # rollout temperature=1.0, top_p=1
    val_sampling_params = tinker.SamplingParams(
        max_tokens=MAX_RESPONSE_TOKENS, stop=renderer.get_stop_sequences(),
        temperature=0.0, top_p=1.0)              # val_kwargs greedy: temperature=0
    adam_params = tinker.AdamParams(
        learning_rate=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY, grad_clip_norm=GRAD_CLIP_NORM)

    # verl uses RandomSampler with drop_last=True -> floor(N/B) steps per epoch.
    steps_per_epoch = len(train_data) // BATCH_SIZE
    n_steps = 2 if args.smoke else TOTAL_EPOCHS * steps_per_epoch
    print(f"Total steps: {n_steps} ({TOTAL_EPOCHS} epochs x {len(train_data)} samples / "
          f"{BATCH_SIZE} batch)")

    use_wandb = not args.smoke
    if use_wandb:
        wandb.init(project="VerIH", name=EXPERIMENT_NAME, config={
            "base_model": BASE_MODEL, "lora_rank": LORA_RANK, "group_size": GROUP_SIZE,
            "batch_size": BATCH_SIZE, "total_epochs": TOTAL_EPOCHS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS, "max_response_tokens": MAX_RESPONSE_TOKENS,
            "learning_rate": LEARNING_RATE, "ppo_clip_low": PPO_CLIP_LOW,
            "ppo_clip_high": PPO_CLIP_HIGH, "train_samples": len(train_data),
        })

    # val_before_train=True (skip in smoke mode)
    if not args.smoke:
        print("\n[val] Running validation before training...")
        await validate(training_client, renderer, tokenizer, test_data,
                       val_sampling_params, step=0, use_wandb=use_wandb)

    smoke_csv, smoke_writer = (None, None)
    if args.smoke:
        smoke_csv, smoke_writer = open_smoke_csv()

    # One persistent batch stream; resuming = consume resume_step batches off it,
    # so the resumed run sees the same batch the original would have at start_step.
    stream = batch_stream(train_data, BATCH_SIZE, DATA_SEED)
    if args.resume_step > 0:
        for _ in range(args.resume_step):
            next(stream)
        print(f"[resume] fast-forwarded data sampler past {args.resume_step} batches")

    start_step = args.resume_step + 1
    end_step   = args.resume_step + n_steps   # inclusive

    for step in range(start_step, end_step + 1):
        batch   = next(stream)
        metrics = await train_step(
            step, batch, training_client=training_client, renderer=renderer,
            tokenizer=tokenizer, sampling_params=sampling_params,
            adam_params=adam_params, smoke_writer=smoke_writer,
        )
        if args.smoke:
            smoke_csv.flush()

        print(f"step {step:4d}/{end_step} | reward {metrics['mean_reward']:.3f} "
              f"(mixed_only {metrics['reward_mixed_only']:.3f}) "
              f"| all_pass {metrics['n_all_pass']} all_fail {metrics['n_all_fail']} "
              f"mixed {metrics['n_mixed']} "
              f"(degenerate {metrics['n_degenerate']}/{metrics['batch_size']}) "
              f"| datums {metrics['n_datums']}")

        if use_wandb:
            wandb.log({
                "train/reward":            metrics["mean_reward"],
                "train/reward_mixed_only": metrics["reward_mixed_only"],
                "train/degenerate_frac":   metrics["degenerate_frac"],
                "train/n_datums":          metrics["n_datums"],
                "train/n_all_pass":        metrics["n_all_pass"],
                "train/n_all_fail":        metrics["n_all_fail"],
                "train/n_mixed":           metrics["n_mixed"],
            }, step=step)

        if not args.smoke and step % TEST_FREQ == 0:
            await validate(training_client, renderer, tokenizer, test_data,
                           val_sampling_params, step=step, use_wandb=use_wandb)

        if not args.smoke and step % SAVE_FREQ == 0:
            await save_checkpoint(training_client, f"{EXPERIMENT_NAME}-step{step}",
                                  use_wandb, step, "checkpoint/path")

    if smoke_csv is not None:
        smoke_csv.close()
        print(f"[smoke] reward detail written to {smoke_csv.name}")

    # Final checkpoints (skip in smoke mode — leave nothing behind)
    if not args.smoke:
        ckpt = await (await training_client.save_state_async(
            f"{EXPERIMENT_NAME}-final")).result_async()
        print(f"[ckpt] training state: {ckpt.path}")
        sampler = await (await training_client.save_weights_for_sampler_async(
            f"{EXPERIMENT_NAME}-final")).result_async()
        print(f"[ckpt] sampler weights: {sampler.path}")
        if use_wandb:
            wandb.log({"checkpoint/training_state": ckpt.path,
                       "checkpoint/sampler": sampler.path})
            wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
