# Tinker GRPO training

GRPO RLVR training of `Qwen/Qwen3-8B` (LoRA) on the VerIH dataset via the [Tinker](https://tinker-docs.thinkingmachines.ai) API, then export the result toa merged Hugging Face model. Reward + advantage logic is reused from verl so runs match the verl reference command.

## Files

| File | Role |
|------|------|
| `tinker_train_v2.py` | GRPO training loop (sample → score → advantage → PPO update). |
| `tinker_to_hf.py`    | Download a Tinker LoRA checkpoint, merge into base weights, export/push HF model. |
| `verl_rewards.py`    | Loads verl's `ih.compute_score` + `compute_grpo_outcome_advantage` without importing verl/ray. |
| `smoke_rewards.csv`  | Per-response reward detail written by `--smoke`. |

## Environment

```bash
conda create -n ih_tinker python=3.12
conda activate ih_tinker

pip install tinker tinker_cookbook peft wandb
```

## Setup

```bash
export TINKER_API_KEY=...        # required for both scripts
hf auth login                    # only for tinker_to_hf.py --push_to_hub
```

Dataset lives at `dataset/verih/{train,test}.json` (already committed).

## Training — `tinker_train_v2.py`

```bash
python tinker_train_v2.py --smoke      # 2 real steps, no val/save, writes smoke_rewards.csv
python tinker_train_v2.py              # full run (logs to wandb project "VerIH")
```

Resume from a checkpoint (weights + Adam state):

```bash
python tinker_train_v2.py \
  --resume tinker://.../weights/<name> \
  --resume_step 200
```


Key config (top of the file, mirrors the verl command):

- `BATCH_SIZE=128`, `GROUP_SIZE=4`, `TOTAL_EPOCHS=12`
- `LORA_RANK=64`, `LEARNING_RATE=1e-5` (raised from verl's 1e-6 because LoRA)
- `MAX_PROMPT_TOKENS=1024`, `MAX_RESPONSE_TOKENS=2048`
- PPO clip 0.8–1.2; `SAVE_FREQ=200`, `TEST_FREQ=100`

Notes worth knowing:

- **Length normalization** (`build_datums`): per-token advantage is divided by
  response length because Tinker's PPO loss *sums* over tokens while verl
  averages. Without it the effective LR scales with response length.
- Full runs save `<EXPERIMENT_NAME>-final` training state and sampler
  weights at the end; `--smoke` saves nothing.

`--smoke` writes per-response detail (step, prompt, reward, advantage, format,
answer) to `smoke_rewards.csv`.

## Export to HF — `tinker_to_hf.py`

Download a Tinker LoRA checkpoint, merge it into the base model, and optionally
push to the Hub.

Sampler-weights checkpoint → merged model:

```bash
python tinker_to_hf.py \
  --checkpoint tinker://.../sampler_weights/<name> \
  --output ./hf_export/merged
```

Training-state checkpoint (`.../weights/<name>`) — add `--from_weights` so it
first resumes a training client and creates a matching sampler-weights snapshot:

```bash
python tinker_to_hf.py --from_weights \
  --checkpoint tinker://.../weights/<EXPERIMENT_NAME>-final \
  --output ./hf_export/merged
```

Push to the Hub (private by default):

```bash
python tinker_to_hf.py --from_weights \
  --checkpoint tinker://.../weights/<name> \
  --push_to_hub your-username/qwen3-8b-verih   # add --public to make it public
```

## Load the merged model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./hf_export/merged")
tok   = AutoTokenizer.from_pretrained("./hf_export/merged")
```
