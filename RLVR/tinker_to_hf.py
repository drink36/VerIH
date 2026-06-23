from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path



def weights_to_sampler_weights(weights_url: str) -> str:
    import asyncio
    import time
    import tinker
    if "/weights/" not in weights_url:
        sys.exit(f"[from_weights] expected tinker://.../weights/<name>, got {weights_url}")
    base_name = weights_url.rsplit("/weights/", 1)[1]
    # Add a timestamp suffix so we never reuse a stuck/partial archive name.
    name = f"{base_name}-sw-{int(time.time())}"

    async def _convert() -> str:
        print(f"[from_weights] resuming training client from {weights_url} ...")
        sc = tinker.ServiceClient()
        tc = await sc.create_training_client_from_state_with_optimizer_async(weights_url)
        print(f"[from_weights] saving sampler_weights as {name!r} ...")
        fut = await tc.save_weights_for_sampler_async(name)
        response = await fut.result_async()
        sampler_url = response.path
        print(f"[from_weights] sampler_weights ready: {sampler_url}")
        return sampler_url

    return asyncio.run(_convert())


def download_checkpoint(checkpoint_path: str, download_root: Path,
                        max_attempts: int = 5) -> Path:
    """Download a Tinker checkpoint via the tinker CLI; return the adapter dir.

    Retries with backoff to tolerate the propagation lag that can occur right
    after `save_weights_for_sampler_async` — the future resolves locally but
    the server's download endpoint may briefly 404 the new checkpoint before
    it's indexed.
    """
    import time
    download_root.mkdir(parents=True, exist_ok=True)
    print(f"[download] {checkpoint_path}")
    print(f"[download] into {download_root}")

    before = set(p.name for p in download_root.iterdir()) if download_root.exists() else set()

    delay = 5
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            ["tinker", "checkpoint", "download", checkpoint_path,
             "--output", str(download_root), "--force"],
            check=False,
        )
        if result.returncode == 0:
            break
        if attempt == max_attempts:
            sys.exit(f"[download] tinker CLI failed (exit {result.returncode}) "
                     f"after {max_attempts} attempts. Is TINKER_API_KEY set "
                     f"and the path correct?")
        print(f"[download] attempt {attempt} failed; retrying in {delay}s "
              f"(checkpoint may still be propagating) ...")
        time.sleep(delay)
        delay *= 2

    # Find the newly-created adapter directory
    after = {p.name for p in download_root.iterdir()}
    new_dirs = [download_root / n for n in (after - before)
                if (download_root / n).is_dir()]
    if not new_dirs:
        # download --force may reuse an existing dir; fall back to any adapter dir
        new_dirs = [p for p in download_root.iterdir()
                    if p.is_dir() and (p / "adapter_config.json").exists()]
    if not new_dirs:
        sys.exit(f"[download] could not locate the extracted adapter dir under {download_root}")

    adapter_dir = new_dirs[0]
    print(f"[download] adapter dir: {adapter_dir}")
    return adapter_dir


def verify_adapter(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        sys.exit(f"[verify] no adapter_config.json in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    print(f"[verify] base_model = {cfg.get('base_model_name_or_path')}")
    print(f"[verify] rank (r)   = {cfg.get('r')}")
    print(f"[verify] lora_alpha = {cfg.get('lora_alpha')}")
    print(f"[verify] target_modules = {cfg.get('target_modules')}")
    has_weights = (adapter_dir / "adapter_model.safetensors").exists() or \
                  (adapter_dir / "adapter_model.bin").exists()
    if not has_weights:
        sys.exit(f"[verify] no adapter weights (.safetensors/.bin) in {adapter_dir}")
    return cfg

def install_canonical_tokenizer(base_model: str, output_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    files = ["tokenizer_config.json", "tokenizer.json", "vocab.json",
             "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    print(f"[normalize] installing canonical tokenizer from {base_model} ...")
    snap_dir = Path(snapshot_download(
        repo_id=base_model,
        allow_patterns=files,
    ))
    import shutil
    for fname in files:
        src = snap_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"[normalize]   copied {fname} ({src.stat().st_size} B)")


def merge(adapter_dir: Path, base_model: str, output_dir: Path, dtype: str,
          push_to_hub: str | None = None, private: bool = True,
          max_shard_size: str = "5GB") -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = {"bfloat16": torch.bfloat16,
                   "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    print(f"[merge] loading base model {base_model} ({dtype}) ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="cpu",
    )

    print(f"[merge] attaching adapter {adapter_dir} ...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[merge] merging LoRA into base weights ...")
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged model to {output_dir} (shard ≤ {max_shard_size}) ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.save_pretrained(str(output_dir), safe_serialization=True,
                          max_shard_size=max_shard_size)
    tokenizer.save_pretrained(str(output_dir))
    install_canonical_tokenizer(base_model, output_dir)
    print(f"[done] merged HF model at: {output_dir}")
    print(f"[done] load it with: AutoModelForCausalLM.from_pretrained('{output_dir}')")

    if push_to_hub:
        print(f"[publish] pushing merged model to https://huggingface.co/{push_to_hub} "
              f"({'private' if private else 'public'}) ...")
        model.push_to_hub(push_to_hub, private=private, max_shard_size=max_shard_size)
        tokenizer.push_to_hub(push_to_hub, private=private)
        from huggingface_hub import HfApi
        api = HfApi()
        for fname in ("tokenizer_config.json", "tokenizer.json", "vocab.json",
                      "merges.txt", "special_tokens_map.json", "added_tokens.json"):
            fpath = output_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=push_to_hub,
                    repo_type="model",
                )
        print(f"[publish] done: https://huggingface.co/{push_to_hub}")


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Tinker checkpoint path (tinker://.../weights/<name>)")
    src.add_argument("--adapter_dir", help="Already-downloaded PEFT adapter directory")

    parser.add_argument("--base_model", default="Qwen/Qwen3-8B",
                        help="Base model the LoRA was trained on (overridden by adapter_config if present)")
    parser.add_argument("--output", default="./hf_export/merged",
                        help="Where to write the merged HF model")
    parser.add_argument("--download_root", default="./hf_export",
                        help="Where to extract the downloaded adapter")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_shard_size", default="5GB",
                        help="Shard the merged safetensors into pieces of at most this size "
                             "(e.g. 5GB). Smaller shards = more resumable uploads.")
    parser.add_argument("--no_merge", action="store_true",
                        help="Only download + verify the adapter, skip merging")
    parser.add_argument("--push_to_hub", default=None,
                        help="HF repo id (e.g. your-username/qwen3-8b-verih) to publish the "
                             "merged model to. Requires `huggingface-cli login` first.")
    parser.add_argument("--public", action="store_true",
                        help="Make the pushed HF repo public (default: private).")
    parser.add_argument("--from_weights", action="store_true",
                        help="The --checkpoint URL is a training-state "
                             "(tinker://.../weights/<name>) snapshot, not a "
                             "sampler_weights snapshot. Resume a training client "
                             "from it and create a matching sampler_weights "
                             "checkpoint first, then download that.")
    args = parser.parse_args()

    # Stage 1: get the adapter dir
    if args.checkpoint:
        checkpoint = args.checkpoint
        if args.from_weights:
            checkpoint = weights_to_sampler_weights(checkpoint)
        adapter_dir = download_checkpoint(checkpoint, Path(args.download_root))
    else:
        adapter_dir = Path(args.adapter_dir)

    cfg = verify_adapter(adapter_dir)

    # Prefer the base model recorded in the adapter config
    base_model = cfg.get("base_model_name_or_path") or args.base_model

    if args.no_merge:
        print("\n[done] adapter ready (no merge requested).")
        print(f"[done] adapter dir: {adapter_dir}")
        print(f"[done] base model:  {base_model}")
        return

    # Stage 2: merge (and optionally publish to HF Hub)
    merge(adapter_dir, base_model, Path(args.output), args.dtype,
          push_to_hub=args.push_to_hub, private=not args.public,
          max_shard_size=args.max_shard_size)


if __name__ == "__main__":
    main()
