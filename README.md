# QwenImage Inference on TPU with PyTorch/XLA

Run [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) inference on TPU v6e using PyTorch/XLA SPMD. We 
take advantage of Diffusers' compatibility with PyTorch/XLA so the heavylifting of model implementation is
already offloaded to it.

## Setup

Follow the [SETUP.md](./SETUP.md) guide.

## Usage

```bash
python qwen_inference.py
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--width` | 1024 | Image width |
| `--height` | 1024 | Image height |
| `--num-inference-steps` | 50 | Denoising steps |
| `--guidance-scale` | 4.5 | True CFG scale (`true_cfg_scale` in the pipeline) |
| `--seed` | None | RNG seed |
| `--itters` | 15 | Inference iterations for benchmarking |
| `--profile` | off | Enable XLA profiler |
| `--profile-duration` | 10000 | Profiling duration (ms) |

## How it works

### SPMD sharding

The QwenImage DiT cannot fully fit on a single chip. That is why we need some kind of sharding. The script uses single-process SPMD instead of multi-process `xmp.spawn`. A 2D mesh `(num_devices // 4, 4)` is created with `("data", "model")` axes. Transformer parameters with `dim >= 2` are sharded along their largest dimension on the `"model"` axis. This follows the approach from [diffusers#13474](https://github.com/huggingface/diffusers/pull/13474/).

Since there are model-level components in the QwenImage pipeline, we had to implement sharding. Other alternative
was to do some kind of offloading between runs but that would have terribly slowed down execution.

### Flash attention

The existing `QwenDoubleStreamAttnProcessor2_0` (joint attention for image + text streams) is used with `_attention_backend = "_native_xla"` to dispatch to the XLA Pallas flash attention kernel. The generic `XLAFlashAttnProcessor2_0` cannot be used here because it doesn't handle the double-stream return format.

### Text encoding

Text encoding runs on CPU via a separate pipeline instance (loaded without transformer/VAE). The resulting prompt embeddings are moved to the XLA device before the denoising loop.

### True CFG with negative prompt

A negative prompt of `" "` (single space) is used for classifier-free guidance. There is a known issue where `encode_prompt` drops `prompt_embeds_mask` to `None` when all values are `True` (common for short prompts), which silently disables CFG. The script works around this by recreating the mask after encoding.

## Required diffusers patch

The QwenImage VAE uses `CACHE_T = 2` for temporal caching, leading to `x[:, :, -2:, :, :]` slices. When the temporal dimension is 1 (single image), XLA raises a `RuntimeError` because `-2` is out of bounds for a size-1 dimension (CPU/CUDA silently clamp this). The fix in `autoencoder_kl_qwenimage.py`:

```diff
- cache_x = x[:, :, -CACHE_T:, :, :].clone()
+ cache_x = x[:, :, -min(CACHE_T, x.shape[2]):, :, :].clone()
```

This is a no-op on CPU/CUDA. PR is available [here](https://github.com/huggingface/diffusers/pull/13480).

## Benchmark results (TPU v6e-8)

| | Time |
|---|---|
| Compilation (first run, 50 steps) | ~95 min |
| Inference (50 steps, steady state) | ~20 sec |

Compiled graphs are cached at `/tmp/data/compiler_cache_tRiLlium_eXp` and reused across runs.

Example output can be found [here](./inference_out.png).

## Profiling

Refer to [PROFILE.md](./PROFILE.md).

## Acknowledgements

Thanks to Claude Code (for OSS) for pairing.

Thanks to the TRC program for providing TPUs.

Thanks to the Google ML Developer Program team for providing Google Cloud credits.
