from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import structlog
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.custom_kernel import FlashAttention

from diffusers import QwenImagePipeline


logger = structlog.get_logger()
metrics_filepath = "/tmp/metrics_report.txt"

num_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(np.arange(num_devices), (num_devices // 4, 4), ("data", "model"))
xs.set_global_mesh(mesh)


def main(args, text_pipe, ckpt_id):
    cache_path = Path("/tmp/data/compiler_cache_tRiLlium_eXp")
    cache_path.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(cache_path), readonly=False)

    profile_path = Path("/tmp/data/profiler_out_tRiLlium_eXp")
    profile_path.mkdir(parents=True, exist_ok=True)
    profiler_port = 9012
    profile_duration = args.profile_duration
    if args.profile:
        logger.info(f"starting profiler on port {profiler_port}")
        _ = xp.start_server(profiler_port)
    device0 = xm.xla_device()

    logger.info(f"loading qwenimage from {ckpt_id}")
    pipe = QwenImagePipeline.from_pretrained(
        ckpt_id, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
    ).to(device0)

    # Shard transformer parameters across the "model" axis (FSDP-style)
    for name, param in pipe.transformer.named_parameters():
        if param.dim() >= 2:
            shard_dim = max(range(param.dim()), key=lambda i: param.shape[i])
            spec = [None] * param.dim()
            spec[shard_dim] = "model"
            xs.mark_sharding(param, mesh, tuple(spec))

    from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
    for module in pipe.transformer.modules():
        if hasattr(module, "processor") and isinstance(module.processor, QwenDoubleStreamAttnProcessor2_0):
            module.processor._attention_backend = "_native_xla"
    FlashAttention.DEFAULT_BLOCK_SIZES = {
        "block_q": 1536,
        "block_k_major": 1536,
        "block_k": 1536,
        "block_b": 1536,
        "block_q_major_dkv": 1536,
        "block_k_major_dkv": 1536,
        "block_q_dkv": 1536,
        "block_k_dkv": 1536,
        "block_q_dq": 1536,
        "block_k_dq": 1536,
        "block_k_major_dq": 1536,
    }

    prompt = "A cat holding a sign that says hello world"
    width = args.width
    height = args.height
    n_steps = args.num_inference_steps

    negative_prompt = " "

    logger.info("starting compilation run...")
    ts = perf_counter()
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = text_pipe.encode_prompt(
            prompt=prompt, max_sequence_length=512
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = text_pipe.encode_prompt(
            prompt=negative_prompt, max_sequence_length=512
        )
    # encode_prompt sets mask to None when all True; restore it so CFG activates
    if negative_prompt_embeds_mask is None:
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[:2], dtype=torch.bool
        )
    prompt_embeds = prompt_embeds.to(device0)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(device0)
    negative_prompt_embeds = negative_prompt_embeds.to(device0)
    negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device0)

    image = pipe(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        true_cfg_scale=args.guidance_scale,
        num_inference_steps=n_steps,
        height=height,
        width=width,
    ).images[0]
    logger.info(f"compilation took {perf_counter() - ts} sec.")
    image.save("/tmp/compile_out.png")

    base_seed = 4096 if args.seed is None else args.seed
    xm.set_rng_state(seed=base_seed, device=device0)
    times = []
    logger.info("starting inference run...")
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = text_pipe.encode_prompt(
            prompt=prompt, max_sequence_length=512
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = text_pipe.encode_prompt(
            prompt=negative_prompt, max_sequence_length=512
        )
    if negative_prompt_embeds_mask is None:
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[:2], dtype=torch.bool
        )
    prompt_embeds = prompt_embeds.to(device0)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(device0)
    negative_prompt_embeds = negative_prompt_embeds.to(device0)
    negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device0)
    for _ in range(args.itters):
        ts = perf_counter()

        if args.profile:
            xp.trace_detached(f"localhost:{profiler_port}", str(profile_path), duration_ms=profile_duration)
        image = pipe(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            true_cfg_scale=args.guidance_scale,
            num_inference_steps=n_steps,
            height=height,
            width=width,
        ).images[0]
        inference_time = perf_counter() - ts
        logger.info(f"inference time: {inference_time}")
        times.append(inference_time)
    logger.info(f"avg. inference over {args.itters} iterations took {sum(times) / len(times)} sec.")
    image.save("/tmp/inference_out.png")
    metrics_report = met.metrics_report()
    with open(metrics_filepath, "w+") as fout:
        fout.write(metrics_report)
    logger.info(f"saved metric information as {metrics_filepath}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--width", type=int, default=1024, help="width of the image to generate")
    parser.add_argument("--height", type=int, default=1024, help="height of the image to generate")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="number of denoising steps")
    parser.add_argument("--seed", type=int, default=None, help="seed for inference")
    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("--profile-duration", type=int, default=10000, help="duration for profiling in msec.")
    parser.add_argument("--itters", type=int, default=15, help="iterations to run inference and get avg time in sec.")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="guidance scale for true CFG")
    args = parser.parse_args()

    xr.use_spmd()

    ckpt_id = "Qwen/Qwen-Image"
    text_pipe = QwenImagePipeline.from_pretrained(ckpt_id, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(
        "cpu"
    )
    main(args, text_pipe, ckpt_id)
