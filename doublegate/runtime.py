"""Backbone loading shared by the training and inference commands."""

import random
import os
from pathlib import Path

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(model_path, device, dtype, trust_remote_code=False):
    # Avoid stale or truncated JIT kernels in a shared account's home cache.
    # This setting affects only the current process and never deletes a cache.
    if device.type == "cuda":
        cache = Path(os.environ.get("PYTORCH_KERNEL_CACHE_PATH", ".cache/torch-kernels"))
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["PYTORCH_KERNEL_CACHE_PATH"] = str(cache.resolve())
    # Imported lazily so core architecture tests and --help work with torch alone.
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not trust_remote_code:
        raise ValueError("ChatGLM-6B uses custom Python code. Review your model files, then pass --trust-remote-code")
    if not Path(model_path).is_dir():
        raise FileNotFoundError("Download ChatGLM-6B first and pass its local directory with --model")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build and check your driver")
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU execution requires --precision fp32")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    original_shape = {"num_layers": 28, "hidden_size": 4096, "inner_hidden_size": 16384, "vocab_size": 130528}
    if config.model_type != "chatglm" or any(getattr(config, key, None) != value for key, value in original_shape.items()):
        raise ValueError("The CLI supports the original ChatGLM-6B, not ChatGLM2/3, GLM-4, or other backbones")
    if getattr(config, "quantization_bit", 0) not in (0, None):
        raise ValueError("Use full-precision ChatGLM-6B weights, not a quantized checkpoint")
    config.pre_seq_len = None
    config.prefix_projection = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    base = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True,
                                     local_files_only=True, torch_dtype=dtype).to(device)
    return base, tokenizer


PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
