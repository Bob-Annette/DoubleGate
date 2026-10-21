"""The paper's full DoubleGate architecture, without baseline or ablation modes."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class DoubleGateConfig:
    rank: int = 16  # Total rank, not the rank of each expert.
    alpha: float = 32.0
    dropout: float = 0.1
    num_experts: int = 8
    num_tasks: int = 16  # IDs are 1..16; row zero is reserved.
    task_embedding_dim: int = 64
    num_heads: int = 4
    target_modules: tuple = ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")

    def __post_init__(self):
        for name in ("rank", "num_experts", "num_tasks", "task_embedding_dim", "num_heads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rank % self.num_experts:
            raise ValueError("rank must be divisible by num_experts")
        if self.task_embedding_dim % self.num_heads:
            raise ValueError("task_embedding_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Require 0 <= dropout < 1 and a positive finite alpha")
        if not self.target_modules or any(not isinstance(x, str) or not x for x in self.target_modules):
            raise ValueError("target_modules must contain nonempty module names")
        self.target_modules = tuple(self.target_modules)


class AttentionTaskGate(nn.Module):
    """Task query attends to learned expert keys/values; shared by all layers."""

    def __init__(self, config):
        super().__init__()
        dim = config.task_embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.num_experts = config.num_experts
        self.query_proj = nn.Linear(dim, dim, bias=False)
        # Retained for the published parameter inventory. The research forward
        # uses expert_keys/expert_values directly, NOT these two projections.
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.expert_keys = nn.Parameter(torch.randn(config.num_experts, dim))
        self.expert_values = nn.Parameter(torch.randn(config.num_experts, dim))
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.output_proj = nn.Linear(dim, config.num_experts, bias=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, task_embedding):
        batch = task_embedding.shape[0]
        queries = self.query_proj(task_embedding).view(batch, self.num_heads, self.head_dim).permute(1, 0, 2)
        keys = self.expert_keys.view(self.num_experts, self.num_heads, self.head_dim).permute(1, 0, 2)
        values = self.expert_values.view(self.num_experts, self.num_heads, self.head_dim).permute(1, 0, 2)
        temperature = self.log_temperature.exp().clamp(min=1e-8, max=1e4)
        scores = queries @ keys.transpose(-2, -1) / (math.sqrt(self.head_dim) * temperature)
        attended = (scores.softmax(dim=-1) @ values).permute(1, 0, 2).contiguous().view(batch, -1)
        return self.output_proj(self.layer_norm(attended + task_embedding)).softmax(dim=-1)


class DoubleGateLinear(nn.Module):
    """Frozen linear + dense expert mixture with one shared SiLU token gate."""

    def __init__(self, base_layer, config, route_context):
        super().__init__()
        self.base_layer = base_layer
        self.route_context = route_context
        rank = config.rank // config.num_experts
        self.scale = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        kwargs = {"device": base_layer.weight.device, "dtype": torch.float32}
        self.expert_A = nn.ModuleList([
            nn.Linear(base_layer.in_features, rank, bias=False, **kwargs) for _ in range(config.num_experts)
        ])
        self.expert_B = nn.ModuleList([
            nn.Linear(rank, base_layer.out_features, bias=False, **kwargs) for _ in range(config.num_experts)
        ])
        self.token_gate = nn.Linear(base_layer.in_features, rank, bias=False, **kwargs)
        # The archived model registers this embedding at each adapted layer but
        # routes using the GLOBAL task embedding. Keep it for exact parameter
        # accounting; it is intentionally not used to compute routing weights.
        self.compat_task_embedding = nn.Embedding(config.num_tasks + 1, config.task_embedding_dim, **kwargs)
        for layer_a, layer_b in zip(self.expert_A, self.expert_B):
            nn.init.normal_(layer_a.weight, std=0.01)
            nn.init.zeros_(layer_b.weight)

    def forward(self, x):
        weights = self.route_context.get()
        if weights is None:
            raise RuntimeError("Call DoubleGateModel with task_id; do not call its backbone directly")
        if x.ndim != 3 or x.shape[1] != weights.shape[0]:
            raise ValueError("This release expects ChatGLM-6B hidden states [sequence, batch, hidden]")
        result = self.base_layer(x)
        features = x.to(self.expert_A[0].weight.dtype)
        modulation = F.silu(self.token_gate(features))
        for index, (layer_a, layer_b) in enumerate(zip(self.expert_A, self.expert_B)):
            # Separate dropout draws per expert, as in the paper implementation.
            update = layer_b(modulation * layer_a(self.dropout(features)))
            update = update * self.scale * weights[:, index][None, :, None]
            # Match the research implementation's per-expert accumulation,
            # including casting into the backbone dtype and preserving strides.
            result += update
        return result


class DoubleGateModel(nn.Module):
    """Task-aware wrapper for the original ChatGLM-6B backbone.

    Only a single device and num_beams=num_return_sequences=1 are supported.
    Adapters stay in float32 when injected into a float16 backbone.
    """

    def __init__(self, base_model, config=None):
        super().__init__()
        self.base_model = base_model
        self.adapter_config = config or DoubleGateConfig()
        self._route = ContextVar(f"doublegate_route_{id(self)}", default=None)
        self._checkpointing = False
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        matched = {}
        for name, module in base_model.named_modules():
            if any(name == suffix or name.endswith("." + suffix) for suffix in self.adapter_config.target_modules):
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"Target {name} must be a torch.nn.Linear, got {type(module).__name__}")
                matched[name] = module
        if not matched:
            raise ValueError("No target projections found; use the original ChatGLM-6B backbone")
        found = {name.rsplit(".", 1)[-1] for name in matched}
        missing = set(self.adapter_config.target_modules) - found
        if missing:
            raise ValueError(f"Missing target projection types: {sorted(missing)}")
        for name, module in matched.items():
            parent_name, _, leaf = name.rpartition(".")
            parent = base_model.get_submodule(parent_name) if parent_name else base_model
            setattr(parent, leaf, DoubleGateLinear(module, self.adapter_config, self._route))
        self.adapted_modules = tuple(matched)
        device = next(base_model.parameters()).device
        self.task_embedding = nn.Embedding(self.adapter_config.num_tasks + 1, self.adapter_config.task_embedding_dim,
                                           device=device, dtype=torch.float32)
        self.task_gate = AttentionTaskGate(self.adapter_config).to(device=device)
        self._adapter_names = frozenset(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def _weights(self, task_id, batch_size):
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.as_tensor(task_id, device=self.task_embedding.weight.device)
        if task_id.ndim != 1 or len(task_id) != batch_size or task_id.dtype not in (torch.int32, torch.int64):
            raise ValueError("task_id must be an integer vector with one ID per input example")
        task_id = task_id.to(device=self.task_embedding.weight.device, dtype=torch.long)
        if bool(((task_id < 1) | (task_id > self.adapter_config.num_tasks)).any()):
            raise ValueError(f"Task IDs must be in [1, {self.adapter_config.num_tasks}]")
        return self.task_gate(self.task_embedding(task_id))

    @contextmanager
    def _routing(self, weights):
        token = self._route.set(weights)
        try:
            yield
        finally:
            self._route.reset(token)

    def forward(self, input_ids, task_id, **kwargs):
        with self._routing(self._weights(task_id, input_ids.shape[0])):
            return self.base_model(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, task_id, **kwargs):
        if self.training:
            raise RuntimeError("Call model.eval() before generation")
        generation_config = getattr(self.base_model, "generation_config", None)
        for key in ("num_beams", "num_return_sequences"):
            value = kwargs.get(key, getattr(generation_config, key, 1))
            if value != 1:
                raise ValueError(f"Only {key}=1 is supported; routing must match each input example")
            kwargs[key] = 1
        with self._routing(self._weights(task_id, input_ids.shape[0])):
            return self.base_model.generate(input_ids=input_ids, **kwargs)

    def enable_gradient_checkpointing(self):
        """Non-reentrant block recomputation with the original routing context.

        No global monkey-patch or mutable per-layer routing cache is needed.
        Saving/loading adapter tensors is independent of checkpointing.
        """
        if self._checkpointing:
            return
        layers = getattr(getattr(self.base_model, "transformer", None), "layers", None)
        if layers is None:
            raise ValueError("Expected ChatGLM-6B transformer.layers for gradient checkpointing")
        for block in layers:
            original_forward = block.forward

            def checkpointed_forward(*args, _forward=original_forward, **kwargs):
                if not self.training or not torch.is_grad_enabled():
                    return _forward(*args, **kwargs)
                weights = self._route.get()

                def recompute(*inputs):
                    with self._routing(weights):
                        return _forward(*inputs, **kwargs)

                return checkpoint(recompute, *args, use_reentrant=False, preserve_rng_state=True)

            block.forward = checkpointed_forward
        self._checkpointing = True

    def adapter_state_dict(self):
        # named_parameters deduplicates aliases and excludes all frozen weights.
        return {name: value.detach().cpu().contiguous() for name, value in self.named_parameters()
                if name in self._adapter_names}

    def save_adapter(self, directory, task_map):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        payload = {"format_version": 1, "architecture": "doublegate", "base_model_type": "chatglm",
                   "config": asdict(self.adapter_config), "task_map": task_map,
                   "adapted_modules": list(self.adapted_modules)}
        (directory / "adapter_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        torch.save(self.adapter_state_dict(), directory / "adapter_model.pt")

    @classmethod
    def from_adapter(cls, base_model, directory, trainable=False):
        directory = Path(directory)
        payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("architecture") != "doublegate":
            raise ValueError("Not a supported DoubleGate adapter")
        model = cls(base_model, DoubleGateConfig(**payload["config"]))
        if list(model.adapted_modules) != payload["adapted_modules"]:
            raise ValueError("Backbone projection layout does not match the saved adapter")
        state = torch.load(directory / "adapter_model.pt", map_location="cpu", weights_only=True)
        expected = model.adapter_state_dict()
        if set(state) != set(expected):
            raise ValueError("Adapter tensor names do not match the model; checkpoint may be incomplete")
        for name in state:
            if state[name].shape != expected[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
        model.load_state_dict(state, strict=False)  # Only backbone tensors are intentionally absent.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(trainable and name in expected)
        model.train(trainable)
        return model, payload["task_map"]
