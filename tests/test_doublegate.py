"""CPU-only correctness checks, not paper experiments or benchmark runners."""

import copy
import tempfile
import unittest
from pathlib import Path
import json

import torch
from torch import nn

from doublegate import DoubleGateConfig, DoubleGateModel
from doublegate.data import load_records, load_task_map, SupervisedDataset, TrainCollator
from doublegate.model import DoubleGateLinear


class Block(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.query_key_value = nn.Linear(hidden, hidden * 3)
        self.dense = nn.Linear(hidden * 3, hidden)
        self.dense_h_to_4h = nn.Linear(hidden, hidden * 4)
        self.dense_4h_to_h = nn.Linear(hidden * 4, hidden)

    def forward(self, x):
        return x + self.dense_4h_to_h(torch.tanh(self.dense_h_to_4h(self.dense(self.query_key_value(x)))))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([Block(), Block()])

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids).transpose(0, 1)
        for block in self.transformer.layers:
            x = block(x)
        return x.transpose(0, 1)

    def generate(self, input_ids, **kwargs):
        self.forward(input_ids)
        return input_ids


def small_config(dropout=0.0):
    return DoubleGateConfig(rank=4, num_experts=2, num_tasks=3, task_embedding_dim=8, num_heads=2, dropout=dropout)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
        self.tasks = torch.tensor([1, 3])

    def test_zero_initialization_is_backbone(self):
        base = TinyBackbone()
        reference = base(self.ids).detach()
        model = DoubleGateModel(base, small_config())
        # Frozen vs. trainable linear kernels may differ by float32 rounding.
        torch.testing.assert_close(model(self.ids, self.tasks), reference, rtol=1e-6, atol=2e-7)

    def test_dense_equation_and_gradients(self):
        model = DoubleGateModel(TinyBackbone(), small_config())
        layer = next(m for m in model.modules() if isinstance(m, DoubleGateLinear))
        for expert in layer.expert_B:
            nn.init.normal_(expert.weight, std=0.1)
        x = torch.randn(5, 2, 8, requires_grad=True)
        weights = model._weights(self.tasks, 2)
        with model._routing(weights):
            output = layer(x)
        expected = layer.base_layer(x)
        gate = torch.nn.functional.silu(layer.token_gate(x))
        for i, (a, b) in enumerate(zip(layer.expert_A, layer.expert_B)):
            expected = expected + layer.scale * weights[:, i][None, :, None] * b(gate * a(x))
        torch.testing.assert_close(output, expected)
        parameters = [x, layer.token_gate.weight, model.task_embedding.weight, model.task_gate.expert_keys]
        actual_grad = torch.autograd.grad(output.square().mean(), parameters, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().mean(), parameters)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all(p.grad is None for p in layer.base_layer.parameters()))

    def test_checkpointed_backward_matches(self):
        base = TinyBackbone()
        model = DoubleGateModel(base, small_config(dropout=0.1))
        for module in model.modules():
            if isinstance(module, DoubleGateLinear):
                for expert in module.expert_B:
                    nn.init.normal_(expert.weight, std=0.1)
        torch.manual_seed(77)
        output = model(self.ids, self.tasks)
        output.square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.enable_gradient_checkpointing()
        torch.manual_seed(77)
        checked = model(self.ids, self.tasks)
        checked.square().mean().backward()
        torch.testing.assert_close(checked, output)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name])
        self.assertIsNone(model._route.get())

    def test_save_reload_is_exact_and_base_is_frozen(self):
        base = TinyBackbone()
        clean_base = copy.deepcopy(base)
        model = DoubleGateModel(base, small_config())
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        model(self.ids, self.tasks).square().mean().backward()
        opt.step()
        model.eval()
        reference = model(self.ids, self.tasks)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "adapter"
            model.save_adapter(directory, {"one": 1, "three": 3})
            restored, mapping = DoubleGateModel.from_adapter(clean_base, directory)
            torch.testing.assert_close(restored(self.ids, self.tasks), reference, rtol=0, atol=0)
            self.assertEqual(mapping, {"one": 1, "three": 3})
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            self.assertEqual(set(restored.adapter_state_dict()), set(model.adapter_state_dict()))
            self.assertFalse(any(".base_layer." in name for name in restored.adapter_state_dict()))
            with self.assertRaises(FileExistsError):
                model.save_adapter(directory, mapping)

    def test_invalid_task_and_generation_expansion_fail_closed(self):
        model = DoubleGateModel(TinyBackbone(), small_config()).eval()
        for tasks in (torch.tensor([0, 1]), torch.tensor([4, 1]), torch.tensor([1]), torch.tensor([1., 2.])):
            with self.assertRaises(ValueError):
                model(self.ids, tasks)
        with self.assertRaises(ValueError):
            model.generate(self.ids, self.tasks, num_beams=2)
        torch.testing.assert_close(model.generate(self.ids, self.tasks), self.ids)
        with self.assertRaises(RuntimeError):
            model.base_model(self.ids)

    def test_configuration_validation(self):
        for kwargs in ({"rank": 7}, {"num_heads": 3}, {"num_experts": 0}, {"dropout": 1}, {"alpha": float("nan")}):
            with self.assertRaises(ValueError):
                DoubleGateConfig(**kwargs)

    def test_supervised_format_and_padding(self):
        class Tokenizer:
            pad_token_id, bos_token_id = 0, 8

            def encode(self, text, add_special_tokens=False):
                return [1] * len(text)

            def build_inputs_with_special_tokens(self, source, target):
                return source + [7, self.bos_token_id] + target + [9]

        rows = [{"input": "abc", "target": "xy", "task_dataset": "one"},
                {"input": "a", "target": "x", "task_dataset": "one"}]
        dataset = SupervisedDataset(rows, Tokenizer(), {"one": 1}, source_length=3, target_length=3)
        first = dataset[0]
        self.assertEqual(first["input_ids"].tolist(), [1, 1, 7, 8, 1, 9])
        self.assertEqual(first["labels"].tolist(), [-100, -100, -100, 8, 1, 9])
        batch = TrainCollator(0)([first, dataset[1]])
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][1, -1].item(), 0)
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["task_id"].tolist(), [1, 1])

    def test_data_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "example.json"
            row = {"input": "synthetic input", "task_dataset": "one"}
            path.write_text(json.dumps([row]), encoding="utf-8")
            self.assertEqual(load_records(path, {"one": 1}, require_target=False), [row])
            with self.assertRaises(ValueError):
                load_records(path, {"one": 1})
            with self.assertRaises(ValueError):
                load_records(path, {"two": 2}, require_target=False)
            for mapping in ([1, 2], {"one": True}, {"one": 0}, {"one": 1, "two": 1}):
                path.write_text(json.dumps(mapping), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_task_map(path, 3)

    def test_published_parameter_count_without_allocating_backbone(self):
        class PaperBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_key_value = nn.Linear(4096, 12288)
                self.dense = nn.Linear(4096, 4096)
                self.dense_h_to_4h = nn.Linear(4096, 16384)
                self.dense_4h_to_h = nn.Linear(16384, 4096)
        with torch.device("meta"):
            base = nn.Module()
            base.layers = nn.ModuleList([PaperBlock() for _ in range(28)])
            model = DoubleGateModel(base)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 31_102_657)


if __name__ == "__main__":
    unittest.main()
