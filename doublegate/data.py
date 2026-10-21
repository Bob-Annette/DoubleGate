"""Small JSON/JSONL loader and original ChatGLM-6B supervised formatting."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_records(path, task_map, require_target=True):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("The input data file is empty")
    rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array or JSONL file")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("input"), str) or not row["input"].strip():
            raise ValueError(f"Row {index + 1}: input must be a nonempty string")
        if row.get("task_dataset") not in task_map:
            raise ValueError(f"Row {index + 1}: task_dataset is absent from the task map")
        if require_target and (not isinstance(row.get("target"), str) or not row["target"].strip()):
            raise ValueError(f"Row {index + 1}: training requires a nonempty target")
    return rows


def load_task_map(path, num_tasks):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Task map must be a name-to-integer mapping")
    mapping = mapping.get("str2id", mapping)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Task map must be a nonempty name-to-integer mapping")
    if any(not isinstance(name, str) or not isinstance(value, int) or isinstance(value, bool)
           or not 1 <= value <= num_tasks for name, value in mapping.items()):
        raise ValueError(f"Task IDs must be integers in [1, {num_tasks}]")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Task IDs must be unique")
    return mapping


class SupervisedDataset(Dataset):
    def __init__(self, rows, tokenizer, task_map, source_length=1024, target_length=196):
        if source_length < 2 or target_length < 3:
            raise ValueError("source_length >= 2 and target_length >= 3 are required")
        self.rows, self.tokenizer, self.task_map = rows, tokenizer, task_map
        self.source_length, self.target_length = source_length, target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, tokenizer = self.rows[index], self.tokenizer
        source = tokenizer.encode(row["input"], add_special_tokens=False)[:self.source_length - 1]
        target = tokenizer.encode(row["target"], add_special_tokens=False)[:self.target_length - 2]
        ids = tokenizer.build_inputs_with_special_tokens(source, target)
        context_length = ids.index(tokenizer.bos_token_id)
        labels = [-100] * context_length + ids[context_length:]
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "task_id": self.task_map[row["task_dataset"]]}


class TrainCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        pad = torch.nn.utils.rnn.pad_sequence
        return {"input_ids": pad([x["input_ids"] for x in rows], batch_first=True, padding_value=self.pad_token_id),
                "labels": pad([x["labels"] for x in rows], batch_first=True, padding_value=-100),
                "task_id": torch.tensor([x["task_id"] for x in rows], dtype=torch.long)}
