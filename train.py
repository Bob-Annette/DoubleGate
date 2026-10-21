"""Single-GPU DoubleGate training. Run python train.py --help for options."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from doublegate import DoubleGateConfig, DoubleGateModel
from doublegate.data import load_records, load_task_map, SupervisedDataset, TrainCollator
from doublegate.runtime import load_backbone, PRECISIONS, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/chatglm-6b")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--task-map", default="configs/task_map.json")
    parser.add_argument("--config", default="configs/doublegate.json")
    parser.add_argument("--output", default="outputs/doublegate")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--source-length", type=int, default=1024)
    parser.add_argument("--target-length", type=int, default=196)
    parser.add_argument("--precision", choices=PRECISIONS, default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.log_every) < 1:
        parser.error("Steps, batch size, accumulation and log interval must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    if args.source_length < 2 or args.target_length < 3:
        parser.error("Source length must be >= 2 and target length >= 3")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists. Choose a new --output directory; existing runs are never overwritten")
    config = DoubleGateConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    task_map = load_task_map(args.task_map, config.num_tasks)
    rows = load_records(args.train_file, task_map)
    if len(rows) < args.batch_size:
        parser.error("Training file has fewer examples than --batch-size; reduce the batch size")
    set_seed(args.seed)
    device = torch.device(args.device)
    base, tokenizer = load_backbone(args.model, device, PRECISIONS[args.precision], args.trust_remote_code)
    model = DoubleGateModel(base, config).train()
    model.base_model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    dataset = SupervisedDataset(rows, tokenizer, task_map, args.source_length, args.target_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=TrainCollator(tokenizer.pad_token_id), num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1 - step / args.steps))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=device.type == "cuda" and args.precision == "fp16")
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in parameters),
                      "effective_batch_size": args.batch_size * args.gradient_accumulation,
                      "optimizer_steps": args.steps, "seed": args.seed}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    # Paths and machine identifiers are deliberately absent from portable metadata.
    settings = {key: value for key, value in vars(args).items()
                if key not in {"model", "train_file", "task_map", "config", "output", "device"}}
    settings["training_examples"] = len(dataset)
    (output / "training_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    iterator = iter(loader)
    completed, skipped = 0, 0
    with (output / "train_log.jsonl").open("x", encoding="utf-8") as log:
        while completed < args.steps:
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for _ in range(args.gradient_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {key: value.to(device) for key, value in batch.items()}
                result = model(**batch, use_cache=False)
                loss = result.loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite loss; retained the run log without writing a completed adapter")
                mean_loss += loss.detach().float().item() / args.gradient_accumulation
                scaler.scale(loss / args.gradient_accumulation).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not scaler.is_enabled() and not bool(torch.isfinite(norm)):
                raise FloatingPointError("Non-finite gradient norm")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= old_scale
            if updated:
                completed += 1
                scheduler.step()
            else:
                skipped += 1
            record = {"step": completed, "loss": mean_loss, "optimizer_updated": updated,
                      "loss_scale": scaler.get_scale(), "skipped_updates": skipped}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if completed % args.log_every == 0 or completed == 1 or not updated:
                print(json.dumps(record), flush=True)
            if skipped > max(100, args.steps):
                raise FloatingPointError("Too many skipped FP16 updates; inspect the log and try bf16 on supported hardware")
    model.save_adapter(output / "adapter", task_map)
    print("Training complete. Adapter saved to the output directory's adapter/ subdirectory.")


if __name__ == "__main__":
    main()
