"""Generate JSONL predictions with a saved DoubleGate adapter."""

import argparse
import json
from pathlib import Path

import torch

from doublegate import DoubleGateModel
from doublegate.data import load_records
from doublegate.runtime import load_backbone, PRECISIONS, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/chatglm-6b")
    parser.add_argument("--adapter", default="outputs/doublegate/adapter")
    parser.add_argument("--input-file", default="data/test.jsonl")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--source-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=196)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=PRECISIONS, default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling; default is top_p=0.7, temperature=0.95")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.source_length < 2 or args.max_new_tokens < 1:
        parser.error("Source length must be >= 2 and max-new-tokens >= 1")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new prediction filename")
    settings = json.loads((Path(args.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    rows = load_records(args.input_file, settings["task_map"], require_target=False)
    set_seed(args.seed)
    device = torch.device(args.device)
    base, tokenizer = load_backbone(args.model, device, PRECISIONS[args.precision], args.trust_remote_code)
    model, task_map = DoubleGateModel.from_adapter(base, args.adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": not args.greedy, "use_cache": True}
    if not args.greedy:
        generation.update(top_p=0.7, temperature=0.95)
    # One prompt at a time: avoids padding/routing ambiguities in legacy ChatGLM.
    with torch.inference_mode(), output.open("x", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            tokens = tokenizer(row["input"], return_tensors="pt", truncation=True, max_length=args.source_length)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            generated = model.generate(**tokens, task_id=torch.tensor([task_map[row["task_dataset"]]], device=device), **generation)
            continuation = generated[0, tokens["input_ids"].shape[1]:]
            answer = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            record = dict(row)
            record["prediction"] = answer  # Keep reference target, if supplied.
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"Generated {index + 1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
