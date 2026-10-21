# DoubleGate

Implementation of **DoubleGate: Dual-Gating Mixture-of-LoRA-Experts for Parameter-Efficient Medical Multi-Task Learning**, accepted by *Expert Systems with Applications*.

This repository contains the paper's full dual-gating method, single-GPU training, and inference for the **original ChatGLM-6B**. The Python package is named `doublegate`. It does not include baselines, ablations, experiment queues, datasets, or model weights.

## 1. Install

Use **Python 3.10** and a CUDA-capable NVIDIA GPU. Run the commands below from the repository root. The supplied environment targets Linux; choose a PyTorch CUDA build compatible with your driver.

```bash
git clone https://github.com/Bob-Annette/DoubleGate.git doublegate
cd doublegate
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The Transformers version is intentionally pinned for the original ChatGLM API. This release does not require a modified PEFT/TRL installation, DeepSpeed, or bitsandbytes. Do not upgrade Transformers independently of the backbone code.

CUDA kernels are cached in `.cache/torch-kernels/` to avoid stale shared-account caches. You can override that location with `PYTORCH_KERNEL_CACHE_PATH`; the scripts never delete an existing cache.

## 2. Prepare the backbone and data

Download the original [ChatGLM-6B](https://huggingface.co/zai-org/chatglm-6b) model, including its Python files, tokenizer, configuration, and weight shards. Accept its license before use. For example:

```bash
hf download zai-org/chatglm-6b --revision bf0f5cfb575eebebf9b655c5861177acfee03f16 --local-dir models/chatglm-6b
```

The scripts load local files only. `--trust-remote-code` explicitly permits execution of the custom Python files in the model directory; review those files first. ChatGLM2/3, GLM-4, and quantized checkpoints are not supported by these entry points.

Use JSONL (one object per line) or a JSON array with the following fields:

```json
{"input":"Your task instruction and input text", "target":"Expected answer", "task_dataset":"CHIP-CTC"}
```

- Training requires all three fields; inference requires `input` and `task_dataset`.
- `configs/task_map.json` contains the paper's 16 task IDs. Keep IDs consistent between training and inference. The adapter saves the mapping automatically.
- For your own tasks, provide `--task-map` and set `num_tasks` in a copy of `configs/doublegate.json`. IDs must be unique integers from 1 to `num_tasks`; zero is reserved.
- Obtain benchmark data from its original provider and follow its license. No patient records or benchmark samples are redistributed here. The few examples in `examples/` are synthetic and only test the command-line workflow.

## 3. Train

Place training data in `data/train.jsonl`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --trust-remote-code
```

Defaults: 8,000 optimizer updates, learning rate 2e-4, effective batch size 64 (micro-batch 8 × accumulation 8), linear learning-rate decay, source/target limits 1,024/196 tokens, FP16 backbone, FP32 adapters, and gradient checkpointing. All paths can be changed with command-line arguments; no server-specific paths are embedded.

If memory is limited, reduce `--batch-size` and increase `--gradient-accumulation` so their product remains 64, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 1 --gradient-accumulation 64 --trust-remote-code
```

The output directory must not already exist. It contains `train_log.jsonl`, portable `training_config.json`, and `adapter/`. Only adapter parameters are saved, not the backbone. This minimal trainer saves the final adapter; it does not resume interrupted training or select checkpoints on a validation set. It preserves the paper's architecture and main training settings, but is not a bit-for-bit replay of historical experimental runs.

To check installation with the synthetic examples (not a performance experiment):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --train-file examples/train.jsonl --output outputs/smoke --steps 2 --batch-size 1 --gradient-accumulation 2 --source-length 128 --target-length 16 --log-every 1 --trust-remote-code
```

## 4. Infer

Put inputs in `data/test.jsonl` and use the trained adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --trust-remote-code
```

Or use the adapter from the installation check:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --adapter outputs/smoke/adapter --input-file examples/infer.jsonl --output outputs/smoke_predictions.jsonl --max-new-tokens 8 --greedy --trust-remote-code
```

Each output JSONL object preserves the input fields and adds `prediction`. Existing output files are never overwritten. Default decoding uses top-p sampling (`top_p=0.7`, `temperature=0.95`); `--greedy` disables sampling. Inference runs one input at a time, with one beam and one returned sequence.

Both commands support `--help`. `--model` selects the local backbone directory, `--adapter` selects a saved adapter, and `--precision` accepts `fp16`, `bf16`, or `fp32`. CPU execution requires `--device cpu --precision fp32` and substantial RAM; GPU execution is recommended.

## Architecture

A frozen projection is augmented as follows (bias omitted):

```text
y = W0 x + (alpha / r) * sum_i p_i(task) B_i [SiLU(G x) * A_i Dropout_i(x)]
```

`p(task)` comes from one global four-head attention task gate. `G` is a separate token gate in each adapted projection, shared by its experts. All experts contribute; this is a dense mixture, not top-k routing. Because the token gate depends on the input, the adapter cannot be merged into a static LoRA weight.

The default configuration uses total rank 16, 8 experts (rank 2 each), alpha 32, dropout 0.1, and 64-dimensional task embeddings. It adapts `query_key_value`, `dense`, `dense_h_to_4h`, and `dense_4h_to_h` in all 28 ChatGLM blocks. The registered trainable-parameter inventory is **31,102,657**, matching the paper. For that accounting, the implementation retains the research model's unused per-layer task embeddings and unused key/value projection parameters; they do not affect the forward computation or receive gradients.

`doublegate/model.py` holds the architecture and adapter save/load API. `doublegate/data.py` handles supervised formatting and padding. `doublegate/runtime.py` loads the backbone. The saved adapter format is specific to this release, not a generic PEFT checkpoint.

## Tests

These CPU checks need PyTorch but do not download a model or dataset:

```bash
python -m unittest discover -s tests -v
```

They check the dense dual-gating equation and gradients, zero-initialized adapter behavior, checkpointed backpropagation, adapter save/load, task validation, and the published parameter inventory.

## License and intended use

Code is provided under the [MIT license](LICENSE). Backbone weights, model code, and datasets retain their respective licenses. This is research software, not a clinically validated system or a source of medical advice. A two-step installation check does not produce a useful medical model; train on appropriate, authorized data before evaluating downstream performance.
