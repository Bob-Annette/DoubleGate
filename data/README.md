# Paper data: eight-task PromptCBLUE subset

These are the prompt-formatted training data and labeled evaluation records for the eight tasks reported in the DoubleGate paper. They are JSONL files: each line contains `input`, reference `target`, `task_dataset`, `task_type`, `sample_id`, and `answer_choices`.

| Task | Training | Evaluation |
| --- | ---: | ---: |
| CMeIE | 2,828 | 600 |
| CHIP-CDN | 2,381 | 600 |
| CHIP-CDEE | 1,562 | 600 |
| CHIP-MDCFNPC | 4,935 | 600 |
| CHIP-CTC | 3,622 | 1,100 |
| KUAKE-QIC | 3,279 | 660 |
| IMCS-V2-MRG | 1,799 | 600 |
| MedDG | 4,964 | 600 |
| **Total** | **25,370** | **5,360** |

`train.jsonl` is an unchanged copy of the archived `train.json`, with only the filename extension changed to reflect its JSONL format. The archive's `test.json` also contains 600 CHIP-STS and 600 KUAKE-IR records, which are outside the paper's eight-task comparison. `test.jsonl` includes only the eight tasks above. Retained lines, answers, sample identifiers, and relative order are unchanged. No records or reference answers were generated or edited for this release.

The task IDs in `../configs/task_map.json` retain the original 16-task mapping, including unused entries, to keep the eight tasks' IDs and the model configuration consistent. `test.jsonl` is a labeled local evaluation split, not the hidden competition test set. Keep its reference answers out of training. No separately verified validation file is available in the archive, so none is reconstructed here.

## Sources and terms

- The archived files match the public [MOELoRA data release](https://github.com/liuqidong07/MOELoRA-peft/tree/master/data): upstream training blob `5780987f825742fc14720bb61fdfa6e3cda5e123` and evaluation blob `ec6739014ce711c85767de40e8c0a73690f707d2`. The upstream repository's [MIT notice](LICENSE-MOELoRA) is retained with this copy.
- The benchmark originates from [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE), built on [CBLUE](https://github.com/CBLUEbenchmark/CBLUE). Please acknowledge these projects and MOELoRA when using the data.
- PromptCBLUE's authors state that project resources are for **academic research only; commercial use is prohibited**. This data release retains that restriction and any applicable original task terms. DoubleGate's root MIT license applies to its code and does not replace the data owners' terms or grant additional dataset rights.

## SHA-256

Git preserves the dataset bytes across platforms through `.gitattributes`.

```text
ac4413e6e6665daeb1a7d36d0ceddd07bc8fc3c41e4aeb4bf1521769201f1436  train.jsonl
47a10c5e253a855ae19fba03bf48d8ba183c22a9539757cc1779451850601bbe  test.jsonl
```
