# gemma4-edit-anchor

Code, data and every raw model transcript behind the paper *(title set when the paper is final)*, written for the Kaggle Gemma 4 Developer Agent Paper Track.

The question: when a small Gemma 4 model (E2B or E4B, 4-bit QAT, on a 6 GB laptop GPU) drives a coding agent through the competition's `edit_file(filepath, old_string, new_string)` tool, how often does the exact-match `old_string` fail, why, and what do the usual fixes buy and cost?

## What is here

| Path | What it is |
|---|---|
| `build_tasks.py` | Builds single-file edit tasks from public git history (1 to 3 hunks, 1 to 20 changed lines, 30 to 800-line file). Deterministic seed. |
| `data/dev100.jsonl`, `data/holdout60.jsonl` | The task sets used in the paper (rich, requests, httpx, fastapi / click, flask, attrs). `data/dev.jsonl` and `data/holdout.jsonl` are the larger pools they were cut from. |
| `appliers.py` | Edit appliers: `strict` (competition semantics), `lineno`, `ws` (unique whitespace/indent-tolerant match), `fuzzy` (Aider-style, for comparison only), plus `nearest_block` for the hint. |
| `run_exp.py` | Agent loop against a local llama.cpp server with the competition's tool signatures. Conditions S, N, R, RN, L. |
| `driver.py` | Runs the experiment queue in batches. |
| `score.py`, `analysis.py` | Every number in the paper comes from these two scripts. |
| `positive_control.py` | Gold edits must apply and reproduce the gold file under every applier and the line-range tool (320/320). |
| `make_figs.py`, `make_fig0.py` | Figures, drawn from `results/summary.json`. |
| `runs/*.jsonl` | Every model transcript: each tool call, raw arguments, applier result, final file. |
| `skill/anchored_edit/` | Drop-in ADK skill for the competition harness (whitespace/line-number repair only when the match is unique, otherwise the nearest block). |

## Reproduce

```
# 1. tasks (needs the seven repos cloned under <repos_dir>)
python build_tasks.py <repos_dir> data/dev.jsonl 50 rich requests httpx fastapi
python build_tasks.py <repos_dir> data/holdout.jsonl 40 click flask attrs
python positive_control.py data/dev.jsonl data/holdout.jsonl

# 2. model runs (llama.cpp b11215, google/gemma-4-E2B-it-qat-q4_0-gguf and E4B)
python driver.py            # edit the paths at the top of driver.py

# 3. numbers and figures
python score.py "runs/*.jsonl" --tasks data/dev.jsonl data/holdout.jsonl --out results/summary.json
python analysis.py --tasks data/dev.jsonl data/holdout.jsonl --out results/paired.json
python make_figs.py results/summary.json figures/
```

Steps 1 and 3 need no GPU. Rescoring the saved transcripts takes a few minutes on a laptop CPU.

## Notes

- No Kaggle competition data is used or included. Tasks are rebuilt from public git history; see `THIRD_PARTY.md` for licenses.
- Two development commits only change trailing whitespace; the scorer excludes them (98 development tasks are scored).
- Tool calls go through llama.cpp's Gemma 4 chat template and tool-call parser, not the competition's vLLM parser. Calls whose arguments the parser could not read are counted separately (`arg_parse_errors`).

## License

Apache-2.0 for the code in this repository. Task data keeps the license of each source project (`THIRD_PARTY.md`).
