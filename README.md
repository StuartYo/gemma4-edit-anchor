# gemma4-edit-anchor

Code, data and every raw model transcript behind the paper **Fixing the Quote Did Not Fix the Edit: Measuring exact-match `edit_file` failures of Gemma 4 E2B and E4B on a laptop GPU, and what each repair costs**, written for the Kaggle Gemma 4 Developer Agent Paper Track.

The question: when a small Gemma 4 model (E2B or E4B, 4-bit QAT, on a 6 GB laptop GPU) drives a coding agent through the competition's `edit_file(filepath, old_string, new_string)` tool, how often does the exact-match `old_string` fail, why, and what do the usual repairs buy and cost?

Short answer (158 scored tasks from seven Python repositories): the first `edit_file` call matched exactly in 46% (E2B) and 70% (E4B) of tasks; most rejected quotes were near misses or partial rewrites, not copy errors; none of silent repair, a nearest-block hint, both, or a line-range tool raised exact task success over strict matching by more than seed noise.

## What is here

| Path | What it is |
|---|---|
| `build_tasks.py` | Builds single-file edit tasks from public git history (1 to 3 hunks, 1 to 20 changed lines, 30 to 800-line file). Deterministic seed. |
| `data/dev100.jsonl`, `data/holdout60.jsonl` | The task sets used in the paper (rich, requests, httpx, fastapi / click, flask, attrs). `data/dev.jsonl` and `data/holdout.jsonl` are the 320-task pool they were cut from (first 25 / 20 per repository). |
| `appliers.py` | Edit appliers: `strict` (competition semantics), `lineno`, `ws` (unique whitespace/indent-tolerant match), `fuzzy` (difflib >= 0.8, for comparison only), plus `nearest_block` for the hint. |
| `run_exp.py` | Agent loop against a local llama.cpp server with the competition's tool signatures. Conditions S, N, R, RN, L. |
| `driver.py`, `start_driver.py` | Runs the experiment queue in batches (`driver.BATCHES`); `start_driver.py` launches it as a detached background process on Windows. |
| `tools/set_stack_reserve.py` | Makes the 256 MB stack copy of `llama-server.exe` used for the runs (see below). |
| `score.py`, `analysis.py`, `extra_analysis.py` | Every number in the paper comes from these three scripts. |
| `positive_control.py` | Gold edits must apply and reproduce the gold file under every applier and the line-range tool (320/320). |
| `make_figs.py`, `make_fig0.py` | Figures, drawn from `results/summary.json`. |
| `results/` | `summary.json`, `paired.json`, `extra.json` as used in the paper. |
| `runs/*.jsonl` | Every model transcript: each tool call, raw arguments, applier result, final file. `runs/driver.log` is the run log. `runs/_discarded_prenudge/` holds partial runs made before the reminder was added (not scored). |
| `skill/anchored_edit/` | Drop-in ADK skill for the competition harness (whitespace/line-number repair only when the match is unique, otherwise the nearest block). Not evaluated inside the competition harness. |

## Reproduce

Steps 1 and 3 need no GPU. Rescoring the saved transcripts takes about 12 minutes on a laptop CPU (mostly the fuzzy replay).

```
# 1. tasks (needs the seven repos cloned under <repos_dir>)
python build_tasks.py <repos_dir> data/dev.jsonl 50 rich requests httpx fastapi
python build_tasks.py <repos_dir> data/holdout.jsonl 40 click flask attrs
python positive_control.py data/dev.jsonl data/holdout.jsonl

# 2. model runs
#    llama.cpp b11215 (Windows, Vulkan) and google/gemma-4-E2B-it-qat-q4_0-gguf, google/gemma-4-E4B-it-qat-q4_0-gguf.
#    The stock llama-server.exe overflows its 1 MB thread stack (exit 0xC00000FD) while parsing very long
#    tool-call strings. The runs used a copy whose PE header stack reserve is 256 MB; nothing else differs:
python tools/set_stack_reserve.py <llama.cpp>/llama-server.exe <llama.cpp>/llama-server-stack256m.exe 256
#    Edit LL, GGUF and the log/lock paths at the top of driver.py, then:
python start_driver.py            # all batches; or e.g. "python start_driver.py 3 4" for some
#    Server flags: -c 49152 -ngl 99 -t 4 -np 3 --cache-ram 512 --jinja. A watchdog restarts the server
#    with the same flags if it exits, and the runner resends the same turn.

# 3. numbers and figures
python score.py "runs/*.jsonl" --tasks data/dev.jsonl data/holdout.jsonl --out results/summary.json
python analysis.py --tasks data/dev.jsonl data/holdout.jsonl --out results/paired.json
python extra_analysis.py --tasks data/dev.jsonl data/holdout.jsonl --out results/extra.json
python make_figs.py results/summary.json figures/
python make_fig0.py
```

Generation with three parallel server slots is not bitwise reproducible even with fixed seeds; the saved transcripts are the record, and scoring them is deterministic.

## Notes

- No Kaggle competition data is used or included. Tasks are rebuilt from public git history; see `THIRD_PARTY.md` for licenses.
- Two development commits only change trailing whitespace; the scorer excludes them (98 development tasks are scored).
- Tool calls go through llama.cpp's Gemma 4 chat template and tool-call parser, not the competition's vLLM parser. Calls whose arguments the parser could not read are counted separately (`arg_parse_errors`).
- When a model passes a bare JSON number as a text argument, the runner (for `edit_lines`) and the scorer (for `edit_file`) read it as that text.
- Condition L was added after the E2B development results of the other conditions and is exploratory.
- AI use: the author chose the topic and decided to enter. Claude (Anthropic's AI model) did most of the experiment design, harness code, running, analysis and writing. The author reviewed and approved the results and the final text.

## License

Apache-2.0 for the code in this repository. Task data and transcripts contain source files from seven public projects, each under its own license (`THIRD_PARTY.md`).
