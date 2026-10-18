## Abstract

Coding agents change code through tools such as the competition's `edit_file(filepath, old_string, new_string)`, which succeeds only if `old_string` occurs in the file exactly once. We measured how often the two smallest Gemma 4 models fail it, why, and what repairs buy. On 158 single-file edits rebuilt from real commits in seven Python repositories, run on one 6 GB laptop GPU, the first `edit_file` call matched exactly in 46% (E2B) and 70% (E4B) of tasks. Most rejected quotes were near misses or partial rewrites (53% and 71% on the development set); copy errors that a safe repair can fix were 24% and 2%. No condition (silent repair of whitespace and line numbers, a nearest-block hint, both, or a line-range tool) changed exact task success by more than seed noise: differences from strict matching ranged from -10 to +3 tasks, no McNemar p was below 0.07, and two seeds of one condition differed by 7. Repairs recovered rejected quotes, but many recovered edits moved the file further from the correct version.

## 1 Introduction

The competition harness changes code through `edit_file(filepath, old_string, new_string)`, which succeeds only if `old_string` appears in the file exactly once, byte for byte. A model that fits on a laptop GPU may know what to change yet fail to reproduce the text it replaces; competitors have reported a Gemma 4 agent re-sending one failing call 27 times (Kaggle discussion 747168). Tool builders respond by silently repairing near matches, showing the closest block, or addressing lines by number, each with known risks: a repair can land in the wrong place, and line numbers fail silently when they drift (Althoubi, 2026).

We measured the problem on real outputs of small models: (Q1) how often is `old_string` rejected, and why; (Q2) does any repair change exact task success by more than seed noise; (Q3) what does each repair cost? Contributions:

1. A measurement of edit failures for Gemma 4 E2B and E4B through a real tool-calling stack, with an eight-class taxonomy computed from transcripts.
2. A paired ablation of four conditions against harness behavior, with a noise floor and an offline replay of identical calls through four appliers. The results are null, and we report them as such.
3. Released tasks, appliers, transcripts, scoring scripts and a drop-in skill for the competition harness.

![Figure 1](https://raw.githubusercontent.com/StuartYo/gemma4-edit-anchor/master/figures/fig1_solved_by_condition.png)
*Figure 1. No repair condition beat strict matching (dashed line) by more than seed noise. Bars: percent solved, labelled with task counts; whiskers: 95% bootstrap intervals.*

## 2 The edit interface

Our tools copy the competition's public signatures, `read_file` and `edit_file(filepath, old_string, new_string, allow_multiple)` (`allow_multiple` is ignored). Under harness semantics (`strict`) a call fails if `old_string` is empty, absent or not unique, and the model may retry.

`score.py` classifies each rejected call against the file at that moment; the first matching check names the class: (1) **not unique**; (2) **empty**; (3) **copied line numbers**: matches once `read_file` number prefixes are removed; (4) **whitespace or indentation**: matches line by line ignoring surrounding whitespace, with one uniform indentation shift; (5) **quoted the edited version** (target mode only): matches the post-commit file; (6) **near miss**: the most similar file block has `difflib` similarity of at least 0.9; (7) **partly rewritten**: 0.6 to 0.9; (8) **not in file**: below 0.6. Classes 3 and 4 can be repaired without choosing between locations; 6 to 8 cannot.

## 3 Repair design

Conditions differ only after an exact match fails (Figure 2). **S (strict)** is harness behavior: on failure, "Error: old_string was not found in the file" (or the match count). The model with its own retries is the baseline.

![Figure 2](https://raw.githubusercontent.com/StuartYo/gemma4-edit-anchor/master/figures/fig0_conditions.png)
*Figure 2. One edit call under each condition.*

**R (silent repair).** On failure the applier (`ws`) removes line-number prefixes if every non-empty line of `old_string` has one, then matches line by line ignoring surrounding whitespace, accepting a window only if all non-blank lines share one indentation shift (re-applied to `new_string`) and only if exactly one window qualifies; otherwise the model sees the S error.

**N (nearest-block feedback).** Matching stays strict. On failure the error adds the file block most similar to `old_string`, with line numbers, and asks the model to copy it, as Aider does when a search block fails (Aider-AI, 2025). **RN** applies R and falls back to the N hint.

**L (line range with an anchor check).** L replaces `edit_file` with `edit_lines(filepath, start_line, end_line, expected_first_line, new_content)`. The edit is rejected, showing nearby lines, unless the stripped first line of `expected_first_line` equals the stripped `start_line` (anchor-and-verify; Althoubi, 2026). L was designed after the E2B development results for the other conditions, so it is exploratory.

**Offline replay.** The first `edit_file` call of an episode always meets the unmodified file, so we replay the S runs' first calls through four nested appliers without sampling noise: `strict`, `lineno` (adds number-prefix removal), `ws` (R) and `fuzzy`, which takes the block of k-1 to k+1 lines with the highest `difflib` ratio if at least 0.8. `fuzzy` reimplements a fallback in Aider's source that Aider's current code does not reach (Aider-AI, 2025).

## 4 Tasks, metrics and setup

### 4.1 Tasks

`build_tasks.py` keeps non-merge commits that change exactly one non-test Python file, with 1 to 20 changed lines in 1 to 3 hunks, at least one non-comment change, and 30 to 800 lines. A fixed-seed shuffle gave a pool of 320 tasks: 50 each from rich, requests, httpx and fastapi, the public repositories behind the competition's tasks (no competition data was used), and 40 each from click, flask and attrs. Experiments use the first 25 per development and 20 per held-out repository; two whitespace-only commits are not scored, leaving 98 development and 60 held-out tasks. Code and task lists were committed (`3151b5d`) before any held-out run.

In *target* mode the prompt shows each block that must appear after the edit, with two lines of context, so the correct file is known and the interface is isolated from deciding what to change. In *intent* mode the prompt gives only the commit subject; we still see whether and where each call applies.

### 4.2 Metrics

**Solved**: the final file equals the post-commit file, ignoring trailing whitespace. **Applied**: at least one edit applied. **First-call strict success**: the first call passes `strict`. **Wrong place**: an applied edit not overlapping any changed gold span widened by 3 lines. **Moved away from gold**: a repaired edit lowers line similarity to the post-commit file. **Cost**: completion tokens per task. We report 95% percentile bootstrap intervals (2,000 resamples; Efron, 1979) and exact two-sided McNemar tests on paired outcomes against S (McNemar, 1947).

### 4.3 Experimental details and reproducibility

One laptop with an RTX 4050 Laptop GPU (6 GB) ran llama.cpp b11215 (Vulkan) `llama-server` with all layers on the GPU, Gemma's chat template and tool-call parser, and three parallel slots of 16,384 context tokens. Models: Google's `gemma-4-E2B-it-qat-q4_0-gguf` and `gemma-4-E4B-it-qat-q4_0-gguf`. Sampling: temperature 1.0, top-p 0.95, top-k 64, thinking off, at most 2,048 tokens per turn.

An episode starts after a `read_file` of the target file with numbered lines and allows 8 turns and 6 edit calls. A model that stops before any edit applies gets one fixed reminder. Context overflow ends the episode; unparseable calls are retried up to three times and counted. All conditions use seed 1; S and L on E2B development also seed 2. Parallel batching makes generation non-deterministic despite fixed seeds: first-call strict success, identical in expectation across S, N, R and RN, ranged from 42 to 49 (E2B) and 66 to 70 (E4B).

Two mid-run fixes changed nothing the model sees: `llama-server` ran from a copy of the b11215 binary with its stack reserve raised from 1 MB to 256 MB, because long tool-call outputs overflowed the default stack, with the driver restarting the server and resending the same turn; and a bare JSON number passed as an `edit_lines` argument is read as text instead of crashing the driver, which affected one task (details in the README). Scripts recompute every number from the saved transcripts on a CPU, and gold edits reproduce the post-commit file under every applier and the line-range tool on all 320 tasks. Code and transcripts: https://github.com/StuartYo/gemma4-edit-anchor (Apache-2.0).

### 4.4 Impact of the repair design

In Table 1, E2B solved 20 of 98 under S and 11 to 17 under the others (p >= 0.078); E4B solved 33 under S, N and R, 35 under RN and 23 under L (p >= 0.087). For E2B, R and RN applied an edit in 74 and 73 tasks against 64 (p = 0.16, 0.18) yet solved fewer.

*Table 1. Target mode, seed 1, tasks, E2B / E4B. "vs S": solved difference with paired 95% bootstrap interval; McNemar p for every pair is in `results/paired.json`.*

| Split | Cond. | Solved | vs S [95% CI] | Applied | Tokens |
|---|---|---|---|---|---|
| dev | S | 20 / 33 | n/a | 64 / 84 | 2313 / 1155 |
| dev | N | 13 / 33 | -7 [-16, 1] / 0 [-7, 7] | 63 / 82 | 1757 / 1217 |
| dev | R | 17 / 33 | -3 [-10, 5] / 0 [-6, 7] | 74 / 79 | 1995 / 1149 |
| dev | RN | 16 / 35 | -4 [-12, 4] / +2 [-4, 8] | 73 / 83 | 1783 / 1266 |
| dev | L | 11 / 23 | -9 [-18, 0] / -10 [-20, 0] | 77 / 89 | 1389 / 1221 |
| held-out | S | 18 / 27 | n/a | 36 / 49 | 1912 / 1163 |
| held-out | RN | 21 / 26 | +3 [-2, 9] / -1 [-6, 4] | 43 / 53 | 1418 / 854 |
| held-out | L | 14 / 24 | -4 [-11, 2] / -3 [-9, 2] | 56 / 60 | 1027 / 965 |

### 4.5 Model size, seed noise and retries

E4B matched exactly on the first call in 70 of 98 development tasks against 42 for E2B, and solved 33 against 20. Copied line numbers and whitespace errors fell from 24% of E2B's rejected calls to 2% of E4B's (Figure 3).

A second seed moved E2B's solved count from 20 to 16 under S (p = 0.50) and from 11 to 18 under L, and its applied count under S from 64 to 51 (p = 0.035). Under S, E2B re-sent an identical failing call 116 times (75 with seed 2); the N hint cut this to 64 and tokens per task from 2313 to 1757, without more solved tasks. E4B's tokens did not fall.

![Figure 3](https://raw.githubusercontent.com/StuartYo/gemma4-edit-anchor/master/figures/fig2_failure_taxonomy.png)
*Figure 3. Near misses and partial rewrites, not copy errors, dominate rejected calls (condition S, development).*

### 4.6 Same calls, different appliers

Strict rejected 76 of E2B's 149 first calls (S, target mode, both splits). Stripping line numbers recovered 11, R 19 and `fuzzy` 60 (Table 2, Figure 4). Wrong-place edits were rare: 2 for R, 7 for `fuzzy`. The larger cost is content: 8 of R's 19 and 31 of `fuzzy`'s 60 recovered edits made the file less similar to the correct one.

*Table 2. First calls of the S runs replayed, target mode, both splits, E2B / E4B. Appliers are nested; all columns except "Applied" count recovered calls. Exact fix: one-hunk tasks only.*

| Applier | Applied | Recovered | Wrong place | Away from gold | Exact fix |
|---|---|---|---|---|---|
| strict | 73 / 111 | n/a | n/a | n/a | n/a |
| lineno | 84 / 113 | 11 / 2 | 0 / 0 | 5 / 2 | 1 / 0 |
| ws (R) | 92 / 117 | 19 / 6 | 2 / 1 | 8 / 4 | 3 / 1 |
| fuzzy | 133 / 148 | 60 / 37 | 7 / 4 | 31 / 24 | 10 / 3 |
| gold edits, each applier | 320 of 320 | n/a | n/a | n/a | n/a |

![Figure 4](https://raw.githubusercontent.com/StuartYo/gemma4-edit-anchor/master/figures/fig3_repair_tradeoff.png)
*Figure 4. Repairs recover quotes, but many recovered edits move the file away from the correct version.*

## 5 Held-out repositories and intent-only tasks

**Held-out repositories.** On click, flask and attrs, run once after the freeze, the pattern held: E2B solved 18 of 60 under S, 21 under RN (p = 0.51) and 14 under L (p = 0.39); E4B solved 27, 26 and 24. L applied an edit in far more tasks (E2B 56 against 36, p < 0.001; E4B 60 against 49, p = 0.001), yet solved fewer, so the extra applied edits were not correct ones.

**Intent-only tasks.** With only the commit subject, copied line numbers became the largest failure class (Table 3), so a unique-match repair has more to fix: replay through R recovers 27 (E2B) and 20 (E4B) first calls. But 10 and 14 of those land outside the lines the real commit changed. Strict calls that applied did so too (8 of 21 and 26 of 56), so this mostly reflects the model choosing another location, which no applier can correct. We did not run R in intent mode, so its effect on outcomes is not tested.

*Table 3. Intent mode, development, condition S, first calls. "Elsewhere": applied outside the real commit's changed lines (±3).*

| Model | Strict applied (elsewhere) | Largest failure classes | R recovers (elsewhere) |
|---|---|---|---|
| E2B | 21 (8) | line numbers 22, near miss 13 of 50 | 27 (10) |
| E4B | 56 (26) | line numbers 13, whitespace 7 of 38 | 20 (14) |

**Where it gets worse.** L solved fewer tasks than S in all four model and split pairs, by 3 to 10, though no single difference was significant. Tasks from 2025 or later (12) were solved at rates not distinguishable from older ones (Fisher p >= 0.29), too few to test memorization.

## 6 Related work

SWE-agent showed that the agent-computer interface matters as much as the model (Yang et al., 2024). Exact string replacement is the edit command of OpenHands (Wang et al., 2024) and Anthropic's text editor tool (Anthropic, 2025); Agentless generates search/replace edits (Xia et al., 2024). Diff-XYZ finds that smaller open models benefit little from any diff format (Glukhov et al., 2025). Althoubi (2026) perturbs 640 edits: content-anchored formats fail visibly, location-anchored ones silently, and an anchor-and-verify applier misapplied 1 edit in 8,320 trials. These studies use large models, gold edits or synthetic perturbations; we found no measurement of edit failures and repair costs on a small open model's own tool calls.

## 7 Threats to validity, limitations and conclusion

**Threats to validity.** The reminder was added after commit `3151b5d` but before any scored run; earlier partial runs are kept in `runs/_discarded_prenudge`. Target-mode prompts contain the post-edit text, which invites class 5, and "solved" measures the interface, not problem solving; intent mode is the check. The seed-noise floor comes from two E2B development seeds only; E4B and held-out noise was not measured. Most conditions have one seed and comparisons are uncorrected, so p-values are descriptive. With the observed discordant pairs, a McNemar test has about 80% power only for differences of roughly 7 to 15 tasks, so smaller effects would go undetected. The repositories are public and may be in training data.

**Limitations.** We tested E2B and E4B only, thinking off, on one laptop, through llama.cpp's parser rather than the competition's vLLM parser, on single-file edits of at most 20 lines. The skill was not evaluated inside the competition harness.

**Conclusion.** The first exact-match edit failed in 54% of tasks for E2B and 30% for E4B, mostly through near misses rather than copy errors. Repairing or hinting the quote did not change exact task success beyond seed noise, and the line-range tool lowered it in every pair we ran, though not significantly in any one. In target mode, unique-match repair rarely misplaced edits and is cheap to add, but should not be expected to solve more tasks; similarity matching adds silent wrong edits.

**AI use.** The author chose the topic and decided to enter. Claude (Anthropic's AI model) did most of the experiment design, harness code, running, analysis and writing. The author reviewed and approved the results and the final text.

## References

- Aider-AI (2025). Search/replace applier. https://github.com/Aider-AI/aider/blob/main/aider/coders/editblock_coder.py
- Althoubi, A. (2026). Look Before You Leap: Pre-Action Verification for LLM Agents. arXiv:2609.11957.
- Anthropic (2025). Text editor tool. https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. Annals of Statistics 7(1). https://doi.org/10.1214/aos/1176344552
- Google. Gemma 4 E2B/E4B QAT Q4_0 GGUF model cards. https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf
- Glukhov, E. et al. (2025). Diff-XYZ: A Benchmark for Evaluating Diff Understanding. arXiv:2510.12487.
- Kaggle (2026). Discussion 747168. https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168
- llama.cpp, build b11215. https://github.com/ggml-org/llama.cpp
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2). https://doi.org/10.1007/BF02295996
- Wang, X. et al. (2024). OpenHands: An Open Platform for AI Software Developers as Generalist Agents. arXiv:2407.16741.
- Xia, C. S. et al. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
- Yang, J. et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv:2405.15793.
