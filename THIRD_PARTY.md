# Third-party code in data/ and runs/

`data/*.jsonl` and `runs/*.jsonl` contain source files copied from the public git history of these projects, used here as benchmark inputs. Each file keeps its original license:

| Project | License | Source |
|---|---|---|
| Textualize/rich | MIT | https://github.com/Textualize/rich |
| psf/requests | Apache-2.0 | https://github.com/psf/requests |
| encode/httpx | BSD-3-Clause | https://github.com/encode/httpx |
| fastapi/fastapi | MIT | https://github.com/fastapi/fastapi |
| pallets/click | BSD-3-Clause | https://github.com/pallets/click |
| pallets/flask | BSD-3-Clause | https://github.com/pallets/flask |
| python-attrs/attrs | MIT | https://github.com/python-attrs/attrs |

Every task records the repository and commit SHA, so `build_tasks.py` can rebuild the set from fresh clones. No data from the Kaggle competition is included.

Model: Gemma 4 E2B and E4B instruction-tuned QAT Q4_0 GGUF files published by Google on Hugging Face (`google/gemma-4-E2B-it-qat-q4_0-gguf`, `google/gemma-4-E4B-it-qat-q4_0-gguf`), Apache-2.0. Inference: llama.cpp build b11215 (Vulkan), MIT.
