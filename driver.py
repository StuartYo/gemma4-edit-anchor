"""One-process experiment driver: runs (model, split, mode, cond) jobs, three at a
time against one llama-server with 3 slots. Restarts the server when the model changes.
Resumable: run_exp skips task ids already in an output file."""
import json, subprocess, sys, time, types, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import run_exp

LL = r"E:\tools\llama.cpp\b11215\llama-server.exe"
GGUF = {"E2B": r"E:\models\gemma4\gemma-4-E2B_q4_0-it.gguf", "E4B": r"E:\models\gemma4\gemma-4-E4B_q4_0-it.gguf"}
PORT = 8091

def server(model):
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"])
    time.sleep(3)
    log = open(r"E:\_tmp_claude\gemma4paper\server.log", "w")
    p = subprocess.Popen([LL, "-m", GGUF[model], "--port", str(PORT), "-c", "49152", "-ngl", "99", "-t", "4",
                          "--jinja", "--no-webui", "-np", "3"], stdout=log, stderr=log,
                         creationflags=0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    run_exp.wait_ready(PORT)
    return p

def job(model, split, mode, cond, seed=1):
    out = Path(f"runs/{model}_{split}_{mode}_{cond}_s{seed}.jsonl")
    tasks = [json.loads(l) for l in open(f"data/{split}.jsonl", encoding="utf-8")]
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    args = types.SimpleNamespace(model=model, cond=cond, mode=mode, seed=seed, port=PORT,
                                 max_turns=8, max_edits=6, keep_states=False)
    with open(out, "a", encoding="utf-8") as f:
        for t in tasks:
            if t["id"] in done:
                continue
            f.write(json.dumps(run_exp.run_task(t, args), ensure_ascii=False) + "\n"); f.flush()
    print(time.strftime("%H:%M"), "done", model, split, mode, cond, flush=True)

QUEUE = {
    "E2B": [("dev100", "target", "S"), ("dev100", "target", "RN"), ("dev100", "target", "N"),
            ("dev100", "target", "R"), ("holdout60", "target", "S"), ("holdout60", "target", "RN"),
            ("dev100", "intent", "S")],
    "E4B": [("dev100", "target", "S"), ("dev100", "target", "RN"), ("holdout60", "target", "S"),
            ("holdout60", "target", "RN"), ("dev100", "intent", "S"), ("dev100", "target", "N"),
            ("dev100", "target", "R")],
}
if __name__ == "__main__":
    models = sys.argv[1:] or ["E2B", "E4B"]
    for m in models:
        p = server(m)
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda j: job(m, *j), QUEUE[m]))
        p.kill()
    print("ALLDONE", flush=True)
