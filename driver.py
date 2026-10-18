"""One-process experiment driver: runs (model, split, mode, cond) jobs, three at a
time against one llama-server with 3 slots. Restarts the server when the model changes.
Resumable: run_exp skips task ids already in an output file."""
import json, subprocess, sys, threading, time, types, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import run_exp

# Same b11215 build; this copy only raises the PE header stack reserve from 1 MB to 256 MB.
# The stock exe dies with 0xC00000FD (stack overflow in llama-common.dll) while parsing a
# long Gemma 4 tool-call string. Code, DLLs, model, server flags and sampling are unchanged.
LL = r"E:\tools\llama.cpp\b11215\llama-server-stack256m.exe"
GGUF = {"E2B": r"E:\models\gemma4\gemma-4-E2B_q4_0-it.gguf", "E4B": r"E:\models\gemma4\gemma-4-E4B_q4_0-it.gguf"}
PORT = 8091
SRVLOG = r"E:\_tmp_claude\gemma4paper\server.log"
KILL_ALL = "Get-Process llama-server,llama-server-stack256m -ErrorAction SilentlyContinue | Stop-Process -Force"

def launch(model, mode):
    log = open(SRVLOG, mode)
    return subprocess.Popen([LL, "-m", GGUF[model], "--port", str(PORT), "-c", "49152", "-ngl", "99", "-t", "4",
                             "--jinja", "--no-webui", "-np", "3", "--cache-ram", "512"], stdout=log, stderr=log,
                            creationflags=0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS

class Server:
    """llama-server plus a watchdog: if it exits during a batch, start it again with the same
    arguments. In-flight requests get a connection error and run_exp retries the same turn
    after wait_ready, so no task is skipped or scored as a failure."""
    def __init__(self, model):
        self.model, self.stop, self.restarts = model, threading.Event(), 0
        self.p = launch(model, "w")
        run_exp.wait_ready(PORT)
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while not self.stop.wait(5):
            rc = self.p.poll()
            if rc is None:
                continue
            self.restarts += 1
            print(time.strftime("%H:%M"), f"llama-server exited rc={hex(rc & 0xffffffff)}, restart #{self.restarts}", flush=True)
            if self.restarts > 30:
                print(time.strftime("%H:%M"), "too many llama-server restarts, watchdog gives up", flush=True)
                return
            time.sleep(3)
            self.p = launch(self.model, "a")

    def kill(self):
        self.stop.set()
        self.p.kill()

def server(model):
    subprocess.run(["powershell", "-NoProfile", "-Command", KILL_ALL])
    time.sleep(3)
    return Server(model)

def job(model, split, mode, cond, part=None, seed=1):
    suffix = f"_p{part[0]}" if part is not None else ""
    out = Path(f"runs/{model}_{split}_{mode}_{cond}_s{seed}{suffix}.jsonl")
    tasks = [json.loads(l) for l in open(f"data/{split}.jsonl", encoding="utf-8")]
    if part is not None:  # (i, n): i-th of n slices, so one batch holds the lock < 1 hour
        i, n = part
        k = (len(tasks) + n - 1) // n
        tasks = tasks[i * k:(i + 1) * k]
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    args = types.SimpleNamespace(model=model, cond=cond, mode=mode, seed=seed, port=PORT,
                                 max_turns=8, max_edits=6, keep_states=False)
    with open(out, "a", encoding="utf-8") as f:
        for t in tasks:
            if t["id"] in done:
                continue
            try:
                rec = run_exp.run_task(t, args)
            except Exception as e:  # harness bug on one task: log it, leave the id undone, keep the batch going
                print(time.strftime("%H:%M"), "HARNESS_ERROR", model, split, mode, cond, t["id"], repr(e), flush=True)
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    print(time.strftime("%H:%M"), "done", model, split, mode, cond, flush=True)

LOCK = Path("E:/_tmp_claude/render.lock")

OTHER = Path("E:/lyric-pv/render.lock")  # video renders lock this (mkdir style)

def acquire():
    while True:
        if OTHER.exists():
            time.sleep(60)
            continue
        try:
            with open(LOCK, "x", encoding="utf-8") as f:
                f.write(f"Claude Gemma4論文 session {time.strftime('%Y-%m-%d %H:%M:%S')} llama-server Gemma4 batch")
            print(time.strftime("%H:%M"), "lock acquired", flush=True)
            return
        except (FileExistsError, PermissionError):
            time.sleep(60)

def release():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass
    print(time.strftime("%H:%M"), "lock released", flush=True)

H0, H1 = (0, 2), (1, 2)
BATCHES = [
    ("E2B", [("dev100", "target", "S"), ("dev100", "target", "RN"), ("dev100", "target", "N")]),
    ("E2B", [("dev100", "target", "R", H0), ("holdout60", "target", "S"), ("holdout60", "target", "RN")]),
    ("E2B", [("dev100", "target", "R", H1), ("dev100", "intent", "S", H0), ("dev100", "intent", "S", H1)]),
    ("E4B", [("dev100", "target", "S", H0), ("dev100", "target", "RN", H0), ("holdout60", "target", "S")]),
    ("E4B", [("dev100", "target", "S", H1), ("dev100", "target", "RN", H1), ("holdout60", "target", "RN")]),
    ("E4B", [("dev100", "intent", "S", H0), ("dev100", "intent", "S", H1), ("dev100", "target", "N", H0)]),
    ("E4B", [("dev100", "target", "N", H1), ("dev100", "target", "R", H0), ("dev100", "target", "R", H1)]),
    # 7-10: line-range interface with first-line anchor check (condition L), added after the E2B dev results
    ("E2B", [("dev100", "target", "L", H0), ("dev100", "target", "L", H1), ("holdout60", "target", "L")]),
    ("E4B", [("dev100", "target", "L", H0), ("dev100", "target", "L", H1), ("holdout60", "target", "L")]),
    # 9: noise floor, a second sampling seed of the baseline
    ("E2B", [("dev100", "target", "S", H0, 2), ("dev100", "target", "S", H1, 2), ("dev100", "target", "L", None, 2)]),
]

if __name__ == "__main__":
    which = [int(x) for x in sys.argv[1:]] or list(range(len(BATCHES)))
    for i in which:
        m, jobs = BATCHES[i]
        acquire()
        p = None
        try:
            p = server(m)
            with ThreadPoolExecutor(3) as ex:
                list(ex.map(lambda j: job(m, *j), jobs))
        finally:
            if p is not None:
                p.kill()
            subprocess.run(["powershell", "-NoProfile", "-Command", KILL_ALL])
            release()
        print(time.strftime("%H:%M"), "batch", i, "done", flush=True)
        time.sleep(90)  # let anyone waiting on the lock take it
    print("ALLDONE", flush=True)
