r"""Start driver.py as an independent background process on Windows (no console, own
process group), so Ctrl+C or closing the shell does not stop it. Resumable: finished task
ids are skipped. Output goes to runs/driver.log and runs/driver.err.

usage: python start_driver.py [batch numbers...]   (default: all batches in driver.BATCHES)
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
batches = sys.argv[1:]
(ROOT / "runs").mkdir(exist_ok=True)
log = open(ROOT / "runs" / "driver.log", "a", encoding="utf-8")
log.write(time.strftime("%H:%M") + " start_driver.py: " + " ".join(batches) + "\n"); log.flush()
err = open(ROOT / "runs" / "driver.err", "a", encoding="utf-8")
flags = (0x08000000 | 0x00000200) if sys.platform == "win32" else 0  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
p = subprocess.Popen([sys.executable, "-u", "driver.py", *batches], cwd=ROOT, stdout=log, stderr=err,
                     stdin=subprocess.DEVNULL, creationflags=flags)
(ROOT / "runs" / "driver.pid").write_text(str(p.pid))
print("driver PID", p.pid)
