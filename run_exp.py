"""Run edit tasks against a local llama-server (OpenAI-compatible) with the
competition's public tool signatures (read_file, edit_file).

usage:
  python run_exp.py --tasks data/dev.jsonl --model E2B --cond S --mode target \
      --seed 1 --out runs/E2B_dev_target_S_s1.jsonl [--limit N] [--port 8091]

cond:
  S  : strict applier, harness-style error text on failure (baseline)
  N  : strict applier, on failure the tool result also shows the nearest block
       of the file (code-side hint), so the model can copy it
  R  : silent repair (lineno + whitespace/indent, unique match only), else S error
  RN : silent repair, else nearest-block hint
mode:
  target : the prompt shows the lines that must appear after the edit (ground truth known)
  intent : the prompt shows only the commit subject (realistic, no ground truth)
"""
import argparse, difflib, json, time, urllib.request, urllib.error
from pathlib import Path
import appliers as A

SYSTEM = (
    "You are a software engineering agent working in a Python repository at /workspace. "
    "Use the tools to make the requested change. edit_file replaces old_string with new_string; "
    "old_string must match the file exactly, including indentation, and must occur exactly once. "
    "Do not include line numbers in old_string or new_string. When the change is complete, reply "
    "with a short summary and no tool call."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Reads a file from /workspace with 1-indexed inclusive line slicing.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replaces old_string with new_string in an existing non-empty file inside /workspace.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "allow_multiple": {"type": "boolean"}},
            "required": ["filepath", "old_string", "new_string"]}}},
]

SYSTEM_L = (
    "You are a software engineering agent working in a Python repository at /workspace. "
    "Use the tools to make the requested change. edit_lines replaces a range of lines, using the "
    "line numbers shown by read_file. expected_first_line must be the current text of start_line, "
    "copied from read_file without the number; the edit is rejected if it does not match. "
    "Do not include line numbers in new_content. When the change is complete, reply with a short "
    "summary and no tool call."
)
TOOLS_L = [TOOLS[0], {"type": "function", "function": {
    "name": "edit_lines",
    "description": "Replaces lines start_line..end_line (1-indexed, inclusive, as numbered by read_file) with new_content. Use end_line = start_line - 1 to insert before start_line.",
    "parameters": {"type": "object", "properties": {
        "filepath": {"type": "string"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
        "expected_first_line": {"type": "string"},
        "new_content": {"type": "string"}},
        "required": ["filepath", "start_line", "end_line", "expected_first_line", "new_content"]}}}]

def apply_lines(text, start, end, expected, new):
    """Location-anchored edit with a content check on the first line (anchor-and-verify)."""
    lines = text.split("\n")
    trail = text.endswith("\n")
    if trail:
        lines = lines[:-1]
    n = len(lines)
    try:
        start, end = int(start), int(end)
    except Exception:
        return None, {"error": "bad_range"}
    if not (1 <= start <= n + 1 and start - 1 <= end <= n):
        return None, {"error": "bad_range", "n": n}
    exp, _ = A.strip_lineno(expected or "")
    exp = exp.split("\n")[0].strip()
    cur = lines[start - 1].strip() if start <= n else ""
    if exp != cur:
        return None, {"error": "anchor_mismatch"}
    new, _ = A.strip_lineno(new or "")
    nl = new.split("\n") if new != "" else []
    if nl and nl[-1] == "" and new.endswith("\n"):
        nl = nl[:-1]
    out = lines[:start - 1] + nl + lines[end:]
    return "\n".join(out) + ("\n" if trail else ""), {"span": (start - 1, max(end, start)), "how": "lines"}

def numbered(text, start=1, end=None):
    lines = text.split("\n")
    if text.endswith("\n"):
        lines = lines[:-1]
    end = len(lines) if end is None else min(end, len(lines))
    return "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(max(1, start), end + 1))

def gold_regions(before, after):
    """Changed line spans in `before` (0-based [s,e)) and target blocks from `after`."""
    b, a = before.split("\n"), after.split("\n")
    sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
    spans, blocks = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        spans.append((i1, max(i2, i1 + 1)))
        lo, hi = max(0, j1 - 2), min(len(a), j2 + 2)
        blocks.append("\n".join(a[lo:hi]))
    return spans, blocks

def norm(t):
    return "\n".join(l.rstrip() for l in t.strip("\n").split("\n"))

def chat(port, messages, seed, max_tokens=2048, tools=None):
    body = {"messages": messages, "tools": tools or TOOLS, "temperature": 1.0, "top_p": 0.95,
            "top_k": 64, "seed": seed, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

def apply_call(cond, text, old, new):
    if cond in ("R", "RN"):
        return A.ws(text, old, new)
    return A.strict(text, old, new)

def error_text(cond, text, old, info):
    e = info.get("error", "")
    if "multiple" in e:
        msg = f"Error: old_string occurs {info.get('n', 'several')} times in the file. Add more surrounding lines so it is unique."
    else:
        msg = "Error: old_string was not found in the file."
    if cond in ("N", "RN"):
        start, block, ratio = A.nearest_block(text, old)
        msg += ("\nThe closest block in the file (lines with numbers, copy it without the numbers):\n" + block)
    return msg

def wait_ready(port):
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server not ready")

def run_task(t, args):
    wait_ready(args.port)
    path = t["path"]
    text = t["before"]
    spans, blocks = gold_regions(t["before"], t["after"])
    if args.mode == "target":
        target = "\n\n".join(f"--- block {i+1} ---\n{b}" for i, b in enumerate(blocks))
        task = (f"Edit {path}. After your change, the file must contain the following lines "
                f"(each block is shown with up to two unchanged lines of context around it):\n\n{target}\n\n"
                "Change only what is needed to make these blocks appear.")
    else:
        task = f"Make this change in {path} (commit message): {t['subject']}"
    L = args.cond == "L"
    msgs = [
        {"role": "system", "content": SYSTEM_L if L else SYSTEM},
        {"role": "user", "content": task},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"filepath": path})}}]},
        {"role": "tool", "tool_call_id": "c0", "content": numbered(text)},
    ]
    log = {"id": t["id"], "repo": t["repo"], "mode": args.mode, "cond": args.cond,
           "model": args.model, "seed": args.seed, "calls": [], "turns": 0,
           "gold_spans": spans, "n_lines": t["n_lines_before"]}
    t0 = time.time()
    n_edit = 0
    toks = 0
    for turn in range(args.max_turns):
        resp = None
        attempt = 0
        while attempt < 3:
            try:
                resp = chat(args.port, msgs, args.seed * 1000 + turn * 10 + attempt, tools=TOOLS_L if L else TOOLS)
                break
            except urllib.error.HTTPError as ex:
                if ex.code == 400:  # request too long for the context window: end the episode
                    log["context_overflow"] = True
                    log["error"] = ex.read().decode("utf-8", "replace")[:300]
                    break
                if ex.code != 500:  # 503 loading etc.: not the model's fault, wait and retry
                    time.sleep(5); wait_ready(args.port); continue
                body = ex.read().decode("utf-8", "replace")
                kind = "parse_errors" if "parse" in body.lower() else "server_errors"
                log[kind] = log.get(kind, 0) + 1
                log["error"] = body[:300]
                attempt += 1
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(5); wait_ready(args.port); continue
        if resp is None:
            break
        log.pop("error", None)
        log["turns"] += 1
        toks += resp.get("usage", {}).get("completion_tokens", 0)
        m = resp["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs} if tcs
                    else {"role": "assistant", "content": m.get("content") or ""})
        if not tcs:
            log["final_msg"] = (m.get("content") or "")[:300]
            applied_any = any(c.get("applied") for c in log["calls"])
            if not applied_any and log.get("nudges", 0) < 1:
                # one uniform nudge, like a harness that rejects a turn with no action
                log["nudges"] = log.get("nudges", 0) + 1
                if n_edit == 0:
                    log["stopped_without_edit_first"] = True
                msgs.append({"role": "user", "content": "You have not changed the file yet. Make the change now by calling " + ("edit_lines." if L else "edit_file.")})
                continue
            break
        for tc in tcs:
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments", "")
            try:
                a = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                a = None
            rec = {"turn": turn, "name": name, "raw": raw if isinstance(raw, str) else json.dumps(raw)}
            if name == "edit_file" and isinstance(a, dict) and "old_string" in a and "new_string" in a:
                n_edit += 1
                before_text = text
                res, info = apply_call(args.cond, text, a["old_string"], a["new_string"])
                rec.update({"old": a["old_string"], "new": a["new_string"], "info": info,
                            "text_before_call": None})
                if res is not None:
                    text = res
                    out = "File edited successfully."
                else:
                    out = error_text(args.cond, before_text, a["old_string"], info)
                rec["applied"] = res is not None
                rec["state_hash"] = hash(before_text)
                rec["before_call"] = before_text if not rec["applied"] or args.keep_states else None
            elif L and name == "edit_lines" and isinstance(a, dict) and "start_line" in a and "new_content" in a:
                n_edit += 1
                before_text = text
                res, info = apply_lines(text, a.get("start_line"), a.get("end_line", a.get("start_line")),
                                        a.get("expected_first_line", ""), a.get("new_content", ""))
                rec.update({"lines": [a.get("start_line"), a.get("end_line")],
                            "expected": a.get("expected_first_line"), "new": a.get("new_content"), "info": info})
                if res is not None:
                    text = res
                    out = "File edited successfully."
                elif info.get("error") == "anchor_mismatch":
                    st = int(a.get("start_line"))
                    out = ("Error: expected_first_line does not match line %d. Current lines:\n" % st) + \
                          numbered(before_text, max(1, st - 2), st + 4)
                else:
                    out = "Error: start_line/end_line out of range (file has %s lines)." % info.get("n", "?")
                rec["applied"] = res is not None
                rec["before_call"] = before_text if not rec["applied"] else None
            elif name == "read_file" and isinstance(a, dict):
                out = numbered(text, a.get("start_line") or 1, a.get("end_line"))
            else:
                out = "Error: invalid tool call or missing parameters."
                rec["malformed"] = True
            log["calls"].append(rec)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", "x"), "content": out})
        if n_edit >= args.max_edits:
            break
    log["secs"] = round(time.time() - t0, 1)
    log["completion_tokens"] = toks
    log["final"] = text
    log["changed"] = norm(text) != norm(t["before"])
    if args.mode == "target":
        log["solved"] = norm(text) == norm(t["after"])
    return log

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks"); ap.add_argument("--model"); ap.add_argument("--cond", default="S")
    ap.add_argument("--mode", default="target"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=8091); ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-edits", type=int, default=6); ap.add_argument("--keep-states", action="store_true")
    args = ap.parse_args()
    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    if args.limit:
        tasks = tasks[:args.limit]
    done = set()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8")}
    with open(outp, "a", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            if t["id"] in done:
                continue
            log = run_task(t, args)
            f.write(json.dumps(log, ensure_ascii=False) + "\n"); f.flush()
            print(i, t["id"], "solved" if log.get("solved") else ("changed" if log["changed"] else "-"),
                  log["secs"], flush=True)

if __name__ == "__main__":
    main()
