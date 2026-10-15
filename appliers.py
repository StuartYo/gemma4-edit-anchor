"""Edit appliers for edit_file(old_string, new_string) calls.

Every applier returns (new_text or None, info). info["span"] is the 0-based
[start, end) line span of the matched region in the original file, so the
caller can check whether the edit landed where it should.

strict    : the competition semantics. old_string must occur exactly once.
lineno    : strip read_file line-number prefixes copied into old_string, then strict.
ws        : line-wise match ignoring trailing spaces and a uniform indentation
            shift; new_string is re-indented by the same shift. Unique match only.
fuzzy     : Aider-style. Best block by difflib ratio >= 0.8. Takes the best one
            even if another block is close (this is the risky variant).
"""
import difflib, re

LINENO = re.compile(r"^\s*\d+(\t|: |\| | \| |→|  )")

def _count_spans(text, old):
    spans, i = [], text.find(old)
    while i != -1:
        spans.append(i)
        i = text.find(old, i + 1)
    return spans

def _char_to_line(text, idx):
    return text.count("\n", 0, idx)

def strict(text, old, new):
    if old == "":
        return None, {"error": "empty"}
    spans = _count_spans(text, old)
    if len(spans) == 0:
        return None, {"error": "not_found"}
    if len(spans) > 1:
        return None, {"error": "multiple", "n": len(spans)}
    i = spans[0]
    s = _char_to_line(text, i)
    e = _char_to_line(text, i + len(old)) + 1
    return text[:i] + new + text[i + len(old):], {"span": (s, e), "how": "exact"}

def strip_lineno(s):
    lines = s.split("\n")
    nonempty = [l for l in lines if l.strip()]
    if nonempty and all(LINENO.match(l) for l in nonempty):
        return "\n".join(LINENO.sub("", l, count=1) for l in lines), True
    return s, False

def lineno(text, old, new):
    r, info = strict(text, old, new)
    if r is not None:
        return r, info
    o2, did_o = strip_lineno(old)
    n2, did_n = strip_lineno(new)
    if not did_o:
        return None, info
    r, info2 = strict(text, o2, n2 if did_n else new)
    if r is not None:
        info2["how"] = "lineno"
    return r, (info2 if r is not None else info)

def _indent(l):
    return len(l) - len(l.lstrip(" "))

def ws(text, old, new):
    r, info = lineno(text, old, new)
    if r is not None:
        return r, info
    old, _ = strip_lineno(old)
    new, _ = strip_lineno(new)
    flines = text.split("\n")
    olines = old.strip("\n").split("\n")
    if not any(l.strip() for l in olines):
        return None, info
    key = [l.strip() for l in olines]
    hits = []
    for s in range(0, len(flines) - len(olines) + 1):
        win = flines[s:s + len(olines)]
        if [l.strip() for l in win] != key:
            continue
        # uniform indentation shift on non-blank lines
        shifts = {_indent(w) - _indent(o) for w, o in zip(win, olines) if o.strip()}
        if len(shifts) == 1:
            hits.append((s, shifts.pop()))
    if len(hits) != 1:
        return None, {"error": "ws_" + ("not_found" if not hits else "multiple"), "n": len(hits)}
    s, shift = hits[0]
    nlines = new.strip("\n").split("\n") if new.strip("\n") else []
    def reind(l):
        if not l.strip():
            return ""
        if shift >= 0:
            return " " * shift + l
        return l[min(-shift, _indent(l)):]
    rep = [reind(l) for l in nlines]
    out = flines[:s] + rep + flines[s + len(olines):]
    return "\n".join(out), {"span": (s, s + len(olines)), "how": "ws", "shift": shift}

def fuzzy(text, old, new, thresh=0.8):
    r, info = ws(text, old, new)
    if r is not None:
        return r, info
    old, _ = strip_lineno(old)
    new, _ = strip_lineno(new)
    flines = text.split("\n")
    olines = old.strip("\n").split("\n")
    k = len(olines)
    if k == 0:
        return None, info
    target = "\n".join(l.strip() for l in olines)
    best, best_s, best_len = 0.0, -1, k
    for L in {k - 1, k, k + 1} - {0}:
        for s in range(0, len(flines) - L + 1):
            cand = "\n".join(l.strip() for l in flines[s:s + L])
            sm = difflib.SequenceMatcher(None, target, cand, autojunk=False)
            if sm.real_quick_ratio() < best or sm.quick_ratio() < best:
                continue
            ratio = sm.ratio()
            if ratio > best:
                best, best_s, best_len = ratio, s, L
    if best < thresh:
        return None, {"error": "fuzzy_not_found", "best": round(best, 3)}
    win = flines[best_s:best_s + best_len]
    shift = _indent(next((w for w in win if w.strip()), "")) - _indent(next((o for o in olines if o.strip()), ""))
    nlines = new.strip("\n").split("\n") if new.strip("\n") else []
    rep = [(" " * shift + l if shift >= 0 else l[min(-shift, _indent(l)):]) if l.strip() else "" for l in nlines]
    out = flines[:best_s] + rep + flines[best_s + best_len:]
    return "\n".join(out), {"span": (best_s, best_s + best_len), "how": "fuzzy", "ratio": round(best, 3)}

APPLIERS = {"strict": strict, "lineno": lineno, "ws": ws, "fuzzy": fuzzy}

def nearest_block(text, old, max_lines=None):
    """Code-side hint: the file block most similar to old_string, with line numbers.

    Returns (start_line_1based, block_text_with_numbers, ratio)."""
    old, _ = strip_lineno(old)
    flines = text.split("\n")
    olines = old.strip("\n").split("\n")
    k = max(1, len(olines))
    target = "\n".join(l.strip() for l in olines)
    best, best_s = 0.0, 0
    for s in range(0, max(1, len(flines) - k + 1)):
        cand = "\n".join(l.strip() for l in flines[s:s + k])
        sm = difflib.SequenceMatcher(None, target, cand, autojunk=False)
        if sm.real_quick_ratio() < best or sm.quick_ratio() < best:
            continue
        r = sm.ratio()
        if r > best:
            best, best_s = r, s
    block = flines[best_s:best_s + k]
    numbered = "\n".join(f"{best_s + 1 + i:6d}\t{l}" for i, l in enumerate(block))
    return best_s + 1, numbered, best
