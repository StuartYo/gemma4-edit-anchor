import appliers as A
F = "def f(x):\n    if x:\n        return 1\n    return 2\n\ndef g():\n    return 2\n"
def test_strict_ok():
    r, i = A.strict(F, "        return 1\n", "        return 3\n"); assert "return 3" in r and i["span"] == (2, 4)
def test_strict_multiple():
    r, i = A.strict(F, "return 2", "x"); assert r is None and i["error"] == "multiple"
def test_lineno():
    old = "     3\t        return 1"
    r, i = A.lineno(F, old, "     3\t        return 9"); assert r and "return 9" in r and i["how"] == "lineno"
def test_ws_indent_shift():
    r, i = A.ws(F, "if x:\n    return 1", "if x:\n    return 5"); assert r and "        return 5" in r and i["shift"] == 4
def test_ws_ambiguous_rejected():
    r, i = A.ws(F, "  return 2", "  return 7"); assert r is None
def test_fuzzy_takes_close():
    r, i = A.fuzzy(F, "if x:\n        return 11", "if x:\n        return 5"); assert r is not None and i["how"] == "fuzzy"
def test_positive_control_all_appliers():
    for name, fn in A.APPLIERS.items():
        r, i = fn(F, "def g():\n    return 2\n", "def g():\n    return 3\n"); assert r and "return 3" in r, name
