---
name: anchored_edit
description: Apply an old_string/new_string edit when edit_file reports "not found". Repairs copied line numbers and indentation drift only when the match is unique; otherwise prints the closest block of the file so you can copy it exactly.
---

# anchored_edit

Use this after `edit_file` fails with "old_string not found".

Run:

    run_skill_script anchored_edit scripts/anchored_edit.py --file <path> --old <old.txt> --new <new.txt>

Write old_string and new_string to two temporary files first (avoids shell quoting).

What it does, in order:
1. Exact match (same as edit_file).
2. Removes read_file line-number prefixes if every line of old_string has one.
3. Matches line by line ignoring trailing spaces and one uniform indentation shift, and re-indents new_string by the same shift.

It writes the file only if exactly one location matches. If nothing or more than one location matches, it changes nothing and prints the closest block with line numbers. Copy that block (without numbers) as your next old_string.

It never uses similarity-threshold (fuzzy) matching, because in our measurements that is where edits land in the wrong place.
