"""Cut the paper's task sets from the pools: the first K tasks of each repository, in file order.

usage: python cut_splits.py data/dev.jsonl data/dev100.jsonl 25
       python cut_splits.py data/holdout.jsonl data/holdout60.jsonl 20
"""
import json, sys
from collections import Counter

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = Counter()
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
    for line in f:
        repo = json.loads(line)["repo"]
        if seen[repo] < k:
            seen[repo] += 1
            g.write(line)
print(dst, dict(seen))
