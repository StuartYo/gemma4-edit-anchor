"""Word count for the Kaggle 3,000-word limit: counts words in the paper body,
excluding the reference list, image lines and table pipes."""
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
body = re.split(r"(?m)^#+\s*References\s*$", t)[0]
body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.%/]*", body)
print(len(words), "words (body, without references)")
print(len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.%/]*", t)), "words (whole file)")
