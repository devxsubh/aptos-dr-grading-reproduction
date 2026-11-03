"""Checks that every relative link in the repository's Markdown and HTML files points at a file that exists.

Usage:  python tools/check_links.py        (run from anywhere; exits 1 if a link is broken)
"""
import posixpath
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r'(?:\]\(|src=["\']|href=["\'])([^)"\'\s#][^)"\']*)')
files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
broken = 0
for f in files:
    if not f.endswith((".md", ".html")):
        continue
    text = (ROOT / f).read_text(errors="ignore")
    for target in LINK.findall(text):
        if re.match(r"[a-z]+:|/_blob|mailto:", target):
            continue
        path = urllib.parse.unquote(target.split("#")[0].split("?")[0])
        if not path:
            continue
        resolved = ROOT / posixpath.normpath(posixpath.join(posixpath.dirname(f), path))
        if not resolved.exists():
            print(f"{f}: {target}")
            broken += 1
print(f"{broken} broken link(s)")
sys.exit(1 if broken else 0)
