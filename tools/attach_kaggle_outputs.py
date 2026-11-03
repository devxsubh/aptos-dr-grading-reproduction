"""Rebuilds an "executed" copy of a Kaggle kernel notebook: the cells exactly as they ran, plus their outputs.

Why this exists: Kaggle's API returns a kernel's log and its saved files, but not the executed notebook. This script
puts the outputs back next to the cells so the record can be read like any Jupyter notebook.

Usage:  python tools/attach_kaggle_outputs.py <kernel notebook.ipynb> <kernel log> <results dir> [<out.ipynb>]

How the outputs are reconstructed (so nobody mistakes them for a native execution):
  * stdout / stderr come from the kernel log. Each printed line is given to the first code cell, from the current
    one onwards, whose code prints that text (directly or through a function it calls). Lines that match nothing
    stay with the cell that printed the line before.
  * tables: when a cell saves a CSV (`.to_csv("name.csv")`), that CSV from the results folder is attached to the
    cell as a plain-text table.
  * figures: when a cell saves a PNG (`.savefig("name.png")`), that image is attached to the cell.
A banner cell at the top of the output notebook says the same.
"""
import ast
import base64
import json
import re
import sys
from pathlib import Path

import pandas as pd

BANNER = (
    "> **Executed record of a Kaggle kernel.** The cells below are the cells that ran. Kaggle does not return the executed "
    "notebook, so the outputs were put back afterwards by `tools/attach_kaggle_outputs.py`: printed text comes from the "
    "kernel's log (each line assigned to the cell whose code prints it), and each table or figure comes from the CSV or PNG "
    "file that cell saved. Everything shown was produced by the run; the placement is a reconstruction. Kernel: `{kernel}`.")


def source_of(cell):
    return "".join(cell["source"])


def parse(src):
    lines = [("pass  # magic" if l.lstrip().startswith("!") else l) for l in src.splitlines()]
    try:
        return ast.parse("\n".join(lines))
    except SyntaxError:
        return None


def fragment(call):
    # The literal text a print() call starts with, used to recognise its output line.
    if not call.args:
        return None
    a = call.args[0]
    if isinstance(a, ast.Call) and ast.unparse(a.func) == "json.dumps":
        return "{"                                       # a pretty-printed JSON object starts with a line holding only "{"
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        text = a.value
    elif isinstance(a, ast.JoinedStr):
        parts = [v.value for v in a.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        text = max(parts, key=len) if parts else ""
    else:
        return None
    text = text.strip("\n").strip()
    return text[:40] if len(text) >= 5 else None


def calls_in(node):
    prints, names = [], set()
    stack = [node]
    while stack:
        n = stack.pop()
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue                                 # definitions run only when called
            stack.append(ch)
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
                if f.id == "print":
                    frag = fragment(n)
                    if frag:
                        prints.append(frag)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return prints, names


def analyse(code_cells):
    funcs = {}                                           # function name -> (fragments printed inside, names it calls)
    info = []
    for c in code_cells:
        tree = parse(source_of(c))
        own, called = [], set()
        if tree:
            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    frs, nm = [], set()
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Call):
                            if isinstance(sub.func, ast.Name) and sub.func.id == "print":
                                fr = fragment(sub)
                                if fr:
                                    frs.append(fr)
                            elif isinstance(sub.func, ast.Name):
                                nm.add(sub.func.id)
                            elif isinstance(sub.func, ast.Attribute):
                                nm.add(sub.func.attr)
                    funcs[stmt.name] = (frs, nm)
                else:
                    p, n = calls_in(stmt)
                    own += p
                    called |= n
        info.append((own, called))
    result = []
    for own, called in info:
        frags, seen, todo = list(own), set(), list(called)
        while todo:
            n = todo.pop()
            if n in seen or n not in funcs:
                continue
            seen.add(n)
            frags += funcs[n][0]
            todo += list(funcs[n][1])
        result.append(frags)
    return result


def executed_calls(cell):
    # Source text of the calls a cell runs at top level (not code inside function definitions, which only runs when called).
    tree = parse(source_of(cell))
    if not tree:
        return ""
    chunks = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack = [stmt]
        while stack:
            n = stack.pop()
            for ch in ast.iter_child_nodes(n):
                if not isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    stack.append(ch)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("to_csv", "savefig"):
                chunks.append(ast.unparse(n))
    return "\n".join(chunks)


def read_log(path):
    text = Path(path).read_text()
    try:
        events = json.loads(text)
    except json.JSONDecodeError:
        events = json.loads(text.rstrip().rstrip(",") + "]")
    return [(e.get("stream_name", "stdout"), e.get("data", "")) for e in events]


def find_file(results, name):
    hits = sorted(Path(results).rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def main(nb_path, log_path, results_dir, out_path=None):
    nb = json.loads(Path(nb_path).read_text())
    code_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    code_cells = [nb["cells"][i] for i in code_idx]
    frags = analyse(code_cells)
    streams = {k: {"stdout": [], "stderr": []} for k in range(len(code_cells))}
    cur = 0
    for stream, data in read_log(log_path):
        if stream == "stdout":
            for line in data.splitlines(True):
                text = line.strip()
                if text:
                    for j in range(cur, len(code_cells)):
                        if any((f == text) if f == "{" else (f in text) for f in frags[j]):
                            cur = j
                            break
                streams[cur]["stdout"].append(line)
        else:
            streams[cur]["stderr"].append(data)
    count = 0
    for k, cell in enumerate(code_cells):
        outs = []
        for name in ("stdout", "stderr"):
            if streams[k][name]:
                outs.append({"output_type": "stream", "name": name, "text": "".join(streams[k][name]).splitlines(True)})
        src = executed_calls(cell)
        for csv in dict.fromkeys(re.findall(r"""to_csv\([^)]*?["']([\w\-.]+\.csv)["']""", src)):
            f = find_file(results_dir, csv)
            if f:
                df = pd.read_csv(f)
                shown = df.head(60).to_string(index=False)
                note = f"{csv}" + (f" (first 60 of {len(df)} rows)" if len(df) > 60 else "")
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"text/plain": (note + "\n" + shown).splitlines(True)}})
        for png in dict.fromkeys(re.findall(r"""savefig\([^)]*?["']([\w\-.]+\.png)["']""", src)):
            f = find_file(results_dir, png)
            if f:
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": base64.b64encode(f.read_bytes()).decode(), "text/plain": [f"<{png}>"]}})
        cell["outputs"] = outs
        if outs:
            count += 1
            cell["execution_count"] = count
        else:
            cell["execution_count"] = None
    kernel = Path(nb_path).parent.name
    banner = {"cell_type": "markdown", "metadata": {}, "source": BANNER.format(kernel=kernel).splitlines(True)}
    nb["cells"].insert(0, banner)
    out = Path(out_path) if out_path else Path(nb_path).with_name("executed.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {count} of {len(code_cells)} code cells have outputs")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
