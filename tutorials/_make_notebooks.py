#!/usr/bin/env python3
"""Build the tutorial notebooks (.ipynb) from the cell-marked tutorial scripts (.py).

Each tutorial is authored ONCE as a plain Python script using lightweight cell markers,
so the runnable script and the notebook never drift apart:

    # %% [markdown]      -> a markdown cell; the following "# " comment lines are its text
    # %%                 -> a code cell, until the next marker

The script's module docstring becomes the notebook's title cell.  A code cell whose first
line is ``if __name__ == "__main__":`` (needed in the .py for multiprocessing on macOS)
has that line stripped and its body dedented for the notebook, where ``__name__`` is
already ``"__main__"`` and the guard is unnecessary.

Run from the repo root:   python tutorials/_make_notebooks.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = 'if __name__ == "__main__":'


def _split_cells(src):
    """Return [(kind, text), ...] with kind in {'markdown', 'code'}."""
    lines = src.splitlines()
    cells, kind, buf = [], None, []
    # module docstring -> title markdown cell
    m = re.match(r'\s*"""(.*?)"""', src, re.S)
    if m:
        cells.append(("markdown", m.group(1).strip()))
        lines = src[m.end():].splitlines()

    def flush():
        if kind and "".join(buf).strip():
            text = "\n".join(buf).strip("\n")
            if kind == "markdown":
                text = "\n".join(re.sub(r"^# ?", "", l) for l in text.splitlines())
            else:
                body = text.splitlines()
                if body and body[0].strip() == GUARD:          # strip the script-only guard
                    body = [l[4:] if l.startswith("    ") else l for l in body[1:]]
                text = "\n".join(body).strip("\n")
            cells.append((kind, text))

    for l in lines:
        if l.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif l.startswith("# %%"):
            flush(); kind, buf = "code", []
        elif kind:
            buf.append(l)
    flush()
    return cells


def build(py_path):
    cells = _split_cells(open(py_path).read())
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                    "name": "python3"},
                                     "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, text in cells:
        src = [l + "\n" for l in text.splitlines()]
        if src: src[-1] = src[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = py_path[:-3] + ".ipynb"
    json.dump(nb, open(out, "w"), indent=1)
    return out, len(cells)


if __name__ == "__main__":
    for py in sorted(glob.glob(os.path.join(HERE, "[0-9][0-9]_*.py"))):
        out, n = build(py)
        print("wrote %s  (%d cells)" % (os.path.basename(out), n))
