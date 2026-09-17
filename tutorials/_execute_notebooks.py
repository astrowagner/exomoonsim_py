#!/usr/bin/env python3
"""Execute the tutorial notebooks and store their outputs (text + figures) in place.

GitHub renders the outputs saved inside an .ipynb, so committing executed notebooks
lets readers see every plot and printed result without running anything.

This is a minimal, dependency-free executor for the tutorials' two output kinds:
printed text (captured as a ``stream`` output) and matplotlib figures (captured at
each ``plt.show()`` as PNG ``display_data``, exactly as the inline backend would).
If you have Jupyter installed the standard tool does the same job:

    jupyter nbconvert --to notebook --execute --inplace tutorials/0*.ipynb

Run from the repo root (after building the notebooks with _make_notebooks.py):

    python tutorials/_execute_notebooks.py            # all tutorials
    python tutorials/_execute_notebooks.py 02         # just tutorial 02
"""
import base64
import contextlib
import glob
import io
import json
import os
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")           # headless; figures are captured, not shown
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
sys.path.insert(0, ROOT)
DPI = 96                                              # keeps embedded PNGs a few tens of kB each


def _capture_open_figures(sink):
    """Render every open figure to PNG display_data, then close them."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        w, h = fig.get_size_inches()
        sink.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(buf.getvalue()).decode("ascii"),
                              "text/plain": ["<Figure size %dx%d with %d Axes>"
                                             % (w * fig.dpi, h * fig.dpi, len(fig.axes))]}})
    plt.close("all")


def execute(nb_path):
    nb = json.load(open(nb_path))
    ns = {"__name__": "__main__", "__builtins__": __builtins__}
    count = 0
    t0 = time.time()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        outputs, out, pos = [], io.StringIO(), [0]

        def flush_text():                       # text printed so far -> one stream output
            text = out.getvalue()[pos[0]:]
            if text:
                outputs.append({"output_type": "stream", "name": "stdout", "text": text.splitlines(True)})
            pos[0] = len(out.getvalue())

        def show(*a, **k):                      # keep text/figure order as a kernel would
            flush_text(); _capture_open_figures(outputs)
        plt.show = show

        count += 1
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(code, "<cell %d>" % count, "exec"), ns)
            flush_text(); _capture_open_figures(outputs)         # anything left open at cell end
            ok = True
        except Exception:
            flush_text(); _capture_open_figures(outputs)
            tb = traceback.format_exc()
            outputs.append({"output_type": "error", "ename": "Exception", "evalue": "",
                            "traceback": tb.splitlines()})
            ok = False
        cell["outputs"] = outputs
        cell["execution_count"] = count
        if not ok:
            print("  !! error in cell %d of %s:\n%s" % (count, os.path.basename(nb_path), tb))
            break
    json.dump(nb, open(nb_path, "w"), indent=1)
    return ok, time.time() - t0


if __name__ == "__main__":
    pat = (sys.argv[1] + "*") if len(sys.argv) > 1 else "0*"
    os.chdir(ROOT)                                    # notebooks save small .npz files here (gitignored)
    for nb in sorted(glob.glob(os.path.join(HERE, pat + ".ipynb"))):
        print("executing", os.path.basename(nb), "...", flush=True)
        ok, dt = execute(nb)
        print("  %s in %.0f s" % ("done" if ok else "FAILED", dt), flush=True)
