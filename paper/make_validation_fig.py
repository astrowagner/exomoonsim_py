#!/usr/bin/env python3
"""Figure: idl_vs_python.png  (Paper III, Appendix -- validation against IDL).

Side-by-side comparison of the original IDL survey and the Python port on the
identical 9 a x 101 mass grid at 50 uas (50 trials/cell):
  top row    -- detection fraction: IDL | Python | (Python - IDL)
  bottom row -- detection significance: IDL | Python | ratio (Python / IDL)

Inputs are both committed to data/:
  - idl_cubes_dump.txt              (IDL cubes, written by validation/dump_cubes.pro)
  - survey_50muas_ntrials50.npz     (Python survey, from run_survey_50muas.py)

This is a statistical/structural comparison, not a bit match: the two codes use
different RNGs and different orbit fitters (IDL Newton-Raphson vs the port's
Levenberg-Marquardt). Prints the agreement statistics quoted in the Appendix.

Run:  python make_validation_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
IDL = os.path.join(HERE, "data", "idl_cubes_dump.txt")
NPZ = os.path.join(HERE, "data", "survey_50muas_ntrials50.npz")
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "idl_vs_python.png")


def parse_idl_dump(path):
    """Read dump_cubes.pro output -> dict(a, mass, sigcube, perrcube, amperrcube, detfrac)."""
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    na = nm = ntrials = None
    vecs, cubes = {}, {}
    names = {"sigcube", "perrcube", "amperrcube", "detfrac"}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("ntrials="):
            ntrials = int(float(s.split("=")[1]))
        elif s.startswith("na="):
            na = int(float(s.split("=")[1]))
        elif s.startswith("nm="):
            nm = int(float(s.split("=")[1]))
        elif s == "#as":
            i += 1; vecs["as"] = np.array(lines[i].split(), float)
        elif s == "#masses":
            i += 1; vecs["masses"] = np.array(lines[i].split(), float)
        elif s[1:] in names:
            name = s[1:]; rows = []
            for _ in range(na):
                i += 1; rows.append([float(x) for x in lines[i].split()])
            cubes[name] = np.array(rows)
        i += 1
    return dict(ntrials=ntrials, a=vecs["as"], mass=vecs["masses"], **cubes)


idl = parse_idl_dump(IDL)
d = np.load(NPZ, allow_pickle=True)
a_p, m_p = d["a_grid"], d["mass_grid"]
assert np.allclose(idl["a"], a_p) and np.allclose(idl["mass"], m_p), "grid mismatch"

di, dp = idl["detfrac"], d["detfrac"]
si, sp = idl["sigcube"], d["sigcube"]
pi, pp = idl["perrcube"], d["perrcube"]
ai, ap = idl["amperrcube"], d["amperrcube"]

same_bin = (di >= 0.5) == (dp >= 0.5)
good = si > 1.0
ratio = sp[good] / si[good]
det = (di >= 0.5) & (dp >= 0.5)
print(f"detection maps agree (>=0.5 cut): {same_bin.mean()*100:.1f}% "
      f"({int(same_bin.sum())}/{same_bin.size} cells)")
print(f"significance ratio Py/IDL: median={np.median(ratio):.2f} "
      f"16-84%=[{np.percentile(ratio,16):.2f},{np.percentile(ratio,84):.2f}]")
if det.any():
    print(f"period error % (both detect): IDL {np.median(pi[det]):.4f} Py {np.median(pp[det]):.4f}")
    print(f"amplitude err % (both detect): IDL {np.median(ai[det]):.2f} Py {np.median(ap[det]):.2f}")

# --------------------------------------------------------------------------- #
def mesh(ax, C, **kw):
    return ax.pcolormesh(a_p, m_p, C.T, shading="nearest", **kw)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
for ax, C, title in [(axes[0, 0], di, "Detection Fraction (IDL)"),
                     (axes[0, 1], dp, "Detection Fraction (Python)")]:
    im = mesh(ax, C, cmap="viridis", vmin=0, vmax=1); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
im = mesh(axes[0, 2], dp - di, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0, 2].set_title(r"Detection Fraction (Python $-$ IDL)", fontsize=10)
fig.colorbar(im, ax=axes[0, 2], shrink=0.85)

smax = float(np.nanmax([np.nanmax(si), np.nanmax(sp)]))
for ax, C, title in [(axes[1, 0], si, "Significance (IDL)"),
                     (axes[1, 1], sp, "Significance (Python)")]:
    im = mesh(ax, np.clip(C, 1, None), cmap="magma", norm=LogNorm(vmin=1, vmax=smax))
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, shrink=0.85)
with np.errstate(divide="ignore", invalid="ignore"):
    rmap = np.where(si > 1, sp / si, np.nan)
im = mesh(axes[1, 2], rmap, cmap="coolwarm", vmin=1.0, vmax=1.8)
axes[1, 2].set_title("Significance Ratio (Python / IDL)", fontsize=10)
fig.colorbar(im, ax=axes[1, 2], shrink=0.85)
for ax in axes.flat:
    ax.set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
    ax.set_ylabel(r"Moon Mass ($M_\oplus$)")
fig.suptitle(r"IDL vs Python Survey --- $50\,\mu$as, 9$\times$101 Grid, 50 Trials/Cell", fontsize=12)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
