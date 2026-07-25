#!/usr/bin/env python3
"""
Compare a FULL-GRID IDL survey (dumped by dump_cubes.pro) against a Python survey
(.npz), for grids too large to print cell-by-cell (e.g. the 9 x 101 production
run).  Prints summary agreement statistics and writes a side-by-side figure
(detection fraction and significance: IDL | Python | difference/ratio).

Usage:
    python compare_maps.py <idl_cubes_dump.txt> <python_survey.npz> [out.png]

Like compare_idl_py.py, this is a statistical/structural comparison, not a bit
match: the two codes use different RNGs and different orbit fitters.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parse_idl_dump(path):
    """Read dump_cubes.pro output -> dict(a, mass, sigcube, perrcube, amperrcube,
    detfrac) with cubes shaped [na, nm]."""
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


def main(argv=None):
    argv = argv or sys.argv[1:]
    idl_path, npz_path = argv[0], argv[1]
    out_png = argv[2] if len(argv) > 2 else "compare_maps.png"

    idl = parse_idl_dump(idl_path)
    d = np.load(npz_path, allow_pickle=True)
    a_i, m_i = idl["a"], idl["mass"]
    a_p, m_p = d["a_grid"], d["mass_grid"]

    print(f"IDL grid : {a_i.size} a x {m_i.size} mass  (ntrials={idl['ntrials']})")
    print(f"Py  grid : {a_p.size} a x {m_p.size} mass  (ntrials={int(d['ntrials'])})")
    if a_i.shape != a_p.shape or m_i.shape != m_p.shape:
        print("!! grid shapes differ -- aborting"); return
    if not (np.allclose(a_i, a_p) and np.allclose(m_i, m_p)):
        print("!! grid VALUES differ -- aborting"); return

    di, dp = idl["detfrac"], d["detfrac"]                 # [na, nm]
    si, sp = idl["sigcube"], d["sigcube"]
    pi, pp = idl["perrcube"], d["perrcube"]
    ai, ap = idl["amperrcube"], d["amperrcube"]

    # ---- detection-map agreement ----
    same_bin = (di >= 0.5) == (dp >= 0.5)
    close_02 = np.abs(di - dp) <= 0.2
    print("\nDetection maps:")
    print(f"  same detected/not (>=0.5 cut) : {same_bin.mean()*100:5.1f}%  "
          f"({int(same_bin.sum())}/{same_bin.size} cells)")
    print(f"  detfrac within 0.2            : {close_02.mean()*100:5.1f}%")

    # ---- significance ratio (where IDL sig is meaningfully positive) ----
    good = si > 1.0
    ratio = sp[good] / si[good]
    print("\nSignificance ratio Py/IDL (cells with IDL sig>1):")
    print(f"  median={np.median(ratio):.2f}  16-84%=[{np.percentile(ratio,16):.2f}, "
          f"{np.percentile(ratio,84):.2f}]  (constant offset expected/OK)")

    # ---- period & amplitude error, where BOTH detect ----
    det = (di >= 0.5) & (dp >= 0.5)
    if det.any():
        print("\nWhere both detect (median values):")
        print(f"  period error %  : IDL {np.median(pi[det]):.4f}  Py {np.median(pp[det]):.4f}")
        print(f"  amplitude err % : IDL {np.median(ai[det]):.2f}   Py {np.median(ap[det]):.2f}")

    # ---- figure: detfrac (IDL|Py|diff) and significance (IDL|Py|ratio) ----
    def mesh(ax, C, **kw):
        return ax.pcolormesh(a_p, m_p, C.T, shading="nearest", **kw)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    # detection fraction
    for ax, C, title in [(axes[0, 0], di, "Detection fraction (IDL)"),
                         (axes[0, 1], dp, "Detection fraction (Python)")]:
        im = mesh(ax, C, cmap="viridis", vmin=0, vmax=1); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    im = mesh(axes[0, 2], dp - di, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 2].set_title("detfrac  (Python - IDL)")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.85)
    # significance (log)
    smax = float(np.nanmax([np.nanmax(si), np.nanmax(sp)]))
    for ax, C, title in [(axes[1, 0], si, "Significance (IDL)"),
                         (axes[1, 1], sp, "Significance (Python)")]:
        im = mesh(ax, np.clip(C, 1, None), cmap="magma",
                  norm=LogNorm(vmin=1, vmax=smax)); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    with np.errstate(divide="ignore", invalid="ignore"):
        rmap = np.where(si > 1, sp / si, np.nan)
    im = mesh(axes[1, 2], rmap, cmap="coolwarm", vmin=1.0, vmax=1.8)
    axes[1, 2].set_title("significance ratio (Py / IDL)")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.85)
    for ax in axes.flat:
        ax.set_xlabel("Moon semimajor axis (R_Jup)")
        ax.set_ylabel("Moon mass (M_Earth)")
    fig.suptitle("IDL vs Python survey — 50 µas", fontsize=13)
    fig.savefig(out_png, dpi=130)
    print("\nsaved figure ->", out_png)


if __name__ == "__main__":
    main()
