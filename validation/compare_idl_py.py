#!/usr/bin/env python3
"""
Compare an IDL survey (dumped by dump_cubes.pro) against a Python survey (.npz).

Usage:
    python compare_idl_py.py <idl_cubes_dump.txt> <python_survey.npz>

Reports per-cell significance / period-error / amplitude-error / detection-fraction
side by side, and whether the detection maps agree.  Because RNG and fitter differ,
this is a statistical / structural comparison, not a bit match.
"""
import sys
import numpy as np


def parse_idl_dump(path):
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
            cubes[name] = np.array(rows)          # [na, nm]
        i += 1
    return dict(ntrials=ntrials, a=vecs["as"], mass=vecs["masses"], **cubes)


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    idl = parse_idl_dump(sys.argv[1])
    py = np.load(sys.argv[2])

    a_i, m_i = idl["a"], idl["mass"]
    a_p, m_p = py["a_grid"], py["mass_grid"]
    print(f"IDL grid : a={a_i}  mass={m_i}  ntrials={idl['ntrials']}")
    print(f"Py  grid : a={a_p}  mass={m_p}  ntrials={int(py['ntrials'])}")
    if not (np.allclose(sorted(a_i), sorted(a_p)) and np.allclose(sorted(m_i), sorted(m_p))):
        print("\nWARNING: grids differ; comparing on the overlapping cells only.")

    print("\n%-6s %-7s | %-9s %-9s %6s | %-8s %-8s | %-8s %-8s | detfrac"
          % ("a", "mass", "sig(IDL)", "sig(Py)", "ratio", "perrI", "perrP", "ampI", "ampP"))
    print("-" * 96)
    det_agree = det_total = 0
    for ia, a in enumerate(a_p):
        for im, mass in enumerate(m_p):
            ji = np.argmin(np.abs(a_i - a)); jm = np.argmin(np.abs(m_i - mass))
            sI = idl["sigcube"][ji, jm]; sP = py["sigcube"][ia, im]
            pI = idl["perrcube"][ji, jm]; pP = py["perrcube"][ia, im]
            aI = idl["amperrcube"][ji, jm]; aP = py["amperrcube"][ia, im]
            dI = idl["detfrac"][ji, jm]; dP = py["detfrac"][ia, im]
            ratio = sP / sI if sI else np.nan
            print("%-6g %-7g | %-9.3g %-9.3g %6.2f | %-8.4g %-8.4g | %-8.3g %-8.3g | %.2f/%.2f"
                  % (a, mass, sI, sP, ratio, pI, pP, aI, aP, dI, dP))
            det_total += 1
            det_agree += (round(dI) == round(dP)) or (abs(dI - dP) <= 0.2)

    print("-" * 96)
    print(f"detection maps agree on {det_agree}/{det_total} cells "
          f"(fraction within 0.2 or same rounded)")
    # summary ratios
    r = py["sigcube"] / np.where(np.array([[idl['sigcube'][np.argmin(abs(a_i-a)),
            np.argmin(abs(m_i-mm))] for mm in m_p] for a in a_p]) == 0, np.nan,
            [[idl['sigcube'][np.argmin(abs(a_i-a)), np.argmin(abs(m_i-mm))] for mm in m_p] for a in a_p])
    print(f"significance ratio Py/IDL: median={np.nanmedian(r):.2f}  "
          f"(a constant offset is expected/OK; look for it being ~consistent)")


if __name__ == "__main__":
    main()
