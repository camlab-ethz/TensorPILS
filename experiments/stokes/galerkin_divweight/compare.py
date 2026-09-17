#!/usr/bin/env python
"""FEM least squares against its continuity weight, next to the strong-form baseline.

Reads output/stokes/galerkin_divweight/dw*/ and prints the two ladders side by side, so the
question the structured table cannot currently answer -- how much of the label-free win is a
two-block rescaling and how much is the velocity preconditioner -- is read off directly.
"""
import glob, json, os
import numpy as np

PINO = "output/stokes/stokes_paper/stokes_benchmark/lr_sweep/pino/lr1e-3_dw{w}/results/*.json"
FEM = "output/stokes/galerkin_divweight/dw{w}/results/*.json"
PLS = ("output/stokes/stokes_paper/stokes_benchmark/final/pls/lr1e-3_om256/seed*/results/*.json")


def read(pat):
    fs = sorted(glob.glob(pat))
    if not fs:
        return None
    u = np.mean([json.load(open(f))["test_rel_l2_u"] for f in fs]) * 100
    p = np.mean([json.load(open(f))["test_rel_l2_p"] for f in fs]) * 100
    return u, p


def main():
    print(f"{'continuity weight w':>20} | {'FEM  ½‖(r_mom, w·r_cont)‖²':>28} | "
          f"{'PINO  strong form':>22}")
    print("-" * 78)
    for w in (1, 3, 10, 30, 100, 300, 1000):
        fem, pino = read(FEM.format(w=w)), read(PINO.format(w=w))
        fs = f"{fem[0]:7.2f} % / {fem[1]:6.2f} %" if fem else "            --"
        ps = f"{pino[0]:7.2f} % / {pino[1]:6.2f} %" if pino else "            --"
        print(f"{w:>20} | {fs:>28} | {ps:>22}")
    pls = read(PLS)
    if pls:
        print("-" * 78)
        print(f"{'PLS, omega=256':>20} | {'(weight + velocity A^-1)':>28} | "
              f"{pls[0]:7.2f} % / {pls[1]:6.2f} %")


if __name__ == "__main__":
    main()
