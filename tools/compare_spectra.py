#!/usr/bin/env python3
"""
compare_spectra.py - confronta gli spettri a DM 0 di due survey (es. prima e dopo lo step 2).

Richiede che entrambi i survey siano stati eseguiti con --spectrum. Produce:
  OUT/compare_spectrum.png  spettri sovrapposti (assoluti e normalizzati al continuo) e loro rapporto
  OUT/compare_lines.csv     per ogni riga del primo survey: rapporto riga/continuo prima e dopo e
                            frazione di potenza della riga (sopra il continuo) rimasta

Esempio:
  python3 compare_spectra.py survey_step1/ survey_step2/ --labels "step 1" "step 2" --out confronto/
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.spectrum import continuum, find_lines  # noqa: E402


def load(d):
    z = np.load(Path(d) / "spectrum.npz")
    return z["freq_hz"], z["psd"].astype(np.float64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--out", default=".")
    p.add_argument("--fmax", type=float, default=None, help="limite superiore del grafico [Hz]")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fa, pa = load(a.a)
    fb, pb = load(a.b)
    if fa.size != fb.size or not np.allclose(fa, fb):
        pb = np.column_stack([np.interp(fa, fb, pb[:, j]) for j in range(pb.shape[1])])
    f = fa
    df = f[1] - f[0]
    A, B = pa[:, 0], pb[:, 0]
    ca, cb = continuum(A, df), continuum(B, df)
    lines = find_lines(f, A)
    rows = []
    for ln in sorted(lines, key=lambda x: -x["ratio"]):
        i = int(np.argmin(np.abs(f - ln["freq_hz"])))
        j0, j1 = max(0, i - 1), i + 2
        ra = (A[j0:j1] / ca[j0:j1]).max()
        rb = (B[j0:j1] / cb[j0:j1]).max()
        left = (rb - 1) / (ra - 1) if ra > 1 else np.nan
        lab = f"{ln['freq_hz']:.3f}" if ln["harmonic"] == 1 else f"{ln['harmonic']} x {ln['family_f0_hz']:.3f}"
        rows.append((ln["freq_hz"], lab, ra, rb, left))
    with open(out / "compare_lines.csv", "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["freq_hz", "family", f"ratio_{a.labels[0]}", f"ratio_{a.labels[1]}", "line_power_left"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}", r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.4f}"])
    print(f"{'freq [Hz]':>10} {'famiglia':>16} {a.labels[0]:>12} {a.labels[1]:>12} {'potenza riga rimasta':>22}")
    for r in rows[:25]:
        print(f"{r[0]:10.3f} {r[1]:>16} {r[2]:12.1f} {r[3]:12.1f} {r[4] * 100:21.2f}%")
    band = (f > 1) & (f < f[-1] * 0.95)
    print(f"Rapporto mediano dei continui ({a.labels[1]}/{a.labels[0]}): {np.median(cb[band] / ca[band]):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fmax = a.fmax or f[-1]
    sel = (f > 0) & (f <= fmax)
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax[0].semilogy(f[sel], A[sel], lw=0.5, label=a.labels[0])
    ax[0].semilogy(f[sel], B[sel], lw=0.5, label=a.labels[1], alpha=0.8)
    ax[0].set_ylabel("densita' spettrale [unita'$^2$/Hz]")
    ax[0].set_title("Spettro della serie a DM 0 (banda intera)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].semilogy(f[sel], (A / ca)[sel], lw=0.5, label=a.labels[0])
    ax[1].semilogy(f[sel], (B / cb)[sel], lw=0.5, label=a.labels[1], alpha=0.8)
    ax[1].axhline(10, color="r", ls=":", lw=0.8)
    ax[1].set_ylabel("spettro / continuo")
    ax[1].legend(fontsize=8)
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = np.where(A > 0, B / np.where(A > 0, A, 1), np.nan)
    ax[2].semilogy(f[sel], rat[sel], lw=0.4, color="k")
    ax[2].axhline(1, color="r", lw=0.8)
    ax[2].set_ylabel(f"{a.labels[1]} / {a.labels[0]}")
    ax[2].set_xlabel("frequenza [Hz]")
    ax[2].set_title("Rapporto degli spettri (valori < 1: potenza rimossa)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "compare_spectrum.png", dpi=110)
    print(f"Grafico: {out / 'compare_spectrum.png'}  Tabella: {out / 'compare_lines.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
