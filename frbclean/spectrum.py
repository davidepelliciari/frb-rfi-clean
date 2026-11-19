"""
Spettro di potenza delle serie a DM 0 (banda intera e sottobande) e individuazione delle righe.

Metodo: media degli spettri (Welch) di segmenti di durata fissa (i blocchi del survey), con
finestra di Hann. Con fmax < Nyquist nativo si sommano r campioni consecutivi: e' un filtro
anti-aliasing scadente, e righe sopra la nuova Nyquist ricadono nello spettro come righe fittizie
(in ag278m_mc_no0004 le armoniche alte di 50 Hz producevano una falsa famiglia a 166.67 Hz).
Per questo il default del survey e' la risoluzione nativa. Densita' spettrale in (unita')^2 / Hz.

Righe: il "continuo" e' la mediana mobile dello spettro su `cont_hz` Hz; una riga e' un massimo
locale del rapporto spettro/continuo oltre `thresh`. Ogni riga viene attribuita, se possibile,
a una famiglia armonica n x f0 (f0 = una riga piu' forte, tolleranza `harm_tol` in frazione).
"""
from __future__ import annotations

import csv

import numpy as np
from scipy.ndimage import median_filter


class SpectrumAccumulator:
    def __init__(self, ts: float, nseries: int, fmax: float):
        self.ts = ts
        self.r = max(1, int(np.floor(1.0 / (2.0 * fmax * ts))))
        self.dt = self.r * ts
        self.nser = nseries
        self.L = None
        self.sum = None
        self.count = 0

    def add(self, X: np.ndarray):
        """X: (campioni, serie). Segmenti piu' corti del primo vengono ignorati."""
        m = X.shape[0] // self.r
        if m < 16:
            return
        Y = X[: m * self.r].reshape(m, self.r, -1).sum(1)
        if self.L is None:
            self.L = m
            self.win = np.hanning(m)
            self.U = (self.win ** 2).mean()
            self.sum = np.zeros((m // 2 + 1, self.nser))
        if m < self.L:
            return
        Y = Y[: self.L]
        Y = (Y - Y.mean(0)) * self.win[:, None]
        F = np.fft.rfft(Y, axis=0)
        self.sum += np.abs(F) ** 2
        self.count += 1

    def result(self):
        if self.count == 0:
            return None, None
        f = np.fft.rfftfreq(self.L, self.dt)
        psd = self.sum / self.count * 2.0 * self.dt / (self.L * self.U)
        psd[0] = 0
        return f, psd


def continuum(psd: np.ndarray, df: float, cont_hz: float = 2.0) -> np.ndarray:
    k = max(5, int(round(cont_hz / df)) | 1)
    c = median_filter(psd, size=(k,) + (1,) * (psd.ndim - 1), mode="nearest")
    c[c <= 0] = np.min(c[c > 0]) if np.any(c > 0) else 1.0
    return c


def find_lines(f, psd, thresh: float = 10.0, cont_hz: float = 2.0, fmin: float = 0.5, nmax: int = 60,
               harm_tol: float = 0.002, nharm: int = 64):
    """Righe nello spettro della banda intera. Restituisce lista di dict ordinata per frequenza."""
    df = f[1] - f[0]
    cont = continuum(psd, df, cont_hz)
    ratio = psd / cont
    ok = (f >= fmin)
    loc = (ratio[1:-1] > ratio[:-2]) & (ratio[1:-1] >= ratio[2:]) & (ratio[1:-1] > thresh) & ok[1:-1]
    idx = np.flatnonzero(loc) + 1
    # unisce picchi entro 2 bin
    keep = []
    for i in idx[np.argsort(-ratio[idx])]:
        if all(abs(i - j) > 2 for j in keep):
            keep.append(i)
        if len(keep) >= nmax:
            break
    lines = []
    for i in keep:
        # frequenza interpolata: parabola sul logaritmo della potenza nei tre bin attorno al picco
        fi = float(f[i])
        if 0 < i < psd.size - 1 and psd[i - 1] > 0 and psd[i + 1] > 0 and psd[i] > 0:
            a, b, c = np.log(psd[i - 1]), np.log(psd[i]), np.log(psd[i + 1])
            den = a - 2 * b + c
            if den < 0:
                fi += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) * df
        lines.append(dict(freq_hz=fi, ratio=float(ratio[i]), psd=float(psd[i]), continuum=float(cont[i])))
    # famiglie armoniche: per frequenza crescente, una riga e' l'armonica n di una fondamentale piu' bassa
    # se f = n * f0 entro max(harm_tol * f, (n + 1) * df); altrimenti e' una nuova fondamentale
    fund = []
    for ln in sorted(lines, key=lambda x: x["freq_hz"]):
        lab = None
        for f0 in fund:
            n = ln["freq_hz"] / f0
            nr = int(round(n))
            if 2 <= nr <= nharm and abs(ln["freq_hz"] - nr * f0) <= max(harm_tol * ln["freq_hz"], (nr + 1) * df):
                lab = (f0, nr)
                break
        if lab is None:
            fund.append(ln["freq_hz"])
            lab = (ln["freq_hz"], 1)
        ln["family_f0_hz"], ln["harmonic"] = float(lab[0]), lab[1]
    return sorted(lines, key=lambda x: x["freq_hz"])


def save(out, f, psd, sub_freq, lines):
    np.savez_compressed(out / "spectrum.npz", freq_hz=f, psd=psd.astype(np.float32), sub_freq=sub_freq)
    with open(out / "spectrum_lines.csv", "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["freq_hz", "ratio_to_continuum", "family_f0_hz", "harmonic", "psd", "continuum"])
        for ln in sorted(lines, key=lambda x: -x["ratio"]):
            w.writerow([f"{ln['freq_hz']:.4f}", f"{ln['ratio']:.1f}", f"{ln['family_f0_hz']:.4f}", ln["harmonic"],
                        f"{ln['psd']:.4g}", f"{ln['continuum']:.4g}"])


def plot(out, f, psd, sub_freq, lines, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = f[1] - f[0]
    full = psd[:, 0]
    cont = continuum(full, df)
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw=dict(height_ratios=[1, 1, 1.2]))
    sel = f > 0
    ax[0].semilogy(f[sel], full[sel], lw=0.5, label="spettro")
    ax[0].semilogy(f[sel], cont[sel], lw=1, color="k", alpha=0.7, label="continuo (mediana mobile 2 Hz)")
    ax[0].set_ylabel("densita' spettrale [unita'$^2$/Hz]")
    ax[0].set_title(f"{title}\nSerie a DM 0, banda intera (risoluzione {df:.3f} Hz)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].semilogy(f[sel], (full / cont)[sel], lw=0.5)
    ax[1].axhline(10, color="r", ls=":", lw=0.8)
    top = sorted(lines, key=lambda x: -x["ratio"])[:15]
    for ln in top:
        lab = f"{ln['freq_hz']:.2f}" if ln["harmonic"] == 1 else f"{ln['harmonic']}x{ln['family_f0_hz']:.2f}"
        ax[1].annotate(lab, (ln["freq_hz"], ln["ratio"]), fontsize=7, rotation=60, xytext=(2, 4),
                       textcoords="offset points")
    ax[1].set_ylabel("spettro / continuo")
    ax[1].set_title("Spettro normalizzato al continuo (riga tratteggiata: soglia 10); etichette: le 15 righe piu' forti",
                    fontsize=9)
    o = np.argsort(sub_freq)
    W = np.vstack([psd[:, 1 + j] / continuum(psd[:, 1 + j], df) for j in o])
    im = ax[2].imshow(np.log10(np.clip(W, 1, None)), aspect="auto", origin="lower", cmap="magma",
                      extent=[f[0], f[-1], sub_freq.min(), sub_freq.max()], vmin=0, vmax=3, interpolation="nearest")
    fig.colorbar(im, ax=ax[2], label="log$_{10}$(spettro / continuo)", pad=0.01)
    ax[2].set_ylabel("sottobanda [MHz]")
    ax[2].set_xlabel("frequenza [Hz]")
    ax[2].set_title("Righe per sottobanda", fontsize=9)
    for a in ax[:2]:
        a.set_xlim(0, f[-1])
    fig.tight_layout()
    fig.savefig(out / "spectrum.png", dpi=110)
    plt.close(fig)
