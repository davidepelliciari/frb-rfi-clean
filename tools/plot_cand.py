#!/usr/bin/env python3
"""
plot_cand.py - ispezione di un candidato direttamente dal filterbank, anche su piu' file affiancati
(es. originale, step 2, step 3), per vedere cosa e' cambiato attorno al candidato.

Per ogni file, una colonna con:
  1. profilo dedispersione al DM dato (boxcar della larghezza data) in unita' di sigma robusta;
  2. spettro dinamico dedisperso (sottobande x tempo);
  3. spettro dinamico a DM 0 (non dedisperso), con la traccia di dispersione attesa (spostata di +2 ms);
  4. piano DM-tempo (S/N del boxcar per ogni DM tra 0 e 2 DM).

Il tempo del candidato e' quello di Heimdall (stime/tcand): arrivo alla frequenza piu' alta della banda.
Ogni canale e' normalizzato con mediana e MAD su +-norm_win (default 1 s), cosi' come i boxcar; canali costanti (mascherati) a zero.

Esempi:
  python3 plot_cand.py step2.fil step3.fil --t 657.412 --dm 45.8 --width 3 -o cand_657.png
  python3 plot_cand.py orig.fil step3.fil --t 657.412 --dm 45.8 --width 3 --labels orig step3 --win 0.2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.sigproc import Filterbank  # noqa: E402

KDM = 4.148808e3  # s MHz^2 / (pc cm^-3)


def load(fb: Filterbank, t: float, dm: float, win: float):
    """Legge [t - win, t + sweep + win] e restituisce dati normalizzati (canali ordinati per freq decrescente)."""
    f = fb.freqs
    fhi = f.max()
    delay = KDM * dm * (f ** -2 - fhi ** -2)  # s, rispetto al top della banda
    s0 = int(np.floor((t - win) / fb.tsamp))
    n = int(np.ceil((2 * win + delay.max()) / fb.tsamp))
    x = fb.read(s0, n).astype(np.float32)
    med = np.median(x, 0)
    mad = np.median(np.abs(x - med), 0) * 1.4826
    live = mad > 0
    x = np.where(live, (x - med) / np.where(live, mad, 1), 0.0)
    o = np.argsort(-f)
    return x[:, o], f[o], live[o], s0, delay[o]


def dedisperse(x, f, dm, tsamp, nout):
    d = np.round(KDM * dm * (f ** -2 - f.max() ** -2) / tsamp).astype(int)
    idx = np.arange(nout)[:, None] + d[None, :]
    idx = np.clip(idx, 0, x.shape[0] - 1)
    return x[idx, np.arange(x.shape[1])]


def _box(series, w):
    c = np.concatenate([[0], np.cumsum(series, dtype=np.float64)])
    return (c[w:] - c[:-w]) / np.sqrt(w)


def box_stats(series, w):
    """Mediana e sigma robusta del boxcar di larghezza w (statistica di riferimento su una finestra lunga)."""
    b = _box(series, w)[::max(1, w // 4)]
    med = np.median(b)
    sig = np.median(np.abs(b - med)) * 1.4826
    return med, (sig if sig > 0 else 1.0)


def boxcar_snr(series, w, stats=None):
    b = _box(series, w)
    med, sig = stats if stats is not None else box_stats(series, w)
    return (b - med) / sig, w


def block(a, r, axis=0):
    m = a.shape[axis] // r
    a = np.take(a, np.arange(m * r), axis=axis)
    sh = list(a.shape)
    sh[axis:axis + 1] = [m, r]
    return a.reshape(sh).mean(axis + 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil", nargs="+")
    p.add_argument("--t", type=float, required=True, help="tempo del candidato [s] (Heimdall, top della banda)")
    p.add_argument("--dm", type=float, required=True)
    p.add_argument("--width", type=int, default=None, help="larghezza Heimdall: log2 dei campioni (default: scelta dal profilo)")
    p.add_argument("--win", type=float, default=0.1, help="semi-finestra mostrata attorno al candidato [s] (default 0.1)")
    p.add_argument("--norm-win", type=float, default=1.0,
                   help="semi-finestra [s] usata per mediana/sigma di canali e boxcar (default 1 s; deve contenere "
                        "molte larghezze del candidato, altrimenti l'S/N dei boxcar larghi e' sottostimato)")
    p.add_argument("--nsub", type=int, default=32, help="sottobande negli spettri dinamici (default 32)")
    p.add_argument("--zoom", type=float, default=None,
                   help="semi-finestra degli spettri dinamici [s] (default: max(20 ms, 40 larghezze))")
    p.add_argument("--tres", type=float, default=None, help="risoluzione temporale dei pannelli [s] (default: larghezza)")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("-o", "--out", default="cand.png")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = a.labels or [Path(x).parent.name + "/" + Path(x).name for x in a.fil]
    nf = len(a.fil)
    fig, ax = plt.subplots(4, nf, figsize=(5.2 * nf, 13), squeeze=False,
                           gridspec_kw=dict(height_ratios=[0.8, 1.3, 1.3, 1.1]))
    for k, path in enumerate(a.fil):
        fb = Filterbank(path)
        # lettura larga (statistiche), poi ritaglio di +-win attorno al candidato per i pannelli
        wr = max(a.win, a.norm_win)
        xw, f, live, sw0, delay = load(fb, a.t, a.dm, wr)
        nlive = max(1, live.sum())
        nw = int(round(2 * wr / fb.tsamp))
        profw = dedisperse(xw, f, a.dm, fb.tsamp, nw).sum(1) / np.sqrt(nlive)
        off = int(round((wr - a.win) / fb.tsamp))
        x = xw[off:]
        s0 = sw0 + off
        nout = int(round(2 * a.win / fb.tsamp))
        dd = dedisperse(x, f, a.dm, fb.tsamp, nout)
        prof = profw[off:off + nout]
        # larghezza: quella data, o la migliore entro 2^0..2^12 nella zona centrale
        ws = [2 ** a.width] if a.width is not None else [2 ** j for j in range(13)]
        best = None
        c0 = nout // 2
        for w in ws:
            if w >= nout // 4:
                continue
            z, _ = boxcar_snr(prof, w, box_stats(profw, w))
            lo, hi = max(0, c0 - 2 * w - int(0.01 / fb.tsamp)), min(z.size, c0 + w + int(0.01 / fb.tsamp))
            j = lo + int(np.argmax(z[lo:hi]))
            if best is None or z[j] > best[0]:
                best = (z[j], w, j)
        snr, w, jpk = best
        print(f"{labels[k]}: DM {a.dm:.2f}, boxcar {w} campioni ({w * fb.tsamp * 1e3:.3g} ms), S/N {snr:.2f}")
        tres = a.tres or w * fb.tsamp
        r = max(1, int(round(tres / fb.tsamp)))
        tt = (np.arange(nout // r) * r + r / 2) * fb.tsamp + s0 * fb.tsamp - a.t
        # 1. profilo
        pr = block(prof, r) * np.sqrt(r)
        prw = block(profw, r) * np.sqrt(r)
        mw = np.median(prw)
        pr = (pr - mw) / (np.median(np.abs(prw - mw)) * 1.4826 or 1)
        ax[0, k].step(tt * 1e3, pr, where="mid", lw=0.8)
        ax[0, k].axvline((jpk + s0) * fb.tsamp - a.t + w * fb.tsamp / 2 - 0 * fb.tsamp, color="r", lw=0.6, ls=":")
        ax[0, k].set_title(f"{labels[k]}\nDM {a.dm:.1f}, boxcar {w} camp. ({w * fb.tsamp * 1e3:.3g} ms): S/N {snr:.1f}",
                           fontsize=9)
        ax[0, k].set_ylabel("S/N per bin")
        # 2. dedisperso
        nsub = min(a.nsub, x.shape[1])
        g = x.shape[1] // nsub
        sub = block(block(dd, g, axis=1), r, axis=0) * np.sqrt(g * r)
        ext = [tt[0] * 1e3, tt[-1] * 1e3, f[nsub * g - 1], f[0]]
        v = np.percentile(sub, [1, 99.5])
        ax[1, k].imshow(sub.T, aspect="auto", extent=ext, cmap="viridis", vmin=v[0], vmax=v[1], interpolation="nearest")
        ax[1, k].set_ylabel("frequenza [MHz]")
        ax[1, k].set_title("dedisperso", fontsize=9)
        # 3. DM 0
        raw = x[:nout + int(round(delay.max() / fb.tsamp))]
        sub0 = block(block(raw, g, axis=1), r, axis=0) * np.sqrt(g * r)
        t0 = (np.arange(sub0.shape[0]) * r + r / 2) * fb.tsamp + s0 * fb.tsamp - a.t
        v = np.percentile(sub0, [1, 99.5])
        ax[2, k].imshow(sub0.T, aspect="auto", extent=[t0[0] * 1e3, t0[-1] * 1e3, f[nsub * g - 1], f[0]],
                        cmap="viridis", vmin=v[0], vmax=v[1], interpolation="nearest")
        ax[2, k].plot((delay + (jpk + s0) * fb.tsamp - a.t + w * fb.tsamp / 2) * 1e3 + 2, f, "w--", lw=0.6, alpha=0.6)
        ax[2, k].set_ylabel("frequenza [MHz]")
        ax[2, k].set_title("DM 0 (tratteggio: traccia attesa, spostata di +2 ms)", fontsize=9)
        # 4. DM-tempo
        dms = np.linspace(0, 2 * a.dm, 81)
        M = np.empty((dms.size, nout // r))
        ref = box_stats(profw, w)   # il rumore del boxcar non dipende dal DM
        for i, d in enumerate(dms):
            s = dedisperse(x, f, d, fb.tsamp, nout).sum(1) / np.sqrt(nlive)
            z, _ = boxcar_snr(s, w, ref)
            z = np.concatenate([z, np.zeros(nout - z.size)])
            M[i] = block(z, r)
        ax[3, k].imshow(M, aspect="auto", origin="lower", extent=[tt[0] * 1e3, tt[-1] * 1e3, dms[0], dms[-1]],
                        cmap="magma", vmin=0, vmax=max(6, snr), interpolation="nearest")
        ax[3, k].set_ylabel("DM [pc cm$^{-3}$]")
        ax[3, k].set_xlabel(f"tempo - {a.t:.4f} s [ms]")
        ax[0, k].set_xlim(tt[0] * 1e3, tt[-1] * 1e3)
        ax[3, k].set_xlim(tt[0] * 1e3, tt[-1] * 1e3)
        zm = (a.zoom or max(0.02, 40 * w * fb.tsamp)) * 1e3
        tc = ((jpk + s0) * fb.tsamp - a.t + w * fb.tsamp / 2) * 1e3
        ax[1, k].set_xlim(tc - zm, tc + zm)
        ax[2, k].set_xlim(tc - zm, tc + zm + delay.max() * 1e3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("scritto", a.out)


if __name__ == "__main__":
    main()
