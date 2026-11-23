#!/usr/bin/env python3
"""
plot_dynspec.py - spettri dinamici (tempo x frequenza) e serie temporali di uno o piu' filterbank,
uno sotto l'altro con lo stesso asse dei tempi, per confrontarli (es. originale e step 4).

Per ogni file: in alto la serie temporale (somma sui canali, dedispersa al DM dato), sotto lo spettro
dinamico. Tutti i pannelli condividono l'asse x (tempo dall'istante di inizio, in ms).

Normalizzazione: ogni canale di ogni file e' riportato in unita' della propria sigma (mediana e MAD su una
finestra di riferimento di --ref secondi centrata sulla finestra mostrata, che conviene sia ampia); i canali
costanti (mascherati) sono in grigio. La scala di colori e' COMUNE a tutti i file (salvo --own-scale), cosi'
i colori sono confrontabili; lo stesso per l'asse y delle serie temporali (salvo --own-scale).
Serie temporale: somma dei canali attivi / sqrt(numero di canali) per campione, mediata sul bin di tempo e
espressa in S/N per bin (rumore bianco atteso = 1).

Risoluzione: --tres in ms; se e' piu' fine del campionamento si usa il campionamento (risoluzione massima).
--nchan: numero di canali mostrati (gruppi di canali adiacenti mediati); non piu' dei canali del file.

Esempi:
  python3 plot_dynspec.py orig.fil step4.fil --labels originale step4 --start 500.4 --window 400 --dm 0 -o cmp.png
  python3 plot_dynspec.py a.fil --start 673.56 --window 100 --dm 50 --tres 0.128 --nchan 64
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.sigproc import Filterbank  # noqa: E402

KDM = 4.148808e3
MAD2SIG = 1.4826


def prepare(path, start, window, dm, tres, nchan, ref):
    """Restituisce (tempo [ms], frequenze dei gruppi [MHz], spettro (nt, ngruppi) in sigma, serie in S/N per bin,
    tres effettiva [ms], gruppi mascherati)."""
    fb = Filterbank(path)
    ts = fb.tsamp
    f = fb.freqs
    delay = KDM * dm * (f ** -2 - f.max() ** -2)                  # [s] rispetto al canale piu' alto
    shift = np.round(delay / ts).astype(int)
    r = max(1, int(round(tres * 1e-3 / ts)))
    if tres * 1e-3 < ts:
        print(f"{Path(path).name}: risoluzione richiesta {tres:g} ms < campionamento: uso {ts * 1e3:.4g} ms")
    if nchan > fb.nchans:
        print(f"{Path(path).name}: richiesti {nchan} canali, il file ne ha {fb.nchans}: li mostro tutti")
    n_show = int(round(window * 1e-3 / ts)) // r * r
    if n_show < r:
        raise SystemExit("finestra piu' corta della risoluzione temporale")
    s0 = int(np.floor(start / ts))
    # statistiche per canale su una finestra di riferimento (piu' ampia), centrata sulla finestra mostrata
    n_ref = max(int(round(ref / ts)), n_show)
    c_ref = s0 + n_show // 2
    xr = fb.read(max(0, c_ref - n_ref // 2), n_ref).astype(np.float32)
    med = np.median(xr, 0)
    sig = MAD2SIG * np.median(np.abs(xr - med), 0)
    del xr
    dead = sig < 0.05 * np.median(sig[sig > 0]) if np.any(sig > 0) else np.ones(fb.nchans, bool)
    # dati: finestra + sweep, poi dedispersione con shift interi
    x = fb.read(s0, n_show + shift.max() + 1).astype(np.float32)
    if x.shape[0] < n_show + shift.max() + 1:
        raise SystemExit(f"{path}: la finestra (piu' lo sweep a DM {dm}) esce dal file")
    z = (x - med) / np.where(dead, 1, sig)
    z[:, dead] = 0.0
    dd = z[np.arange(n_show)[:, None] + shift[None, :], np.arange(fb.nchans)[None, :]]
    del z, x
    # serie temporale: somma sui canali attivi / sqrt(N), media su r campioni, in S/N per bin
    live = ~dead
    ts_ = dd[:, live].sum(1) / np.sqrt(max(1, live.sum()))
    ts_bin = ts_.reshape(-1, r).mean(1) * np.sqrt(r)
    # spettro: gruppi di canali adiacenti (in ordine di frequenza decrescente), media; tempo mediato su r
    order = np.argsort(-f)
    groups = np.array_split(order, min(nchan, fb.nchans))
    spec = np.empty((n_show // r, len(groups)), np.float32)
    gdead = np.zeros(len(groups), bool)
    for j, g in enumerate(groups):
        gl = g[live[g]]
        if gl.size == 0:
            gdead[j] = True
            spec[:, j] = np.nan
            continue
        # media dei canali attivi del gruppo, in sigma del singolo canale: rumore per bin 1/sqrt(n r)
        spec[:, j] = dd[:, gl].mean(1).reshape(-1, r).mean(1)
    fg = np.array([f[g].mean() for g in groups])
    t_ms = (np.arange(n_show // r) + 0.5) * r * ts * 1e3
    return t_ms, fg, spec, ts_bin, r * ts * 1e3, gdead, fb


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil", nargs="+", help="uno o piu' filterbank (mostrati uno sotto l'altro)")
    p.add_argument("--start", type=float, required=True, help="istante iniziale [s] (al canale piu' alto)")
    p.add_argument("--window", type=float, required=True, help="durata mostrata [ms]")
    p.add_argument("--dm", type=float, default=0.0, help="DM di dedispersione (default 0)")
    p.add_argument("--tres", type=float, default=0.0,
                   help="risoluzione temporale [ms] (default e se piu' fine del campionamento: risoluzione massima)")
    p.add_argument("--nchan", type=int, default=64, help="numero di canali mostrati (default 64)")
    p.add_argument("--ref", type=float, default=2.0,
                   help="finestra di riferimento per mediana e sigma dei canali [s] (default 2)")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--vmax", type=float, default=None,
                   help="scala di colori simmetrica +-vmax [sigma] (default: percentile 99.5 di |valori|, comune)")
    p.add_argument("--own-scale", action="store_true", help="scala di colori e asse y separati per ogni file")
    p.add_argument("--width", type=float, default=14.0, help="larghezza della figura [pollici] (default 14)")
    p.add_argument("-o", "--out", default="dynspec_cmp.png")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = a.labels or [Path(x).name for x in a.fil]
    data = [prepare(x, a.start, a.window, a.dm, a.tres, a.nchan, a.ref) for x in a.fil]
    tres_eff = data[0][4]
    nf = len(data)
    hr = []
    for k in range(nf):
        hr += ([0.45] if k else []) + [1.0, 2.6]          # riga vuota tra un file e l'altro (per il titolo)
    fig, axs = plt.subplots(len(hr), 1, figsize=(a.width, 3.6 * nf + 0.6), sharex=True,
                            gridspec_kw=dict(height_ratios=hr, hspace=0.08))
    for k in range(1, nf):
        axs[3 * k - 1].set_visible(False)
    ax = [x for i, x in enumerate(axs) if not (i % 3 == 2)]
    allv = np.concatenate([np.abs(d[2][np.isfinite(d[2])]) for d in data])
    vcommon = a.vmax or float(np.percentile(allv, 99.5))
    ycommon = (min(d[3].min() for d in data), max(d[3].max() for d in data))
    cmap = matplotlib.colormaps["RdBu_r"].copy() if hasattr(matplotlib, "colormaps") else plt.get_cmap("RdBu_r")
    cmap.set_bad("0.75")
    for k, (t, fg, spec, tsb, tr, gdead, fb) in enumerate(data):
        a_ts, a_sp = ax[2 * k], ax[2 * k + 1]
        a_ts.step(t, tsb, where="mid", lw=0.8, color="k")
        a_ts.axhline(0, color="0.6", lw=0.5)
        if not a.own_scale:
            pad = 0.05 * (ycommon[1] - ycommon[0])
            a_ts.set_ylim(ycommon[0] - pad, ycommon[1] + pad)
        a_ts.set_ylabel("S/N per bin", fontsize=8)
        a_ts.set_title(f"{labels[k]}   (DM {a.dm:g}, risoluzione {tr:.3g} ms, {len(fg)} canali, "
                       f"campionamento {fb.tsamp * 1e3:.3g} ms)", fontsize=9, loc="left")
        v = vcommon if not a.own_scale else float(np.percentile(np.abs(spec[np.isfinite(spec)]), 99.5))
        df = abs(fg[0] - fg[1]) if len(fg) > 1 else abs(fb.foff)
        im = a_sp.imshow(spec.T, aspect="auto", origin="upper", cmap=cmap, vmin=-v, vmax=v,
                         extent=[0, t[-1] + tr / 2, fg.min() - df / 2, fg.max() + df / 2], interpolation="nearest")
        a_sp.set_ylabel("frequenza [MHz]", fontsize=8)
        cb = fig.colorbar(im, ax=[a_ts, a_sp], pad=0.01, fraction=0.025)
        cb.set_label("media del gruppo di canali [σ del canale]", fontsize=7)
    ax[-1].set_xlabel(f"tempo dall'inizio [ms]   (inizio = {a.start:.6f} s, al canale piu' alto)")
    ax[-1].set_xlim(0, data[0][0][-1] + tres_eff / 2)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print("scritto", a.out)


if __name__ == "__main__":
    main()
