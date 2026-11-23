#!/usr/bin/env python3
"""
plot_dynspec.py - spettri dinamici (tempo x frequenza) e serie temporali di uno o piu' filterbank,
uno sotto l'altro con lo stesso asse dei tempi, per confrontarli (es. originale e step 4).

Per ogni file: in alto la serie temporale (somma sui canali, dedispersa al DM dato), sotto lo spettro
dinamico. Tutti i pannelli condividono l'asse x (tempo dall'istante di inizio, in ms).

Unita' (--units):
  counts (default): i valori del file (conteggi), mediati sui canali del gruppo e sul bin di tempo, come in un
    waterfall usuale. Serie temporale: media sui canali attivi. Spettro: media su tutta la finestra (il bandpass).
  sigma: ogni canale in unita' della propria sigma (mediana e MAD su una finestra di riferimento di --ref
    secondi centrata sulla finestra mostrata). Serie temporale: somma dei canali attivi / sqrt(N) per campione,
    in S/N per bin; spettro: media su tutta la finestra, in S/N (rumore bianco atteso = 1).
I canali costanti (mascherati) sono in grigio (in rosso tenue con la colormap in scala di grigi). La scala di
colori e' COMUNE a tutti i file (salvo --own-scale), cosi' i colori sono confrontabili; lo stesso per gli assi
delle serie temporali e degli spettri.

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


def prepare(path, start, window, dm, tres, nchan, ref, units="counts"):
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
    if units == "counts":
        z = x
    else:
        z = (x - med) / np.where(dead, 1, sig)
    z[:, dead] = 0.0
    dd = z[np.arange(n_show)[:, None] + shift[None, :], np.arange(fb.nchans)[None, :]]
    del z, x
    # serie temporale: somma sui canali attivi / sqrt(N), media su r campioni, in S/N per bin
    live = ~dead
    if units == "counts":
        ts_bin = dd[:, live].mean(1).reshape(-1, r).mean(1)
    else:
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
    # spettro: media su tutta la finestra di ogni gruppo, in S/N (rumore bianco atteso = 1)
    nlg = np.array([live[g].sum() for g in groups])
    band = np.full(len(groups), np.nan)
    band[~gdead] = spec[:, ~gdead].mean(0) * (1.0 if units == "counts" else np.sqrt(nlg[~gdead] * n_show))
    t_ms = (np.arange(n_show // r) + 0.5) * r * ts * 1e3
    return t_ms, fg, spec, ts_bin, r * ts * 1e3, gdead, fb, band


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
    p.add_argument("--units", choices=["counts", "sigma"], default="counts",
                   help="counts: valori del file (default); sigma: ogni canale in unita' della propria sigma")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--vmin", type=float, default=None, help="minimo della scala di colori (solo --units counts)")
    p.add_argument("--vmax", type=float, default=None,
                   help="massimo della scala di colori (counts: default percentile 99.5, comune; sigma: scala "
                        "simmetrica +-vmax, default percentile 99.5 di |valori|)")
    p.add_argument("--own-scale", action="store_true", help="scala di colori e asse y separati per ogni file")
    p.add_argument("--cmap", choices=["gray", "viridis", "RdBu_r"], default="gray",
                   help="colormap: gray (scala di grigi, default), viridis, RdBu_r (blu-bianco-rosso)")
    p.add_argument("--width", type=float, default=14.0, help="larghezza della figura [pollici] (default 14)")
    p.add_argument("-o", "--out", default="dynspec_cmp.png")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = a.labels or [Path(x).name for x in a.fil]
    data = [prepare(x, a.start, a.window, a.dm, a.tres, a.nchan, a.ref, a.units) for x in a.fil]
    counts = a.units == "counts"
    tres_eff = data[0][4]
    nf = len(data)
    hr = []
    for k in range(nf):
        hr += ([0.45] if k else []) + [1.0, 2.6]          # riga vuota tra un file e l'altro (per il titolo)
    fig = plt.figure(figsize=(a.width, 3.6 * nf + 0.6))
    gs = fig.add_gridspec(len(hr), 3, height_ratios=hr, width_ratios=[1.0, 0.12, 0.015], hspace=0.08, wspace=0.02)
    rows = [k * 3 for k in range(nf)]                     # riga della serie temporale di ogni file
    ax, ax_band, ax_cb = [], [], []
    for k, r0 in enumerate(rows):
        a_ts = fig.add_subplot(gs[r0, 0], sharex=ax[0] if ax else None)
        a_sp = fig.add_subplot(gs[r0 + 1, 0], sharex=a_ts)
        a_bd = fig.add_subplot(gs[r0 + 1, 1], sharey=a_sp)
        a_cb = fig.add_subplot(gs[r0 + 1, 2])
        ax += [a_ts, a_sp]
        ax_band.append(a_bd)
        ax_cb.append(a_cb)
    for x in ax[:-1]:
        x.tick_params(labelbottom=False)
    allv = np.concatenate([d[2][np.isfinite(d[2])] for d in data])

    def climits(vals):
        if counts:
            lo_, hi_ = np.percentile(vals, [0.5, 99.5])
            return (a.vmin if a.vmin is not None else float(lo_)), (a.vmax if a.vmax is not None else float(hi_))
        v_ = a.vmax or float(np.percentile(np.abs(vals), 99.5))
        return -v_, v_

    ccommon = climits(allv)
    ycommon = (min(d[3].min() for d in data), max(d[3].max() for d in data))
    bcommon = (min(np.nanmin(d[7]) for d in data), max(np.nanmax(d[7]) for d in data))
    cmap = matplotlib.colormaps[a.cmap].copy() if hasattr(matplotlib, "colormaps") else plt.get_cmap(a.cmap)
    # canali mascherati: grigio chiaro con le colormap colorate, rosso tenue con quella in scala di grigi
    cmap.set_bad("#f4c7c3" if a.cmap == "gray" else "0.75")
    for k, (t, fg, spec, tsb, tr, gdead, fb, band) in enumerate(data):
        a_ts, a_sp = ax[2 * k], ax[2 * k + 1]
        a_ts.step(t, tsb, where="mid", lw=0.8, color="k")
        if not counts:
            a_ts.axhline(0, color="0.6", lw=0.5)
        if not a.own_scale:
            pad = 0.05 * (ycommon[1] - ycommon[0])
            a_ts.set_ylim(ycommon[0] - pad, ycommon[1] + pad)
        a_ts.set_ylabel("conteggi" if counts else "S/N per bin", fontsize=8)
        a_ts.set_title(f"{labels[k]}   (DM {a.dm:g}, risoluzione {tr:.3g} ms, {len(fg)} canali, "
                       f"campionamento {fb.tsamp * 1e3:.3g} ms)", fontsize=9, loc="left")
        c_lo, c_hi = ccommon if not a.own_scale else climits(spec[np.isfinite(spec)])
        df = abs(fg[0] - fg[1]) if len(fg) > 1 else abs(fb.foff)
        im = a_sp.imshow(spec.T, aspect="auto", origin="upper", cmap=cmap, vmin=c_lo, vmax=c_hi,
                         extent=[0, t[-1] + tr / 2, fg.min() - df / 2, fg.max() + df / 2], interpolation="nearest")
        a_sp.set_ylabel("frequenza [MHz]", fontsize=8)
        # spettro (media su tutta la finestra) a destra, con lo stesso asse delle frequenze
        a_bd = ax_band[k]
        ok = np.isfinite(band)
        a_bd.step(band[ok], fg[ok], where="mid", lw=0.8, color="k")
        if not counts:
            a_bd.axvline(0, color="0.6", lw=0.5)
        if not a.own_scale:
            pad = 0.05 * (bcommon[1] - bcommon[0])
            a_bd.set_xlim(bcommon[0] - pad, bcommon[1] + pad)
        a_bd.tick_params(labelleft=False, labelsize=7)
        a_bd.set_xlabel("conteggi" if counts else "S/N", fontsize=7)
        a_bd.set_title("spettro", fontsize=8)
        cb = fig.colorbar(im, cax=ax_cb[k])
        cb.set_label("conteggi (media del gruppo di canali)" if counts else "media del gruppo di canali [σ del canale]",
                     fontsize=7)
        cb.ax.tick_params(labelsize=7)
    ax[-1].set_xlabel(f"tempo dall'inizio [ms]   (inizio = {a.start:.6f} s, al canale piu' alto)")
    ax[-1].set_xlim(0, data[0][0][-1] + tres_eff / 2)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print("scritto", a.out)


if __name__ == "__main__":
    main()
