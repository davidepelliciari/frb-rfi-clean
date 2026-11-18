"""
Ricerca a singolo impulso semplificata, su CPU, per test rapidi (file sintetici, tagli di pochi secondi).

NON sostituisce Heimdall: serve a confrontare "prima/dopo" la pulizia con una procedura simile
(dedispersione incoerente con shift interi, baseline mobile, normalizzazione con rumore misurato,
boxcar in potenze di 2, raggruppamento dei candidati).

Normalizzazione: per ogni serie dedispersa si sottrae una baseline (mediana su blocchi di
`baseline` secondi, interpolata) e si divide per la sigma robusta del blocco, cosi' che le
variazioni rapide di varianza producano falsi positivi come nelle pipeline reali.
"""
from __future__ import annotations

import numpy as np

from .sigproc import Filterbank

KDM = 4.148808e3
MAD2SIG = 1.4826


def dm_grid(fb: Filterbank, dm_min: float, dm_max: float, tol: float = 1.25):
    """Griglia di DM con passo tale che lo sweep cambi di circa tol campioni tra trial successivi."""
    k = KDM * (fb.freqs.min() ** -2 - fb.freqs.max() ** -2)
    step = tol * fb.tsamp / k
    return np.arange(dm_min, dm_max + step, step)


def _local_norm(ts: np.ndarray, nb: int):
    n = ts.size
    m = n // nb
    if m < 2:
        med = np.median(ts)
        sig = MAD2SIG * np.median(np.abs(ts - med)) or ts.std() or 1.0
        return (ts - med) / sig
    blocks = ts[: m * nb].reshape(m, nb)
    meds = np.median(blocks, 1)
    sigs = MAD2SIG * np.median(np.abs(blocks - meds[:, None]), 1)
    sigs[sigs <= 0] = np.median(sigs[sigs > 0]) if np.any(sigs > 0) else 1.0
    c = (np.arange(m) + 0.5) * nb
    x = np.arange(n)
    return (ts - np.interp(x, c, meds)) / np.interp(x, c, sigs)


def search(data: np.ndarray, fb: Filterbank, dms, thresh: float = 8.0, max_width: int = 1024,
           baseline: float = 2.0, mask: np.ndarray | None = None):
    """
    data: (n, nchan) float32. Restituisce lista di (t [s], dm, S/N, larghezza in campioni).
    Il tempo e' riferito al canale a frequenza piu' alta.
    """
    n, nch = data.shape
    ts_ = fb.tsamp
    fmax = fb.freqs.max()
    use = np.ones(nch, bool) if mask is None else ~mask
    nb = max(16, int(baseline / ts_))
    out = []
    for dm in dms:
        sh = np.round(KDM * dm * (fb.freqs ** -2 - fmax ** -2) / ts_).astype(int)
        L = n - sh.max()
        if L <= 0:
            continue
        s = np.zeros(L, np.float64)
        for c in np.flatnonzero(use):
            s += data[sh[c]:sh[c] + L, c]
        z = _local_norm(s, nb)
        cs = np.concatenate([[0.0], np.cumsum(z)])
        W = 1
        while W <= max_width and W < L // 4:
            b = (cs[W:] - cs[:-W]) / np.sqrt(W)
            # normalizza la serie dei boxcar con la sua sigma robusta (come il rumore misurato)
            sb = MAD2SIG * np.median(np.abs(b - np.median(b)))
            b = b / (sb if sb > 0 else 1.0)
            idx = np.flatnonzero(b > thresh)
            if idx.size:
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                for r in runs:
                    p = r[np.argmax(b[r])]
                    out.append(((p + W / 2) * ts_, float(dm), float(b[p]), W))
            W *= 2
    return out


def cluster(cands, dt: float = 0.05):
    """Raggruppa i candidati vicini nel tempo (entro dt); per ogni gruppo tiene il piu' forte."""
    if not cands:
        return []
    c = sorted(cands, key=lambda x: x[0])
    groups = [[c[0]]]
    for x in c[1:]:
        if x[0] - groups[-1][-1][0] <= dt:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [dict(t=max(g, key=lambda x: x[2])[0], dm=max(g, key=lambda x: x[2])[1],
                 snr=max(x[2] for x in g), width=max(g, key=lambda x: x[2])[3], n=len(g)) for g in groups]


def run(path, dm_min=25, dm_max=700, thresh=8.0, max_width=1024, tol=1.25, tmax=None):
    fb = Filterbank(path)
    n = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / fb.tsamp))
    data = fb.read(0, n).astype(np.float32)
    dms = dm_grid(fb, dm_min, dm_max, tol)
    raw = search(data, fb, dms, thresh, max_width)
    return raw, cluster(raw)
