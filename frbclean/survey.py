"""
Censimento degli RFI in un filterbank SIGPROC, in un solo passaggio a blocchi.

Funziona con qualsiasi filterbank: tutti i parametri sono in unita' fisiche
(secondi, MHz) e vengono convertiti in campioni e canali a partire dall'header.

Prodotti (nella cartella di output):
  survey.npz          statistiche per canale e per blocco, spettri dinamici ridotti,
                      serie temporale a DM 0 a ~1 ms
  events.csv          catalogo degli eventi impulsivi a DM 0 (banda intera e sottobande)
  periodicity.csv     periodicita' trovate nella serie a DM 0 e nel treno di eventi
  flagged_channels.txt  canali sospetti in formato "min max" (stesso formato del flag file
                      della pipeline, indici dei canali nell'ordine del file)
  summary.json        riassunto numerico
  *.png               grafici diagnostici (se matplotlib e' disponibile)
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .sigproc import Filterbank

MAD2SIG = 1.4826


# --------------------------------------------------------------------------- utilita'
def robust_sigma(x: np.ndarray, axis=0, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mediana e sigma robusta (1.4826 * MAD) lungo axis, eventualmente su un sottocampione."""
    xs = x[::step] if axis == 0 else x
    med = np.median(xs, axis=axis)
    sig = MAD2SIG * np.median(np.abs(xs - med), axis=axis)
    bad = sig <= 0
    if np.any(bad):
        sig = np.where(bad, np.std(xs, axis=axis), sig)
    return med, sig


class Decimator:
    """Accumula somme, conteggi e massimi in bin temporali fissi, con blocchi di ingresso arbitrari."""

    def __init__(self, nbins: int, binsize: int, ncol: int, keep_max: bool = False):
        self.binsize = binsize
        self.sum = np.zeros((nbins, ncol))
        self.cnt = np.zeros(nbins)
        self.max = np.full((nbins, ncol), -np.inf) if keep_max else None

    def add(self, start: int, data: np.ndarray):
        n = data.shape[0]
        bs = self.binsize
        i = 0
        # parte iniziale fino al primo bordo di bin (metodo generico)
        head = min(n, (-start) % bs)
        if head:
            self._add_slow(start, data[:head])
            i = head
        # bin completi: reshape e somma (veloce)
        m = (n - i) // bs
        if m:
            b0 = (start + i) // bs
            b1 = min(b0 + m, self.cnt.size)
            m = b1 - b0
            if m > 0:
                seg = data[i:i + m * bs].reshape(m, bs, -1)
                self.sum[b0:b1] += seg.sum(axis=1, dtype=np.float64)
                self.cnt[b0:b1] += bs
                if self.max is not None:
                    self.max[b0:b1] = np.maximum(self.max[b0:b1], seg.max(axis=1))
                i += m * bs
        if i < n:
            self._add_slow(start + i, data[i:])

    def _add_slow(self, start: int, data: np.ndarray):
        """Pochi segmenti: un ciclo sui bin toccati."""
        n = data.shape[0]
        b = start // self.binsize
        i = 0
        while i < n:
            j = min(n, (b + 1) * self.binsize - start)
            bb = min(b, self.cnt.size - 1)
            self.sum[bb] += data[i:j].sum(axis=0, dtype=np.float64)
            self.cnt[bb] += j - i
            if self.max is not None:
                self.max[bb] = np.maximum(self.max[bb], data[i:j].max(axis=0))
            i, b = j, b + 1

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum / self.cnt[:, None]


def subband_edges(nchan: int, nsub: int) -> np.ndarray:
    return np.linspace(0, nchan, nsub + 1).round().astype(int)[:-1]


# --------------------------------------------------------------------------- parametri
@dataclass
class SurveyConfig:
    nsub: int = 16               # sottobande per gli eventi a banda parziale
    max_width: float = 0.032     # larghezza massima degli impulsi cercati [s]
    thresh: float = 6.0          # soglia S/N per gli eventi
    sub_hit: float = 3.0         # soglia S/N per dire che una sottobanda e' "colpita"
    block: float | None = None   # durata dei blocchi [s] (default: automatico)
    dyn_nt: int = 2000           # bin temporali dello spettro dinamico ridotto
    ts_res: float = 1e-3         # risoluzione della serie a DM 0 salvata [s]
    max_runs: int = 20000        # massimo numero di eventi per livello e per blocco
    chan_z: float = 5.0          # soglia (z robusto) per segnalare canali sospetti


@dataclass
class Event:
    t: float            # centro dell'evento [s]
    width: float        # larghezza del boxcar migliore [s]
    snr_full: float     # S/N nella banda intera
    snr_max: float      # S/N massimo tra banda intera e sottobande
    best: str           # 'full' oppure 'subNN' (NN = sottobanda in ordine di frequenza crescente)
    n_sub_hit: int      # sottobande sopra sub_hit al picco
    frac_band: float    # n_sub_hit / nsub
    sub_hits: str       # maschera delle sottobande (in ordine di frequenza crescente)


# --------------------------------------------------------------------------- ricerca eventi
def _detect_block(X: np.ndarray, start: int, kmax: int, cfg: SurveyConfig, sub_order: np.ndarray):
    """
    X: (n, 1 + nsub) serie a DM 0 (banda intera + sottobande), con la baseline gia' sottratta.
    Restituisce una lista di record (start_box, W, snr_full, snr_max, col, hits) in campioni.
    """
    n, ncol = X.shape
    cs = np.concatenate([np.zeros((1, ncol)), np.cumsum(X, axis=0)])
    recs = []
    step = max(1, n // 10000)
    for k in range(kmax + 1):
        W = 1 << k
        if W >= n // 4:
            break
        B = cs[W:] - cs[:-W]                       # boxcar sovrapposti, B[i] = somma di X[i:i+W]
        _, sig = robust_sigma(B, axis=0, step=step)
        sig[sig <= 0] = np.inf
        Z = B / sig
        zmax = Z.max(axis=1)
        idx = np.flatnonzero(zmax > cfg.thresh)
        if idx.size == 0:
            continue
        cut = np.flatnonzero(np.diff(idx) > 1) + 1
        runs = np.split(idx, cut)
        if len(runs) > cfg.max_runs:
            peaks = np.array([zmax[r].max() for r in runs])
            keep = np.argsort(peaks)[-cfg.max_runs:]
            runs = [runs[i] for i in np.sort(keep)]
        for r in runs:
            p = r[np.argmax(zmax[r])]
            z = Z[p]
            col = int(np.argmax(z))
            hits = z[1:] > cfg.sub_hit
            recs.append((start + p, W, float(z[0]), float(z[col]), col, hits[sub_order]))
    # nota: col e' nell'ordine del file; viene convertito in ordine di frequenza in _merge
    return recs


def _merge(recs, nsub: int, tsamp: float, sub_rank: np.ndarray) -> list[Event]:
    """Unisce i record sovrapposti nel tempo (anche tra larghezze diverse) tenendo il piu' forte."""
    if not recs:
        return []
    st = np.array([r[0] for r in recs])
    en = st + np.array([r[1] for r in recs])
    order = np.argsort(st, kind="stable")
    st, en = st[order], en[order]
    runmax = np.maximum.accumulate(en)
    new = np.concatenate([[True], st[1:] >= runmax[:-1]])
    cid = np.cumsum(new) - 1
    out = []
    recs = [recs[i] for i in order]
    snr_max = np.array([r[3] for r in recs])
    snr_full = np.array([r[2] for r in recs])
    bounds = np.flatnonzero(new).tolist() + [len(recs)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        j = a + int(np.argmax(snr_max[a:b]))
        s0, W, _, smax, col, hits = recs[j]
        out.append(Event(
            t=(s0 + W / 2) * tsamp, width=W * tsamp, snr_full=float(snr_full[a:b].max()),
            snr_max=float(smax), best="full" if col == 0 else f"sub{int(sub_rank[col - 1]):02d}",
            n_sub_hit=int(hits.sum()), frac_band=float(hits.sum()) / nsub,
            sub_hits="".join("1" if h else "0" for h in hits)))
    return out


# --------------------------------------------------------------------------- periodicita'
def _norm_spectrum(x: np.ndarray, dt: float, nharm: int = 8, fmin: float = 0.05):
    """Spettro di potenza normalizzato con mediana mobile e somma armonica. Restituisce (f, P, P_sum)."""
    x = np.nan_to_num(x - np.nanmedian(x))
    n = x.size
    if n < 64:
        return np.array([]), np.array([]), np.array([])
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, dt)
    P[0] = 0
    # normalizzazione a blocchi logaritmici (tiene conto del rumore rosso)
    norm = np.ones_like(P)
    edges = np.unique(np.geomspace(1, P.size, 200).astype(int))
    for a, b in zip(edges[:-1], edges[1:]):
        half = max(b - a, 64)                   # almeno ~128 bin, cosi' un picco non sposta la mediana
        lo, hi = max(1, a - half), min(P.size, b + half)
        m = np.median(P[lo:hi])
        norm[a:b] = m if m > 0 else 1
    Pn = P / norm / np.log(2)           # potenza normalizzata: media ~1 per rumore
    Ps = Pn.copy()
    for h in range(2, nharm + 1):
        idx = np.arange(P.size) * h
        ok = idx < P.size
        Ps[ok] += Pn[idx[ok]]
    sel = f >= fmin
    return f[sel], Pn[sel], Ps[sel]


def _top_peaks(f, Pn, Ps, ntop=10, nharm=8, max_div=32):
    """
    Picchi piu' forti della somma armonica, riportati alla frequenza fondamentale.
    Per un treno di impulsi stretti tutte le armoniche sono forti: per ogni picco si cerca
    il divisore n piu' grande per cui anche f/n e' forte nello spettro NON sommato (la somma
    armonica a f/n conterrebbe f stessa), e dopo
    ogni detezione si azzerano tutte le armoniche della fondamentale trovata.
    """
    if f.size == 0:
        return []
    out = []
    P = Ps.copy()
    df = f[1] - f[0] if f.size > 1 else 1.0
    i0 = int(round(f[0] / df))                      # indice di f[0] nello spettro completo

    def at(freq):
        j = int(round(freq / df)) - i0
        return j if 0 <= j < P.size else None

    for _ in range(ntop):
        i = int(np.argmax(P))
        if P[i] <= 0:
            break
        fund = f[i]
        for n in range(max_div, 1, -1):
            j = at(f[i] / n)
            if j is None:
                continue
            loc = Pn[max(0, j - 1):j + 2].max()
            ref = Pn[max(0, i - 1):i + 2].max()
            if loc >= 0.3 * ref and loc > 10:
                fund = f[i] / n
                break
        sig = (P[i] - nharm) / np.sqrt(nharm)       # S/N approssimato (chi^2 con 2*nharm g.d.l.)
        out.append(dict(freq_hz=float(fund), period_s=float(1 / fund), power=float(P[i]), sigma=float(sig),
                        strongest_harmonic_hz=float(f[i])))
        # azzera la fondamentale, le sue armoniche e le sub-armoniche vicine
        for h in list(np.arange(1, int(f[-1] / fund) + 2)) + [0.5, 1 / 3, 0.25]:
            j = at(fund * h)
            if j is not None:
                P[max(0, j - 2):j + 3] = 0
    return out


def refine_period(times: np.ndarray, f0: float, tspan: float, nbins: float = 2.0, ngrid: int = 401):
    """
    Affina una frequenza f0 su tempi di arrivo esatti con il test di Rayleigh:
    Z(f) = |sum exp(2 pi i f t)|^2 / N, massimizzato in f0 +- nbins/tspan.
    Restituisce (f_migliore, Z).
    """
    if times.size < 3 or f0 <= 0:
        return f0, 0.0
    fs = f0 + np.linspace(-nbins, nbins, ngrid) / tspan
    fs = fs[fs > 0]
    Z = np.empty(fs.size)
    for a in range(0, fs.size, 50):
        ph = np.exp(2j * np.pi * np.outer(fs[a:a + 50], times))
        Z[a:a + 50] = np.abs(ph.sum(1)) ** 2 / times.size
    i = int(np.argmax(Z))
    return float(fs[i]), float(Z[i])


# --------------------------------------------------------------------------- canali
def flag_channels(noise: np.ndarray, kurt: np.ndarray, outfrac: np.ndarray, zthr: float,
                  min_rel: float = 0.1):
    """
    Canali sospetti: rumore anomalo rispetto ai vicini (z robusto > zthr e differenza relativa
    > min_rel), kurtosis alta, molti outlier, canali morti.
    noise: deviazione standard tipica del canale (mediana nel tempo). Non si usa la MAD perche'
    con dati a pochi bit e' quantizzata e crea salti finti tra canali vicini.
    """
    nch = noise.size
    med_mad = noise
    reasons = {}
    dead = med_mad <= 0
    for c in np.flatnonzero(dead):
        reasons.setdefault(int(c), []).append("morto")
    live = ~dead
    if live.sum() >= 8:
        # rumore: confronto con una mediana mobile su ~nch/32 canali (IQRM-like)
        w = max(3, (nch // 32) | 1)
        lg = np.log(np.where(live, med_mad, np.nan))
        pad = np.pad(lg, w // 2, mode="edge")
        smooth = np.array([np.nanmedian(pad[i:i + w]) for i in range(nch)])
        d = lg - smooth
        _, s = robust_sigma(d[live])
        z = d / s if s > 0 else np.zeros(nch)
        for c in np.flatnonzero(live & (np.abs(z) > zthr) & (np.abs(d) > np.log1p(min_rel))):
            reasons.setdefault(int(c), []).append(f"rumore z={z[c]:+.1f}")
        mk, sk = robust_sigma(kurt[live])
        zk = (kurt - mk) / sk if sk > 0 else np.zeros(nch)
        for c in np.flatnonzero(live & (zk > zthr)):
            reasons.setdefault(int(c), []).append(f"kurtosis z={zk[c]:+.1f}")
        mo, so = robust_sigma(outfrac[live])
        zo = (outfrac - mo) / so if so > 0 else np.zeros(nch)
        for c in np.flatnonzero(live & (zo > zthr) & (outfrac > 1e-3)):
            reasons.setdefault(int(c), []).append(f"outlier {outfrac[c] * 100:.2f}%")
    return dict(sorted(reasons.items()))


def group_ranges(chans) -> list[tuple[int, int]]:
    chans = sorted(chans)
    out = []
    for c in chans:
        if out and c == out[-1][1] + 1:
            out[-1] = (out[-1][0], c)
        else:
            out.append((c, c))
    return out


# --------------------------------------------------------------------------- survey
def run_survey(path, outdir, cfg: SurveyConfig | None = None, plot: bool = True, verbose: bool = True,
               tmax: float | None = None) -> dict:
    cfg = cfg or SurveyConfig()
    t_start = time.time()
    fb = Filterbank(path)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    nch, ts = fb.nchans, fb.tsamp
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    nsub = max(1, min(cfg.nsub, nch))
    sub_edges = subband_edges(nch, nsub)
    # ordine delle sottobande per frequenza crescente (i canali possono essere in ordine decrescente)
    sub_freq = np.array([fb.freqs[a:b].mean() for a, b in zip(sub_edges, list(sub_edges[1:]) + [nch])])
    sub_order = np.argsort(sub_freq)
    sub_rank = np.argsort(sub_order)          # posizione di ogni sottobanda in ordine di frequenza
    sub_matrix = np.zeros((nch, nsub), np.float32)
    for j, (a, b) in enumerate(zip(sub_edges, list(sub_edges[1:]) + [nch])):
        sub_matrix[a:b, j] = 1

    kmax = max(0, int(np.ceil(np.log2(max(cfg.max_width / ts, 1)) - 1e-9)))
    Wmax = 1 << kmax
    Lb = 8 * Wmax                                       # finestra per la baseline mobile
    block = cfg.block or max(1.0, 128 * Wmax * ts)
    nper = max(Lb, int(round(block / ts)) // Lb * Lb)
    nblocks = int(np.ceil(nsamp / nper))
    if verbose:
        print(f"{fb.path}: {nch} canali, tsamp {ts * 1e6:.3g} us, {nsamp * ts:.1f} s, "
              f"banda {fb.f_lo:.2f}-{fb.f_hi:.2f} MHz, {fb.nbits} bit")
        print(f"  blocchi da {nper * ts:.2f} s ({nblocks}), boxcar 1..{Wmax} campioni "
              f"({ts * 1e3:.3g}..{Wmax * ts * 1e3:.3g} ms), {nsub} sottobande")

    # accumulatori
    st = {k: np.zeros((nblocks, nch), np.float32) for k in ("mean", "std", "median", "mad", "kurt", "skew", "outfrac")}
    block_t = np.zeros(nblocks)
    dyn_bin = max(1, int(np.ceil(nsamp / cfg.dyn_nt)))
    dyn = Decimator(int(np.ceil(nsamp / dyn_bin)), dyn_bin, nch, keep_max=True)
    ts_bin = 1 << max(0, int(round(np.log2(max(cfg.ts_res / ts, 1)))))
    tsd = Decimator(int(np.ceil(nsamp / ts_bin)), ts_bin, 1 + nsub)
    events: list[Event] = []

    for ib, (s0, raw) in enumerate(fb.blocks(nper, 0, nsamp)):
        n = raw.shape[0]
        block_t[ib] = (s0 + n / 2) * ts
        # --- statistiche per canale (momenti esatti a fette, mediane su sottocampione)
        xf = raw.astype(np.float32)
        c = xf[:: max(1, n // 4096)].mean(0)            # centro approssimato: momenti centrati (precisi in float32)
        s1 = np.zeros(nch); s2 = np.zeros(nch); s3 = np.zeros(nch); s4 = np.zeros(nch)
        for a in range(0, n, 32768):
            x = xf[a:a + 32768] - c
            x2 = x * x
            s1 += x.sum(0, dtype=np.float64); s2 += x2.sum(0, dtype=np.float64)
            s3 += (x2 * x).sum(0, dtype=np.float64); s4 += (x2 * x2).sum(0, dtype=np.float64)
        mu = s1 / n                                      # media rispetto a c
        var = np.maximum(s2 / n - mu * mu, 0)
        sd = np.sqrt(var)
        m = c + mu
        with np.errstate(invalid="ignore", divide="ignore"):
            m3 = s3 / n - 3 * mu * s2 / n + 2 * mu ** 3
            m4 = s4 / n - 4 * mu * s3 / n + 6 * mu * mu * s2 / n - 3 * mu ** 4
            st["skew"][ib] = np.where(sd > 0, m3 / sd ** 3, 0)
            st["kurt"][ib] = np.where(sd > 0, m4 / var ** 2 - 3, 0)
        step = max(1, n // 8192)
        sub = xf[::step]
        med = np.median(sub, 0)
        mad = MAD2SIG * np.median(np.abs(sub - med), 0)
        scale = np.where(mad > 0, mad, sd)
        with np.errstate(invalid="ignore", divide="ignore"):
            st["outfrac"][ib] = np.where(scale > 0, (np.abs(sub - med) > 6 * scale).mean(0), 0)
        st["mean"][ib], st["std"][ib], st["median"][ib], st["mad"][ib] = m, sd, med, mad

        # --- spettro dinamico ridotto
        dyn.add(s0, raw)

        # --- serie a DM 0: banda intera + sottobande
        S = (xf @ sub_matrix).astype(np.float64)        # somme per sottobanda (BLAS)
        X = np.empty((n, nsub + 1))
        X[:, 0] = S.sum(1)
        X[:, 1:] = S
        del xf
        tsd.add(s0, X)
        # baseline mobile: mediane su finestre di Lb campioni, interpolate
        nb = n // Lb
        if nb >= 2:
            sstep = max(1, Lb // 2048)
            meds = np.median(X[:nb * Lb].reshape(nb, Lb, -1)[:, ::sstep], axis=1)
            centers = (np.arange(nb) + 0.5) * Lb
            base = np.column_stack([np.interp(np.arange(n), centers, meds[:, j]) for j in range(X.shape[1])])
        else:
            base = np.median(X, axis=0)[None, :]
        X = X - base
        recs = _detect_block(X, s0, kmax, cfg, sub_order)
        ev = _merge(recs, nsub, ts, sub_rank)
        events.extend(ev)
        if verbose:
            print(f"  blocco {ib + 1}/{nblocks}: t={s0 * ts:8.1f} s, eventi {len(ev):6d}", flush=True)

    events.sort(key=lambda e: e.t)

    # --- canali
    noise = np.median(st["std"], 0)
    kurt_med = np.median(st["kurt"], 0)
    out_med = np.median(st["outfrac"], 0)
    flagged = flag_channels(noise, kurt_med, out_med, cfg.chan_z)

    # --- periodicita'
    ts_series = tsd.mean()
    tres = ts_bin * ts
    per_zero = _top_peaks(*_norm_spectrum(ts_series[:, 0], tres))
    def _train(sel):
        tr = np.zeros(ts_series.shape[0])
        tt = []
        for e in events:
            if e.snr_max >= cfg.thresh and sel(e):
                i = int(e.t / tres)
                if i < tr.size:
                    tr[i] += 1
                    tt.append(e.t)
        return tr, np.array(tt)

    def _train_peaks(sel):
        tr, tt = _train(sel)
        peaks = _top_peaks(*_norm_spectrum(tr, tres))
        if tt.size > 20000:                              # Rayleigh su un sottoinsieme se gli eventi sono tanti
            tt = np.sort(np.random.default_rng(0).choice(tt, 20000, replace=False))
        out_ = []
        for p in peaks:
            f, z = refine_period(tt, p["freq_hz"], nsamp * ts)
            p.update(freq_hz=f, period_s=1 / f, rayleigh_z=z)
            if all(abs(f / q["freq_hz"] - 1) > 0.005 for q in out_):   # scarta i doppioni dopo l'affinamento
                out_.append(p)
        return out_
    # treno di tutti gli eventi e solo di quelli a banda larga (>= meta' delle sottobande: tipico dei radar)
    per_train = _train_peaks(lambda e: True)
    per_train_bb = _train_peaks(lambda e: e.frac_band >= 0.5)

    # --- tempo occupato dagli eventi
    occ = np.zeros(int(np.ceil(nsamp * ts / 1e-3)) + 1, bool)
    for e in events:
        a, b = int((e.t - e.width / 2) / 1e-3), int(np.ceil((e.t + e.width / 2) / 1e-3))
        occ[max(0, a):b + 1] = True

    # --- salvataggi
    np.savez_compressed(
        out / "survey.npz", freqs=fb.freqs, block_t=block_t, **{f"chan_{k}": v for k, v in st.items()},
        dyn_mean=dyn.mean().astype(np.float32), dyn_max=dyn.max.astype(np.float32), dyn_dt=dyn_bin * ts,
        zerodm=ts_series.astype(np.float32), zerodm_dt=tres, sub_edges=sub_edges, sub_freq=sub_freq)
    with open(out / "events.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(list(Event.__dataclass_fields__))
        for e in events:
            d = asdict(e)
            w.writerow([f"{d['t']:.6f}", f"{d['width']:.6g}", f"{d['snr_full']:.2f}", f"{d['snr_max']:.2f}",
                        d["best"], d["n_sub_hit"], f"{d['frac_band']:.3f}", d["sub_hits"]])
    with open(out / "periodicity.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["series", "freq_hz", "period_s", "power", "sigma", "strongest_harmonic_hz", "rayleigh_z"])
        for name, lst in (("zerodm", per_zero), ("event_train", per_train),
                          ("event_train_broadband", per_train_bb)):
            for p in lst:
                w.writerow([name, f"{p['freq_hz']:.6f}", f"{p['period_s']:.6f}", f"{p['power']:.2f}", f"{p['sigma']:.2f}",
                            f"{p['strongest_harmonic_hz']:.6f}", f"{p.get('rayleigh_z', float('nan')):.1f}"])
    with open(out / "flagged_channels.txt", "w") as ff:
        ff.write("# canali sospetti (indici nell'ordine del file), formato: min max\n")
        for a, b in group_ranges(flagged):
            ff.write(f"{a} {b}\n")

    snr = np.array([e.snr_max for e in events])
    hours = nsamp * ts / 3600
    summary = dict(
        file=fb.path, nchans=nch, tsamp=ts, nbits=fb.nbits, duration_s=nsamp * ts,
        band_mhz=[fb.f_lo, fb.f_hi], config=asdict(cfg), nsub=nsub,
        boxcar_widths_s=[(1 << k) * ts for k in range(kmax + 1)],
        n_events=len(events),
        events_per_hour_above={str(t): float((snr >= t).sum() / hours) for t in (6, 8, 10, 20, 50, 100, 1000)},
        fraction_time_in_events=float(occ.mean()),
        events_by_band_fraction={
            "<=25%": int(sum(e.frac_band <= 0.25 for e in events)),
            "25-75%": int(sum(0.25 < e.frac_band <= 0.75 for e in events)),
            ">75%": int(sum(e.frac_band > 0.75 for e in events))},
        events_by_width_ms={f"{w * 1e3:.3g}": int(sum(abs(e.width - w) < 1e-12 for e in events))
                            for w in [(1 << k) * ts for k in range(kmax + 1)]},
        top_events=[asdict(e) for e in sorted(events, key=lambda e: -e.snr_max)[:20]],
        flagged_channels={str(k): v for k, v in flagged.items()},
        periodicity_zerodm=per_zero, periodicity_event_train=per_train,
        periodicity_event_train_broadband=per_train_bb,
        runtime_s=None)
    summary["runtime_s"] = time.time() - t_start
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if plot:
        try:
            from . import survey_plots
            survey_plots.make_all(out, summary, events, fb)
        except ImportError as e:
            print(f"  grafici saltati ({e})")
    if verbose:
        print(f"Fatto in {summary['runtime_s']:.0f} s: {len(events)} eventi, "
              f"{len(flagged)} canali sospetti, prodotti in {out}")
    return summary
