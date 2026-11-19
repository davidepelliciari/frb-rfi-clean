"""
Step 2: individuazione automatica e sottrazione delle componenti periodiche (RFI periodico).

Passaggio 1 - analisi (una lettura del file)
  * serie a DM 0: banda intera a risoluzione nativa e 4 quarti di banda (decimati di un fattore 4),
    tenute in memoria;
  * spettro di potenza (Welch, finestra di Hann) della banda intera fino alla frequenza di Nyquist
    nativa, quindi senza aliasing;
  * righe: massimi locali di spettro/continuo (mediana mobile su 2 Hz) oltre `line_thresh`;
  * famiglie: per ogni frequenza candidata f0 (riga/k) si sommano i pesi log10(rapporto) delle righe
    che cadono su un'armonica n*f0 (entro 2 bin), meno il contributo atteso per caso
    (peso totale x 4 df / f0); si sceglie la f0 con l'eccesso maggiore, si tolgono le sue righe e si
    ripete. Se due famiglie hanno un sottomultiplo comune (es. 50 Hz e 166.67 Hz -> 16.67 Hz) viene
    scelto quello, perche' spiega piu' righe;
  * affinamento della frequenza: si piega la serie a tratti, si misura lo spostamento del profilo
    nel tempo (correlazione circolare) e si corregge f0 (precisione ~ tsamp / (P * durata));
  * test di dispersione: si piegano i 4 quarti di banda e si misura il ritardo tra i profili; se e'
    compatibile con una sorgente dispersa con DM >= `protect_dm` la famiglia NON viene sottratta
    (possibile pulsar) e viene segnalata.

Passaggio 2 - sottrazione
  per ogni canale e tratto (almeno `min_cycles` periodi) il profilo medio clippato a risoluzione
  nativa (un intervallo di fase per campione) viene sottratto in base alla fase. Sottrarre la media
  per fase equivale a togliere la potenza alle armoniche n*f0 (tutte, fino a Nyquist) nella banda
  1/durata del tratto; un burst contribuisce al profilo per ~1/(numero di periodi) della sua
  ampiezza, e i suoi campioni brillanti sono esclusi dal clipping. Uscite intere: arrotondamento
  stocastico.

Passaggio 3 - verifica
  spettro dell'output, righe residue, grafico prima/dopo. Con iterations > 1, se restano famiglie
  sottraibili si ripete.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .sigproc import Filterbank, pack_bits, write_header
from .spectrum import continuum, find_lines

KDM = 4.148808e3


@dataclass
class PeriodicConfig:
    freqs: list | None = None      # frequenze fondamentali manuali [Hz]; None = ricerca automatica
    fmin_fund: float = 5.0         # fondamentale minima considerata [Hz]
    line_thresh: float = 8.0       # soglia spettro/continuo per le righe
    nlines: int = 300              # righe piu' forti considerate
    max_families: int = 8
    welch_seg: float = 4.0         # durata (circa) dei segmenti dello spettro [s]
    segment: float = 10.0          # durata minima dei tratti di sottrazione [s]
    min_cycles: int = 200          # periodi minimi per tratto
    clip: float = 4.0              # clipping del profilo (deviazioni standard per intervallo di fase)
    refine: bool = True
    refine_seg_max: float = 60.0   # durata massima dei tratti per l'affinamento della frequenza [s]
    match_tol: float = 0.5         # tolleranza armonica (in bin di frequenza dello spettro)
    protect: bool = True           # test di dispersione (non sottrarre segnali dispersi)
    protect_dm: float = 2.0
    protect_min_period: float = 0.0015   # periodi piu' corti non vengono testati (non sono pulsar)
    protect_untested: bool = True  # se il test non e' conclusivo, la famiglia non viene sottratta
    min_period_samples: float = 8.0  # periodo minimo (in campioni) di una famiglia da sottrarre
    min_power_frac: float = 1e-4   # contributo minimo alla varianza della serie a DM 0
    iterations: int = 2
    dither: bool = True
    seed: int = 0
    chan_group: int = 32           # canali elaborati insieme (memoria)


# =========================================================================== analisi
def analyze(path, cfg: PeriodicConfig, verbose=True, tmax=None):
    fb = Filterbank(path)
    ts, nch = fb.tsamp, fb.nchans
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    L = 1 << int(round(np.log2(max(64, cfg.welch_seg / ts))))
    q_edges = np.linspace(0, nch, 5).round().astype(int)
    Mq = np.zeros((nch, 4), np.float32)
    for j in range(4):
        Mq[q_edges[j]:q_edges[j + 1], j] = 1
    qfreq = np.array([fb.freqs[q_edges[j]:q_edges[j + 1]].mean() for j in range(4)])
    full = np.empty(nsamp, np.float32)
    quart = np.zeros((nsamp // 4, 4), np.float32)
    win = np.hanning(L)
    U = (win ** 2).mean()
    psd = np.zeros(L // 2 + 1)
    nps = 0
    t0 = time.time()
    for s0, raw in fb.blocks(L, 0, nsamp):
        n = raw.shape[0]
        xf = raw.astype(np.float32)
        full[s0:s0 + n] = xf.sum(1)
        q = xf @ Mq
        m = (n // 4)
        if m and s0 % 4 == 0:
            quart[s0 // 4:s0 // 4 + m] = q[: m * 4].reshape(m, 4, 4).sum(1)
        if n == L:
            y = (full[s0:s0 + n] - full[s0:s0 + n].mean()) * win
            psd += np.abs(np.fft.rfft(y)) ** 2
            nps += 1
    f = np.fft.rfftfreq(L, ts)
    psd = psd / max(nps, 1) * 2 * ts / (L * U)
    psd[0] = 0
    if verbose:
        print(f"  analisi: {nsamp * ts:.1f} s, spettro fino a {f[-1]:.0f} Hz (risoluzione {f[1]:.3f} Hz), "
              f"{time.time() - t0:.0f} s", flush=True)
    return dict(fb=fb, ts=ts, nsamp=nsamp, full=full, quart=quart, qfreq=qfreq, f=f, psd=psd)


def _fit_family(fl, w, f0, df, fnyq, tol=0.5):
    nmax = max(1, int(fnyq / f0))
    ok = np.zeros(fl.size, bool)
    for _ in range(4):
        n = np.round(fl / f0)
        ok = (n >= 1) & (n <= nmax) & (np.abs(fl - n * f0) <= tol * df)
        if not ok.any():
            return f0, ok
        f0 = float(np.sum(w[ok] * n[ok] * fl[ok]) / np.sum(w[ok] * n[ok] ** 2))
    n = np.round(fl / f0)
    ok = (n >= 1) & (n <= nmax) & (np.abs(fl - n * f0) <= tol * df)
    return f0, ok


def find_families(lines, df, fnyq, cfg: PeriodicConfig):
    if not lines:
        return []
    lines = sorted(lines, key=lambda x: -x["ratio"])[: cfg.nlines]
    fl = np.array([x["freq_hz"] for x in lines])
    w = np.log10(np.array([x["ratio"] for x in lines]))
    rem = np.ones(fl.size, bool)
    fams = []
    while rem.any() and len(fams) < cfg.max_families:
        idx = np.flatnonzero(rem)
        top = idx[np.argsort(-w[idx])][:40]
        best = None
        for i in top:
            for k in range(1, 61):
                f0 = fl[i] / k
                if f0 < cfg.fmin_fund:
                    break
                f0r, ok = _fit_family(fl[rem], w[rem], f0, df, fnyq, cfg.match_tol)
                score = w[rem][ok].sum()
                # contributo atteso per caso: ogni riga cade entro +-tol*df da un'armonica con prob. 2*tol*df/f0
                excess = score - w[rem].sum() * min(1.0, 2 * cfg.match_tol * df / f0r)
                key = (excess, f0r)
                if best is None or key[0] > best[0] + 1e-9:
                    best = (excess, f0r, ok)
        if best is None or best[0] <= 0.5:
            break
        excess, f0r, ok = best
        sel = idx[ok]
        nmatched = np.round(fl[sel] / f0r)
        fams.append(dict(f0_hz=f0r, n_lines=int(ok.sum()), n_max=int(nmatched.max()), weight=float(w[sel].sum()),
                         excess=float(excess),
                         lines_hz=sorted(float(x) for x in fl[sel])[:50]))
        rem[sel] = False
    return fams


def _fold_series(x, start, ts, f, K):
    t = (start + np.arange(x.size) + 0.5) * ts
    b = np.minimum((np.mod(t * f, 1.0) * K).astype(np.int64), K - 1)
    cnt = np.bincount(b, minlength=K)
    return np.bincount(b, x, K) / np.maximum(cnt, 1), cnt


def _fold_long(x, ts, f, K, chunk=1 << 22):
    """Piegatura di una serie lunga, a pezzi (memoria contenuta)."""
    S = np.zeros(K)
    C = np.zeros(K)
    for a in range(0, x.size, chunk):
        xs = x[a:a + chunk].astype(np.float64)
        t = (a + np.arange(xs.size) + 0.5) * ts
        b = np.minimum((np.mod(t * f, 1.0) * K).astype(np.int64), K - 1)
        S += np.bincount(b, xs, K)
        C += np.bincount(b, minlength=K)
    return S / np.maximum(C, 1), C


def _xcorr_lag(a, b):
    """Spostamento circolare (in intervalli) che allinea b ad a, con interpolazione parabolica."""
    A = np.fft.rfft(a - a.mean())
    B = np.fft.rfft(b - b.mean())
    c = np.fft.irfft(A * np.conj(B), a.size)
    i = int(np.argmax(c))
    y0, y1, y2 = c[i - 1], c[i], c[(i + 1) % a.size]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    lag = (i + d) % a.size
    if lag > a.size / 2:
        lag -= a.size
    peak = y1 / (np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2)) + 1e-30)
    return lag, peak


def refine_frequency(full, ts, f0, cfg: PeriodicConfig, df=None, n_max=1):
    """
    Affina f0 misurando lo spostamento del profilo piegato nel tempo, a stadi: si parte da tratti
    brevi, adatti all'incertezza iniziale (la deriva in un tratto deve restare << 1 ciclo), e li si
    allunga di un fattore 4 a ogni stadio fino a refine_seg_max.
    """
    P = 1.0 / f0
    K = max(8, int(round(P / ts)))
    dur = full.size * ts
    sig_f = (cfg.match_tol * df / max(n_max, 1)) if df else 0.05 * f0     # incertezza iniziale
    T = max(20 * P, min(0.1 / sig_f, cfg.refine_seg_max))
    f = f0
    stages = []
    while True:
        nseg = int(round(T / ts))
        if full.size // nseg < 4:
            break
        profs, tc = [], []
        for s0 in range(0, full.size - nseg + 1, nseg):
            p, _ = _fold_series(full[s0:s0 + nseg].astype(np.float64), s0, ts, f, K)
            profs.append(p)
            tc.append((s0 + nseg / 2) * ts)
        profs = np.array(profs)
        ref = profs.sum(0)
        res = [_xcorr_lag(ref, p) for p in profs]
        lags = np.unwrap(np.array([r[0] for r in res]) / K * 2 * np.pi) / (2 * np.pi)
        tc = np.array(tc)
        b = np.polyfit(tc, lags, 1)[0]
        cands = [f, f + b, f - b]
        score = [_fold_long(full, ts, fc, K)[0].var() for fc in cands]
        f = cands[int(np.argmax(score))]
        stages.append(dict(segment_s=T, correction_hz=f - cands[0]))
        if T >= cfg.refine_seg_max or T * 4 > dur / 4:
            break
        T = min(T * 4, cfg.refine_seg_max)
    return f, dict(f_initial=f0, f_refined=f, rel_change=(f - f0) / f0, stages=stages)


def dispersion_test(quart, qfreq, ts, f0, cfg: PeriodicConfig):
    """Ritardo tra i profili dei quarti di banda; DM equivalente."""
    P = 1.0 / f0
    out = dict(tested=False)
    if P < cfg.protect_min_period:
        out["reason"] = "periodo troppo corto per una pulsar"
        return out
    ts4 = 4 * ts
    K = max(8, int(round(P / ts4)))
    profs, snr = [], []
    for j in range(4):
        x = quart[:, j]
        p, cnt = _fold_long(x, ts4, f0, K)
        sig = float(x[::16].std()) / np.sqrt(max(cnt.mean(), 1))
        profs.append(p)
        snr.append(np.sqrt(max(p.var(), 0)) / sig if sig > 0 else 0.0)
    jref = int(np.argmax(qfreq))
    good = [j for j in range(4) if j != jref and snr[j] >= 5 and snr[jref] >= 5]
    if not good:
        out["reason"] = "profili dei quarti di banda troppo deboli per il test"
        out["snr_quarters"] = [float(s) for s in snr]
        return out
    dms = []
    for j in good:
        lag, peak = _xcorr_lag(profs[j], profs[jref])     # >0: il quarto j arriva dopo il riferimento
        dt = lag * ts4
        k = KDM * (qfreq[j] ** -2 - qfreq[jref] ** -2)
        dms.append(dt / k)
    dm = float(np.median(dms))
    out.update(tested=True, dm_equiv=dm, dm_quarters=[float(x) for x in dms],
               snr_quarters=[float(s) for s in snr], dispersed=bool(abs(dm) >= cfg.protect_dm and dm > 0))
    return out


# =========================================================================== sottrazione
def _phase_groups(s0, n, ts, f):
    """Intervalli di fase (uno per campione del periodo) e ordinamento dei campioni per fase."""
    K = max(2, int(round(1.0 / (f * ts))))
    t = (s0 + np.arange(n) + 0.5) * ts
    b = np.minimum((np.mod(t * f, 1.0) * K).astype(np.int64), K - 1)
    order = np.argsort(b, kind="stable")
    cnt = np.bincount(b, minlength=K)
    nz = np.flatnonzero(cnt)
    starts = np.concatenate([[0], np.cumsum(cnt)])[nz]
    return b, order, b[order], nz, starts, cnt[nz], K


def _subtract_segment(raw, s0, ts, fams, cfg, rng, lo, hi, dtype, nbits):
    """
    Sottrazione del profilo medio clippato per ogni famiglia, vettorizzata su gruppi di canali:
    i campioni vengono ordinati per fase una volta per tratto, e somme, clipping e profilo sono
    calcolati con np.add.reduceat sugli intervalli di fase.
    """
    n, nch = raw.shape
    out = np.empty((n, nch), dtype if nbits >= 8 else np.uint8)
    removed_var = []
    groups = []
    for fam in fams:
        f = fam["f_use"]
        K = max(2, int(round(1.0 / (f * ts))))
        if abs(1.0 - f * K * ts) * n < 0.3:
            # periodo = K campioni esatti (entro 0.3 campioni sul tratto): raggruppamento con reshape
            ph0 = int(np.floor(np.mod((s0 + 0.5) * ts * f, 1.0) * K)) % K
            ncyc = int(np.ceil((ph0 + n) / K))
            valid = np.zeros(ncyc * K, bool)
            valid[ph0:ph0 + n] = True
            groups.append(("exact", K, ph0, ncyc, valid.reshape(ncyc, K)))
        else:
            groups.append(("general",) + _phase_groups(s0, n, ts, f))
    for g0 in range(0, nch, cfg.chan_group):
        g1 = min(nch, g0 + cfg.chan_group)
        x = raw[:, g0:g1].astype(np.float32)
        v0 = x[::8].var(0)
        for grp in groups:
            if grp[0] == "exact":
                _, K, ph0, ncyc, V = grp
                buf = np.zeros((ncyc * K, g1 - g0), np.float32)
                buf[ph0:ph0 + n] = x
                A = buf.reshape(ncyc, K, g1 - g0)
                Vf = V[:, :, None]
                c = np.maximum(V.sum(0), 1)[:, None].astype(np.float32)
                m = A.sum(0) / c
                sd = np.sqrt(np.maximum((A * A).sum(0) / c - m * m, 0))
                sd[sd <= 0] = np.inf
                ok = (np.abs(A - m[None]) <= cfg.clip * sd[None]) & Vf
                T = (A * ok).sum(0) / np.maximum(ok.sum(0), 1)
                T -= T.mean(0, keepdims=True)
                A -= T[None]
                x = buf[ph0:ph0 + n].copy()
                continue
            _, b, order, bs, nz, starts, cnt, K = grp
            xs = x[order]                                         # campioni ordinati per fase
            c = cnt[:, None].astype(np.float64)
            S1 = np.add.reduceat(xs, starts, axis=0, dtype=np.float64)
            S2 = np.add.reduceat(xs * xs, starts, axis=0, dtype=np.float64)
            m = S1 / c
            sd = np.sqrt(np.maximum(S2 / c - m * m, 0))
            sd[sd <= 0] = np.inf
            pos = np.searchsorted(nz, bs)                         # indice dell'intervallo per ogni campione ordinato
            ok = np.abs(xs - m[pos].astype(np.float32)) <= (cfg.clip * sd[pos]).astype(np.float32)
            Sc = np.add.reduceat(np.where(ok, xs, 0), starts, axis=0, dtype=np.float64)
            Nc = np.add.reduceat(ok, starts, axis=0, dtype=np.float64)
            T = np.where(Nc > 0, Sc / np.maximum(Nc, 1), m)
            T -= T.mean(0, keepdims=True)
            Tfull = np.zeros((K, g1 - g0), np.float32)
            Tfull[nz] = T
            x -= Tfull[b]
        v1 = x[::8].var(0)
        live = v0 > 0
        if live.any():
            removed_var.extend((1 - v1[live] / v0[live]).tolist())
        if nbits == 32:
            out[:, g0:g1] = x
        else:
            y = np.floor(x + rng.random(x.shape, dtype=np.float32)) if cfg.dither else np.rint(x)
            out[:, g0:g1] = np.clip(y, lo, hi)
    return out, (float(np.median(removed_var)) if removed_var else 0.0)


def subtract(path, out, fams, cfg: PeriodicConfig, verbose=True, tmax=None):
    fb = Filterbank(path)
    ts = fb.tsamp
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    if fb.nbits == 32:
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = (-128, 127) if fb.dtype == np.int8 else (0, 2 ** fb.nbits - 1)
    Pmax = max(1.0 / f["f_use"] for f in fams)
    nseg = int(round(max(cfg.segment, cfg.min_cycles * Pmax) / ts))
    rng = np.random.default_rng(cfg.seed)
    items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fb.items]
    seg_info = []
    t0 = time.time()
    with open(out, "wb") as fo:
        fo.write(write_header(items))
        for s0 in range(0, nsamp, nseg):
            n = min(nseg, nsamp - s0)
            raw = fb.read(s0, n)
            y, vr = _subtract_segment(raw, s0, ts, fams, cfg, rng, lo, hi, fb.dtype, fb.nbits)
            if fb.nbits < 8:
                fo.write(pack_bits(y.astype(np.uint8).ravel(), fb.nbits).tobytes())
            else:
                fo.write(y.tobytes())
            seg_info.append(dict(t0=float(s0 * ts), frac_var_removed_median=vr))
            if verbose:
                print(f"  sottrazione: t={s0 * ts:8.1f} s, varianza per canale rimossa {vr * 100:.3f}%  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    return seg_info


# =========================================================================== driver
def _families_for(an, cfg, verbose, manual=None):
    f, psd = an["f"], an["psd"]
    df = f[1] - f[0]
    lines = find_lines(f, psd, thresh=cfg.line_thresh, fmin=cfg.fmin_fund, nmax=cfg.nlines)
    if manual:
        fams = [dict(f0_hz=float(x), n_lines=None, manual=True) for x in manual]
    else:
        fams = find_families(lines, df, f[-1], cfg)
    tot_var = float(np.sum(psd[1:]) * df)
    cont = continuum(psd, df)
    for fam in fams:
        fu = fam["f0_hz"]
        # contributo della famiglia alla varianza della serie a DM 0 (potenza delle righe sopra il continuo)
        nh = np.arange(1, int(f[-1] / fu) + 1)
        ib = np.clip(np.round(nh * fu / df).astype(int), 1, psd.size - 2)
        exc = sum(float(np.maximum(psd[i - 1:i + 2] - cont[i - 1:i + 2], 0).sum()) for i in ib) * df
        fam["power_frac"] = exc / tot_var if tot_var > 0 else 0.0
        fam["period_samples"] = 1.0 / (fu * an["ts"])
        skip = None
        if fam["period_samples"] < cfg.min_period_samples:
            skip = f"periodo di {fam['period_samples']:.1f} campioni (< {cfg.min_period_samples})"
        elif fam["power_frac"] < cfg.min_power_frac:
            skip = f"contributo trascurabile ({fam['power_frac']:.1e} della varianza)"
        if skip and not fam.get("manual"):
            fam.update(f_use=fu, protected=True, skipped=skip)
            if verbose:
                print(f"  famiglia f0 = {fu:.6f} Hz: ignorata, {skip}", flush=True)
            continue
        if cfg.refine:
            fu, rinfo = refine_frequency(an["full"], an["ts"], fu, cfg, df=an["f"][1] - an["f"][0],
                                         n_max=fam.get("n_max") or 1)
            fam["refine"] = rinfo
        fam["f_use"] = fu
        fam["period_ms"] = 1e3 / fu
        if cfg.protect:
            dt_ = dispersion_test(an["quart"], an["qfreq"], an["ts"], fu, cfg)
            fam["dispersion"] = dt_
            fam["protected"] = bool(dt_.get("dispersed", False)) or (
                cfg.protect_untested and not dt_.get("tested", False) and "troppo corto" not in dt_.get("reason", ""))
        else:
            fam["protected"] = False
        if verbose:
            if not fam["protected"]:
                st = f"da sottrarre (circa {fam['power_frac'] * 100:.2f}% della varianza a DM 0)"
            elif fam.get("dispersion", {}).get("tested"):
                st = "PROTETTA (segnale disperso, possibile pulsar)"
            else:
                st = "NON sottratta (test di dispersione non conclusivo)"
            nl = fam.get("n_lines")
            print(f"  famiglia f0 = {fu:.6f} Hz (P = {1e3 / fu:.4f} ms)"
                  + (f", {nl} righe" if nl is not None else ", manuale")
                  + (f", DM equivalente {fam['dispersion'].get('dm_equiv', float('nan')):.1f}"
                     if cfg.protect and fam["dispersion"].get("tested") else "")
                  + f": {st}", flush=True)
    return lines, fams


def _plot_compare(path_png, f, psd_a, psd_b, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = f[1] - f[0]
    ca, cb = continuum(psd_a, df), continuum(psd_b, df)
    sel = f > 0
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax[0].loglog(f[sel], psd_a[sel], lw=0.4, label="prima")
    ax[0].loglog(f[sel], psd_b[sel], lw=0.4, label="dopo", alpha=0.8)
    ax[0].set_ylabel("densita' spettrale")
    ax[0].legend(fontsize=8)
    ax[0].set_title(title, fontsize=10)
    ax[1].loglog(f[sel], (psd_a / ca)[sel], lw=0.4, label="prima")
    ax[1].loglog(f[sel], (psd_b / cb)[sel], lw=0.4, label="dopo", alpha=0.8)
    ax[1].axhline(8, color="r", ls=":", lw=0.8)
    ax[1].set_ylabel("spettro / continuo")
    ax[1].legend(fontsize=8)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(psd_a > 0, psd_b / np.where(psd_a > 0, psd_a, 1), np.nan)
    ax[2].loglog(f[sel], r[sel], lw=0.3, color="k")
    ax[2].axhline(1, color="r", lw=0.8)
    ax[2].set_ylabel("dopo / prima")
    ax[2].set_xlabel("frequenza [Hz]")
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


def _top_lines(lines, n=30):
    return [dict(freq_hz=x["freq_hz"], ratio=x["ratio"]) for x in sorted(lines, key=lambda x: -x["ratio"])[:n]]


def run_periodic(path, out, cfg: PeriodicConfig | None = None, verbose=True, report=None, tmax=None) -> dict:
    cfg = cfg or PeriodicConfig()
    t_start = time.time()
    out = Path(out)
    if verbose:
        fb = Filterbank(path)
        print(f"{fb.path}: {fb.nchans} canali, tsamp {fb.tsamp * 1e6:.3g} us, {fb.duration:.1f} s, {fb.nbits} bit")
        print("Analisi dello spettro e delle famiglie di righe")
    an0 = analyze(path, cfg, verbose, tmax)
    lines0, fams = _families_for(an0, cfg, verbose, manual=cfg.freqs)
    an0 = dict(f=an0["f"], psd=an0["psd"])          # libera le serie temporali (memoria)
    history = []
    src = Path(path)
    it = 0
    all_fams = []
    while True:
        todo = [f for f in fams if not f["protected"]]
        all_fams.extend(fams)
        if not todo:
            if src != Path(path):
                break
            # niente da sottrarre: copia identica
            if verbose:
                print("Nessuna famiglia da sottrarre: l'output e' una copia dell'input")
            fb = Filterbank(path)
            nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / fb.tsamp))
            items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fb.items]
            with open(out, "wb") as fo:
                fo.write(write_header(items))
                for _, raw in fb.blocks(1 << 20, 0, nsamp):
                    fo.write(raw.tobytes() if fb.nbits >= 8 else pack_bits(raw.ravel(), fb.nbits).tobytes())
            src = out
            break
        it += 1
        if verbose:
            print(f"Sottrazione (iterazione {it}): {len(todo)} famiglie")
        tmp = out if src == Path(path) else out.with_name(out.name + ".tmp")
        seg = subtract(src, tmp, todo, cfg, verbose, tmax)
        if tmp != out:
            os.replace(tmp, out)
        src = out
        history.append(dict(iteration=it, families=[f["f_use"] for f in todo], segments=seg))
        if it >= cfg.iterations or cfg.freqs:
            break
        an = analyze(out, cfg, verbose, tmax)
        _, fams = _families_for(an, cfg, verbose)
        del an
    if verbose:
        print("Verifica: spettro dell'output")
    an1 = analyze(out, cfg, False, tmax) if history else an0
    lines1 = find_lines(an1["f"], an1["psd"], thresh=cfg.line_thresh, fmin=cfg.fmin_fund, nmax=cfg.nlines)
    rpath = Path(report) if report else Path(str(out) + ".periodic.json")
    base = str(rpath)[:-5] if str(rpath).endswith(".json") else str(rpath)
    np.savez_compressed(base + "_spectrum.npz", freq_hz=an0["f"], psd_before=an0["psd"].astype(np.float32),
                        psd_after=an1["psd"].astype(np.float32))
    try:
        _plot_compare(base + "_spectrum.png", an0["f"], an0["psd"], an1["psd"],
                      f"{Path(path).name}: serie a DM 0 prima e dopo lo step 2")
    except ImportError:
        pass
    summary = dict(input=str(path), output=str(out), config=asdict(cfg),
                   n_lines_before=len(lines0), n_lines_after=len(lines1),
                   lines_before=_top_lines(lines0), lines_after=_top_lines(lines1),
                   families=[{k: v for k, v in f.items()} for f in all_fams],
                   iterations=history, runtime_s=time.time() - t_start)
    rpath.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if verbose:
        print(f"Fatto in {summary['runtime_s']:.0f} s. Righe oltre {cfg.line_thresh}x il continuo: "
              f"{len(lines0)} prima, {len(lines1)} dopo.")
        if lines1:
            print("  righe residue piu' forti:", ", ".join(f"{x['freq_hz']:.2f} Hz ({x['ratio']:.0f}x)"
                                                         for x in _top_lines(lines1, 8)))
        print(f"Riassunto: {rpath}; grafico: {base}_spectrum.png")
    return summary
