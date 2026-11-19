"""
Step 2: sottrazione di componenti periodiche (es. 50 Hz e armoniche) canale per canale.

Per ogni tratto di `segment` secondi e per ogni canale si stima il profilo medio su un periodo
P = 1/f, diviso in `nbins` intervalli di fase, con una media clippata (si escludono i campioni oltre
`clip` deviazioni standard dal valore medio del loro intervallo di fase, cosi' impulsi e burst non
entrano nel profilo). Il profilo, privato della sua media (la sottrazione non cambia il livello
del canale), viene sottratto da ogni campione in base alla sua fase. Tutte le armoniche di f fino
a nbins/2 sono rimosse insieme.

La fase e' riferita all'inizio del file: phi(t) = frac(f * t), t = (indice + 0.5) * tsamp.
Con --search la frequenza viene affinata per ogni tratto in [f - search_width, f + search_width]
massimizzando la varianza del profilo piegato della somma dei canali.

Sicurezza per gli FRB: in un tratto da 10 s a 50 Hz ogni intervallo di fase e' la media di
~500 periodi; un burst contribuisce al profilo per una frazione ~ (larghezza / P) / 500 della sua
ampiezza, e i campioni piu' brillanti sono comunque esclusi dal clipping.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .sigproc import Filterbank, pack_bits, write_header


@dataclass
class PeriodicConfig:
    freqs: list = field(default_factory=lambda: [50.0])   # frequenze fondamentali [Hz]
    nbins: int = 64               # intervalli di fase per periodo
    segment: float = 10.0         # durata dei tratti su cui si stima il profilo [s]
    clip: float = 4.0             # taglio della media clippata (deviazioni standard)
    search: bool = False          # affina la frequenza per ogni tratto
    search_width: float = 0.05    # semi-ampiezza della ricerca [Hz]
    search_step: float = 0.0005   # passo della ricerca [Hz]
    sub: int = 65536              # campioni per sotto-blocco (memoria)
    seed: int = 0                 # seme per l'arrotondamento stocastico
    dither: bool = True           # arrotondamento stocastico per uscite intere (consigliato)


def _phase_bins(start: int, n: int, ts: float, f: float, nb: int) -> np.ndarray:
    t = (start + np.arange(n) + 0.5) * ts
    return np.minimum((np.mod(t * f, 1.0) * nb).astype(np.int64), nb - 1)


def estimate_template(x, start, ts, f, cfg: PeriodicConfig):
    """Profilo medio clippato (nbins x canali), a media nulla per canale."""
    nb = cfg.nbins
    bins = _phase_bins(start, x.shape[0], ts, f, nb)
    order = np.argsort(bins, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(bins, minlength=nb))])
    T = np.zeros((nb, x.shape[1]), np.float64)
    ntot = 0
    for k in range(nb):
        rows = order[bounds[k]:bounds[k + 1]]
        if rows.size == 0:
            continue
        xb = x[rows]
        m = xb.mean(0, dtype=np.float64)
        sd = xb.std(0, dtype=np.float64)
        sd[sd <= 0] = np.inf
        ok = np.abs(xb - m) <= cfg.clip * sd
        n_ok = ok.sum(0)
        T[k] = np.where(n_ok > 0, (xb * ok).sum(0, dtype=np.float64) / np.maximum(n_ok, 1), m)
        ntot += rows.size
    T = T - T.mean(0, keepdims=True)
    return T.astype(np.float32), bins, ntot / nb


def refine_frequency(x, start, ts, f0, cfg: PeriodicConfig, live):
    """Frequenza che massimizza la varianza del profilo piegato della somma dei canali attivi."""
    s = x[:, live].sum(1, dtype=np.float64) if live.any() else x.sum(1, dtype=np.float64)
    s = s - s.mean()
    t = (start + np.arange(s.size) + 0.5) * ts
    nb = cfg.nbins
    best, fb = -1.0, f0
    for f in np.arange(f0 - cfg.search_width, f0 + cfg.search_width + cfg.search_step / 2, cfg.search_step):
        b = np.minimum((np.mod(t * f, 1.0) * nb).astype(np.int64), nb - 1)
        cnt = np.bincount(b, minlength=nb)
        prof = np.bincount(b, s, nb) / np.maximum(cnt, 1)
        v = prof.var()
        if v > best:
            best, fb = v, float(f)
    return fb


def run_periodic(path, out, cfg: PeriodicConfig | None = None, verbose=True, report=None, tmax=None) -> dict:
    cfg = cfg or PeriodicConfig()
    t_start = time.time()
    fb = Filterbank(path)
    ts = fb.tsamp
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    nseg = max(1, int(round(cfg.segment / ts)))
    if fb.nbits == 32:
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = (-128, 127) if fb.dtype == np.int8 else (0, 2 ** fb.nbits - 1)
    if verbose:
        print(f"{fb.path}: {fb.nchans} canali, tsamp {fb.tsamp * 1e6:.3g} us, {nsamp * ts:.1f} s, {fb.nbits} bit")
        print(f"Frequenze: {cfg.freqs} Hz, {cfg.nbins} intervalli di fase, tratti da {cfg.segment} s"
              + (", ricerca della frequenza" if cfg.search else ""))
    items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fb.items]
    segs = []
    n_sat = 0
    rng = np.random.default_rng(cfg.seed)
    with open(out, "wb") as fo:
        fo.write(write_header(items))
        for s0 in range(0, nsamp, nseg):
            n = min(nseg, nsamp - s0)
            raw = fb.read(s0, n)
            x = raw.astype(np.float32)
            xs = x[::8]                                   # sottocampione per le statistiche di controllo
            sd_ch = xs.std(0)
            live = sd_ch > 0
            info = dict(t0=float(s0 * ts), dur=float(n * ts), comps=[])
            var_before = float(xs.var(0)[live].mean()) if live.any() else 0.0
            for f0 in cfg.freqs:
                f = refine_frequency(x, s0, ts, f0, cfg, live) if cfg.search else float(f0)
                if n * ts * f < 20:          # meno di 20 periodi: profilo non affidabile
                    info["comps"].append(dict(f=f, skipped=True))
                    continue
                T, bins, nper = estimate_template(x, s0, ts, f, cfg)
                T[:, ~live] = 0
                for a in range(0, n, cfg.sub):            # sottrazione a sotto-blocchi (memoria)
                    x[a:a + cfg.sub] -= T[bins[a:a + cfg.sub]]
                rms = np.sqrt((T[:, live] ** 2).mean(0)) / np.where(sd_ch[live] > 0, sd_ch[live], 1)
                info["comps"].append(dict(f=f, samples_per_bin=float(nper),
                                          template_rms_over_sigma_median=float(np.median(rms)) if rms.size else 0.0,
                                          template_rms_over_sigma_max=float(rms.max()) if rms.size else 0.0))
            var_after = float(x[::8].var(0)[live].mean()) if live.any() else 0.0
            info["frac_var_removed_per_channel"] = float(1 - var_after / var_before) if var_before > 0 else 0.0
            segs.append(info)
            if fb.nbits != 32:
                # arrotondamento stocastico: la correzione sottratta e' spesso < 1 conteggio per canale
                # (ma coerente su tutti i canali); un arrotondamento deterministico la cancellerebbe.
                # In media il valore e' conservato; il rumore aggiunto e' <= 1/4 di conteggio^2.
                y = np.floor(x + rng.random(x.shape, dtype=np.float32)) if cfg.dither else np.rint(x)
                n_sat += int(((y < lo) | (y > hi)).sum())
                y = np.clip(y, lo, hi)
                if fb.nbits < 8:
                    fo.write(pack_bits(y.astype(np.uint8).ravel(), fb.nbits).tobytes())
                else:
                    fo.write(y.astype(fb.dtype).tobytes())
            else:
                fo.write(x.astype(np.float32).tobytes())
            if verbose:
                c = info["comps"][0] if info["comps"] else {}
                print(f"  t={s0 * ts:8.1f} s  f={c.get('f', float('nan')):.4f} Hz  "
                      f"profilo/rumore (mediana canali) {c.get('template_rms_over_sigma_median', 0):.3f}  "
                      f"varianza rimossa {info['frac_var_removed_per_channel'] * 100:.2f}%", flush=True)

    fr = [c["f"] for s_ in segs for c in s_["comps"] if not c.get("skipped")]
    vr = [s_["frac_var_removed_per_channel"] for s_ in segs]
    summary = dict(input=str(fb.path), output=str(out), duration_s=nsamp * ts, config=asdict(cfg),
                   freq_min=min(fr) if fr else None, freq_max=max(fr) if fr else None,
                   frac_var_removed_median=float(np.median(vr)) if vr else 0.0,
                   frac_var_removed_max=float(np.max(vr)) if vr else 0.0,
                   samples_saturated=n_sat, segments=segs, runtime_s=time.time() - t_start)
    rpath = Path(report) if report else Path(str(out) + ".periodic.json")
    rpath.write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"Fatto in {summary['runtime_s']:.0f} s. Varianza per canale rimossa: mediana "
              f"{summary['frac_var_removed_median'] * 100:.2f}%, massimo {summary['frac_var_removed_max'] * 100:.2f}%")
        print(f"Riassunto: {rpath}")
    return summary
