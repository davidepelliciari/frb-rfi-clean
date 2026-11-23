"""
Step 4: sottrazione del rumore di modo comune (segnale comune a tutti i canali).

Modello, per ogni canale attivo c, sui dati normalizzati z_c = (x_c - mu_c(t)) / sigma_c:
    z_c(t) = g_c m(t) + n_c(t)
con m(t) la media di z su tutti i canali attivi (il modo comune) e g_c l'accoppiamento del canale.
L'output e' x_c - sigma_c g_c m(t). Il livello mu_c(t) e' la MEDIA del canale nel blocco (`block`
secondi), interpolata linearmente tra i centri dei blocchi vicini: con la media il modo comune ha media
nulla, quindi la sottrazione non introduce offset (con la mediana, per dati interi o con RFI positivo,
media e mediana differiscono e la differenza verrebbe sottratta come uno scalino diverso per blocco e
canale); con l'interpolazione il livello e' continuo, quindi non ci sono scalini tra i blocchi. Le
variazioni del modo comune piu' lente di qualche blocco restano nel livello mu_c(t) e quindi nei dati
(Heimdall sottrae la linea di base); quelle piu' rapide vengono sottratte.

  * mode="regress" (default): g_c stimato per regressione di z_c su m nel blocco, a risoluzione
    `fit_res` (dove il modo comune domina sul rumore bianco), con m calcolato escludendo il canale
    stesso, per non avere un bias verso 1 quando il modo comune e' debole;
  * mode="zerodm": g_c = 1 per tutti i canali (filtro a DM zero di Eatough et al. 2009).

Va applicato dopo gli step 1-3: richiede canali normalizzati e mascherati (step 1), e azzera la serie a
banda intera a DM 0 su cui lavorano gli step 2 e 3. Un FRB perde circa la frazione larghezza / sweep del
proprio segnale (in ogni istante occupa solo quella frazione dei canali, che entra nella media); per i
burst stretti ad alto DM la perdita e' trascurabile, per quelli larghi a basso DM no: va misurata con le
iniezioni.

I canali costanti (mascherati) restano identici. Con dati interi l'arrotondamento e' stocastico
(dithering), come nello step 2.

Diagnostica (OUT.commonmode.json, OUT.commonmode.npz): per blocco, il rumore della serie a banda intera
a DM 0 (media su `fit_res`) diviso quello atteso per canali indipendenti, prima e dopo (dopo e' ~0 per
costruzione); lo stesso per la differenza tra meta' alta e meta' bassa della banda, che la sottrazione non
azzera e misura la correlazione residua (1 = nessuna); i guadagni g_c.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .sigproc import Filterbank, pack_bits, write_header

MAD2SIG = 1.4826


@dataclass
class CommonModeConfig:
    block: float = 4.0            # durata dei blocchi [s]
    mode: str = "regress"         # "regress" o "zerodm"
    fit_res: float = 1e-3         # risoluzione per la stima dei guadagni e per la diagnostica [s]
    gain_min: float = -1.0
    gain_max: float = 3.0
    dither: bool = True
    seed: int = 0


def _robust(x, step):
    xs = x[::step]
    med = np.median(xs, 0)
    sig = MAD2SIG * np.median(np.abs(xs - med), 0)
    return med, sig


def _excess(m, N, r):
    """Rumore di m (media su N canali a rumore unitario) mediato su r campioni / atteso per canali indipendenti."""
    k = m.size // r
    if k < 16:
        return np.nan
    mm = m[: k * r].reshape(k, r).mean(1)
    mm = mm - np.median(mm)
    s = MAD2SIG * np.median(np.abs(mm))
    return float(s / (1.0 / np.sqrt(N * r)))


def _split_excess(z, r):
    """Come _excess, per la differenza tra la media della meta' alta e quella della meta' bassa dei canali:
    misura la parte correlata che un solo modo comune non descrive (1 = nessuna)."""
    h = z.shape[1] // 2
    if h < 1:
        return np.nan
    d = z[:, :h].mean(1) - z[:, h:].mean(1)
    # varianza attesa della differenza per canali indipendenti a rumore unitario: 1/h + 1/(N-h)
    k = d.size // r
    if k < 16:
        return np.nan
    dd = d[: k * r].reshape(k, r).mean(1)
    dd = dd - np.median(dd)
    s = MAD2SIG * np.median(np.abs(dd))
    return float(s / np.sqrt((1.0 / h + 1.0 / (z.shape[1] - h)) / r))


def run_commonmode(path, out, cfg: CommonModeConfig | None = None, verbose=True, report=None, tmax=None) -> dict:
    cfg = cfg or CommonModeConfig()
    if cfg.mode not in ("regress", "zerodm"):
        raise ValueError(f"mode sconosciuto: {cfg.mode}")
    t_start = time.time()
    fb = Filterbank(path)
    ts, nch = fb.tsamp, fb.nchans
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    is_int = fb.nbits != 32
    lo, hi = ((-128, 127) if fb.dtype == np.int8 else (0, 2 ** fb.nbits - 1)) if is_int else (-np.inf, np.inf)
    nper = max(1024, int(round(cfg.block / ts)))
    r = max(1, int(round(cfg.fit_res / ts)))
    probe = fb.read(0, min(nsamp, nper)).astype(np.float32)
    sd = probe.std(0)
    live = sd > 0.05 * np.median(sd[sd > 0]) if np.any(sd > 0) else sd > 0
    N = int(live.sum())
    if N < 2:
        raise ValueError("servono almeno due canali attivi")
    rng = np.random.default_rng(cfg.seed)
    if verbose:
        print(f"{fb.path}: {nch} canali ({N} attivi), tsamp {ts * 1e6:.3g} us, {nsamp * ts:.1f} s; modo comune: {cfg.mode}")
    items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fb.items]
    gains, ex_before, ex_after, sp_before, sp_after, tblk = [], [], [], [], [], []
    last = 0.0
    starts = list(range(0, nsamp, nper))

    def stats(c0):
        n = min(nper, nsamp - c0)
        raw = fb.read(c0, n)
        x = raw.astype(np.float32)
        mu = x.mean(0, dtype=np.float64)
        _, sig = _robust(x, max(1, n // 8192))
        return dict(c0=c0, n=n, raw=raw, mu=mu, sig=sig, tc=c0 + n / 2.0)

    def write(res):
        if fb.nbits < 8:
            fo.write(pack_bits(res.astype(np.uint8).ravel(), fb.nbits).tobytes())
        else:
            fo.write(res.tobytes())

    with open(out, "wb") as fo:
        fo.write(write_header(items))
        prev, cur = None, stats(starts[0]) if starts else None
        for ib in range(len(starts)):
            nxt = stats(starts[ib + 1]) if ib + 1 < len(starts) else None
            c0, n, raw, sig = cur["c0"], cur["n"], cur["raw"], cur["sig"]
            ok = live & (sig > 0)
            Nb = int(ok.sum())
            if Nb < 2 or n < 4 * r:
                write(raw)
                prev, cur = cur, nxt
                continue
            # livello di ogni canale: media del blocco, interpolata linearmente tra i centri dei blocchi
            # vicini (continua da un blocco all'altro). Con la media, e non la mediana, il modo comune ha
            # media nulla e la sottrazione non introduce offset; con l'interpolazione non introduce scalini.
            tt = c0 + np.arange(n) + 0.5
            mu = np.empty((n, Nb), np.float32)
            mc = cur["mu"][ok]
            left, right = tt < cur["tc"], tt >= cur["tc"]
            for side, nb in ((left, prev), (right, nxt)):
                if not side.any():
                    continue
                if nb is None:
                    mu[side] = mc
                else:
                    f = ((tt[side] - cur["tc"]) / (nb["tc"] - cur["tc"])).astype(np.float32)[:, None]
                    mu[side] = mc + (nb["mu"][ok] - mc) * f
            x = raw[:, ok].astype(np.float32)
            s_ = sig[ok].astype(np.float32)
            z = (x - mu) / s_
            del mu
            m = z.mean(1)
            if cfg.mode == "zerodm":
                g = np.ones(Nb, np.float32)
            else:
                k = n // r
                Z1 = z[: k * r].reshape(k, r, Nb).mean(1)
                M1 = Z1.mean(1)
                Z1 = Z1 - Z1.mean(0)
                M1 = M1 - M1.mean()
                # modo comune senza il canale stesso: (N M1 - Z1_c) / (N - 1)
                Mloo = (Nb * M1[:, None] - Z1) / (Nb - 1)
                g = (Z1 * Mloo).sum(0) / np.maximum((Mloo * Mloo).sum(0), 1e-12)
                g = np.clip(g, cfg.gain_min, cfg.gain_max).astype(np.float32)
            y = z - m[:, None] * g[None, :]
            ex_before.append(_excess(m, Nb, r))
            ex_after.append(_excess(y.mean(1), Nb, r))
            sp_before.append(_split_excess(z, r))
            sp_after.append(_split_excess(y, r))
            del z, y
            gfull = np.full(nch, np.nan, np.float32)
            gfull[ok] = g
            gains.append(gfull)
            tblk.append(c0 * ts)
            outv = x - (s_ * g)[None, :] * m[:, None]
            if is_int:
                outv = np.floor(outv + rng.random(outv.shape, dtype=np.float32)) if cfg.dither else np.rint(outv)
                outv = np.clip(outv, lo, hi)
            res = raw.copy()
            res[:, ok] = outv.astype(raw.dtype)
            write(res)
            if verbose and (time.time() - last > 30 or c0 + n >= nsamp):
                last = time.time()
                print(f"  t={(c0 + n) * ts:8.1f} s ({(c0 + n) / nsamp * 100:5.1f}%): eccesso di rumore a DM 0 "
                      f"(scala {cfg.fit_res * 1e3:g} ms) {ex_before[-1]:.2f} -> {ex_after[-1]:.2f}  "
                      f"({time.time() - t_start:.0f} s)", flush=True)
            prev, cur = cur, nxt
    G = np.array(gains) if gains else np.zeros((0, nch))
    eb, ea = np.array(ex_before), np.array(ex_after)
    sb, sa = np.array(sp_before), np.array(sp_after)
    summary = dict(input=str(path), output=str(out), duration_s=nsamp * ts, config=asdict(cfg),
                   n_live_channels=N, n_blocks=len(gains),
                   zerodm_excess_before_median=float(np.nanmedian(eb)) if eb.size else None,
                   zerodm_excess_after_median=float(np.nanmedian(ea)) if ea.size else None,
                   split_excess_before_median=float(np.nanmedian(sb)) if sb.size else None,
                   split_excess_after_median=float(np.nanmedian(sa)) if sa.size else None,
                   gain_median=float(np.nanmedian(G)) if G.size else None,
                   gain_p10_p90=[float(np.nanpercentile(G, 10)), float(np.nanpercentile(G, 90))] if G.size else None,
                   runtime_s=time.time() - t_start,
                   note="zerodm_excess: rumore della serie a banda intera a DM 0 (media su fit_res) / atteso per "
                        "canali indipendenti (1 = nessuna correlazione; dopo la sottrazione e' ~0 per costruzione). "
                        "split_excess: lo stesso per la differenza meta' alta - meta' bassa della banda, che la "
                        "sottrazione non azzera: misura la correlazione residua non descritta da un solo modo comune")
    rpath = Path(report) if report else Path(str(out) + ".commonmode.json")
    rpath.write_text(json.dumps(summary, indent=2))
    base = str(rpath)[:-5] if str(rpath).endswith(".json") else str(rpath)
    np.savez_compressed(base + ".npz", t_block=np.array(tblk), gains=G.astype(np.float32),
                        excess_before=eb.astype(np.float32), excess_after=ea.astype(np.float32),
                        split_before=sb.astype(np.float32), split_after=sa.astype(np.float32), freqs=fb.freqs)
    if verbose:
        print(f"Fatto in {summary['runtime_s']:.0f} s. Eccesso di rumore a DM 0 (mediana sui blocchi): "
              f"{summary['zerodm_excess_before_median']:.2f} -> {summary['zerodm_excess_after_median']:.2f}; "
              f"differenza alta-bassa {summary['split_excess_before_median']:.2f} -> "
              f"{summary['split_excess_after_median']:.2f}; "
              f"guadagni mediana {summary['gain_median']:.2f} (p10-p90 {summary['gain_p10_p90'][0]:.2f}-"
              f"{summary['gain_p10_p90'][1]:.2f})")
        print(f"Riassunto: {rpath}")
    return summary
