"""
Primo stadio di pulizia: maschera dei canali, normalizzazione mobile per canale,
sostituzione delle celle tempo-frequenza con varianza anomala.

Due passaggi sul file, entrambi a blocchi (memoria indipendente dalla dimensione del file):

1. Statistiche. Per ogni canale e finestra di durata `stat_window` si stimano livello e rumore
   con media e deviazione standard "clippate" (campioni entro clip_sigma volte la sigma robusta),
   su un sottocampione. Le serie temporali di livello e rumore vengono poi filtrate con una mediana
   mobile su `norm_span` finestre: la mediana conserva i gradini (salti di guadagno) ma ignora i
   disturbi piu' brevi di circa meta' dello span. Dalle stesse statistiche si ricava la maschera
   automatica dei canali (stessi criteri del survey).

2. Pulizia. Ogni campione e' normalizzato, z = (x - livello(t)) / rumore(t), con livello e rumore
   interpolati linearmente tra i centri delle finestre. I dati vengono divisi in celle di
   `cell` secondi per canale; si sostituiscono con rumore gaussiano le celle con varianza media
   (stimata dalle differenze tra campioni consecutivi, insensibile ai segnali lisci come gli FRB)
   oltre la mediana + `cell_sigma` sigma robuste (calcolate sul blocco). Se in un
   intervallo e' anomala piu' di `row_frac` dei canali, si sostituisce l'intero intervallo.
   I canali mascherati sono posti al valore medio di uscita (costanti, nessun rumore).
   In uscita: y = mean_out + sigma_out * z, quantizzato nel tipo del file (arrotondamento e taglio).
   Per 8 bit: mean_out = 128, sigma_out = 8, cioe' dinamica di +-16 sigma.

Sensibilita' agli FRB: la varianza delle celle e' stimata dalle differenze tra campioni consecutivi,
quindi un burst piu' largo di pochi campioni (per canale) quasi non la modifica; i salti di livello
richiedono che il nuovo livello persista per cp_persist secondi, cosa che un burst non fa. La
conservazione dell'S/N e' verificata nei test con burst da 0.5 a 50 ms e S/N fino a 300.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .sigproc import Filterbank, pack_bits, write_header
from .survey import flag_channels, group_ranges

MAD2SIG = 1.4826
# varianza di una normale troncata a +-3 sigma: std(troncata) = 0.9866 * sigma
TRUNC3 = 0.98658


@dataclass
class CleanConfig:
    stat_window: float = 0.2      # durata delle finestre per livello e rumore [s]
    norm_span: int = 5            # finestre nella mediana mobile (span = norm_span * stat_window)
    stat_step: int = 4            # sottocampionamento fisso (usato solo se stat_nmax = 0)
    stat_nmax: int = 1024         # campioni per finestra e canale usati per le statistiche
    norm_res: float = 0.001       # risoluzione temporale con cui si applicano livello e rumore [s]
    clip_sigma: float = 3.0       # taglio per media/std clippate
    auto_mask: bool = True        # maschera automatica dei canali
    chan_z: float = 5.0           # soglia (z robusto) per la maschera automatica
    mask_block: float = 4.0       # durata dei blocchi per le statistiche della maschera [s]
    mask_stride: int = 4          # un campione ogni mask_stride per le statistiche della maschera
    mask_chans: list = field(default_factory=list)   # canali da mascherare (indici del file)
    cell: float = 0.008           # durata delle celle per il test di varianza [s]
    cell_sigma: float = 6.0       # soglia del test di varianza (sigma robuste)
    row_frac: float = 0.5         # oltre questa frazione di canali anomali si sostituisce tutto l'intervallo
    sample_clip: float | None = None  # opzionale: sostituisce i singoli campioni con |z| > sample_clip
    out_mean: float | None = None     # default: centro della dinamica (interi) o 0 (float)
    out_sigma: float | None = None    # default: 1/32 della dinamica (>= 8 bit), 1/4 (< 8 bit), 1 (float)
    chunk: float = 3.0            # durata dei blocchi del passaggio di pulizia [s]
    protect: bool = True          # veto di dispersione prima di sostituire eventi su molti canali
    protect_frac: float = 0.02    # frazione minima di canali anomali in un intervallo per applicare il veto
    protect_dm_min: float = 15.0  # DM minimo considerato "disperso"
    protect_dm_max: float = 1000.0
    protect_res: float = 0.0005   # risoluzione temporale del veto [s]
    protect_step: float = 4.0     # passo della griglia di DM, in bin di risoluzione di sweep
    protect_ratio: float = 1.2    # protegge se S/N(DM migliore) >= ratio * S/N(DM 0) ...
    protect_snr: float = 7.0      # ... e S/N(DM migliore) >= protect_snr
    changepoints: bool = True     # individua i salti di livello e non mediare a cavallo di essi
    cp_res: float = 0.0005        # risoluzione della ricerca dei salti [s]
    cp_len: float = 0.02          # lunghezza dei due lati del filtro a gradino [s]
    cp_nsub: int = 16             # sottobande usate per la ricerca dei salti
    cp_sigma: float = 6.0         # soglia del filtro a gradino (sigma robuste)
    cp_gap: float = 0.1           # distanza dal salto delle finestre del test di persistenza [s]
    cp_persist: float = 0.2       # durata delle finestre del test di persistenza [s]
    cp_keep: float = 0.5          # frazione del salto che deve persistere
    seed: int = 0


def parse_chan_list(s: str) -> list[int]:
    """'121-127,110,185' -> [110, 121, ..., 127, 185]"""
    out = set()
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def read_flag_file(path) -> list[int]:
    """File nel formato 'min max' per riga (come flagged_channels.txt o il flag file di FRB_HF_Finder)."""
    out = set()
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        v = [int(float(x)) for x in line.split()[:2]]
        out.update(range(v[0], (v[1] if len(v) > 1 else v[0]) + 1))
    return sorted(out)


# --------------------------------------------------------------------------- primo passaggio
def _clipped_stats(x: np.ndarray, clip: float):
    """Media e std clippate per colonna (x: campioni x canali, float32)."""
    med = np.median(x, axis=0)
    mad = MAD2SIG * np.median(np.abs(x - med), axis=0)
    sd0 = x.std(axis=0)
    s = np.where(mad > 0, mad, sd0)
    good = np.abs(x - med) <= clip * np.where(s > 0, s, 1)
    n = good.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, (x * good).sum(0) / n, med)
        v = np.where(n > 1, ((x - m) ** 2 * good).sum(0) / np.maximum(n - 1, 1), 0)
    sd = np.sqrt(v) / (TRUNC3 if clip == 3.0 else 1.0)
    # momenti per la maschera automatica (sul sottocampione, senza clipping)
    xc = x - x.mean(0)
    var = (xc ** 2).mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kurt = np.where(var > 0, (xc ** 4).mean(0) / var ** 2 - 3, 0)
        outf = np.where(s > 0, (np.abs(x - med) > 6 * s).mean(0), 0)
    return m, sd, kurt, outf


def detect_changepoints(fb: Filterbank, cfg: CleanConfig, verbose=True) -> np.ndarray:
    """
    Salti di livello simultanei in piu' canali (salti di guadagno), con risoluzione cp_res.

    Si costruiscono le serie a DM 0 di `cp_nsub` sottobande, decimate a cp_res. Su ognuna si
    applica un filtro a gradino D(t) = media[t, t+L) - media[t-L, t), con L = cp_len, normalizzato
    con la sua sigma robusta. Un salto e' un massimo locale di max_g |D_g| sopra cp_sigma.
    Un salto viene accettato solo se il livello resta diverso anche lontano (test di persistenza):
    un impulso (RFI o FRB), per quanto brillante, non supera il test.
    Restituisce gli indici di campione dei salti (ordinati).
    """
    ts = fb.tsamp
    r = max(1, int(round(cfg.cp_res / ts)))
    L = max(2, int(round(cfg.cp_len / (r * ts))))
    nsub = max(1, min(cfg.cp_nsub, fb.nchans))
    edges = np.linspace(0, fb.nchans, nsub + 1).round().astype(int)
    M = np.zeros((fb.nchans, nsub), np.float32)
    for j in range(nsub):
        M[edges[j]:edges[j + 1], j] = 1
    nper = max(r, (2 ** 26 // max(1, fb.bytes_per_samp)) // r * r)
    series = []
    for s0, raw in fb.blocks(nper):
        S = raw.astype(np.float32) @ M
        m = S.shape[0] // r
        series.append(S[: m * r].reshape(m, r, nsub).mean(1))
    X = np.concatenate(series).astype(np.float64)
    n = X.shape[0]
    if n < 4 * L:
        return np.zeros(0, int)
    cs = np.vstack([np.zeros((1, nsub)), np.cumsum(X, 0)])
    t = np.arange(L, n - L + 1)
    D = (cs[t + L] - cs[t]) / L - (cs[t] - cs[t - L]) / L
    sig = MAD2SIG * np.median(np.abs(D - np.median(D, 0)), 0)
    sig[sig <= 0] = 1
    Dz = D / sig
    Q = np.abs(Dz).max(1)
    idx = np.flatnonzero(Q > cfg.cp_sigma)
    # persistenza: il livello deve restare diverso anche lontano dal salto, tra
    # [t+G, t+G+P) e [t-G-P, t-G), con G = cp_gap (oltre la durata di un burst) e P = cp_persist,
    # e la differenza deve essere almeno meta' di quella vista dal filtro corto (per un burst e' << 1)
    G = max(L, int(round(cfg.cp_gap / (r * ts))))
    P = max(L, int(round(cfg.cp_persist / (r * ts))))
    cps = []
    if idx.size:
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > L) + 1)
        for run in runs:
            k = run[np.argmax(Q[run])]
            tk = t[k]
            a0, a1 = max(0, tk - G - P), max(0, tk - G)
            b0, b1 = min(n, tk + G), min(n, tk + G + P)
            if a1 - a0 < L or b1 - b0 < L:
                continue
            g = int(np.argmax(np.abs(Dz[k])))
            D2 = (cs[b1, g] - cs[b0, g]) / (b1 - b0) - (cs[a1, g] - cs[a0, g]) / (a1 - a0)
            if np.sign(D2) == np.sign(D[k, g]) and abs(D2) >= cfg.cp_keep * abs(D[k, g]):
                cps.append(int(tk * r))
    cps = np.array(cps, int)
    if verbose:
        print(f"  salti di livello trovati: {cps.size}"
              + (f" (primi: {', '.join(f'{c * ts:.3f}' for c in cps[:8])} s)" if cps.size else ""))
    return cps


def compute_stats(fb: Filterbank, cfg: CleanConfig, changepoints=None, verbose=True):
    """
    Statistiche per canale su finestre di durata stat_window che non attraversano i salti di livello.
    Le finestre di ogni segmento (tra due salti) sono filtrate con la mediana mobile solo all'interno
    del segmento, e l'interpolazione nel secondo passaggio non attraversa i salti.
    """
    ts = fb.tsamp
    wlen = max(1, int(round(cfg.stat_window / ts)))
    cps = np.zeros(0, int) if changepoints is None else np.asarray(changepoints, int)
    seg_edges = np.unique(np.concatenate([[0], cps[(cps > 0) & (cps < fb.nsamp)], [fb.nsamp]]))
    # finestre: griglia regolare dentro ogni segmento; l'ultima finestra corta si unisce alla precedente
    wins, segs = [], []
    for si, (a, b) in enumerate(zip(seg_edges[:-1], seg_edges[1:])):
        cuts = list(range(a, b, wlen)) + [b]
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < wlen // 2:
            cuts.pop(-2)
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            wins.append((x0, x1))
            segs.append(si)
    wins = np.array(wins, int)
    segs = np.array(segs, int)
    nwin = len(wins)
    loc = np.zeros((nwin, fb.nchans), np.float32)
    sca = np.zeros((nwin, fb.nchans), np.float32)
    kur = np.zeros((nwin, fb.nchans), np.float32)
    ouf = np.zeros((nwin, fb.nchans), np.float32)
    t0 = time.time()
    blk = dict(std=[], kurt=[], outfrac=[])     # statistiche per blocco di lettura (~4 s), per la maschera
    per_read = max(wlen, 2 ** 26 // max(1, fb.bytes_per_samp))
    w = 0
    while w < nwin:
        r0 = wins[w, 0]
        w1 = w
        while w1 < nwin and wins[w1, 1] - r0 <= per_read:
            w1 += 1
        w1 = max(w1, w + 1)
        raw = fb.read(r0, wins[w1 - 1, 1] - r0)
        nbm = max(64, int(round(cfg.mask_block / ts)))
        for a0 in range(0, raw.shape[0], nbm):
            _block_moments(raw[a0:a0 + nbm:cfg.mask_stride], blk)
        for k in range(w, w1):
            a, b = wins[k] - r0
            step = max(1, (b - a) // cfg.stat_nmax) if cfg.stat_nmax else cfg.stat_step
            x = raw[a:b:step].astype(np.float32)
            if x.shape[0] < 4:
                if k > 0:
                    loc[k], sca[k] = loc[k - 1], sca[k - 1]
                continue
            loc[k], sca[k], kur[k], ouf[k] = _clipped_stats(x, cfg.clip_sigma)
        if verbose:
            print(f"  statistiche: {w1 / nwin * 100:5.1f}%  ({time.time() - t0:.0f} s)", flush=True)
        w = w1
    # mediana mobile dentro ogni segmento
    if cfg.norm_span > 1:
        for si in np.unique(segs):
            sel = np.flatnonzero(segs == si)
            if sel.size > 1:
                span = min(cfg.norm_span, sel.size)
                loc[sel] = median_filter(loc[sel], size=(span, 1), mode="nearest")
                sca[sel] = median_filter(sca[sel], size=(span, 1), mode="nearest")
    centers = wins.mean(1)
    return dict(loc=loc, scale=sca, kurt=kur, outfrac=ouf, centers=centers, wins=wins, segs=segs,
                seg_edges=seg_edges, changepoints=cps,
                blk_std=np.array(blk["std"]), blk_kurt=np.array(blk["kurt"]), blk_outfrac=np.array(blk["outfrac"]))


def _block_moments(raw, blk):
    """Deviazione standard, kurtosis e frazione di outlier su TUTTI i campioni del blocco (come il survey)."""
    n = raw.shape[0]
    if n < 64:
        return
    sub = raw[:: max(1, n // 8192)].astype(np.float32)
    c = sub.mean(0)
    med = np.median(sub, 0)
    mad = MAD2SIG * np.median(np.abs(sub - med), 0)
    s1 = np.zeros(raw.shape[1]); s2 = np.zeros_like(s1); s3 = np.zeros_like(s1); s4 = np.zeros_like(s1)
    nout = np.zeros_like(s1)
    for a in range(0, n, 32768):
        x = raw[a:a + 32768].astype(np.float32)
        y = x - c
        y2 = y * y
        s1 += y.sum(0, dtype=np.float64); s2 += y2.sum(0, dtype=np.float64)
        s3 += (y2 * y).sum(0, dtype=np.float64); s4 += (y2 * y2).sum(0, dtype=np.float64)
        sc = np.where(mad > 0, mad, 1.0)
        nout += (np.abs(x - med) > 6 * sc).sum(0)
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m4 = s4 / n - 4 * mu * s3 / n + 6 * mu * mu * s2 / n - 3 * mu ** 4
        kurt = np.where(var > 0, m4 / var ** 2 - 3, 0)
    blk["std"].append(np.sqrt(var)); blk["kurt"].append(kurt)
    blk["outfrac"].append(np.where(mad > 0, nout / n, 0))


def build_mask(fb: Filterbank, stats: dict, cfg: CleanConfig):
    reasons = {}
    for c in cfg.mask_chans:
        if 0 <= c < fb.nchans:
            reasons.setdefault(int(c), []).append("utente")
    if cfg.auto_mask:
        if stats["blk_std"].size:
            auto = flag_channels(np.median(stats["blk_std"], 0), np.median(stats["blk_kurt"], 0),
                                 np.median(stats["blk_outfrac"], 0), cfg.chan_z)
        else:
            auto = flag_channels(np.median(stats["scale"], 0), np.median(stats["kurt"], 0),
                                 np.median(stats["outfrac"], 0), cfg.chan_z)
        for c, r in auto.items():
            reasons.setdefault(int(c), []).extend(r)
    dead = np.median(stats["scale"], 0) <= 0
    for c in np.flatnonzero(dead):
        reasons.setdefault(int(c), []).append("rumore nullo")
    mask = np.zeros(fb.nchans, bool)
    mask[list(reasons)] = True
    return mask, dict(sorted(reasons.items()))


# --------------------------------------------------------------------------- secondo passaggio
def _interp_rows(arr: np.ndarray, centers: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Interpolazione lineare di arr (finestre x canali) ai campioni dati."""
    if centers.size == 1:
        return np.repeat(arr[:1], samples.size, 0)
    pos = np.interp(samples, centers, np.arange(centers.size))
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, centers.size - 1)
    w = (pos - i0).astype(np.float32)[:, None]
    return arr[i0] * (1 - w) + arr[i1] * w


def _interp_segments(arr, stats, s0, n):
    """Come _interp_rows ma senza attraversare i salti: ogni segmento usa solo le sue finestre."""
    out = np.empty((n, arr.shape[1]), np.float32)
    for a, b, sel in _segment_parts(stats, s0, n):
        out[a - s0:b - s0] = _interp_rows(arr[sel], stats["centers"][sel], np.arange(a, b) + 0.5)
    return out


def _segment_parts(stats, s0, n):
    edges = stats["seg_edges"]
    si0 = max(0, np.searchsorted(edges, s0, side="right") - 1)
    si1 = max(0, np.searchsorted(edges, s0 + n, side="left") - 1)
    for si in range(si0, si1 + 1):
        a = max(s0, edges[si])
        b = min(s0 + n, edges[si + 1])
        if b > a:
            yield a, b, np.flatnonzero(stats["segs"] == si)


def _normalize(raw, stats, loc, scale, s0, K):
    """
    z = (x - livello) / rumore, con livello e rumore interpolati al centro di blocchetti di K campioni
    (costanti dentro il blocchetto) e mai attraverso un salto.
    """
    n, nch = raw.shape
    z = np.empty((n, nch), np.float32)
    for a, b, sel in _segment_parts(stats, s0, n):
        c = stats["centers"][sel]
        m = (b - a) // K
        if m:
            ctr = a + (np.arange(m) + 0.5) * K
            lo = _interp_rows(loc[sel], c, ctr)[:, None, :]
            sc = _interp_rows(scale[sel], c, ctr)[:, None, :]
            x = raw[a - s0:a - s0 + m * K].reshape(m, K, nch).astype(np.float32)
            z[a - s0:a - s0 + m * K] = ((x - lo) / sc).reshape(m * K, nch)
        r0 = a + m * K
        if r0 < b:
            smp = np.arange(r0, b) + 0.5
            z[r0 - s0:b - s0] = (raw[r0 - s0:b - s0].astype(np.float32) - _interp_rows(loc[sel], c, smp)) \
                / _interp_rows(scale[sel], c, smp)
    return z


def output_scaling(fb: Filterbank, cfg: CleanConfig):
    if fb.nbits == 32:
        lo, hi = -np.inf, np.inf
        mean = 0.0 if cfg.out_mean is None else cfg.out_mean
        sig = 1.0 if cfg.out_sigma is None else cfg.out_sigma
    else:
        lo, hi = (-128, 127) if fb.dtype == np.int8 else (0, 2 ** fb.nbits - 1)
        rng = hi - lo + 1
        mean = (lo + hi + 1) / 2 if cfg.out_mean is None else cfg.out_mean
        # 8 bit o piu': rumore = 1/32 della dinamica, cioe' taglio a +-16 sigma (i burst brillanti e
        # stretti superano facilmente 8 sigma per canale); errore di quantizzazione trascurabile
        # (varianza 1/(12*32^2/256^2...) ~ 0.1%). Meno di 8 bit: non c'e' dinamica, 1/4 dell'intervallo.
        sig = (rng / 32 if fb.nbits >= 8 else rng / 4) if cfg.out_sigma is None else cfg.out_sigma
    return mean, sig, lo, hi


class _DispersionVeto:
    """
    Per gli intervalli in cui molti canali hanno celle anomale, confronta l'S/N della serie
    dedispersa a DM 0 con il massimo su DM in [protect_dm_min, protect_dm_max]. Se il segnale e'
    piu' forte a un DM dispersivo, l'intervallo non viene sostituito (possibile FRB).
    La serie usa z con segno, quindi un aumento di varianza a media nulla (rumore RFI) da' S/N
    basso a tutti i DM e viene sostituito comunque.
    """
    KDM = 4.148808e3

    def __init__(self, fb, cfg, live):
        self.cfg = cfg
        self.ts = fb.tsamp
        self.live = np.flatnonzero(live)
        f = fb.freqs[self.live] if self.live.size else fb.freqs
        fmax = fb.freqs.max()
        self.dv = max(1, int(round(cfg.protect_res / self.ts)))
        kb = self.KDM * (fb.freqs.min() ** -2 - fmax ** -2)
        self.sw_max = int(np.ceil(kb * cfg.protect_dm_max / self.ts))
        step = cfg.protect_step * self.dv * self.ts / kb
        # la griglia parte da 0: un segnale e' "disperso" solo se il suo DM migliore e' >= protect_dm_min
        self.dms = np.arange(0.0, cfg.protect_dm_max + step, step)
        self.lo = self.dms < cfg.protect_dm_min
        self.shifts = np.round(self.KDM * self.dms[:, None] * (f[None, :] ** -2 - fmax ** -2)
                               / (self.dv * self.ts)).astype(int)
        self.wmax = 32
        self.n_checked = 0
        self.time_s = 0.0

    def _best(self, series, sig):
        """S/N massimo sui boxcar; restituisce (S/N, posizione, larghezza) in bin decimati."""
        cs = np.concatenate([[0.0], np.cumsum(series)])
        best, pos, wb, W = -np.inf, 0, 1, 1
        while W <= self.wmax and W < series.size:
            b = (cs[W:] - cs[:-W]) / (sig * np.sqrt(W))
            i = int(np.argmax(b))
            if b[i] > best:
                best, pos, wb = float(b[i]), i, W
            W *= 2
        return best, pos, wb

    def check(self, z_e, off, bad, ncell, v=None):
        """Restituisce (celle da salvare [(canale, riga0, riga1)], info) per gli eventi da proteggere."""
        out = []
        nl = self.live.size
        if nl < 4:
            return out
        frac = bad[:, self.live].mean(1)
        rows = np.flatnonzero(frac >= self.cfg.protect_frac)
        if rows.size == 0:
            return out
        gap = max(1, self.sw_max // ncell)
        t_in = time.time()
        for g in np.split(rows, np.flatnonzero(np.diff(rows) > gap) + 1):
            self.n_checked += 1
            a = off + g[0] * ncell
            b = off + (g[-1] + 1) * ncell
            w0 = max(0, a - self.sw_max - ncell)
            w1 = min(z_e.shape[0], b + self.sw_max + ncell)
            seg = z_e[w0:w1][:, self.live]
            nd = seg.shape[0] // self.dv
            if nd < 8:
                continue
            Zd = seg[: nd * self.dv].reshape(nd, self.dv, nl).sum(1)
            # rumore della serie: si usa la varianza misurata nelle righe dell'evento, cosi' un
            # semplice aumento di varianza (glitch) non supera la soglia per effetto del massimo su
            # molti DM; un segnale disperso brillante resta comunque ben sopra
            vrow = 1.0 if v is None else max(1.0, float(np.mean(v[g[0]:g[-1] + 1][:, self.live])))
            sig = np.sqrt(self.dv * nl * vrow)
            # DM 0: solo attorno all'evento
            i0, i1 = max(0, (a - w0) // self.dv - 4), min(nd, (b - w0) // self.dv + 4)
            snr0 = self._best(Zd[i0:i1].sum(1), sig)[0]
            best, bdm, bpos, bw, bsh = -np.inf, 0.0, 0, 1, None
            best_lo = snr0
            cols = np.arange(nl)
            for d, sh in enumerate(self.shifts):
                L = nd - sh.max()
                if L <= 4:
                    continue
                srs = Zd[sh[None, :] + np.arange(L)[:, None], cols].sum(1)
                snr_d, p_, w_ = self._best(srs, sig)
                if self.lo[d]:
                    best_lo = max(best_lo, snr_d)
                elif snr_d > best:
                    best, bdm, bpos, bw, bsh = snr_d, float(self.dms[d]), p_, w_, sh
            if bsh is not None and best >= self.cfg.protect_snr and best >= self.cfg.protect_ratio * max(best_lo, 0):
                # celle da salvare: per ogni canale, quelle attraversate dalla traccia dispersa
                t_ch0 = w0 + (bpos + bsh) * self.dv - off            # inizio, in campioni del blocco
                t_ch1 = t_ch0 + bw * self.dv
                r0 = np.clip(t_ch0 // ncell - 1, 0, bad.shape[0])
                r1 = np.clip(t_ch1 // ncell + 2, 0, bad.shape[0])
                keep = []
                for c, a_, b_ in zip(self.live, r0, r1):
                    if b_ > a_:
                        keep.append((c, int(a_), int(b_)))
                out.append((keep, dict(dm=bdm, snr_dm=float(best), snr_dm0=float(snr0), snr_lowdm=float(best_lo),
                                       var_event=vrow,
                                       t_top_s=float((w0 - off + bpos * self.dv)))))
        self.time_s += time.time() - t_in
        return out


def run_clean(path, out, cfg: CleanConfig | None = None, verbose: bool = True, report: str | None = None,
              tmax: float | None = None) -> dict:
    cfg = cfg or CleanConfig()
    t_start = time.time()
    fb = Filterbank(path)
    if verbose:
        print(f"{fb.path}: {fb.nchans} canali, tsamp {fb.tsamp * 1e6:.3g} us, {fb.duration:.1f} s, {fb.nbits} bit")
        print("Primo passaggio: salti di livello")
    cps = detect_changepoints(fb, cfg, verbose) if cfg.changepoints else None
    if verbose:
        print("Secondo passaggio: livello e rumore per canale")
    stats = compute_stats(fb, cfg, cps, verbose)
    mask, reasons = build_mask(fb, stats, cfg)
    if verbose:
        print(f"  canali mascherati: {int(mask.sum())} "
              f"({', '.join(f'{a}-{b}' if a != b else str(a) for a, b in group_ranges(np.flatnonzero(mask)))})")
    mean_o, sig_o, lo, hi = output_scaling(fb, cfg)

    ts = fb.tsamp
    ncell = max(1, int(round(cfg.cell / ts)))
    K = max(1, int(round(cfg.norm_res / ts)))
    nper = max(ncell, int(round(cfg.chunk / ts)) // ncell * ncell)
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    rng = np.random.default_rng(cfg.seed)
    live = ~mask
    scale = stats["scale"].copy()
    scale[:, ~live] = 1
    scale[scale <= 0] = 1

    n_cells = n_cells_flag = n_rows = n_rows_flag = n_clip = n_sat = 0
    flag_time = []          # frazione di celle sostituite per blocco
    items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fb.items]
    if verbose:
        print("Terzo passaggio: normalizzazione e sostituzione delle celle anomale")
    t0 = time.time()
    # veto di dispersione: margine di lettura e griglia di DM
    veto = _DispersionVeto(fb, cfg, live) if cfg.protect else None
    margin = 0 if veto is None else ((veto.sw_max + ncell) // ncell + 1) * ncell
    protected = []
    with open(out, "wb") as fo:
        fo.write(write_header(items))
        for c0 in range(0, nsamp, nper):
            c1 = min(nsamp, c0 + nper)
            e0, e1 = max(0, c0 - margin), min(nsamp, c1 + margin)
            raw_e = fb.read(e0, e1 - e0)
            z_e = _normalize(raw_e, stats, stats["loc"], scale, e0, K)
            z_e[:, ~live] = 0
            off = c0 - e0
            n = c1 - c0
            z = z_e[off:off + n]                            # vista sul blocco centrale
            s0 = c0
            # celle tempo x canale
            m = n // ncell
            if m > 0:
                zc = z[: m * ncell].reshape(m, ncell, -1)
                # varianza dalle differenze tra campioni consecutivi: insensibile ai segnali lisci
                # (un FRB piu' largo di pochi campioni), sensibile al rumore aggiunto dall'RFI
                dz = np.diff(zc, axis=1)
                v = 0.5 * (dz * dz).mean(1)                 # (m, nchan), ~1 per rumore bianco
                vl = v[:, live]
                med = np.median(vl)
                sv = MAD2SIG * np.median(np.abs(vl - med))
                sv = sv if sv > 0 else (vl.std() or 1.0)
                bad = (v > med + cfg.cell_sigma * sv) & live[None, :]
                # eventi che coinvolgono molti canali: un FRB brillante e' disperso, l'RFI no
                if veto is not None:
                    for keep, info in veto.check(z_e, off, bad, ncell, v):
                        for c, ra, rb in keep:
                            bad[ra:rb, c] = False
                        info["t"] = float((c0 + info.pop("t_top_s")) * ts)
                        protected.append(info)
                rows = bad[:, live].mean(1) > cfg.row_frac
                bad[rows] = live[None, :]
                nb = int(bad.sum())
                if nb:
                    ii, cc = np.nonzero(bad)
                    zc[ii, :, cc] = rng.standard_normal((nb, ncell)).astype(np.float32)
                n_cells += int(live.sum()) * m
                n_cells_flag += nb
                n_rows += m
                n_rows_flag += int(rows.sum())
                flag_time.append((s0 * ts, nb / max(1, int(live.sum()) * m)))
            if cfg.sample_clip:
                big = np.abs(z) > cfg.sample_clip
                big[:, ~live] = False
                k = int(big.sum())
                if k:
                    z[big] = rng.standard_normal(k).astype(np.float32)
                n_clip += k
            y = z * sig_o + mean_o
            if fb.nbits != 32:
                y = np.rint(y)
                n_sat += int(((y < lo) | (y > hi)).sum())
                y = np.clip(y, lo, hi)
                if fb.nbits < 8:
                    fo.write(pack_bits(y.astype(np.uint8).ravel(), fb.nbits).tobytes())
                else:
                    fo.write(y.astype(fb.dtype).tobytes())
            else:
                fo.write(y.astype(np.float32).tobytes())
            if verbose and (c0 // nper) % 50 == 0:
                print(f"  pulizia: {min(1, c1 / nsamp) * 100:5.1f}%  ({time.time() - t0:.0f} s)", flush=True)

    ft = np.array(flag_time) if flag_time else np.zeros((0, 2))
    summary = dict(
        input=str(fb.path), output=str(out), duration_s=nsamp * ts, config=asdict(cfg),
        output_mean=mean_o, output_sigma=sig_o,
        masked_channels={str(k): v for k, v in reasons.items()},
        masked_channel_ranges=[[int(a), int(b)] for a, b in group_ranges(np.flatnonzero(mask).tolist())],
        frac_channels_masked=float(mask.mean()),
        n_changepoints=int(len(stats["changepoints"])),
        changepoints_s=[float(c * ts) for c in stats["changepoints"]],
        frac_cells_replaced=n_cells_flag / max(1, n_cells),
        frac_rows_replaced=n_rows_flag / max(1, n_rows),
        samples_clipped=n_clip, samples_saturated=n_sat,
        frac_saturated=n_sat / max(1, nsamp * fb.nchans),
        protected_events=protected,
        veto_events_checked=0 if veto is None else veto.n_checked,
        veto_time_s=0.0 if veto is None else veto.time_s,
        worst_blocks=[dict(t=float(a), frac_cells=float(b)) for a, b in ft[np.argsort(-ft[:, 1])[:20]]] if ft.size else [],
        runtime_s=time.time() - t_start)
    rpath = Path(report) if report else Path(str(out) + ".clean.json")
    rpath.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    np.savez_compressed(str(rpath)[:-5] + ".npz" if str(rpath).endswith(".json") else str(rpath) + ".npz",
                        loc=stats["loc"], scale=stats["scale"], centers_s=stats["centers"] * ts,
                        changepoints_s=stats["changepoints"] * ts,
                        mask=mask, flag_time=ft)
    if verbose:
        print(f"Fatto in {summary['runtime_s']:.0f} s. Celle sostituite: {summary['frac_cells_replaced'] * 100:.2f}%, "
              f"intervalli interi: {summary['frac_rows_replaced'] * 100:.2f}%, saturati: {summary['frac_saturated'] * 100:.3f}%")
        if protected:
            print(f"Eventi protetti dal veto di dispersione (possibili segnali dispersi): {len(protected)}")
            for p_ in protected[:10]:
                print(f"   t={p_['t']:.3f} s  DM~{p_['dm']:.0f}  S/N disperso {p_['snr_dm']:.1f}  S/N a DM 0 {p_['snr_dm0']:.1f}")
        print(f"Riassunto: {rpath}")
    return summary
