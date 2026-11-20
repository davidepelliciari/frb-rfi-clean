"""
Step 3: rimozione degli impulsi a DM 0 (radar e altri impulsi a banda intera o parziale), con veto
di dispersione.

Un solo passaggio a blocchi (con margini di lettura per boxcar e veto):

1. Normalizzazione per canale (mediana e sigma robuste del blocco); i canali costanti (mascherati)
   sono esclusi. Serie a DM 0: banda intera e `nsub` sottobande, ciascuna con rumore unitario,
   baseline mobile sottratta.
2. Rivelazione con soglia iterativa: per ogni larghezza di boxcar W = 2^k campioni (fino a
   `max_width`) e ogni serie, la sigma robusta viene stimata, si segnalano i tratti oltre la prima
   soglia, si ristima la sigma escludendoli, e cosi' via per le soglie successive (`thresholds`,
   default 8 e 6). I tratti segnalati a larghezze diverse e nella stessa serie vengono uniti;
   eventi contemporanei in serie diverse formano un solo evento, con l'insieme delle sottobande
   colpite (Z > `sub_hit` al picco).
3. Veto di dispersione:
   * scorciatoia: un FRB a DM >= dm_min non puo' apparire a DM 0 piu' stretto della propria
     dispersione dentro la serie in cui e' visto (sottobanda o banda intera); gli eventi con
     larghezza < `narrow_frac` x quella dispersione sono RFI senza altri calcoli;
   * altrimenti si dedispergono le serie delle sottobande (boxcar alla larghezza dell'evento) su una
     griglia di DM da 0 a `dm_max`, cercando una traccia dispersa che passi per l'evento; l'evento
     viene PROTETTO se il miglior S/N a DM >= dm_min supera `protect_ratio` volte il miglior S/N a
     DM < dm_min e la soglia `protect_snr`.
   Dentro un gruppo di rilevamenti sovrapposti (es. un treno di impulsi radar, o le coppie a 38 ms)
   ogni impulso disgiunto e' un evento a se': si prendono i boxcar in ordine di S/N decrescente,
   scartando quelli che si sovrappongono a uno gia' preso. Ognuno passa dal veto; un evento che cade
   sulla traccia dispersa di un evento gia' protetto dello stesso gruppo e' protetto anch'esso
   (un FRB visto a DM 0 in piu' sottobande produce piu' eventi lungo la stessa traccia).
4. Sostituzione: per gli eventi non protetti, il rettangolo tempo (con margine) x canali delle
   sottobande colpite (tutti i canali se l'evento e' a banda intera o colpisce almeno meta' delle
   sottobande) viene sostituito con rumore gaussiano al livello del blocco, oppure al livello locale
   (media delle righe senza eventi ai due lati, interpolata) se questo differisce significativamente
   da quello del blocco, come per un impulso su un fondo rialzato;
   la sigma e' quella robusta del canale nel blocco.
   Il rettangolo viene esteso finche' le righe adiacenti hanno ancora segnale nelle sottobande
   sostituite (i bordi di un impulso possono essere sfasati tra sottobande). Cosi' un impulso su un fondo rialzato (emissione
   lunga, variazioni lente) non lascia un "buco" con bordi netti, che dedisperso creerebbe nuovi
   candidati. Tutti gli altri campioni restano identici all'input.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .sigproc import Filterbank, pack_bits, write_header

KDM = 4.148808e3
MAD2SIG = 1.4826


@dataclass
class ImpulseConfig:
    nsub: int = 16
    max_width: float = 0.032          # larghezza massima dei boxcar [s]
    thresholds: list = field(default_factory=lambda: [8.0, 6.0])   # soglie iterative (S/N)
    sub_hit: float = 3.0              # S/N per dire che una sottobanda e' colpita (sostituzione)
    veto_hit: float = 2.0             # S/N oltre cui una sottobanda e' esclusa dal veto (colpita anche in parte)
    block: float = 4.0                # durata dei blocchi [s]
    margin_frac: float = 0.25         # margine di sostituzione (frazione della larghezza, min 2 campioni)
    full_frac: float = 0.5            # oltre questa frazione di sottobande colpite si sostituisce tutta la banda
    veto: bool = True
    dm_min: float = 15.0
    dm_max: float = 1000.0
    protect_ratio: float = 1.05
    protect_snr: float = 7.0
    narrow_frac: float = 0.5          # scorciatoia: larghezza < narrow_frac x dispersione a dm_min -> RFI
    track_z: float = 2.0              # S/N per sottobanda lungo la traccia per contare come "segnale"
    track_frac: float = 0.4           # frazione minima di sottobande con segnale lungo la traccia protetta
    grow: bool = True                 # estende il rettangolo finche' il segnale nelle sottobande sostituite e' sopra soglia
    grow_snr: float = 3.5             # soglia per campione sulla serie combinata delle sottobande sostituite
    grow_sub: float = 5.0             # ... o su una singola sottobanda sostituita
    local_level: bool = True          # livello di sostituzione dai lati del rettangolo (False: mediana del blocco)
    flank: float = 0.004              # lunghezza minima di ciascun lato per il livello locale [s]
    flank_w: float = 2.0              # ... e in unita' della larghezza dell'evento
    level_z: float = 5.0              # S/N oltre cui i lati differiscono dal livello del blocco -> livello locale
    seed: int = 0


def _robust_sigma(x, step=1):
    xs = x[::step]
    med = np.median(xs, 0)
    s = MAD2SIG * np.median(np.abs(xs - med), 0)
    return med, np.where(s > 0, s, 1.0)


class _Geometry:
    def __init__(self, fb, cfg, live):
        self.fb, self.cfg = fb, cfg
        nch = fb.nchans
        self.nsub = max(1, min(cfg.nsub, nch))
        self.edges = np.linspace(0, nch, self.nsub + 1).round().astype(int)
        self.sub_of = np.repeat(np.arange(self.nsub), np.diff(self.edges))
        self.nlive = np.array([live[a:b].sum() for a, b in zip(self.edges[:-1], self.edges[1:])])
        self.M = np.zeros((nch, self.nsub), np.float32)
        for g in range(self.nsub):
            a, b = self.edges[g], self.edges[g + 1]
            if self.nlive[g]:
                self.M[a:b, g] = live[a:b] / np.sqrt(self.nlive[g])
        self.fsub = np.array([fb.freqs[a:b].mean() for a, b in zip(self.edges[:-1], self.edges[1:])])
        bw_sub = np.array([abs(fb.freqs[a] - fb.freqs[b - 1]) + abs(fb.foff) for a, b in zip(self.edges[:-1], self.edges[1:])])
        fmax = fb.freqs.max()
        ts = fb.tsamp
        # ritardo di ogni sottobanda rispetto alla piu' alta, per DM unitario [campioni]
        self.delay1 = KDM * (self.fsub ** -2 - self.fsub.max() ** -2) / ts
        # dispersione dentro una sottobanda e nella banda intera a dm_min [campioni]
        fl = self.fsub - bw_sub / 2
        fh = self.fsub + bw_sub / 2
        self.smear_sub_min = float(np.min(KDM * cfg.dm_min * (fl ** -2 - fh ** -2) / ts))
        self.smear_full_min = float(KDM * cfg.dm_min * (fb.f_lo ** -2 - fb.f_hi ** -2) / ts)
        self.sweep_max = int(np.ceil(KDM * cfg.dm_max * (fb.f_lo ** -2 - fmax ** -2) / ts))
        self.kmax = max(0, int(np.ceil(np.log2(max(cfg.max_width / ts, 1)) - 1e-9)))


def _series(z, geo):
    """(n, 1+nsub): banda intera e sottobande, rumore unitario; baseline mobile sottratta."""
    S = z @ geo.M
    live_sub = geo.nlive > 0
    full = S[:, live_sub].sum(1) / np.sqrt(max(live_sub.sum(), 1))
    X = np.column_stack([full, S]).astype(np.float64)
    Lb = max(64, 8 << geo.kmax)
    n = X.shape[0]
    nb = n // Lb
    if nb >= 2:
        meds = np.median(X[: nb * Lb].reshape(nb, Lb, -1)[:, :: max(1, Lb // 1024)], axis=1)
        c = (np.arange(nb) + 0.5) * Lb
        X -= np.column_stack([np.interp(np.arange(n), c, meds[:, j]) for j in range(X.shape[1])])
    else:
        X -= np.median(X, 0)
    return X


def _detect(X, geo, cfg):
    """Restituisce dict W -> (Z, flag): Z (n-W+1, 1+nsub) normalizzato, flag dei tratti oltre l'ultima soglia.

    Soglia iterativa: la sigma robusta di ogni serie viene ristimata escludendo i tratti gia' oltre
    la soglia precedente (e i W campioni vicini), su un sottocampione.
    """
    n = X.shape[0]
    cs = np.vstack([np.zeros((1, X.shape[1]), np.float64), np.cumsum(X, 0)])
    out = {}
    step = max(1, n // 20000)
    for k in range(geo.kmax + 1):
        W = 1 << k
        if W >= n // 4:
            break
        B = (cs[W:] - cs[:-W]).astype(np.float32)
        Bs = B[::step]
        keep = np.ones(Bs.shape, bool)
        med = np.zeros(B.shape[1], np.float32)
        sig = np.ones(B.shape[1], np.float32)
        for thr in cfg.thresholds:
            for j in range(B.shape[1]):
                v = Bs[keep[:, j], j]
                if v.size < 16:
                    v = Bs[:, j]
                med[j] = np.median(v)
                s_ = MAD2SIG * np.median(np.abs(v - med[j]))
                sig[j] = s_ if s_ > 0 else np.inf
            hit = (Bs - med) > thr * sig
            if hit.any():
                # esclude i punti vicini (il boxcar e' largo W): dilatazione sul sottocampione
                r = max(1, int(np.ceil(W / step)))
                hd = hit.copy()
                for sh in range(1, r + 1):
                    hd[sh:] |= hit[:-sh]
                    hd[:-sh] |= hit[sh:]
                keep &= ~hd
        Z = (B - med) / sig
        out[W] = (Z, Z > cfg.thresholds[-1])
    return out


def _events(det, geo, cfg, core0, core1):
    """Eventi (inizio, fine, larghezza, serie migliore, S/N, sottobande colpite) con inizio nel blocco centrale."""
    recs = []
    for W, (Z, flag) in det.items():
        any_ = flag.any(1)
        idx = np.flatnonzero(any_)
        if idx.size == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        zmax = np.where(flag, Z, -np.inf).max(1)
        for r in runs:
            p = r[np.argmax(zmax[r])]
            col = int(np.argmax(np.where(flag[p], Z[p], -np.inf)))
            recs.append((int(r[0]), int(r[-1]) + W, W, col, float(Z[p, col]), Z[p, 1:] > cfg.sub_hit, p,
                         Z[p, 1:] > cfg.veto_hit))
    if not recs:
        return []
    recs.sort(key=lambda x: x[0])
    groups, cur = [], [recs[0]]
    end = recs[0][1]
    for r in recs[1:]:
        if r[0] <= end:
            cur.append(r)
            end = max(end, r[1])
        else:
            groups.append(cur)
            cur, end = [r], r[1]
    groups.append(cur)
    ev = []
    for gi, g in enumerate(groups):
        # impulsi disgiunti del gruppo: boxcar in ordine di S/N, scartando quelli sovrapposti a uno gia' preso
        taken = []
        for x in sorted(g, key=lambda x: -x[4]):
            s0, s1 = x[6], x[6] + x[2]
            if any(s0 < b and a < s1 for a, b in taken):
                continue
            taken.append((s0, s1))
            if not (core0 <= s0 < core1):
                continue
            hits = np.zeros(geo.nsub, bool)
            vhits = np.zeros(geo.nsub, bool)
            for y in g:           # sottobande colpite: dai record il cui boxcar di picco si sovrappone
                if y[6] < s1 and s0 < y[6] + y[2]:
                    hits |= y[5]
                    vhits |= y[7]
                    if y[3] > 0:
                        hits[y[3] - 1] = True
                        vhits[y[3] - 1] = True
            ev.append(dict(a=s0, b=s1, W=x[2], col=x[3], snr=x[4], hits=hits, vhits=vhits, peak=s0, group=gi))
    return ev


def _on_track(ev, tracks, geo, cfg):
    """True se l'evento cade sulla traccia dispersa (t0, dm, w) di un evento gia' protetto dello stesso gruppo."""
    p, W = ev["peak"], ev["W"]
    smear1 = geo.smear_sub_min / cfg.dm_min
    broadband = ev["col"] == 0 or ev["hits"].mean() >= cfg.full_frac
    for t0, dm, w in tracks:
        if broadband:
            if t0 - W - w <= p <= t0 + dm * geo.delay1.max() + w:
                return True
        else:
            g = ev["col"] - 1 if ev["col"] > 0 else int(np.argmax(ev["hits"]))
            if abs(p - (t0 + dm * geo.delay1[g])) <= W + w + dm * smear1:
                return True
    return False


def _veto(ev, det, geo, cfg, excl=None):
    """
    True se l'evento va protetto: esiste una traccia dispersa con DM >= dm_min che passa per l'evento
    e da' un S/N maggiore (di protect_ratio) di quello delle tracce con DM < dm_min.

    La statistica e' la somma delle serie di boxcar (normalizzate) delle sottobande lungo la traccia.
    Si provano tutte le larghezze w = 2^k fino al doppio di quella dell'evento: a DM 0 un burst
    disperso e' spalmato sullo sweep, ma alla giusta DM e' piu' stretto. Per una larghezza w si
    provano solo i DM per cui la dispersione dentro una sottobanda non supera ~4w (oltre, una
    larghezza maggiore e' comunque migliore): cosi' il costo resta piccolo.
    Per un evento a banda parziale si usano solo le sottobande NON colpite: un FRB lascia segnale
    lungo la traccia anche li', un impulso RFI stretto in frequenza no.
    `excl`: intervalli (inizi, fini) degli altri impulsi a banda larga dello stesso gruppo, azzerati
    nella somma: in un treno di impulsi radar una traccia dispersa che attraversa impulsi diversi in
    sottobande diverse imita un FRB (un FRB, invece, a DM 0 appare come un solo evento a banda larga
    o come pezzi a banda parziale, che non vengono azzerati).
    """
    W = ev["W"]
    series_smear = geo.smear_full_min if ev["col"] == 0 else geo.smear_sub_min
    if W < cfg.narrow_frac * series_smear:
        return False, dict(reason="stretto")
    live = geo.nlive > 0
    gdet = ev["col"] - 1
    hits = ev["vhits"].copy()
    if gdet >= 0:
        hits[gdet] = True
    broadband = ev["col"] == 0 or ev["hits"].mean() >= cfg.full_frac
    use = live & (~hits if not broadband else np.ones_like(live))
    cols_all = np.flatnonzero(live)
    cols = np.flatnonzero(use)
    if cols.size < 3:
        return False, dict(reason="poche sottobande libere")
    jdet = int(np.flatnonzero(cols_all == gdet)[0]) if (gdet >= 0 and not broadband) else None
    p = ev["peak"]
    smear1 = max(geo.smear_sub_min / cfg.dm_min, 1e-6)      # dispersione in una sottobanda per DM unitario [campioni]
    # larghezze utili: da W/16 a 2W (anche se il burst vero fosse piu' stretto, a W/16 il guadagno rispetto
    # a DM 0 e' gia' ampiamente sufficiente per la decisione)
    widths = sorted(w for w in det if max(1, W // 16) <= w <= 2 * W)
    # DM plausibili: un evento largo W a DM 0 puo' venire da un burst con dispersione (nella serie in cui e'
    # visto) al piu' ~2.5 W; oltre, il burst sarebbe stato visto piu' largo
    # la scala dipende dalla serie in cui l'evento e' visto meglio: banda intera (sweep) o sottobanda
    per_dm = geo.delay1.max() if ev["col"] == 0 else smear1
    dm_event = min(cfg.dm_max, cfg.dm_min + 2.5 * W / max(per_dm, 1e-9))
    best_lo, best_hi, bdm, bw, bt0 = -np.inf, -np.inf, 0.0, 0, p
    track_frac = 0.0
    sq = np.sqrt(cols.size)
    for w in widths:
        Z = det[w][0]
        n = Z.shape[0]
        sub = Z[:, 1:]
        stride = max(1, w // 4)
        dm_cap = dm_event if w == widths[-1] else min(dm_event, 4.0 * w / smear1)
        dstep = stride / max(geo.delay1.max(), 1e-9)
        dms = np.arange(0.0, dm_cap + dstep, dstep)
        # tutte le DM e tutti gli istanti di partenza in un'unica operazione vettoriale
        dall = np.round(dms[:, None] * geo.delay1[cols_all][None, :]).astype(np.int64)    # (nD, nall)
        d = np.round(dms[:, None] * geo.delay1[cols][None, :]).astype(np.int64)           # (nD, ncols)
        dmax = dall.max(1)
        if jdet is not None:
            lo_t, hi_t = p - dall[:, jdet] - w, p - dall[:, jdet] + W
        else:
            lo_t, hi_t = p - dmax - w, p + W
        lo_t = np.maximum(0, lo_t)
        hi_t = np.minimum(n - 1 - dmax, hi_t)
        ok_dm = hi_t >= lo_t
        if not ok_dm.any():
            continue
        dms, d, lo_t, hi_t = dms[ok_dm], d[ok_dm], lo_t[ok_dm], hi_t[ok_dm]
        nT = int(((hi_t - lo_t) // stride).max()) + 1
        k = np.arange(nT)
        t0 = lo_t[:, None] + stride * k[None, :]                                           # (nD, nT)
        valid = t0 <= hi_t[:, None]
        t0 = np.where(valid, t0, lo_t[:, None])
        idx = t0[:, :, None] + d[:, None, :]                                               # (nD, nT, ncols)
        M = sub[idx, cols[None, None, :]]
        if excl is not None:
            st, en = excl[0] - w + 1, excl[1]
            j = np.searchsorted(st, idx, side="right") - 1
            inside = (j >= 0) & (idx < en[np.maximum(j, 0)])
            M = np.where(inside, 0.0, M)
        tot = np.where(valid, M.sum(2), -np.inf) / sq                                      # (nD, nT)
        per_dm = tot.max(1)
        lo = dms < cfg.dm_min
        if lo.any():
            best_lo = max(best_lo, float(per_dm[lo].max()))
        if (~lo).any():
            j = int(np.flatnonzero(~lo)[np.argmax(per_dm[~lo])])
            if per_dm[j] > best_hi:
                best_hi, bdm, bw = float(per_dm[j]), float(dms[j]), w
                i = int(np.argmax(tot[j]))
                bt0 = int(t0[j, i])
                track_frac = float(np.mean(M[j, i] > cfg.track_z))
    # coerenza: lungo la traccia migliore una buona frazione delle sottobande deve mostrare segnale
    # (un FRB e' esteso in frequenza; una traccia che raccoglie poche sottobande molto forti e' di
    # solito un'altra RFI attraversata per caso)
    protect = (best_hi >= cfg.protect_snr and best_hi >= cfg.protect_ratio * max(best_lo, 0)
               and track_frac >= cfg.track_frac)
    return protect, dict(snr_lowdm=best_lo, snr_dm=best_hi, dm=bdm, width_dm=float(bw), broadband=float(broadband),
                         track_frac=track_frac, t0=bt0)


def _grow(X, subs, A, B, maxext, cfg):
    """Estende [A, B) a sinistra e a destra finche' le righe adiacenti mostrano segnale nelle sottobande
    `subs` (serie a rumore unitario per campione): i bordi di un impulso possono essere sfasati tra
    sottobande, e un bordo rimasto fuori dal rettangolo diventa un nuovo impulso di 1-2 campioni."""
    n = X.shape[0]
    sub = X[:, 1 + subs]
    k = np.sqrt(subs.size)

    def hot(i):
        r = sub[i]
        return r.sum() / k > cfg.grow_snr or r.max() > cfg.grow_sub

    A, B = max(0, A), min(n, B)
    a0, b0 = A, B
    while A > 0 and a0 - A < maxext and hot(A - 1):
        A -= 1
    while B < n and B - b0 < maxext and hot(B):
        B += 1
    # un campione di margine oltre l'ultimo campione caldo
    return max(0, A - 1 if A < a0 else A), min(n, B + 1 if B > b0 else B)


def _local_level(xf, chans, A, B, L, med, sig, hot, zthr):
    """Livello per canale nelle righe [A, B) di xf.

    Di default il livello del blocco (mediana robusta del canale): dopo lo step 1 i dati sono gia'
    normalizzati e questo livello e' privo di bias. Un errore di livello anche piccolo e' pericoloso,
    perche' si somma coerentemente su tutti i canali e campioni del rettangolo (0.1 sigma per canale su
    250 canali e 32 campioni danno S/N ~ 13).
    Si passa al livello locale solo se le righe ai lati del rettangolo (entro 2L righe, escluse quelle
    coperte da eventi: maschera `hot`) differiscono in modo significativo dal livello del blocco
    (S/N > zthr sulla somma dei canali): e' il caso di un impulso su un fondo rialzato, dove il livello
    del blocco lascerebbe un "buco". Il livello locale e' la media delle righe pulite piu' vicine di
    ciascun lato (cercate entro 2L righe, poi fino a 8L), interpolata linearmente tra i due lati. Restituisce (nchans,) o (B-A, nchans)."""
    n = xf.shape[0]
    m0, s0 = med[chans], sig[chans]

    def side(edge, direction):
        # righe senza eventi piu' vicine al rettangolo: prima entro 2L righe, poi fino a 8L
        for span in (2 * L, 8 * L):
            lo, hi = (edge - span, edge) if direction < 0 else (edge, edge + span)
            lo, hi = max(0, lo), min(n, hi)
            if hi - lo < 8:
                return None, None, 0.0
            rows = np.arange(lo, hi)[~hot[lo:hi]]
            rows = rows[-L:] if direction < 0 else rows[:L]
            if rows.size >= 8:
                mu = xf[rows][:, chans].mean(0)
                z = ((mu - m0) / s0).sum() * np.sqrt(rows.size / chans.size)
                return mu, rows.mean(), z
        return None, None, 0.0

    l, cl, zl = side(A, -1)
    r, cr, zr = side(B, +1)
    if max(abs(zl), abs(zr)) < zthr:
        return m0
    if l is None or r is None:
        return l if r is None else r
    f = ((np.arange(A, B) + 0.5 - cl) / (cr - cl)).astype(np.float32)[:, None]
    return l + (r - l) * f


def run_impulses(path, out, cfg: ImpulseConfig | None = None, verbose=True, report=None, tmax=None) -> dict:
    cfg = cfg or ImpulseConfig()
    t_start = time.time()
    fb = Filterbank(path)
    ts, nch = fb.tsamp, fb.nchans
    nsamp = fb.nsamp if tmax is None else min(fb.nsamp, int(tmax / ts))
    if fb.nbits == 32:
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = (-128, 127) if fb.dtype == np.int8 else (0, 2 ** fb.nbits - 1)
    # canali costanti (mascherati) dal primo blocco lungo
    probe = fb.read(0, min(nsamp, int(round(cfg.block / ts)))).astype(np.float32)
    sd = probe.std(0)
    live = sd > 0.05 * np.median(sd[sd > 0]) if np.any(sd > 0) else sd > 0   # quasi costanti = mascherati
    geo = _Geometry(fb, cfg, live)
    Wmax = 1 << geo.kmax
    nper = max(4 * Wmax, int(round(cfg.block / ts)))
    margin = geo.sweep_max + 2 * Wmax + 16
    rng = np.random.default_rng(cfg.seed)
    if verbose:
        print(f"{fb.path}: {nch} canali ({int(live.sum())} attivi), tsamp {ts * 1e6:.3g} us, {nsamp * ts:.1f} s, {fb.nbits} bit")
        print(f"  soglie {cfg.thresholds}, boxcar 1..{Wmax} campioni, {geo.nsub} sottobande, "
              f"veto DM {cfg.dm_min:g}-{cfg.dm_max:g}")
    items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fb.items]
    stats = dict(events=0, removed=0, protected=0, narrow=0, replaced_samples=0)
    protected_list, ev_rows = [], []
    t0 = time.time()
    last_print = 0.0
    veto_time = 0.0
    with open(out, "wb") as fo:
        fo.write(write_header(items))
        for c0 in range(0, nsamp, nper):
            c1 = min(nsamp, c0 + nper)
            e0, e1 = max(0, c0 - margin), min(nsamp, c1 + margin)
            raw = fb.read(e0, e1 - e0)
            xf = raw.astype(np.float32)
            med, sig = _robust_sigma(xf, max(1, xf.shape[0] // 8192))
            z = (xf - med) / sig
            z[:, ~live] = 0
            X = _series(z, geo)
            del z
            det = _detect(X, geo, cfg)
            evs = _events(det, geo, cfg, c0 - e0, c1 - e0)
            out_blk = raw[c0 - e0:c1 - e0].copy()
            tracks = {}
            # impulsi a banda larga di ogni gruppo (per il veto degli altri eventi dello stesso gruppo)
            bb = {}
            for ev in evs:
                if ev["col"] == 0 or ev["hits"].mean() >= cfg.full_frac:
                    mh = max(2, int(cfg.margin_frac * ev["W"]))
                    bb.setdefault(ev["group"], []).append((ev["a"] - mh, ev["b"] + mh, id(ev)))
            # rettangolo di ogni evento (boxcar + margine, esteso sui bordi con segnale) e righe "calde"
            # (coperte da eventi, con qualche campione in piu'), escluse dalla stima del livello locale
            hot = np.zeros(xf.shape[0], bool)
            for ev in evs:
                W = ev["W"]
                m = max(2, int(cfg.margin_frac * W))
                full = ev["col"] == 0 or ev["hits"].mean() >= cfg.full_frac or not ev["hits"].any()
                subs = np.flatnonzero(geo.nlive > 0) if full else np.flatnonzero(ev["hits"] & (geo.nlive > 0))
                A_, B_ = ev["a"] - m, ev["b"] + m
                if cfg.grow and subs.size:
                    A_, B_ = _grow(X, subs, A_, B_, max(4 * W, 64), cfg)
                ev["rect"], ev["full"] = (A_, B_), full
                pad = 4      # il rettangolo e' gia' esteso sui bordi con segnale
                hot[max(0, A_ - pad):B_ + pad] = True
            for ev in evs:
                stats["events"] += 1
                tv = time.time()
                tr = tracks.setdefault(ev["group"], [])
                if cfg.veto and tr and _on_track(ev, tr, geo, cfg):
                    prot, info = True, dict(reason="traccia")
                elif cfg.veto:
                    others = sorted((a_, b_) for a_, b_, k_ in bb.get(ev["group"], []) if k_ != id(ev))
                    prot, info = _veto(ev, det, geo, cfg)
                    if prot and others:
                        # gli altri impulsi a banda larga del gruppo devono stare tutti dentro lo sweep della
                        # traccia (pezzi dello stesso burst); altrimenti la traccia potrebbe solo collegare
                        # impulsi diversi di un treno: si ripete il veto con gli altri impulsi azzerati
                        w_ = info["width_dm"]
                        lo_ = info["t0"] - w_ - ev["W"]
                        hi_ = info["t0"] + info["dm"] * geo.delay1.max() + w_ + ev["W"]
                        if not all(lo_ <= a_ and b_ <= hi_ for a_, b_ in others):
                            excl = (np.array([o_[0] for o_ in others]), np.array([o_[1] for o_ in others]))
                            prot, info = _veto(ev, det, geo, cfg, excl)
                else:
                    prot, info = False, {}
                veto_time += time.time() - tv
                if info.get("reason") == "stretto":
                    stats["narrow"] += 1
                if prot:
                    stats["protected"] += 1
                    if "t0" in info:
                        tr.append((info["t0"], info["dm"], info["width_dm"]))
                    rec = dict(t=float((e0 + ev["a"]) * ts), width_s=float(ev["W"] * ts), snr_dm0=float(ev["snr"]))
                    rec.update({k: (v if isinstance(v, str) else float(v)) for k, v in info.items() if k != "t0"})
                    protected_list.append(rec)
                    continue
                stats["removed"] += 1
                W = ev["W"]
                full = ev["full"]
                A_, B_ = ev["rect"]
                a = max(c0 - e0, A_) - (c0 - e0)
                b = min(c1 - e0, B_) - (c0 - e0)
                if b <= a:
                    continue
                if full:
                    chans = np.flatnonzero(live)
                else:
                    chans = np.flatnonzero(live & ev["hits"][geo.sub_of])
                if chans.size == 0:
                    continue
                level = med[chans]
                if cfg.local_level:
                    level = _local_level(xf, chans, a + (c0 - e0), b + (c0 - e0),
                                         max(int(cfg.flank / ts), int(cfg.flank_w * W), 8), med, sig, hot,
                                         cfg.level_z)
                noise = level + sig[chans] * rng.standard_normal((b - a, chans.size)).astype(np.float32)
                if fb.nbits != 32:
                    noise = np.clip(np.rint(noise), lo, hi)
                out_blk[a:b][:, chans] = noise.astype(out_blk.dtype)
                stats["replaced_samples"] += (b - a) * chans.size
                if len(ev_rows) < 200000:
                    ev_rows.append((float((e0 + ev["a"]) * ts), float(W * ts), int(ev["col"]), float(ev["snr"]),
                                    int(ev["hits"].sum())))
            if fb.nbits < 8:
                fo.write(pack_bits(out_blk.astype(np.uint8).ravel(), fb.nbits).tobytes())
            else:
                fo.write(out_blk.tobytes())
            if verbose and (time.time() - last_print > 30 or c1 >= nsamp):
                last_print = time.time()
                print(f"  t={c1 * ts:8.1f} s ({c1 / nsamp * 100:5.1f}%): eventi {stats['events']}, rimossi "
                      f"{stats['removed']}, protetti {stats['protected']}  ({time.time() - t0:.0f} s, veto "
                      f"{veto_time:.0f} s)", flush=True)
    frac = stats["replaced_samples"] / max(1, nsamp * int(live.sum()))
    summary = dict(input=str(path), output=str(out), duration_s=nsamp * ts, config=asdict(cfg),
                   n_live_channels=int(live.sum()), **stats, frac_data_replaced=frac,
                   protected_events=protected_list[:500], veto_time_s=veto_time, runtime_s=time.time() - t_start)
    rpath = Path(report) if report else Path(str(out) + ".impulses.json")
    rpath.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    base = str(rpath)[:-5] if str(rpath).endswith(".json") else str(rpath)
    with open(base + "_events.csv", "w") as f:
        f.write("t_s,width_s,series,snr_dm0,n_sub_hit\n")
        for r in ev_rows:
            f.write(f"{r[0]:.6f},{r[1]:.6g},{'full' if r[2] == 0 else 'sub%02d' % (r[2] - 1)},{r[3]:.2f},{r[4]}\n")
    if verbose:
        print(f"Fatto in {summary['runtime_s']:.0f} s. Eventi {stats['events']}: rimossi {stats['removed']} "
              f"(di cui {stats['narrow']} per la scorciatoia 'stretti'), protetti {stats['protected']}. "
              f"Dati sostituiti: {frac * 100:.3f}%")
        print(f"Riassunto: {rpath}")
    return summary
