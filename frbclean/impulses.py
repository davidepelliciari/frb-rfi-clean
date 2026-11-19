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
4. Sostituzione: per gli eventi non protetti, il rettangolo tempo (con margine) x canali delle
   sottobande colpite (tutti i canali se l'evento e' a banda intera o colpisce almeno meta' delle
   sottobande) viene sostituito con rumore gaussiano con mediana e sigma robuste del canale. Tutti
   gli altri campioni restano identici all'input.
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
    for g in groups:
        best = max(g, key=lambda x: x[4])
        a, b = min(x[0] for x in g), max(x[1] for x in g)
        if not (core0 <= best[6] < core1):
            continue
        hits = np.zeros(geo.nsub, bool)
        vhits = np.zeros(geo.nsub, bool)
        for x in g:
            hits |= x[5]
            vhits |= x[7]
            if x[3] > 0:
                hits[x[3] - 1] = True
                vhits[x[3] - 1] = True
        # a, b: intervallo del boxcar migliore (quello che si sostituisce); span: unione di tutti i record
        ev.append(dict(a=best[6], b=best[6] + best[2], W=best[2], col=best[3], snr=best[4], hits=hits, vhits=vhits,
                       peak=best[6],
                       span=(a, b)))
    return ev


def _veto(ev, det, geo, cfg):
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
    widths = sorted(w for w in det if w <= 2 * W)
    best_lo, best_hi, bdm, bw = -np.inf, -np.inf, 0.0, 0
    track_frac = 0.0
    for w in widths:
        Z = det[w][0]
        n = Z.shape[0]
        sub = Z[:, 1:]
        stride = max(1, w // 4)
        dm_cap = cfg.dm_max if w == widths[-1] else min(cfg.dm_max, 4.0 * w / smear1)
        dstep = stride / max(geo.delay1.max(), 1e-9)
        for dm in np.arange(0.0, dm_cap + dstep, dstep):
            dall = np.round(dm * geo.delay1[cols_all]).astype(int)
            d = np.round(dm * geo.delay1[cols]).astype(int)
            if jdet is not None:
                lo_t, hi_t = p - dall[jdet] - w, p - dall[jdet] + W
            else:
                lo_t, hi_t = p - dall.max() - w, p + W
            lo_t, hi_t = max(0, lo_t), min(n - 1 - dall.max(), hi_t)
            if hi_t < lo_t:
                continue
            t0 = np.arange(lo_t, hi_t + 1, stride)
            M = sub[t0[:, None] + d[None, :], cols]
            tot = M.sum(1)
            i = int(np.argmax(tot))
            v = float(tot[i] / np.sqrt(cols.size))
            if dm < cfg.dm_min:
                best_lo = max(best_lo, v)
            elif v > best_hi:
                best_hi, bdm, bw = v, float(dm), w
                track_frac = float(np.mean(M[i] > cfg.track_z))
    # coerenza: lungo la traccia migliore una buona frazione delle sottobande deve mostrare segnale
    # (un FRB e' esteso in frequenza; una traccia che raccoglie poche sottobande molto forti e' di
    # solito un'altra RFI attraversata per caso)
    protect = (best_hi >= cfg.protect_snr and best_hi >= cfg.protect_ratio * max(best_lo, 0)
               and track_frac >= cfg.track_frac)
    return protect, dict(snr_lowdm=best_lo, snr_dm=best_hi, dm=bdm, width_dm=float(bw), broadband=float(broadband),
                         track_frac=track_frac)


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
    probe = fb.read(0, min(nsamp, int(round(cfg.block / ts))))
    live = probe.min(0) != probe.max(0)
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
            for ev in evs:
                stats["events"] += 1
                prot, info = _veto(ev, det, geo, cfg) if cfg.veto else (False, {})
                if info.get("reason") == "stretto":
                    stats["narrow"] += 1
                if prot:
                    stats["protected"] += 1
                    protected_list.append(dict(t=float((e0 + ev["a"]) * ts), width_s=float(ev["W"] * ts),
                                               snr_dm0=float(ev["snr"]), **{k: float(v) for k, v in info.items()}))
                    continue
                stats["removed"] += 1
                W = ev["W"]
                m = max(2, int(cfg.margin_frac * W))
                a = max(c0 - e0, ev["a"] - m) - (c0 - e0)
                b = min(c1 - e0, ev["b"] + m) - (c0 - e0)
                if b <= a:
                    continue
                if ev["col"] == 0 or ev["hits"].mean() >= cfg.full_frac or not ev["hits"].any():
                    chans = np.flatnonzero(live)
                else:
                    chans = np.flatnonzero(live & ev["hits"][geo.sub_of])
                if chans.size == 0:
                    continue
                noise = med[chans] + sig[chans] * rng.standard_normal((b - a, chans.size)).astype(np.float32)
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
            if verbose and (c0 // nper) % 25 == 0:
                print(f"  t={c0 * ts:8.1f} s: eventi {stats['events']}, rimossi {stats['removed']}, "
                      f"protetti {stats['protected']}  ({time.time() - t0:.0f} s)", flush=True)
    frac = stats["replaced_samples"] / max(1, nsamp * int(live.sum()))
    summary = dict(input=str(path), output=str(out), duration_s=nsamp * ts, config=asdict(cfg),
                   n_live_channels=int(live.sum()), **stats, frac_data_replaced=frac,
                   protected_events=protected_list[:500], runtime_s=time.time() - t_start)
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
