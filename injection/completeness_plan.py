#!/usr/bin/env python3
"""
completeness_plan.py - pianifica una campagna di completezza: scansioni a parametri fissati,
mescolate a caso in piu' realizzazioni (file iniettati) con ToA casuali.

Scansioni (default; ogni punto ripetuto --per-point volte in totale):
  A  S/N         w = 1 ms, DM = 250, S/N = 5 6 7 8 9 10 12 15 20 30
  B  larghezza   FLUSSO fisso (S/N_1ms = 20), DM = 250, w = 0.1 0.25 0.5 1 2 4 8 16 32 ms
  C  larghezza   S/N fisso = 20, DM = 250, stesse larghezze
  D  DM          w = 1 ms, S/N = 20, DM = 30 50 75 100 150 250 400 550 700
  E  DM x w      S/N = 20, DM = 30 50 100, w = 2 4 8 16 ms

Flusso fisso: S/N_1ms e' l'S/N che il burst avrebbe se fosse largo 1 ms (stesso DM). L'ampiezza per
canale (in unita' della sigma del canale) resta quella del burst da 1 ms, e l'S/N atteso a larghezza w
viene calcolato con lo stesso modello dell'iniettore (dedispersione a shift interi, boxcar migliore,
smearing intra-canale), su un rumore piatto: il rapporto tra larghezze non dipende dal bandpass.
Con --snr-calib gli S/N sono intesi come S/N di Heimdall (come nell'iniettore).

Prodotti in OUTDIR:
  plan.csv             un burst per riga: realization, toa_s, scan, point, snr_expected, snr_1ms, width_s, dm
  real_XX.params       per inject_frb.py (ToA SNR WIDTH DM), uno per realizzazione
  campaign.json        parametri e seed

Esempi:
  # campagna completa (realizzazioni calcolate dalla capienza del file)
  python3 completeness_plan.py --fil obs.fil --outdir camp --seed 1
  # pilota: 300 s, scansioni A e D, 7 burst per punto
  python3 completeness_plan.py --fil cut300.fil --outdir pilot --scans A D --per-point 7 --seed 1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from fil_cut import Fil  # noqa: E402
from inject_frb import Burst, Injector  # noqa: E402
from plan_injections import band_edges, dm_delay, sample_toas  # noqa: E402

MS = 1e-3
DEFAULT_SCANS = {
    "A": dict(kind="snr", fixed=dict(width=1 * MS, dm=250.0), values=[5, 6, 7, 8, 9, 10, 12, 15, 20, 30]),
    "B": dict(kind="width_fluence", fixed=dict(snr_1ms=20.0, dm=250.0),
              values=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]),
    "C": dict(kind="width", fixed=dict(snr=20.0, dm=250.0), values=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]),
    "D": dict(kind="dm", fixed=dict(width=1 * MS, snr=20.0), values=[30, 50, 75, 100, 150, 250, 400, 550, 700]),
    "E": dict(kind="dm_width", fixed=dict(snr=20.0), values=[(d, w) for d in (30, 50, 100) for w in (2, 4, 8, 16)]),
}


class FluenceModel:
    """S/N per ampiezza unitaria (rumore piatto), con lo stesso modello dell'iniettore."""

    def __init__(self, fil: Fil, smear: bool = True):
        self.inj = Injector(fil, smear=smear, seed=0)
        self.t_mid = fil.duration / 2
        self.cache = {}

    def snr_unit(self, width: float, dm: float) -> float:
        key = (round(width, 9), round(dm, 6))
        if key not in self.cache:
            b = Burst(toa=self.t_mid, snr=1.0, width=width, dm=dm)
            self.inj.build_template(b, np.ones(self.inj.nchan), np.random.default_rng(0))
            b._tmpl = None
            self.cache[key] = 1.0 / b.amplitude
        return self.cache[key]

    def snr_at_width(self, snr_1ms: float, width: float, dm: float, calib=None) -> float:
        """S/N atteso a larghezza `width` per un burst con lo stesso flusso di uno da 1 ms a S/N snr_1ms."""
        r = self.snr_unit(width, dm) / self.snr_unit(1 * MS, dm)
        if calib:
            slope, off = calib
            return slope * ((snr_1ms - off) / slope) * r + off
        return snr_1ms * r


def build_points(scans, model: FluenceModel | None, calib):
    pts = []
    for name in scans:
        sc = DEFAULT_SCANS[name]
        fx = sc["fixed"]
        for v in sc["values"]:
            if sc["kind"] == "snr":
                p = dict(snr=float(v), width=fx["width"], dm=fx["dm"], snr_1ms=np.nan, point=f"snr={v:g}")
            elif sc["kind"] == "width":
                p = dict(snr=fx["snr"], width=v * MS, dm=fx["dm"], snr_1ms=np.nan, point=f"w={v:g}ms")
            elif sc["kind"] == "width_fluence":
                s = model.snr_at_width(fx["snr_1ms"], v * MS, fx["dm"], calib)
                p = dict(snr=float(s), width=v * MS, dm=fx["dm"], snr_1ms=fx["snr_1ms"], point=f"w={v:g}ms")
            elif sc["kind"] == "dm":
                p = dict(snr=fx["snr"], width=fx["width"], dm=float(v), snr_1ms=np.nan, point=f"dm={v:g}")
            else:
                d, w = v
                p = dict(snr=fx["snr"], width=w * MS, dm=float(d), snr_1ms=np.nan, point=f"dm={d:g},w={w:g}ms")
            p["scan"] = name
            pts.append(p)
    return pts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fil", required=True, type=Path, help="filterbank (header e durata)")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--scans", nargs="+", default=list(DEFAULT_SCANS), choices=list(DEFAULT_SCANS))
    p.add_argument("--per-point", type=int, default=40, help="burst per punto, in totale (default 40)")
    p.add_argument("--realizations", type=int, default=None,
                   help="numero di realizzazioni (default: il minimo che rispetta --min-gap)")
    p.add_argument("--min-gap", type=float, default=2.0, help="separazione minima tra burst [s] (default 2)")
    p.add_argument("--start-margin", type=float, default=1.0)
    p.add_argument("--end-margin", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=None, help="sovrascrive la durata dell'header [s]")
    p.add_argument("--snr-calib", default=None, metavar="PENDENZA,OFFSET",
                   help="come in inject_frb.py: gli S/N sono S/N di Heimdall")
    p.add_argument("--no-smear", action="store_true", help="modello senza smearing intra-canale")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)

    fil = Fil(str(a.fil))
    calib = tuple(float(x) for x in a.snr_calib.split(",")) if a.snr_calib else None
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    model = FluenceModel(fil, smear=not a.no_smear) if "B" in a.scans else None
    pts = build_points(a.scans, model, calib)

    duration = a.duration or fil.duration
    f_lo, f_hi = band_edges(fil.h)
    dm_max = max(pt["dm"] for pt in pts)
    w_max = max(pt["width"] for pt in pts)
    t_min = a.start_margin
    t_max = duration - a.end_margin - 1.1 * dm_delay(dm_max, f_lo, f_hi) - 5 * w_max
    cap = int((t_max - t_min) // a.min_gap) + 1
    ntot = len(pts) * a.per_point
    nreal = a.realizations or int(np.ceil(ntot / cap))
    if ntot > nreal * cap:
        print(f"ERRORE: {ntot} burst non entrano in {nreal} realizzazioni da {cap} (min-gap {a.min_gap} s)",
              file=sys.stderr)
        return 2

    # ogni punto ripetuto per-point volte, mescolato e distribuito a turno tra le realizzazioni
    order = np.repeat(np.arange(len(pts)), a.per_point)
    rng.shuffle(order)
    groups = [order[r::nreal] for r in range(nreal)]
    a.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, g in enumerate(groups, 1):
        toa = sample_toas(len(g), t_min, t_max, a.min_gap, rng)
        rng.shuffle(g)
        with open(a.outdir / f"real_{r:02d}.params", "w") as f:
            for t, k in zip(toa, g):
                pt = pts[k]
                f.write(f"{t:.6f} {pt['snr']:.3f} {pt['width']:.6g} {pt['dm']:.4f}\n")
                rows.append((r, t, pt))
    with open(a.outdir / "plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["realization", "toa_s", "scan", "point", "snr_expected", "snr_1ms", "width_s", "dm"])
        for r, t, pt in sorted(rows, key=lambda x: (x[0], x[1])):
            w.writerow([r, f"{t:.6f}", pt["scan"], pt["point"], f"{pt['snr']:.3f}",
                        "" if np.isnan(pt["snr_1ms"]) else f"{pt['snr_1ms']:g}", f"{pt['width']:.6g}",
                        f"{pt['dm']:.4f}"])
    meta = dict(fil=str(a.fil), seed=seed, scans={k: DEFAULT_SCANS[k] for k in a.scans}, per_point=a.per_point,
                n_points=len(pts), n_bursts=ntot, realizations=nreal, capacity_per_file=cap,
                min_gap_s=a.min_gap, toa_range_s=[t_min, t_max], snr_calib=calib, smear=not a.no_smear,
                duration_s=duration)
    (a.outdir / "campaign.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"{len(pts)} punti x {a.per_point} = {ntot} burst in {nreal} realizzazioni "
          f"(capienza {cap} per file con min-gap {a.min_gap} s); seed {seed}")
    if "B" in a.scans:
        print("Scansione B (flusso fisso, S/N_1ms = 20): S/N atteso per larghezza:")
        for pt in pts:
            if pt["scan"] == "B":
                print(f"   w = {pt['width'] * 1e3:6.3g} ms -> S/N {pt['snr']:.2f}")
    print(f"Scritti {a.outdir / 'plan.csv'} e {nreal} file .params")
    return 0


if __name__ == "__main__":
    sys.exit(main())
