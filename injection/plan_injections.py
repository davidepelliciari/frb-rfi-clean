#!/usr/bin/env python3
"""
plan_injections.py - pianifica (ed eventualmente esegue) una campagna di iniezione
di FRB sintetici in un file SIGPROC filterbank.

Genera file .params compatibili con FRB_injector.py (una riga per impulso:
"ToA[s] SNR WIDTH[s] DM") piu' un catalogo unico (injections.csv) e i metadati
della campagna (campaign.json), cosi' ogni iniezione e' riproducibile dal seed.

Durata e banda sono lette dall'header del filterbank.

Specifica dei parametri (--snr, --dm, --width):
    "500"          valore fisso
    "100:1000"     uniforme in [100, 1000)
    "log:10:2000"  log-uniforme in [10, 2000)
    "50,200,800"   scelta casuale tra i valori elencati

Esempi:
  # solo pianificazione: 200 FRB, S/N 6-30 in un unico file, DM log-uniforme
  python3 plan_injections.py --fil obs.fil --outdir camp01 --nfrb 200 \\
      --snr 6:30 --dm log:20:2000 --width log:0.0005:0.02 --seed 1

  # comportamento "storico": un file per bin di S/N, DM e larghezza fissi
  python3 plan_injections.py --fil obs.fil --outdir camp02 --nfrb 50 \\
      --layout bins --snr 4:30 --snr-bin 1 --dm 500 --width 0.01

  # pianifica ed esegue il nuovo iniettore (un solo passaggio per file)
  python3 plan_injections.py ... --run --smear

  # oppure con il vecchio iniettore esterno
  python3 plan_injections.py ... --run --injector /jbod/codes/injections/FRB_injector.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from fil_cut import Fil  # noqa: E402

KDM = 4.148808e3  # costante di dispersione [s MHz^2 pc^-1 cm^3]


# --------------------------------------------------------------------------- fisica
def dm_delay(dm: float, f_lo: float, f_hi: float) -> float:
    """Ritardo dispersivo [s] tra f_lo e f_hi [MHz]."""
    return KDM * dm * (f_lo ** -2 - f_hi ** -2)


def band_edges(header: dict) -> tuple[float, float]:
    """Bordi (inferiore, superiore) della banda [MHz], inclusa la mezza larghezza di canale."""
    fch1, foff, nchans = header["fch1"], header["foff"], header["nchans"]
    lo, hi = sorted((fch1, fch1 + (nchans - 1) * foff))
    half = abs(foff) / 2
    return lo - half, hi + half


# --------------------------------------------------------------------------- campionamento
@dataclass
class Spec:
    """Distribuzione di un parametro, costruita da una stringa (vedi docstring del modulo)."""
    kind: str
    values: tuple[float, ...]

    @classmethod
    def parse(cls, s: str) -> "Spec":
        s = s.strip()
        try:
            if s.startswith("log:"):
                lo, hi = map(float, s[4:].split(":"))
                if not 0 < lo < hi:
                    raise ValueError
                return cls("log", (lo, hi))
            if ":" in s:
                lo, hi = map(float, s.split(":"))
                if not lo < hi:
                    raise ValueError
                return cls("uniform", (lo, hi))
            if "," in s:
                return cls("choice", tuple(float(x) for x in s.split(",")))
            return cls("fixed", (float(s),))
        except ValueError:
            raise argparse.ArgumentTypeError(f"specifica non valida: {s!r}") from None

    @property
    def min(self) -> float:
        return min(self.values)

    @property
    def max(self) -> float:
        return max(self.values)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "fixed":
            return np.full(n, self.values[0])
        if self.kind == "uniform":
            return rng.uniform(*self.values, n)
        if self.kind == "log":
            lo, hi = np.log(self.values)
            return np.exp(rng.uniform(lo, hi, n))
        return rng.choice(np.array(self.values), n)


def sample_toas(n: int, t_min: float, t_max: float, min_gap: float,
                rng: np.random.Generator) -> np.ndarray:
    """
    n tempi in [t_min, t_max] con separazione minima min_gap, distribuiti
    uniformemente su tutte le configurazioni ammissibili.

    Metodo: n punti uniformi ordinati in [0, slack], con slack = (t_max-t_min) - (n-1)*min_gap,
    poi l'i-esimo viene spostato di i*min_gap.
    """
    if n < 1:
        raise ValueError("serve almeno un impulso")
    span = t_max - t_min
    slack = span - (n - 1) * min_gap
    if slack < 0:
        nmax = int(span // min_gap) + 1 if span >= 0 else 0
        raise ValueError(f"impossibile piazzare {n} impulsi distanti >= {min_gap:g} s "
                         f"in {span:.2f} s (massimo {nmax})")
    return t_min + np.sort(rng.uniform(0.0, slack, n)) + min_gap * np.arange(n)


# --------------------------------------------------------------------------- pianificazione
def plan(fil: Fil, *, nfrb: int, snr: Spec, dm: Spec, width: Spec, layout: str,
         snr_bin: float, toa_mode: str, min_gap: float | None, start_margin: float,
         end_margin: float, rng: np.random.Generator,
         duration: float | None = None, band: tuple[float, float] | None = None):
    """Restituisce (lista di (nome_file, array righe [toa, snr, width, dm]), info)."""
    f_lo, f_hi = band or band_edges(fil.h)
    duration = duration or fil.duration
    sweep = dm_delay(dm.max, f_lo, f_hi)
    if min_gap is None:
        min_gap = max(0.2, 10 * width.max)
    t_min = start_margin
    t_max = duration - end_margin - 1.1 * sweep - width.max

    if layout == "bins":
        if snr.kind not in ("uniform", "fixed"):
            raise ValueError("--layout bins richiede --snr fisso o nella forma min:max")
        lo, hi = (snr.values[0], snr.values[0] + snr_bin) if snr.kind == "fixed" else snr.values
        edges = np.arange(lo, hi, snr_bin)
        groups = [(f"SyntheticFRB_{a:g}_{a + snr_bin:g}", Spec("uniform", (a, a + snr_bin)))
                  for a in edges]
    else:
        groups = [("SyntheticFRB", snr)]

    out = []
    for name, snr_spec in groups:
        if toa_mode == "uniform":
            toa = np.linspace(t_min, t_max, nfrb)
            if nfrb > 1 and (t_max - t_min) / (nfrb - 1) < min_gap:
                raise ValueError("griglia uniforme piu' fitta di --min-gap")
        else:
            toa = sample_toas(nfrb, t_min, t_max, min_gap, rng)
        rows = np.column_stack([toa, np.round(snr_spec.sample(nfrb, rng), 2),
                                width.sample(nfrb, rng), dm.sample(nfrb, rng)])
        out.append((name, rows))

    info = dict(duration_s=duration, band_mhz=[f_lo, f_hi], max_dm_sweep_s=sweep,
                toa_range_s=[t_min, t_max], min_gap_s=min_gap)
    return out, info


def write_campaign(outdir: Path, groups, info: dict, meta: dict) -> list[Path]:
    pdir = outdir / "params"
    pdir.mkdir(parents=True, exist_ok=True)
    written = []
    with open(outdir / "injections.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["params_file", "toa_s", "snr", "width_s", "dm"])
        for name, rows in groups:
            p = pdir / f"{name}.params"
            with open(p, "w") as f:
                for toa, s, wd, d in rows:
                    f.write(f"{toa:.6f} {s:.2f} {wd:.6g} {d:.4f}\n")
                    w.writerow([p.name, f"{toa:.6f}", f"{s:.2f}", f"{wd:.6g}", f"{d:.4f}"])
            written.append(p)
    (outdir / "campaign.json").write_text(json.dumps({**meta, **info}, indent=2))
    return written


# --------------------------------------------------------------------------- esecuzione
def run_injector(injector: Path, fil_path: Path, params: list[Path], fil_outdir: Path,
                 smear: bool) -> int:
    fil_outdir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, p in enumerate(params, 1):
        cmd = [sys.executable, str(injector), "--FIL", str(fil_path),
               "--outfile", str(fil_outdir / f"INJECTED_{p.stem}"), "--FRBparam", str(p)]
        if smear:
            cmd.append("--smear")
        print(f"[{i}/{len(params)}] {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(p.name)
    if failed:
        print(f"ERRORE: iniezione fallita per {len(failed)} file: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fil", required=True, type=Path, help="filterbank in cui iniettare")
    p.add_argument("--outdir", required=True, type=Path, help="cartella della campagna (params, csv, json)")
    p.add_argument("--nfrb", type=int, required=True, help="numero di FRB per file .params")
    p.add_argument("--snr", type=Spec.parse, default=Spec.parse("4:30"), help="S/N (default 4:30)")
    p.add_argument("--dm", type=Spec.parse, default=Spec.parse("500"), help="DM [pc cm^-3] (default 500)")
    p.add_argument("--width", type=Spec.parse, default=Spec.parse("0.01"), help="larghezza [s] (default 0.01)")
    p.add_argument("--layout", choices=["single", "bins"], default="single",
                   help="single: un solo file con tutti gli S/N (default); bins: un file per bin di S/N")
    p.add_argument("--snr-bin", type=float, default=1.0, help="ampiezza dei bin di S/N con --layout bins")
    p.add_argument("--toa", choices=["random", "uniform"], default="random", help="distribuzione dei ToA")
    p.add_argument("--min-gap", type=float, default=None,
                   help="separazione minima tra ToA [s] (default max(0.2, 10*larghezza massima))")
    p.add_argument("--start-margin", type=float, default=0.5, help="margine iniziale [s]")
    p.add_argument("--end-margin", type=float, default=0.5, help="margine finale [s], oltre allo sweep del DM")
    p.add_argument("--duration", type=float, default=None, help="sovrascrive la durata letta dall'header [s]")
    p.add_argument("--seed", type=int, default=None, help="seed (se omesso ne viene scelto e salvato uno)")
    g = p.add_argument_group("esecuzione")
    g.add_argument("--run", action="store_true", help="esegue l'iniettore dopo aver scritto i .params")
    g.add_argument("--injector", type=Path,
                   help="iniettore esterno compatibile con FRB_injector.py (default: inject_frb.py, in un passaggio)")
    g.add_argument("--snr-calib", default=None, metavar="PENDENZA,OFFSET",
                   help="calibrazione S/N su Heimdall passata a inject_frb.py")
    g.add_argument("--fil-outdir", type=Path, help="dove scrivere i .fil iniettati (default OUTDIR/fil)")
    g.add_argument("--smear", action="store_true", help="passa --smear all'iniettore")
    g.add_argument("--force", action="store_true", help="ignora il controllo dello spazio su disco")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.outdir.exists() and any(a.outdir.iterdir()):
        print(f"ERRORE: {a.outdir} esiste e non e' vuota (evito di mescolare campagne)", file=sys.stderr)
        return 2

    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    fil = Fil(str(a.fil))
    try:
        groups, info = plan(fil, nfrb=a.nfrb, snr=a.snr, dm=a.dm, width=a.width, layout=a.layout,
                            snr_bin=a.snr_bin, toa_mode=a.toa, min_gap=a.min_gap,
                            start_margin=a.start_margin, end_margin=a.end_margin,
                            rng=rng, duration=a.duration)
    except ValueError as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 2

    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                fil=str(a.fil.resolve()), seed=seed, nfrb=a.nfrb, layout=a.layout,
                snr=a.snr.__dict__, dm=a.dm.__dict__, width=a.width.__dict__,
                toa=a.toa, smear=a.smear, argv=sys.argv)
    params = write_campaign(a.outdir, groups, info, meta)

    print(f"Campagna in {a.outdir}: {len(params)} file .params x {a.nfrb} FRB, seed={seed}")
    print(f"  banda {info['band_mhz'][0]:.2f}-{info['band_mhz'][1]:.2f} MHz, "
          f"durata {info['duration_s']:.2f} s, sweep al DM massimo {info['max_dm_sweep_s']:.3f} s")
    print(f"  ToA in [{info['toa_range_s'][0]:.2f}, {info['toa_range_s'][1]:.2f}] s, "
          f"gap minimo {info['min_gap_s']:.3f} s (maxCsec > {2 / info['min_gap_s']:.2f})")

    if not a.run:
        return 0
    fil_outdir = a.fil_outdir or a.outdir / "fil"
    fil_outdir.mkdir(parents=True, exist_ok=True)
    need = len(params) * a.fil.stat().st_size
    free = shutil.disk_usage(fil_outdir).free
    print(f"  spazio richiesto ~{need / 1e9:.1f} GB, libero {free / 1e9:.1f} GB in {fil_outdir}")
    if need > 0.95 * free and not a.force:
        print("ERRORE: spazio insufficiente (usa --force per ignorare)", file=sys.stderr)
        return 3
    if a.injector is not None:
        return run_injector(a.injector, a.fil, params, fil_outdir, a.smear)
    import inject_frb
    rc = 0
    for i, pf in enumerate(params):
        out = fil_outdir / f"INJECTED_{pf.stem}.fil"
        argv = ["--fil", str(a.fil), "--params", str(pf), "--out", str(out), "--seed", str(seed + i)]
        if not a.smear:
            argv.append("--no-smear")
        if a.snr_calib:
            argv += ["--snr-calib", a.snr_calib]
        rc |= inject_frb.main(argv)
    return rc


if __name__ == "__main__":
    sys.exit(main())
