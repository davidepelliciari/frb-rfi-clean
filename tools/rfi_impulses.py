#!/usr/bin/env python3
"""
rfi_impulses.py - step 3: rimuove gli impulsi a DM 0 (radar e simili, a banda intera o parziale),
con veto di dispersione per non toccare i segnali dispersi.

Si applica di norma all'output dello step 2. Solo le zone tempo x sottobande degli impulsi
riconosciuti come RFI vengono sostituite con rumore; tutti gli altri campioni restano identici.
Prodotti: OUT.impulses.json (riassunto, eventi protetti) e OUT.impulses_events.csv (eventi rimossi).

Esempi:
  python3 rfi_impulses.py step2.fil step3.fil
  python3 rfi_impulses.py step2.fil step3.fil --thresholds 10 8 6 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.impulses import ImpulseConfig, run_impulses  # noqa: E402


def main(argv=None):
    d = ImpulseConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil")
    p.add_argument("out")
    p.add_argument("--thresholds", type=float, nargs="+", default=d.thresholds, help=f"soglie iterative S/N (default {d.thresholds})")
    p.add_argument("--max-width", type=float, default=d.max_width, help=f"larghezza massima [s] (default {d.max_width})")
    p.add_argument("--nsub", type=int, default=d.nsub)
    p.add_argument("--no-veto", action="store_true", help="disattiva il veto di dispersione (sconsigliato)")
    p.add_argument("--dm-min", type=float, default=d.dm_min, help=f"DM minimo considerato disperso (default {d.dm_min})")
    p.add_argument("--dm-max", type=float, default=d.dm_max)
    p.add_argument("--protect-ratio", type=float, default=d.protect_ratio)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve() == Path(a.fil).resolve():
        sys.exit("ERRORE: l'output coincide con l'input")
    cfg = ImpulseConfig(thresholds=a.thresholds, max_width=a.max_width, nsub=a.nsub, veto=not a.no_veto,
                        dm_min=a.dm_min, dm_max=a.dm_max, protect_ratio=a.protect_ratio)
    run_impulses(a.fil, a.out, cfg, verbose=not a.quiet, tmax=a.tmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
