#!/usr/bin/env python3
"""
rfi_commonmode.py - step 4: sottrae il rumore di modo comune (segnale comune a tutti i canali).

Si applica all'output dello step 3, come ultimo passo prima della ricerca. Per ogni blocco e canale
stima l'accoppiamento g_c con la media su tutti i canali (il modo comune) e sottrae g_c x modo comune;
con --mode zerodm sottrae la media con g_c = 1 (filtro a DM zero). Mediana e sigma di ogni canale
restano quelle dell'input; i canali mascherati restano identici.
Prodotti: OUT.commonmode.json (eccesso di rumore a DM 0 prima e dopo, guadagni) e OUT.commonmode.npz.

Esempi:
  python3 rfi_commonmode.py step3.fil step4.fil
  python3 rfi_commonmode.py step3.fil step4.fil --mode zerodm
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.commonmode import CommonModeConfig, run_commonmode  # noqa: E402


def main(argv=None):
    d = CommonModeConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil")
    p.add_argument("out")
    p.add_argument("--mode", choices=["regress", "zerodm"], default=d.mode,
                   help="regress: accoppiamento stimato per canale (default); zerodm: g = 1")
    p.add_argument("--block", type=float, default=d.block, help=f"durata dei blocchi [s] (default {d.block})")
    p.add_argument("--fit-res", type=float, default=d.fit_res,
                   help=f"risoluzione per la stima dei guadagni [s] (default {d.fit_res})")
    p.add_argument("--no-dither", action="store_true", help="arrotondamento deterministico (sconsigliato)")
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve() == Path(a.fil).resolve():
        sys.exit("ERRORE: l'output coincide con l'input")
    cfg = CommonModeConfig(mode=a.mode, block=a.block, fit_res=a.fit_res, dither=not a.no_dither)
    run_commonmode(a.fil, a.out, cfg, verbose=not a.quiet, tmax=a.tmax)


if __name__ == "__main__":
    main()
