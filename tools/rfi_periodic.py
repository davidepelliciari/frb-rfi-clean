#!/usr/bin/env python3
"""
rfi_periodic.py - step 2: sottrae le componenti periodiche (es. 50 Hz e armoniche) canale per canale.

Si applica di norma all'output dello step 1 (rfi_clean.py). Per ogni tratto di SEGMENT secondi e
canale si stima il profilo medio (clippato) su un periodo 1/f e lo si sottrae in base alla fase.
L'output ha lo stesso header e numero di bit. Riassunto in OUT.periodic.json.

Esempi:
  python3 rfi_periodic.py step1.fil step2.fil                      # 50 Hz
  python3 rfi_periodic.py step1.fil step2.fil --freq 50 --search   # affina la frequenza per tratto
  python3 rfi_periodic.py step1.fil step2.fil --freq 50 --freq 1.6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.periodic import PeriodicConfig, run_periodic  # noqa: E402


def main(argv=None):
    d = PeriodicConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil")
    p.add_argument("out")
    p.add_argument("--freq", type=float, action="append", help="frequenza fondamentale [Hz] (ripetibile; default 50)")
    p.add_argument("--nbins", type=int, default=d.nbins, help=f"intervalli di fase (default {d.nbins})")
    p.add_argument("--segment", type=float, default=d.segment, help=f"durata dei tratti [s] (default {d.segment})")
    p.add_argument("--clip", type=float, default=d.clip, help=f"clipping in sigma (default {d.clip})")
    p.add_argument("--search", action="store_true", help="affina la frequenza per ogni tratto")
    p.add_argument("--search-width", type=float, default=d.search_width, help=f"semi-ampiezza della ricerca [Hz] (default {d.search_width})")
    p.add_argument("--tmax", type=float, default=None, help="elabora solo i primi TMAX secondi")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve() == Path(a.fil).resolve():
        sys.exit("ERRORE: l'output coincide con l'input")
    cfg = PeriodicConfig(freqs=a.freq or [50.0], nbins=a.nbins, segment=a.segment, clip=a.clip,
                         search=a.search, search_width=a.search_width)
    run_periodic(a.fil, a.out, cfg, verbose=not a.quiet, tmax=a.tmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
