#!/usr/bin/env python3
"""
rfi_periodic.py - step 2: individua e sottrae automaticamente le componenti periodiche (RFI periodico).

  1. analisi: spettro della serie a DM 0 a risoluzione nativa, righe, famiglie armoniche (fondamentale
     comune quando possibile), affinamento della frequenza, test di dispersione (le famiglie disperse,
     es. pulsar, non vengono toccate)
  2. sottrazione per canale del profilo medio a risoluzione nativa (un punto per campione)
  3. verifica: spettro dell'output; se restano famiglie sottraibili si ripete (--iterations)

L'output ha lo stesso header e numero di bit. Prodotti accanto all'output:
  OUT.periodic.json           righe prima/dopo, famiglie (frequenza, test di dispersione, decisione)
  OUT.periodic_spectrum.png   spettro prima/dopo, normalizzato al continuo, rapporto
  OUT.periodic_spectrum.npz   spettri prima/dopo

Esempi:
  python3 rfi_periodic.py step1.fil step2.fil                    # automatico
  python3 rfi_periodic.py step1.fil step2.fil --freq 50          # solo le frequenze indicate
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
    p.add_argument("--freq", type=float, action="append", help="frequenza fondamentale manuale [Hz] (ripetibile)")
    p.add_argument("--line-thresh", type=float, default=d.line_thresh, help=f"soglia righe spettro/continuo (default {d.line_thresh})")
    p.add_argument("--fmin", type=float, default=d.fmin_fund, help=f"fondamentale minima [Hz] (default {d.fmin_fund})")
    p.add_argument("--max-families", type=int, default=d.max_families)
    p.add_argument("--iterations", type=int, default=d.iterations)
    p.add_argument("--no-protect", action="store_true", help="disattiva il test di dispersione")
    p.add_argument("--protect-dm", type=float, default=d.protect_dm, help=f"DM minimo per proteggere una famiglia (default {d.protect_dm})")
    p.add_argument("--segment", type=float, default=d.segment, help=f"durata minima dei tratti di sottrazione [s] (default {d.segment})")
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve() == Path(a.fil).resolve():
        sys.exit("ERRORE: l'output coincide con l'input")
    cfg = PeriodicConfig(freqs=a.freq, line_thresh=a.line_thresh, fmin_fund=a.fmin, max_families=a.max_families,
                         iterations=a.iterations, protect=not a.no_protect, protect_dm=a.protect_dm, segment=a.segment)
    run_periodic(a.fil, a.out, cfg, verbose=not a.quiet, tmax=a.tmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
