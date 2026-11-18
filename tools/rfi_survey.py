#!/usr/bin/env python3
"""
rfi_survey.py - censimento degli RFI in un filterbank SIGPROC (qualsiasi banda, tsamp, nbits).

Un solo passaggio a blocchi sul file; produce statistiche per canale, catalogo degli
eventi impulsivi a DM 0 (banda intera e sottobande), periodicita', canali sospetti e grafici.

Esempio:
  python3 rfi_survey.py obs.fil --out survey_obs/
  python3 rfi_survey.py obs.fil --out prova/ --tmax 120      # solo i primi 120 s
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.survey import SurveyConfig, run_survey  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil")
    p.add_argument("--out", required=True, help="cartella dei prodotti")
    d = SurveyConfig()
    p.add_argument("--nsub", type=int, default=d.nsub, help=f"sottobande (default {d.nsub})")
    p.add_argument("--max-width", type=float, default=d.max_width, help=f"larghezza massima [s] (default {d.max_width})")
    p.add_argument("--thresh", type=float, default=d.thresh, help=f"soglia S/N degli eventi (default {d.thresh})")
    p.add_argument("--block", type=float, default=None, help="durata dei blocchi [s] (default automatica)")
    p.add_argument("--tmax", type=float, default=None, help="analizza solo i primi TMAX secondi")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    cfg = SurveyConfig(nsub=a.nsub, max_width=a.max_width, thresh=a.thresh, block=a.block)
    run_survey(a.fil, a.out, cfg, plot=not a.no_plots, verbose=not a.quiet, tmax=a.tmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
