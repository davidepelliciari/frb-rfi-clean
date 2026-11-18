#!/usr/bin/env python3
"""
rfi_clean.py - primo stadio di pulizia di un filterbank SIGPROC (qualsiasi banda, tsamp, nbits).

  1. maschera dei canali (automatica e/o da file/lista)
  2. normalizzazione mobile per canale (livello e rumore robusti, mediana mobile nel tempo)
  3. sostituzione con rumore delle celle tempo-frequenza con varianza anomala

L'output ha lo stesso header e lo stesso numero di bit: si passa a Heimdall al posto dell'originale.
Accanto all'output vengono scritti OUT.clean.json (riassunto) e OUT.clean.npz (statistiche e maschera).

Esempi:
  python3 rfi_clean.py obs.fil obs_clean.fil
  python3 rfi_clean.py obs.fil obs_clean.fil --mask-file survey_obs/flagged_channels.txt
  python3 rfi_clean.py obs.fil obs_clean.fil --mask-chans 110,121-127,185 --no-auto-mask
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frbclean.clean import CleanConfig, parse_chan_list, read_flag_file, run_clean  # noqa: E402


def main(argv=None):
    d = CleanConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fil")
    p.add_argument("out")
    p.add_argument("--mask-file", help="file con righe 'min max' di canali da mascherare")
    p.add_argument("--mask-chans", default="", help="lista di canali, es. 110,121-127,185")
    p.add_argument("--no-auto-mask", action="store_true", help="disattiva la maschera automatica")
    p.add_argument("--chan-z", type=float, default=d.chan_z, help=f"soglia della maschera automatica (default {d.chan_z})")
    p.add_argument("--stat-window", type=float, default=d.stat_window, help=f"finestra delle statistiche [s] (default {d.stat_window})")
    p.add_argument("--norm-span", type=int, default=d.norm_span, help=f"finestre nella mediana mobile (default {d.norm_span})")
    p.add_argument("--cell", type=float, default=d.cell, help=f"durata delle celle [s] (default {d.cell})")
    p.add_argument("--cell-sigma", type=float, default=d.cell_sigma, help=f"soglia sulle celle (default {d.cell_sigma})")
    p.add_argument("--row-frac", type=float, default=d.row_frac, help=f"frazione di canali per sostituire l'intervallo (default {d.row_frac})")
    p.add_argument("--sample-clip", type=float, default=None, help="sostituisce i campioni con |z| oltre questa soglia (default: no)")
    p.add_argument("--tmax", type=float, default=None, help="elabora solo i primi TMAX secondi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve() == Path(a.fil).resolve():
        sys.exit("ERRORE: l'output coincide con l'input")
    chans = parse_chan_list(a.mask_chans)
    if a.mask_file:
        chans = sorted(set(chans) | set(read_flag_file(a.mask_file)))
    cfg = CleanConfig(stat_window=a.stat_window, norm_span=a.norm_span, auto_mask=not a.no_auto_mask,
                      chan_z=a.chan_z, mask_chans=chans, cell=a.cell, cell_sigma=a.cell_sigma,
                      row_frac=a.row_frac, sample_clip=a.sample_clip, seed=a.seed)
    run_clean(a.fil, a.out, cfg, verbose=not a.quiet, tmax=a.tmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
