#!/usr/bin/env python3
"""
make_noise_fil.py - crea un filterbank di solo rumore gaussiano, con rumore noto,
usando l'header di un file reale (stessi canali, banda, tsamp, nbits).

Serve per calibrare l'S/N dell'iniettore su Heimdall: rumore noto e nessun RFI.

  # rumore bianco con bandpass piatto (media e sigma in conteggi)
  python3 make_noise_fil.py --like obs.fil --out noise.fil --duration 300 --mean 128 --sigma 12

  # bandpass e sigma per canale misurati (robustamente) dal file reale
  python3 make_noise_fil.py --like obs.fil --out noise.fil --duration 300 --bandpass-from-data
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fil_cut import Fil, write_header  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--like", required=True, help="filterbank reale da cui copiare l'header")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, required=True, help="durata [s]")
    p.add_argument("--mean", type=float, default=None, help="media (default: centro del range a nbits)")
    p.add_argument("--sigma", type=float, default=None, help="sigma (default: 1/16 del range a nbits)")
    p.add_argument("--bandpass-from-data", action="store_true",
                   help="media e sigma per canale stimate (mediana, MAD) da 8192 campioni del file reale")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    fil = Fil(a.like)
    h, nchan, nbits = fil.h, fil.h["nchans"], fil.h["nbits"]
    if nbits not in (8, 16, 32):
        sys.exit("nbits supportati: 8, 16, 32")
    dtype = {8: np.uint8, 16: np.uint16, 32: np.float32}[nbits]
    rng = np.random.default_rng(a.seed)

    if a.bandpass_from_data:
        n = min(8192, fil.nsamp)
        start = max(0, fil.nsamp // 2 - n // 2)
        with open(a.like, "rb") as f:
            f.seek(fil.hdr_len + start * fil.bytes_per_samp)
            d = np.fromfile(f, dtype=dtype, count=n * nchan).reshape(-1, nchan).astype(np.float64)
        mean = np.median(d, axis=0)
        sigma = 1.4826 * np.median(np.abs(d - mean), axis=0)
    else:
        full = 2.0 ** nbits if nbits < 32 else 256.0
        mean = np.full(nchan, a.mean if a.mean is not None else full / 2)
        sigma = np.full(nchan, a.sigma if a.sigma is not None else full / 16)

    nsamp = int(round(a.duration / h["tsamp"]))
    items = [(k, (nsamp if k == "nsamples" else v)) for k, v in fil.items]
    step = max(1, (64 * 2**20) // fil.bytes_per_samp)
    with open(a.out, "wb") as f:
        f.write(write_header(items))
        for s in range(0, nsamp, step):
            k = min(step, nsamp - s)
            x = mean + rng.normal(size=(k, nchan)) * sigma
            if nbits < 32:
                x = np.clip(np.round(x), 0, 2**nbits - 1)
            f.write(x.astype(dtype).tobytes())
    np.savetxt(Path(a.out).with_suffix(".bandpass.txt"), np.column_stack([mean, sigma]),
               header="mean sigma (per canale, nell'ordine del file)")
    print(f"{a.out}: {nsamp} campioni, {nchan} canali, {nbits} bit")


if __name__ == "__main__":
    sys.exit(main())
