#!/usr/bin/env python3
"""
fil_cut.py - estrae porzioni di un file SIGPROC filterbank senza caricarlo in memoria.

Solo libreria standard di Python: si puo' copiare ed eseguire su qualunque macchina.

Esempi:
  # mostra l'header e la durata
  python3 fil_cut.py info obs.fil

  # estrae 60 s a partire da t = 300 s
  python3 fil_cut.py cut obs.fil cut_300s.fil --start 300 --duration 60

  # estrae N tagli da 30 s distribuiti uniformemente nell'osservazione
  python3 fil_cut.py sample obs.fil outdir/ --n 10 --duration 30

  # estrae una finestra centrata su un tempo (es. da un candidato heimdall/FETCH)
  python3 fil_cut.py cut obs.fil cand.fil --center 1234.56 --duration 4

Il tstart (MJD) dell'header viene aggiornato all'inizio di ogni taglio, e
nsamples (se presente) viene corretto.
"""
import argparse
import os
import struct
import sys

INT_KEYS = {"telescope_id", "machine_id", "data_type", "barycentric", "pulsarcentric",
            "nbits", "nsamples", "nchans", "nifs", "nbeams", "ibeam"}
DOUBLE_KEYS = {"az_start", "za_start", "src_raj", "src_dej", "tstart", "tsamp",
               "fch1", "foff", "refdm", "period", "fchannel"}
STR_KEYS = {"rawdatafile", "source_name"}
CHAR_KEYS = {"signed"}
FLAG_KEYS = {"HEADER_START", "HEADER_END", "FREQUENCY_START", "FREQUENCY_END"}

COPY_CHUNK = 64 * 1024 * 1024  # 64 MB


def _read_str(f):
    raw = f.read(4)
    if len(raw) < 4:
        raise ValueError("Fine file inattesa durante la lettura dell'header")
    (n,) = struct.unpack("<i", raw)
    if not 0 < n < 256:
        raise ValueError(f"Lunghezza stringa header non valida ({n}): file non SIGPROC?")
    return f.read(n).decode("ascii", errors="replace")


def read_header(path):
    """Restituisce (lista ordinata di (chiave, valore), dimensione header in byte)."""
    items = []
    with open(path, "rb") as f:
        if _read_str(f) != "HEADER_START":
            raise ValueError(f"{path}: manca HEADER_START, non e' un filterbank SIGPROC")
        items.append(("HEADER_START", None))
        while True:
            key = _read_str(f)
            if key in FLAG_KEYS:
                items.append((key, None))
                if key == "HEADER_END":
                    break
            elif key in INT_KEYS:
                items.append((key, struct.unpack("<i", f.read(4))[0]))
            elif key in DOUBLE_KEYS:
                items.append((key, struct.unpack("<d", f.read(8))[0]))
            elif key in STR_KEYS:
                items.append((key, _read_str(f)))
            elif key in CHAR_KEYS:
                items.append((key, struct.unpack("<b", f.read(1))[0]))
            else:
                raise ValueError(f"Chiave header sconosciuta: {key!r}")
        return items, f.tell()


def write_header(items):
    def s(x):
        b = x.encode("ascii")
        return struct.pack("<i", len(b)) + b
    out = b""
    for key, val in items:
        out += s(key)
        if key in FLAG_KEYS:
            continue
        if key in INT_KEYS:
            out += struct.pack("<i", val)
        elif key in DOUBLE_KEYS:
            out += struct.pack("<d", val)
        elif key in STR_KEYS:
            out += s(val)
        elif key in CHAR_KEYS:
            out += struct.pack("<b", val)
    return out


class Fil:
    def __init__(self, path):
        self.path = path
        self.items, self.hdr_len = read_header(path)
        self.h = {k: v for k, v in self.items if v is not None}
        nifs = self.h.get("nifs", 1)
        bits = self.h["nchans"] * nifs * self.h["nbits"]
        if bits % 8:
            raise ValueError("nchans*nifs*nbits non multiplo di 8: taglio a byte non supportato")
        self.bytes_per_samp = bits // 8
        data_bytes = os.path.getsize(path) - self.hdr_len
        self.nsamp = data_bytes // self.bytes_per_samp
        self.tsamp = self.h["tsamp"]
        self.duration = self.nsamp * self.tsamp

    def info(self):
        lines = [f"File:        {self.path}"]
        for k, v in self.items:
            if v is not None:
                lines.append(f"  {k:<14} {v}")
        bw = abs(self.h["foff"]) * self.h["nchans"]
        lines += [
            f"Header:      {self.hdr_len} byte",
            f"Campioni:    {self.nsamp}",
            f"Durata:      {self.duration:.3f} s",
            f"Banda:       {bw:.2f} MHz",
            f"Data rate:   {self.bytes_per_samp / self.tsamp / 1e6:.2f} MB/s "
            f"({self.bytes_per_samp / self.tsamp * 60 / 1e9:.2f} GB/min)",
        ]
        return "\n".join(lines)

    def cut(self, out, start_s, duration_s):
        s0 = max(0, int(round(start_s / self.tsamp)))
        n = int(round(duration_s / self.tsamp))
        s1 = min(self.nsamp, s0 + n)
        if s1 <= s0:
            raise ValueError(f"Intervallo vuoto: start={start_s}s oltre la fine ({self.duration:.2f}s)")
        items = []
        for k, v in self.items:
            if k == "tstart":
                v = v + s0 * self.tsamp / 86400.0
            elif k == "nsamples":
                v = s1 - s0
            items.append((k, v))
        remaining = (s1 - s0) * self.bytes_per_samp
        with open(self.path, "rb") as fin, open(out, "wb") as fout:
            fout.write(write_header(items))
            fin.seek(self.hdr_len + s0 * self.bytes_per_samp)
            while remaining:
                buf = fin.read(min(COPY_CHUNK, remaining))
                if not buf:
                    break
                fout.write(buf)
                remaining -= len(buf)
        return s0 * self.tsamp, (s1 - s0) * self.tsamp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="mostra header e dimensioni")
    pi.add_argument("fil")

    pc = sub.add_parser("cut", help="estrae un intervallo")
    pc.add_argument("fil")
    pc.add_argument("out")
    g = pc.add_mutually_exclusive_group(required=True)
    g.add_argument("--start", type=float, help="inizio [s]")
    g.add_argument("--center", type=float, help="centro della finestra [s]")
    pc.add_argument("--duration", type=float, required=True, help="durata [s]")

    ps = sub.add_parser("sample", help="estrae N tagli distribuiti uniformemente")
    ps.add_argument("fil")
    ps.add_argument("outdir")
    ps.add_argument("--n", type=int, default=10)
    ps.add_argument("--duration", type=float, default=30.0)

    a = p.parse_args(argv)
    fil = Fil(a.fil)

    if a.cmd == "info":
        print(fil.info())
    elif a.cmd == "cut":
        start = a.start if a.start is not None else a.center - a.duration / 2
        t0, dt = fil.cut(a.out, start, a.duration)
        print(f"{a.out}: t0={t0:.3f}s durata={dt:.3f}s")
    elif a.cmd == "sample":
        os.makedirs(a.outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(a.fil))[0]
        span = max(0.0, fil.duration - a.duration)
        for i in range(a.n):
            start = span * i / max(1, a.n - 1) if a.n > 1 else 0.0
            out = os.path.join(a.outdir, f"{base}_t{start:09.2f}s.fil")
            t0, dt = fil.cut(out, start, a.duration)
            print(f"{out}: t0={t0:.3f}s durata={dt:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
