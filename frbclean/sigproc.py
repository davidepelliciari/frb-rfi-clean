"""
Lettura e scrittura di file SIGPROC filterbank a blocchi, per qualsiasi nbits (1, 2, 4, 8, 16, 32).

I dati sono restituiti come array (n_campioni, n_canali) nell'ordine dei canali del file.
Per nbits < 8 si segue la convenzione SIGPROC: il primo campione nei bit meno significativi.
"""
from __future__ import annotations

import os
import struct
from typing import Iterator

import numpy as np

INT_KEYS = {"telescope_id", "machine_id", "data_type", "barycentric", "pulsarcentric",
            "nbits", "nsamples", "nchans", "nifs", "nbeams", "ibeam"}
DOUBLE_KEYS = {"az_start", "za_start", "src_raj", "src_dej", "tstart", "tsamp",
               "fch1", "foff", "refdm", "period", "fchannel"}
STR_KEYS = {"rawdatafile", "source_name"}
CHAR_KEYS = {"signed"}
FLAG_KEYS = {"HEADER_START", "HEADER_END", "FREQUENCY_START", "FREQUENCY_END"}


def _read_str(f) -> str:
    raw = f.read(4)
    if len(raw) < 4:
        raise ValueError("fine file inattesa nell'header")
    (n,) = struct.unpack("<i", raw)
    if not 0 < n < 256:
        raise ValueError(f"lunghezza stringa non valida ({n}): non e' un filterbank SIGPROC?")
    return f.read(n).decode("ascii", errors="replace")


def read_header(path) -> tuple[list[tuple[str, object]], int]:
    items: list[tuple[str, object]] = []
    with open(path, "rb") as f:
        if _read_str(f) != "HEADER_START":
            raise ValueError(f"{path}: manca HEADER_START")
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
                raise ValueError(f"chiave header sconosciuta: {key!r}")
        return items, f.tell()


def write_header(items) -> bytes:
    def s(x: str) -> bytes:
        b = x.encode("ascii")
        return struct.pack("<i", len(b)) + b
    out = b""
    for key, val in items:
        out += s(key)
        if key in FLAG_KEYS:
            continue
        if key in INT_KEYS:
            out += struct.pack("<i", int(val))
        elif key in DOUBLE_KEYS:
            out += struct.pack("<d", float(val))
        elif key in STR_KEYS:
            out += s(val)
        elif key in CHAR_KEYS:
            out += struct.pack("<b", int(val))
    return out


def unpack_bits(raw: np.ndarray, nbits: int) -> np.ndarray:
    """Spacchetta byte con campioni da 1, 2 o 4 bit (primo campione nei bit bassi)."""
    per = 8 // nbits
    mask = (1 << nbits) - 1
    shifts = (np.arange(per, dtype=np.uint8) * nbits).astype(np.uint8)
    return ((raw[:, None] >> shifts) & mask).astype(np.uint8).reshape(-1)


def pack_bits(vals: np.ndarray, nbits: int) -> np.ndarray:
    per = 8 // nbits
    v = vals.astype(np.uint8).reshape(-1, per)
    shifts = (np.arange(per, dtype=np.uint8) * nbits).astype(np.uint8)
    return np.bitwise_or.reduce(v << shifts, axis=1).astype(np.uint8)


class Filterbank:
    """Accesso a blocchi a un filterbank SIGPROC (nifs = 1)."""

    def __init__(self, path):
        self.path = str(path)
        self.items, self.hdr_len = read_header(self.path)
        self.header = {k: v for k, v in self.items if v is not None}
        h = self.header
        self.nchans = int(h["nchans"])
        self.nbits = int(h["nbits"])
        self.nifs = int(h.get("nifs", 1))
        if self.nifs != 1:
            raise NotImplementedError("supportato solo nifs = 1")
        if self.nbits not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"nbits = {self.nbits} non supportato")
        bits = self.nchans * self.nbits
        if bits % 8:
            raise ValueError("nchans * nbits non e' multiplo di 8")
        self.bytes_per_samp = bits // 8
        self.nsamp = (os.path.getsize(self.path) - self.hdr_len) // self.bytes_per_samp
        self.tsamp = float(h["tsamp"])
        self.duration = self.nsamp * self.tsamp
        self.fch1 = float(h["fch1"])
        self.foff = float(h["foff"])
        self.freqs = self.fch1 + self.foff * np.arange(self.nchans)
        self.bandwidth = abs(self.foff) * self.nchans
        self.f_lo = self.freqs.min() - abs(self.foff) / 2
        self.f_hi = self.freqs.max() + abs(self.foff) / 2
        signed = bool(h.get("signed", 0))
        self.dtype = {8: np.int8 if signed else np.uint8, 16: np.uint16, 32: np.float32}.get(self.nbits, np.uint8)

    def read(self, start: int, n: int) -> np.ndarray:
        """Legge n campioni da start; restituisce (n_letti, nchans) nel tipo nativo (uint8 per nbits < 8)."""
        start = max(0, int(start))
        n = max(0, min(int(n), self.nsamp - start))
        with open(self.path, "rb") as f:
            f.seek(self.hdr_len + start * self.bytes_per_samp)
            if self.nbits >= 8:
                a = np.fromfile(f, dtype=self.dtype, count=n * self.nchans)
            else:
                raw = np.fromfile(f, dtype=np.uint8, count=n * self.bytes_per_samp)
                a = unpack_bits(raw, self.nbits)
        return a.reshape(-1, self.nchans)

    def blocks(self, nper: int, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        stop = self.nsamp if stop is None else min(stop, self.nsamp)
        s = start
        while s < stop:
            n = min(nper, stop - s)
            yield s, self.read(s, n)
            s += n
