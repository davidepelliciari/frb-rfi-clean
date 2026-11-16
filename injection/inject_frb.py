#!/usr/bin/env python3
"""
inject_frb.py - inietta FRB sintetici in un file SIGPROC filterbank, in UN SOLO PASSAGGIO.

Il file viene letto e scritto a blocchi, quindi la memoria non dipende dalla
dimensione del filterbank e tutti gli impulsi vengono iniettati con una sola copia.

Modello del segnale (per ogni canale i):
  - profilo gaussiano con FWHM = WIDTH, centrato in ToA + ritardo dispersivo d_i;
    il ToA e' riferito al centro del canale a frequenza piu' alta (come in FRBfaker);
  - smearing intra-canale (opzionale, default attivo): convoluzione con un boxcar
    centrato di durata 8.3e3 * DM * |foff| / f_i^3 s;
  - integrazione esatta sul campione (forma chiusa di gaussiana * boxcar);
  - spettro: indice spettrale alpha e/o inviluppo gaussiano in frequenza (FRB a banda parziale);
  - ampiezza in ogni canale proporzionale al rumore di quel canale (sigma_i), cioe'
    al guadagno: un segnale astrofisico reale segue il bandpass. I canali morti
    (sigma_i = 0) non ricevono segnale.

Definizione di S/N (equivalente a 'box_conv' di FRBfaker, ma con rumore robusto):
  S/N = massimo, sulle larghezze di boxcar, della serie temporale ideale (senza rumore)
  dedispersa al DM vero con shift interi e sommata sui canali senza pesi (come fa
  Heimdall), divisa per sigma_TS * sqrt(W), con sigma_TS = sqrt(sum_i sigma_i^2).
  sigma_i e' stimata con la MAD in una finestra di dati attorno al burst, quindi e'
  robusta agli RFI (FRBfaker usa la deviazione standard, che gli RFI gonfiano).

Calibrazione su Heimdall (--snr-calib): come per FRBfaker, si inietta in un filterbank
finto di rumore noto (tools/make_noise_fil.py), si misura con Heimdall
S/N_heimdall = pendenza * S/N_modello + offset e si passa la coppia all'iniettore.

Dati interi: arrotondamento stocastico (dithering) del segnale, che conserva il
segnale in media anche quando l'ampiezza per canale e' << 1 conteggio, e
saturazione esplicita ai limiti del tipo (i campioni saturati vengono contati).

Formato .params (una riga per burst, separatori spazi):
  ToA[s] SNR WIDTH_FWHM[s] DM [ALPHA [F_CENTER_MHz F_FWHM_MHz]]

Esempio:
  python3 inject_frb.py --fil obs.fil --params camp/params/SyntheticFRB.params \\
      --out obs_inj.fil --truth obs_inj_truth.csv --seed 1
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from fil_cut import Fil  # noqa: E402

KDM = 4.148808e3          # costante di dispersione [s MHz^2 pc^-1 cm^3]
KSMEAR = 8.3e3            # smearing intra-canale: KSMEAR * DM * df / f^3 [s], f e df in MHz
FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_SQRT2PI = np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------- burst
@dataclass
class Burst:
    toa: float              # s, al centro del canale piu' alto in frequenza
    snr: float              # S/N richiesto
    width: float            # FWHM [s]
    dm: float               # pc cm^-3
    alpha: float = 0.0      # indice spettrale
    f_center: float = np.nan  # MHz, inviluppo gaussiano in frequenza (nan = nessuno)
    f_fwhm: float = np.nan    # MHz
    # risultati
    amplitude: float = np.nan
    snr_model: float = np.nan
    best_width_samples: int = 0
    n_clipped: int = 0
    signal_sum: float = 0.0
    first_sample: int = 0
    last_sample: int = 0
    _tmpl: object = field(default=None, repr=False)


def read_params(path: Path) -> list[Burst]:
    bursts = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.split("#")[0].strip()
            if not line:
                continue
            v = [float(x) for x in line.split()]
            if len(v) not in (4, 5, 7):
                raise ValueError(f"{path}:{ln}: attese 4, 5 o 7 colonne, trovate {len(v)}")
            bursts.append(Burst(*v))
    return bursts


# --------------------------------------------------------------------------- profilo
def _cdf_gauss_box(x: np.ndarray, sig: float, tau: np.ndarray) -> np.ndarray:
    """CDF della convoluzione di una gaussiana (sigma sig) con un boxcar centrato di durata tau."""
    tau = np.broadcast_to(tau, x.shape)
    out = ndtr(x / sig)
    m = tau > 1e-3 * sig
    if np.any(m):
        def G(y):  # primitiva di Phi(y/sig)
            z = y / sig
            return y * ndtr(z) + sig * np.exp(-0.5 * z * z) / _SQRT2PI
        xm, tm = x[m], tau[m]
        out[m] = (G(xm + tm / 2) - G(xm - tm / 2)) / tm
    return out


class Injector:
    def __init__(self, fil: Fil, smear: bool = True, noise_window: int = 4096,
                 seed: int | None = None, snr_calib: tuple[float, float] | None = None):
        h = fil.h
        if h.get("nifs", 1) != 1:
            raise ValueError("supportato solo nifs = 1")
        if h["nbits"] not in (8, 16, 32):
            raise ValueError(f"nbits = {h['nbits']} non supportato (8, 16, 32)")
        self.fil = fil
        self.nchan = h["nchans"]
        self.tsamp = h["tsamp"]
        self.nbits = h["nbits"]
        self.foff = h["foff"]
        self.freqs = h["fch1"] + self.foff * np.arange(self.nchan)
        self.f_ref = self.freqs.max()
        self.dtype = {8: np.uint8, 16: np.uint16, 32: np.float32}[self.nbits]
        self.is_int = self.nbits != 32
        if self.is_int:
            info = np.iinfo(self.dtype)
            self.vmin, self.vmax = info.min, info.max
        self.smear = smear
        self.noise_window = noise_window
        self.seed = seed if seed is not None else int(np.random.SeedSequence().entropy % 2**32)
        self.snr_calib = snr_calib  # (pendenza, offset): S/N_heimdall = pendenza * S/N_modello + offset

    # ------------------------------------------------------------------ dati
    def _read(self, fh, start: int, n: int) -> np.ndarray:
        start = max(0, start)
        n = max(0, min(n, self.fil.nsamp - start))
        fh.seek(self.fil.hdr_len + start * self.fil.bytes_per_samp)
        a = np.fromfile(fh, dtype=self.dtype, count=n * self.nchan)
        return a.reshape(-1, self.nchan)

    def channel_sigma(self, fh, center_sample: int) -> np.ndarray:
        """Rumore robusto per canale (1.4826 * MAD) in una finestra attorno al campione dato."""
        n = self.noise_window
        start = min(max(0, center_sample - n // 2), max(0, self.fil.nsamp - n))
        d = self._read(fh, start, n).astype(np.float32)
        med = np.median(d, axis=0)
        sig = 1.4826 * np.median(np.abs(d - med), axis=0)
        # con dati interi e poco rumore la MAD puo' valere 0: ripiego su IQR e std clippata
        bad = sig == 0
        if np.any(bad):
            q75, q25 = np.percentile(d[:, bad], [75, 25], axis=0)
            sig[bad] = (q75 - q25) / 1.349
            still = bad.copy()
            still[bad] = sig[bad] == 0
            if np.any(still):
                x = d[:, still]
                m, s = x.mean(0), x.std(0)
                clip = np.abs(x - m) < 5 * s + 1e-12
                sig[still] = np.array([x[clip[:, j], j].std() if clip[:, j].any() else 0.0
                                       for j in range(x.shape[1])])
        return sig.astype(np.float64)

    # ------------------------------------------------------------------ template
    def build_template(self, b: Burst, sigma: np.ndarray, rng: np.random.Generator):
        ts = self.tsamp
        sig_t, delay, tau, center, k0, L = self._geometry(b)
        edges = (k0[:, None] + np.arange(L + 1)[None, :]) * ts - center[:, None]
        cdf = _cdf_gauss_box(edges, sig_t, np.broadcast_to(tau[:, None], edges.shape))
        prof = np.diff(cdf, axis=1)                                    # massa per campione, somma ~1

        # spettro
        w = (self.freqs / self.f_ref) ** b.alpha
        if np.isfinite(b.f_center) and np.isfinite(b.f_fwhm):
            w = w * np.exp(-0.5 * ((self.freqs - b.f_center) / (b.f_fwhm * FWHM2SIG)) ** 2)
        live = sigma > 0
        if not live.any():
            raise ValueError(f"burst a {b.toa:.3f} s: tutti i canali hanno rumore nullo")
        w = w / w[live].mean()
        unit = prof * (w * sigma)[:, None]                             # template per ampiezza 1

        # S/N ideale: dedispersione con shift interi (come Heimdall), somma, boxcar migliore
        shift = np.round(delay / ts).astype(np.int64)
        pos = (k0 - shift)[:, None] + np.arange(L)[None, :]
        p0 = pos.min()
        tseries = np.zeros(pos.max() - p0 + 1)
        np.add.at(tseries, (pos - p0).ravel(), unit.ravel())
        cs = np.concatenate([[0.0], np.cumsum(tseries)])
        best, bestw = 0.0, 1
        for W in range(1, len(tseries) + 1):
            v = (cs[W:] - cs[:-W]).max() / np.sqrt(W)
            if v > best:
                best, bestw = v, W
        sigma_ts = np.sqrt(np.sum(sigma ** 2))
        snr_unit = best / sigma_ts

        if self.snr_calib:
            slope, offset = self.snr_calib
            target = (b.snr - offset) / slope   # S/N di modello che Heimdall vedra' come b.snr
        else:
            target = b.snr
        A = target / snr_unit
        tmpl = unit * A
        if self.is_int:
            tmpl = np.floor(tmpl + rng.random(tmpl.shape))            # arrotondamento stocastico
        b.amplitude = A
        b.snr_model = target
        b.best_width_samples = bestw
        b.signal_sum = float(tmpl.sum())
        b.first_sample = int(k0.min())
        b.last_sample = int(k0.max() + L)
        b._tmpl = (tmpl.astype(np.float32), k0, L)

    # ------------------------------------------------------------------ iniezione
    @staticmethod
    def _add(chunk: np.ndarray, c0: int, b: Burst, is_int: bool, vmin=None, vmax=None) -> int:
        tmpl, k0, L = b._tmpl
        nch = chunk.shape[0]
        cols = k0[:, None] + np.arange(L)[None, :] - c0               # (nchan, L) indici nel blocco
        m = (cols >= 0) & (cols < nch) & (tmpl != 0)
        if not m.any():
            return 0
        rows = np.broadcast_to(np.arange(tmpl.shape[0])[:, None], cols.shape)[m]
        t = cols[m]
        val = chunk[t, rows].astype(np.float64) + tmpl[m]
        n_clip = 0
        if is_int:
            n_clip = int(np.count_nonzero((val < vmin) | (val > vmax)))
            val = np.clip(val, vmin, vmax)
        chunk[t, rows] = val
        return n_clip

    def run(self, bursts: list[Burst], out: Path, chunk_mb: float = 256.0, verbose: bool = True):
        bursts = sorted(bursts, key=lambda b: b.toa)
        for i, b in enumerate(bursts):
            if not 0 <= b.toa < self.fil.duration:
                raise ValueError(f"burst {i}: ToA {b.toa} s fuori dal file ({self.fil.duration:.2f} s)")
        seq = np.random.SeedSequence(self.seed)
        rngs = [np.random.default_rng(s) for s in seq.spawn(len(bursts))]

        nper = max(1, int(chunk_mb * 2**20 // self.fil.bytes_per_samp))
        pending = list(range(len(bursts)))     # ordinati per ToA
        active: list[int] = []
        with open(self.fil.path, "rb") as fin, open(self.fil.path, "rb") as fnoise, open(out, "wb") as fout:
            fin.seek(0)
            fout.write(fin.read(self.fil.hdr_len))
            c0 = 0
            while c0 < self.fil.nsamp:
                n = min(nper, self.fil.nsamp - c0)
                chunk = self._read(fin, c0, n)
                c1 = c0 + n
                # attiva i burst che possono iniziare prima della fine del blocco
                while pending:
                    b = bursts[pending[0]]
                    if self._first_sample(b) >= c1:
                        break
                    j = pending.pop(0)
                    sig = self.channel_sigma(fnoise, int(b.toa / self.tsamp))
                    self.build_template(b, sig, rngs[j])
                    active.append(j)
                    if verbose:
                        print(f"  burst {j + 1}/{len(bursts)}: ToA={b.toa:.4f}s DM={b.dm:g} "
                              f"W={b.width * 1e3:.3g}ms S/N={b.snr:g} -> ampiezza {b.amplitude:.4g}", flush=True)
                for j in list(active):
                    b = bursts[j]
                    b.n_clipped += self._add(chunk, c0, b, self.is_int,
                                             getattr(self, "vmin", None), getattr(self, "vmax", None))
                    if b.last_sample <= c1:
                        b._tmpl = None
                        active.remove(j)
                fout.write(np.ascontiguousarray(chunk, dtype=self.dtype).tobytes())
                c0 = c1
        return bursts

    def _geometry(self, b: Burst):
        """Ritardi, smearing, centri, primo campione per canale e lunghezza del template."""
        ts = self.tsamp
        sig_t = b.width * FWHM2SIG
        delay = KDM * b.dm * (self.freqs ** -2 - self.f_ref ** -2)
        tau = KSMEAR * b.dm * abs(self.foff) / self.freqs ** 3 if self.smear else np.zeros(self.nchan)
        center = b.toa + delay
        half = 5 * sig_t + tau / 2 + 2 * ts
        L = int(np.ceil(2 * half.max() / ts)) + 2
        k0 = np.floor((center - half) / ts).astype(np.int64)
        return sig_t, delay, tau, center, k0, L

    def _first_sample(self, b: Burst) -> int:
        return int(self._geometry(b)[4].min())


def write_truth(path: Path, bursts: list[Burst], seed: int, src: Path, out: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["toa_s", "snr_requested", "snr_model", "width_fwhm_s", "dm", "alpha",
                    "f_center_mhz", "f_fwhm_mhz", "amplitude", "best_boxcar_samples",
                    "signal_sum", "n_clipped", "seed", "source_fil", "injected_fil"])
        for b in bursts:
            w.writerow([f"{b.toa:.6f}", f"{b.snr:.3f}", f"{b.snr_model:.3f}", f"{b.width:.6g}",
                        f"{b.dm:.4f}", f"{b.alpha:g}", f"{b.f_center:g}", f"{b.f_fwhm:g}",
                        f"{b.amplitude:.6g}", b.best_width_samples, f"{b.signal_sum:.6g}",
                        b.n_clipped, seed, str(src), str(out)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fil", required=True, type=Path)
    p.add_argument("--params", required=True, type=Path, nargs="+", help="uno o piu' file .params")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--truth", type=Path, help="CSV con i parametri iniettati (default OUT.truth.csv)")
    p.add_argument("--no-smear", action="store_true", help="disattiva lo smearing intra-canale")
    p.add_argument("--noise-window", type=int, default=4096, help="campioni per stimare il rumore (default 4096)")
    p.add_argument("--chunk-mb", type=float, default=256, help="dimensione dei blocchi in MB (default 256)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--snr-calib", type=lambda x: tuple(float(v) for v in x.split(",")), default=None,
                   metavar="PENDENZA,OFFSET",
                   help="calibrazione su Heimdall misurata su dati finti: S/N_heimdall = PENDENZA*S/N_modello + OFFSET. "
                        "Con questa opzione SNR nei .params e' interpretato come S/N atteso da Heimdall. "
                        "NB: la vecchia 0.84,0.52 vale per FRBfaker, non per questo iniettore")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    if a.out.resolve() == a.fil.resolve():
        print("ERRORE: --out coincide con --fil", file=sys.stderr)
        return 2
    fil = Fil(str(a.fil))
    bursts = [b for pf in a.params for b in read_params(pf)]
    inj = Injector(fil, smear=not a.no_smear, noise_window=a.noise_window, seed=a.seed,
                   snr_calib=a.snr_calib)
    if not a.quiet:
        print(f"Iniezione di {len(bursts)} burst in {a.fil} -> {a.out} (seed {inj.seed})")
    inj.run(bursts, a.out, chunk_mb=a.chunk_mb, verbose=not a.quiet)
    truth = a.truth or a.out.with_suffix(".truth.csv")
    write_truth(truth, bursts, inj.seed, a.fil, a.out)
    clipped = sum(b.n_clipped for b in bursts)
    if not a.quiet:
        print(f"Fatto. Catalogo: {truth}")
    if clipped:
        print(f"ATTENZIONE: {clipped} campioni saturati ai limiti a {inj.nbits} bit", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
