"""Filterbank sintetico con le patologie trovate in ag278m_mc_no0004 (versione ridotta per i test)."""
import numpy as np

from frbclean.sigproc import write_header

TS = 128e-6
NCH = 64
FCH1 = 1445.5          # 64 canali da 1 MHz: 1382-1446 MHz
FOFF = -1.0
EDGE = np.r_[0:8, 56:64]               # bordi della banda (>1438 e <1390 MHz)
BURST_CH = np.arange(28, 32)           # 1414-1418 MHz
IMPULSIVE_CH = 30
GAIN = [(10.0, 10.9, 0.74, 0.94), (10.9, 11.9, 0.93, 0.99)]   # (t0, t1, bordi, centro)
VAR_BURST = (20.0, 20.25, 20.0)        # (t0, t1, fattore sul rumore) nei canali BURST_CH
GLITCH = (30.0, 30.1, 3.0)             # varianza x3 in tutti i canali
RADAR_T = [3.6, 13.6, 23.6, 33.6]


def make(path, dur=40.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur / TS)
    sig = 22 + 6 * np.sin(np.linspace(0, 3, NCH))
    mu = 130 + 8 * np.cos(np.linspace(0, 4, NCH))
    items = [("HEADER_START", None), ("source_name", "SYNTH_MC"), ("telescope_id", 0), ("machine_id", 0),
             ("data_type", 1), ("fch1", FCH1), ("foff", FOFF), ("nchans", NCH), ("nbits", 8), ("nifs", 1),
             ("tstart", 61318.0), ("tsamp", TS), ("HEADER_END", None)]
    with open(path, "wb") as f:
        f.write(write_header(items))
        step = 50000
        for s in range(0, n, step):
            k = min(step, n - s)
            t = (s + np.arange(k)) * TS
            noise = rng.standard_normal((k, NCH)).astype(np.float32) * sig
            # lampo di rumore e glitch
            m = (t >= VAR_BURST[0]) & (t < VAR_BURST[1])
            noise[np.ix_(m, BURST_CH)] *= VAR_BURST[2]
            m = (t >= GLITCH[0]) & (t < GLITCH[1])
            noise[m] *= GLITCH[2]
            # canale impulsivo
            sp = rng.random(k) < 2e-3
            noise[sp, IMPULSIVE_CH] += 6 * sig[IMPULSIVE_CH]
            # radar: impulsi a banda intera da 0.5 ms
            for tr in RADAR_T:
                m = np.abs(t - tr) < 2.5e-4
                noise[m] += 5 * sig
            x = mu + noise
            # salti di guadagno (moltiplicativi su livello e rumore)
            for t0, t1, ge, gm in GAIN:
                m = (t >= t0) & (t < t1)
                g = np.full(NCH, gm, np.float32)
                g[EDGE] = ge
                x[m] *= g
            f.write(np.clip(np.rint(x), 0, 255).astype(np.uint8).tobytes())
