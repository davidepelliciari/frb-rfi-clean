"""Filterbank sintetico (come output dello step 1: media 128, sigma 8) con impulsi RFI a DM 0 e FRB."""
import numpy as np

from frbclean.sigproc import write_header

KDM = 4.148808e3


def make(path, ts=64e-6, nch=128, dur=30.0, rfi=True, frbs=(), seed=0, sigma=8.0, masked=(60, 61), trains=False,
         plateaus=False):
    """frbs: (t0, DM, FWHM [s], ampiezza per canale e campione [sigma]).

    trains: treni di impulsi a banda intera come il radar A reale (3 gruppi a -30, 0, +55 ms, ciascuno di
    6 impulsi da 0.5 ms ogni 2.5 ms), ogni 5 s a partire da 2.6 s.
    plateaus: emissione a banda intera lunga 150 ms (+0.4 sigma) con un impulso forte (+3 sigma, 1 ms) al
    centro, ogni 5 s a partire da 4.1 s.
    """
    rng = np.random.default_rng(seed)
    n = int(dur / ts)
    fch1, foff = 1446.0, -64.0 / nch
    freqs = fch1 + foff * np.arange(nch)
    items = [("HEADER_START", None), ("nchans", nch), ("nbits", 8), ("nifs", 1), ("tsamp", ts), ("fch1", fch1),
             ("foff", foff), ("tstart", 60000.0), ("data_type", 1), ("HEADER_END", None)]
    d = lambda dm: KDM * dm * (freqs ** -2 - freqs.max() ** -2)
    rr = np.random.default_rng(seed + 100)
    narrow = [(rr.uniform(0.5, dur - 0.5), rr.integers(0, nch - 6)) for _ in range(int(3 * dur))]   # 3/s
    truth = dict(radarA=[], radarB=[], narrow=narrow)
    with open(path, "wb") as fo:
        fo.write(write_header(items))
        for s0 in range(0, n, 100000):
            k = min(100000, n - s0)
            t = (s0 + np.arange(k) + 0.5) * ts
            x = 128 + sigma * rng.standard_normal((k, nch)).astype(np.float32)
            if rfi:
                # radar A: banda intera, ogni 3 s, 2 impulsi da 0.5 ms a 38 ms di distanza
                for tp in np.arange(0.7, dur, 3.0):
                    for dt in (0.0, 0.038):
                        m = np.abs(t - tp - dt) < 2.5e-4
                        x[m] += 3.0 * sigma
                # radar B: meta' bassa della banda, ogni 4 s, larghezza 2 ms
                for tp in np.arange(1.9, dur, 4.0):
                    m = np.abs(t - tp) < 1e-3
                    x[np.ix_(m, np.arange(nch // 2, nch))] += 1.5 * sigma
                # impulsi stretti in pochi canali (6 canali, 0.2 ms)
                for tn, c in narrow:
                    m = np.abs(t - tn) < 1e-4
                    x[np.ix_(m, np.arange(c, c + 6))] += 5 * sigma
            if trains:
                for tp in np.arange(2.6, dur, 5.0):
                    for g0 in (-0.030, 0.0, 0.055):
                        for k in range(6):
                            m = np.abs(t - tp - g0 - 0.0025 * k) < 2.5e-4
                            x[m] += 3.0 * sigma
            if plateaus:
                for tp in np.arange(4.1, dur, 5.0):
                    x[np.abs(t - tp) < 0.075] += 0.4 * sigma
                    x[np.abs(t - tp) < 5e-4] += 3.0 * sigma
            for (tb, dm, w, a) in frbs:
                dd = d(dm)
                for c in range(nch):
                    m = np.abs(t - tb - dd[c]) < 5 * w
                    if m.any():
                        x[m, c] += a * sigma * np.exp(-0.5 * ((t[m] - tb - dd[c]) / (w / 2.3548)) ** 2)
            x[:, list(masked)] = 128
            fo.write(np.clip(np.rint(x), 0, 255).astype(np.uint8).tobytes())
    truth["radarA"] = list(np.arange(0.7, dur, 3.0))
    truth["radarB"] = list(np.arange(1.9, dur, 4.0))
    truth["trains"] = [tp + g0 + 0.0025 * k for tp in np.arange(2.6, dur, 5.0) for g0 in (-0.030, 0.0, 0.055)
                       for k in range(6)] if trains else []
    truth["plateaus"] = list(np.arange(4.1, dur, 5.0)) if plateaus else []
    return truth
