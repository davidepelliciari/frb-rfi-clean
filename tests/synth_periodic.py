"""Filterbank sintetico con RFI periodico di due famiglie (50 Hz e 166.67 Hz), una pulsar e burst."""
import numpy as np

from frbclean.sigproc import write_header

KDM = 4.148808e3


def make(path, ts=50e-6, nch=128, dur=30.0, f50=True, f166=True, pulsar=None, bursts=(), seed=0, sigma=8.0):
    """pulsar: (P [s], DM, larghezza [s], ampiezza per canale [sigma]); bursts: (t0, DM, larghezza, ampiezza)."""
    rng = np.random.default_rng(seed)
    n = int(dur / ts)
    fch1, foff = 1446.0, -64.0 / nch
    freqs = fch1 + foff * np.arange(nch)
    items = [("HEADER_START", None), ("nchans", nch), ("nbits", 8), ("nifs", 1), ("tsamp", ts), ("fch1", fch1),
             ("foff", foff), ("tstart", 60000.0), ("data_type", 1), ("HEADER_END", None)]
    amp50 = sigma * np.linspace(0.1, 0.3, nch)
    amp166 = sigma * np.linspace(1.0, 0.5, nch)
    delays = lambda dm: KDM * dm * (freqs ** -2 - freqs.max() ** -2)
    with open(path, "wb") as fo:
        fo.write(write_header(items))
        for s0 in range(0, n, 100000):
            k = min(100000, n - s0)
            t = (s0 + np.arange(k) + 0.5) * ts
            x = 128 + sigma * rng.standard_normal((k, nch)).astype(np.float32)
            if f50:
                ph = 2 * np.pi * 50.0 * t
                x += (np.sin(ph) + 0.6 * np.sin(2 * ph + 1) + 0.8 * np.sin(3 * ph + 2))[:, None] * amp50[None, :]
            if f166:
                idx = (s0 + np.arange(k)) % int(round(0.006 / ts)) == 0     # un campione ogni 6 ms
                x[idx] += amp166[None, :]
            if pulsar:
                P, dm, w, a = pulsar
                d = delays(dm)
                for c in range(nch):
                    ph = np.mod(t - d[c], P)
                    ph = np.minimum(ph, P - ph)
                    x[:, c] += a * sigma * np.exp(-0.5 * (ph / (w / 2.3548)) ** 2)
            for (tb, dm, w, a) in bursts:
                d = delays(dm)
                for c in range(nch):
                    m = np.abs(t - tb - d[c]) < 5 * w
                    if m.any():
                        x[m, c] += a * sigma * np.exp(-0.5 * ((t[m] - tb - d[c]) / (w / 2.3548)) ** 2)
            fo.write(np.clip(np.rint(x), 0, 255).astype(np.uint8).tobytes())
    return freqs
