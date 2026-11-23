"""Filterbank sintetico con rumore di modo comune (rosso, accoppiamento variabile per canale) e FRB."""
import numpy as np

from frbclean.sigproc import write_header

KDM = 4.148808e3


def red_series(n, ts, rng, alpha=1.0, fmin=0.2):
    """Serie gaussiana con spettro di potenza ~ 1/f^alpha (sopra fmin), varianza unitaria."""
    w = rng.standard_normal(n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, ts)
    F *= 1.0 / np.maximum(f, fmin) ** (alpha / 2)
    F[0] = 0
    x = np.fft.irfft(F, n)
    return x / x.std()


def make(path, ts=64e-6, nch=128, dur=20.0, cm_amp=0.12, cm_white=0.03, frbs=(), seed=0, sigma=8.0,
         masked=(60, 61)):
    """cm_amp: ampiezza (in sigma del canale) della parte rossa del modo comune; cm_white: parte bianca.
    frbs: (t0, DM, FWHM [s], ampiezza per canale e campione [sigma])."""
    rng = np.random.default_rng(seed)
    n = int(dur / ts)
    fch1, foff = 1446.0, -64.0 / nch
    freqs = fch1 + foff * np.arange(nch)
    cm = cm_amp * red_series(n, ts, rng) + cm_white * rng.standard_normal(n)
    g = 1.0 + 0.4 * np.sin(np.linspace(0, 3, nch))          # accoppiamento diverso per canale
    items = [("HEADER_START", None), ("nchans", nch), ("nbits", 8), ("nifs", 1), ("tsamp", ts), ("fch1", fch1),
             ("foff", foff), ("tstart", 60000.0), ("data_type", 1), ("HEADER_END", None)]
    with open(path, "wb") as fo:
        fo.write(write_header(items))
        for s0 in range(0, n, 100000):
            k = min(100000, n - s0)
            t = (s0 + np.arange(k) + 0.5) * ts
            x = rng.standard_normal((k, nch)).astype(np.float32) + (cm[s0:s0 + k, None] * g[None, :]).astype(np.float32)
            for (tb, dm, w, a) in frbs:
                dd = KDM * dm * (freqs ** -2 - freqs.max() ** -2)
                for c in range(nch):
                    mm = np.abs(t - tb - dd[c]) < 5 * w
                    if mm.any():
                        x[mm, c] += a * np.exp(-0.5 * ((t[mm] - tb - dd[c]) / (w / 2.3548)) ** 2)
            x = 128 + sigma * x
            x[:, list(masked)] = 128
            fo.write(np.clip(np.rint(x), 0, 255).astype(np.uint8).tobytes())
    return dict(gains=g, freqs=freqs)


def snr(data, freqs, ts, toa, dm, W, live, half=0.4):
    """S/N del burst: dedispersione a shift interi, boxcar W, rumore dalla stessa serie (MAD) su +-half s."""
    shift = np.round(KDM * dm * (freqs ** -2 - freqs.max() ** -2) / ts).astype(int)
    i0 = int(toa / ts)
    h = int(half / ts)
    lo, hi = i0 - h, i0 + h
    s = np.zeros(hi - lo)
    for c in np.flatnonzero(live):
        s += data[lo + shift[c]:hi + shift[c], c]
    cs = np.concatenate([[0], np.cumsum(s - np.median(s))])
    b = (cs[W:] - cs[:-W])
    off = np.r_[b[: h - 4 * W], b[h + 4 * W:]]
    off = off - np.median(off)
    noise = 1.4826 * np.median(np.abs(off))
    return (b[h - 2 * W:h + W].max() - np.median(b)) / noise
