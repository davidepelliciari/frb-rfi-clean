import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from frbclean.sigproc import write_header  # noqa: E402
from frbclean.periodic import PeriodicConfig, run_periodic  # noqa: E402
from test_inject_frb import load, measure_snr  # noqa: E402

TS, NCH, DUR = 64e-6, 64, 20.0


def _make(path, f=None, signal=None, seed=0):
    rng = np.random.default_rng(seed)
    n = int(DUR / TS)
    items = [("HEADER_START", None), ("nchans", NCH), ("nbits", 8), ("nifs", 1), ("tsamp", TS), ("fch1", 1446.0),
             ("foff", -1.0), ("tstart", 60000.0), ("data_type", 1), ("HEADER_END", None)]
    t = (np.arange(n) + 0.5) * TS
    x = 128 + 8 * rng.standard_normal((n, NCH)).astype(np.float32)
    if f:
        ph = 2 * np.pi * f * t
        w = np.sin(ph) + 0.6 * np.sin(2 * ph + 1) + 0.8 * np.sin(3 * ph + 2)
        x += w[:, None] * (8 * np.linspace(0.1, 0.4, NCH))[None, :]
    if signal is not None:
        x += signal
    with open(path, "wb") as fo:
        fo.write(write_header(items))
        fo.write(np.clip(np.rint(x), 0, 255).astype(np.uint8).tobytes())


def _burst(t0, snr_amp, width, dm):
    """Segnale disperso semplice (gaussiano), uguale per i file con e senza modulazione."""
    n = int(DUR / TS)
    freqs = 1446.0 - np.arange(NCH)
    delay = 4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2)
    t = (np.arange(n) + 0.5) * TS
    s = np.zeros((n, NCH), np.float32)
    sig = width / 2.3548
    for c in range(NCH):
        m = np.abs(t - t0 - delay[c]) < 5 * sig
        s[m, c] = snr_amp * np.exp(-0.5 * ((t[m] - t0 - delay[c]) / sig) ** 2)
    return s


def _line_ratio(d, f):
    s = d.sum(1) - d.sum(1).mean()
    P = np.abs(np.fft.rfft(s)) ** 2
    F = np.fft.rfftfreq(s.size, TS)
    i = np.argmin(abs(F - f))
    return P[i - 3:i + 4].max() / np.median(P[(F > f - 5) & (F < f + 5)])


def test_lines_removed_and_frbs_kept(tmp_path):
    sig = _burst(5.0, 3.0, 0.002, 100) + _burst(12.0, 1.0, 0.02, 30)
    _make(tmp_path / "ref.fil", None, sig, seed=1)
    _make(tmp_path / "mod.fil", 50.0003, sig, seed=1)
    summ = run_periodic(tmp_path / "mod.fil", tmp_path / "out.fil", PeriodicConfig(segment=5.0), verbose=False)
    fr, dr = load(tmp_path / "ref.fil")
    fo, do = load(tmp_path / "out.fil")
    _, dm = load(tmp_path / "mod.fil")
    for f in (50, 100, 150):
        assert _line_ratio(dm, f) > 1000
        assert _line_ratio(do, f) < 50
    assert summ["frac_var_removed_median"] > 0.01
    for t0, dm_, w in ((5.0, 100, 0.002), (12.0, 30, 0.02)):
        W = max(1, int(w / TS))
        a = measure_snr(dr, fr, t0, dm_, W)
        b = measure_snr(do, fo, t0, dm_, W)
        assert 0.93 < b / a < 1.07, (t0, a, b)


def test_no_periodicity_no_change(tmp_path):
    _make(tmp_path / "n.fil", None, _burst(8.0, 2.0, 0.005, 50), seed=2)
    summ = run_periodic(tmp_path / "n.fil", tmp_path / "o.fil", PeriodicConfig(segment=5.0, search=True), verbose=False)
    assert summ["frac_var_removed_median"] < 0.003
    f1, d1 = load(tmp_path / "n.fil")
    f2, d2 = load(tmp_path / "o.fil")
    W = int(0.005 / TS)
    assert abs(measure_snr(d2, f2, 8.0, 50, W) / measure_snr(d1, f1, 8.0, 50, W) - 1) < 0.03
    assert abs(d2.std() - d1.std()) < 0.02
