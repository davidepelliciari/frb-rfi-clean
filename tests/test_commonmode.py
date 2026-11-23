import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import synth_commonmode as sc  # noqa: E402
from frbclean.commonmode import CommonModeConfig, run_commonmode  # noqa: E402
from test_inject_frb import load  # noqa: E402

TS = 64e-6
FRBS = [(0.8, 50, 0.001, 0.35), (2.4, 100, 0.001, 0.35), (4.0, 300, 0.004, 0.2), (5.6, 700, 0.001, 0.35)]


def _snrs(path, live):
    fil, d = load(path)
    h = fil.h
    freqs = h["fch1"] + h["foff"] * np.arange(h["nchans"])
    return [sc.snr(d, freqs, TS, tb, dm, max(1, int(round(w / TS))), live) for tb, dm, w, a in FRBS]


def test_commonmode_restores_snr(tmp_path):
    sc.make(tmp_path / "in.fil", dur=6.5, frbs=FRBS)
    sc.make(tmp_path / "ref.fil", dur=6.5, frbs=FRBS, cm_amp=0, cm_white=0)
    s = run_commonmode(tmp_path / "in.fil", tmp_path / "out.fil", CommonModeConfig(), verbose=False)
    assert s["zerodm_excess_before_median"] > 3
    live = np.ones(128, bool)
    live[[60, 61]] = False
    ref, inp, out = _snrs(tmp_path / "ref.fil", live), _snrs(tmp_path / "in.fil", live), _snrs(tmp_path / "out.fil", live)
    for r, i, o in zip(ref, inp, out):
        assert o > 0.88 * r and o > 1.5 * i, (r, i, o)
    _, d = load(tmp_path / "out.fil")
    assert np.all(d[:, 60:62] == 128)
    # livello e scala dei canali conservati
    _, di = load(tmp_path / "in.fil")
    assert np.allclose(np.median(d[:, live], 0), np.median(di[:, live], 0), atol=1.0)


def test_white_noise_untouched_snr(tmp_path):
    sc.make(tmp_path / "w.fil", dur=6.5, frbs=FRBS, cm_amp=0, cm_white=0)
    run_commonmode(tmp_path / "w.fil", tmp_path / "o.fil", CommonModeConfig(), verbose=False)
    live = np.ones(128, bool)
    live[[60, 61]] = False
    a, b = _snrs(tmp_path / "w.fil", live), _snrs(tmp_path / "o.fil", live)
    for x, y in zip(a, b):
        assert y > 0.95 * x, (x, y)


def test_no_block_offsets_with_skewed_noise(tmp_path):
    """Con media != mediana nei canali (RFI positivo sparso, variabile nel tempo) la sottrazione non deve
    introdurre offset diversi per blocco e canale (gli scalini che la prima versione lasciava)."""
    from frbclean.sigproc import write_header
    ts, nch, dur = 64e-6, 64, 40.0
    n = int(dur / ts)
    rng = np.random.default_rng(7)
    items = [("HEADER_START", None), ("nchans", nch), ("nbits", 8), ("nifs", 1), ("tsamp", ts), ("fch1", 1446.0),
             ("foff", -1.0), ("tstart", 60000.0), ("data_type", 1), ("HEADER_END", None)]
    with open(tmp_path / "s.fil", "wb") as fo:
        fo.write(write_header(items))
        for s0 in range(0, n, 100000):
            k = min(100000, n - s0)
            x = 8 * rng.standard_normal((k, nch)).astype(np.float32)
            rate = 0.01 * (1 + np.sin(2 * np.pi * (s0 * ts) / 23.0))
            x += (rng.random((k, nch)) < rate) * rng.exponential(30, (k, nch))
            fo.write(np.clip(np.rint(128 + x), 0, 255).astype(np.uint8).tobytes())
    run_commonmode(tmp_path / "s.fil", tmp_path / "o.fil", CommonModeConfig(), verbose=False)
    _, a = load(tmp_path / "s.fil")
    _, b = load(tmp_path / "o.fil")
    B = int(4.0 / ts)
    k = n // B
    bm = ((b - a) / 8)[: k * B].reshape(k, B, -1).mean(1)
    assert np.abs(np.diff(bm, axis=0)).mean() < 0.003, np.abs(np.diff(bm, axis=0)).mean()
