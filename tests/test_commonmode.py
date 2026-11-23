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
