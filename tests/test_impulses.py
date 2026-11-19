import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import synth_impulses as si  # noqa: E402
from frbclean.impulses import ImpulseConfig, run_impulses  # noqa: E402
from frbclean.survey import SurveyConfig, run_survey  # noqa: E402
from test_inject_frb import load, measure_snr  # noqa: E402

FRBS = [(2.2, 30, 0.001, 1.0), (5.3, 100, 0.002, 0.6), (8.4, 300, 0.005, 0.4), (11.5, 700, 0.002, 0.6),
        (14.1, 50, 0.0005, 1.5)]


def test_rfi_removed_frbs_protected(tmp_path):
    si.make(tmp_path / "rfi.fil", dur=16.5, frbs=FRBS)
    si.make(tmp_path / "ref.fil", dur=16.5, rfi=False, frbs=FRBS)
    si.make(tmp_path / "rfi_only.fil", dur=16.5, frbs=())
    summ = run_impulses(tmp_path / "rfi.fil", tmp_path / "out.fil", ImpulseConfig(), verbose=False)
    assert summ["frac_data_replaced"] < 0.01
    # nessuna protezione lontana dagli FRB
    for p in summ["protected_events"]:
        assert min(abs(p["t"] - f[0]) for f in FRBS) < 0.4, p
    fr, dr = load(tmp_path / "ref.fil")
    fo, do = load(tmp_path / "out.fil")
    for tb, dm, w, a in FRBS:
        W = max(1, int(round(w / 64e-6)))
        assert measure_snr(do, fo, tb, dm, W) / measure_snr(dr, fr, tb, dm, W) > 0.95, (tb, dm, w)
    # canali mascherati intatti
    assert np.all(do[:, 60:62] == 128)
    # l'RFI a DM 0 e' in gran parte rimosso
    run_impulses(tmp_path / "rfi_only.fil", tmp_path / "rfi_only_out.fil", ImpulseConfig(), verbose=False)
    a = run_survey(tmp_path / "rfi_only.fil", tmp_path / "sa", SurveyConfig(), plot=False, verbose=False)
    b = run_survey(tmp_path / "rfi_only_out.fil", tmp_path / "sb", SurveyConfig(), plot=False, verbose=False)
    assert b["n_events"] < 0.25 * a["n_events"], (a["n_events"], b["n_events"])


def test_pure_noise_untouched(tmp_path):
    si.make(tmp_path / "n.fil", dur=10.0, rfi=False, frbs=())
    summ = run_impulses(tmp_path / "n.fil", tmp_path / "o.fil", ImpulseConfig(), verbose=False)
    assert summ["removed"] == 0
    assert (tmp_path / "n.fil").read_bytes() == (tmp_path / "o.fil").read_bytes()
