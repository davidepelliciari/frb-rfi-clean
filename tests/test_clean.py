import csv, json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "injection"))
import synth_medicina as sm  # noqa: E402
import inject_frb as ij  # noqa: E402
from test_inject_frb import make_noise_fil, load, measure_snr  # noqa: E402
from frbclean import minisearch as ms  # noqa: E402
from frbclean.clean import CleanConfig, parse_chan_list, read_flag_file, run_clean  # noqa: E402
from frbclean.sigproc import Filterbank  # noqa: E402

FRBS = [(6.0, 12, 0.002, 300), (16.0, 12, 0.002, 300), (26.0, 12, 0.001, 100), (36.0, 12, 0.005, 500),
        (11.5, 12, 0.002, 300)]     # l'ultimo cade tra due salti di guadagno


@pytest.fixture(scope="module")
def pathological(tmp_path_factory):
    d = tmp_path_factory.mktemp("cl")
    sm.make(d / "dirty0.fil")
    (d / "frb.params").write_text("".join(f"{a} {b} {c} {e}\n" for a, b, c, e in FRBS))
    ij.main(["--fil", str(d / "dirty0.fil"), "--params", str(d / "frb.params"), "--out", str(d / "dirty.fil"),
             "--seed", "1", "-q"])
    summ = run_clean(d / "dirty.fil", d / "clean.fil", CleanConfig(), verbose=False)
    return d, summ


def _search(path):
    _, cl = ms.run(path, 25, 400, thresh=7, max_width=512, tol=6)
    return cl


def _known(t):
    return any(abs(t - f[0]) < 0.02 for f in FRBS) or any(abs(t - r) < 0.02 for r in sm.RADAR_T)


def test_changepoints_and_mask(pathological):
    _, summ = pathological
    cps = np.array(summ["changepoints_s"])
    for t0 in (10.0, 10.9, 11.9):
        assert np.min(np.abs(cps - t0)) < 0.005, cps
    assert summ["masked_channel_ranges"] == [[sm.IMPULSIVE_CH, sm.IMPULSIVE_CH]]
    # glitch a 30 s: blocco con la frazione di celle sostituite piu' alta
    assert abs(summ["worst_blocks"][0]["t"] - 30.0) < 1.0


def test_spurious_candidates_removed_frbs_kept(pathological):
    d, _ = pathological
    before = _search(d / "dirty.fil")
    after = _search(d / "clean.fil")
    spurious_before = [c for c in before if not _known(c["t"])]
    spurious_after = [c for c in after if not _known(c["t"])]
    assert len(spurious_before) >= 3          # il test e' sensibile: i disturbi producono candidati
    assert spurious_after == [], spurious_after
    for f in FRBS:
        if f[3] <= 400:
            assert any(abs(c["t"] - f[0]) < 0.02 for c in after), f


def test_header_and_size(pathological):
    d, _ = pathological
    a, b = Filterbank(d / "dirty.fil"), Filterbank(d / "clean.fil")
    assert a.header == b.header and a.nsamp == b.nsamp


def test_snr_preserved_on_clean_noise(tmp_path):
    src = tmp_path / "n.fil"
    make_noise_fil(src, nchan=64, nsamp=int(30 / 128e-6), fch1=1445.5, foff=-1.0)
    rows, t = [], 1.0
    for W in (0.0005, 0.005, 0.03):
        for snr in (15, 300):
            rows.append(f"{t} {snr} {W} 100\n"); t += 2.0
    (tmp_path / "v.params").write_text("".join(rows))
    ij.main(["--fil", str(src), "--params", str(tmp_path / "v.params"), "--out", str(tmp_path / "inj.fil"),
             "--seed", "3", "-q"])
    summ = run_clean(tmp_path / "inj.fil", tmp_path / "cl.fil", CleanConfig(), verbose=False)
    assert summ["changepoints_s"] == []
    assert summ["frac_cells_replaced"] < 1e-3
    f1, d1 = load(tmp_path / "inj.fil")
    f2, d2 = load(tmp_path / "cl.fil")
    for r in csv.DictReader(open(tmp_path / "inj.truth.csv")):
        w = int(r["best_boxcar_samples"])
        a = measure_snr(d1, f1, float(r["toa_s"]), float(r["dm"]), w)
        b = measure_snr(d2, f2, float(r["toa_s"]), float(r["dm"]), w)
        assert b / a > 0.95, (r["snr_requested"], r["width_fwhm_s"], a, b)


def test_bright_narrow_frbs_preserved(tmp_path):
    """FRB molto brillanti e stretti: il veto di dispersione e la dinamica dell'output li conservano."""
    src = tmp_path / "b.fil"
    make_noise_fil(src, nchan=128, nsamp=int(8 / 32e-6), fch1=1446.0, foff=-0.5, tsamp=32e-6)
    rows, t = [], 0.8
    for W in (0.0001, 0.0005):
        for snr in (300, 1000):
            rows.append(f"{t} {snr} {W} 150\n"); t += 1.7
    (tmp_path / "b.params").write_text("".join(rows))
    ij.main(["--fil", str(src), "--params", str(tmp_path / "b.params"), "--out", str(tmp_path / "inj.fil"),
             "--seed", "5", "-q"])
    summ = run_clean(tmp_path / "inj.fil", tmp_path / "cl.fil", CleanConfig(), verbose=False)
    f1, d1 = load(tmp_path / "inj.fil")
    f2, d2 = load(tmp_path / "cl.fil")
    for r in csv.DictReader(open(tmp_path / "inj.truth.csv")):
        w = int(r["best_boxcar_samples"])
        a = measure_snr(d1, f1, float(r["toa_s"]), float(r["dm"]), w)
        b = measure_snr(d2, f2, float(r["toa_s"]), float(r["dm"]), w)
        assert b / a > 0.9, (r["snr_requested"], r["width_fwhm_s"], a, b, summ["protected_events"])


def test_float_output_and_user_mask(tmp_path):
    src = tmp_path / "f.fil"
    make_noise_fil(src, nchan=32, nsamp=20000, nbits=32)
    summ = run_clean(src, tmp_path / "c.fil", CleanConfig(mask_chans=[3, 4, 5], auto_mask=False), verbose=False)
    assert summ["masked_channel_ranges"] == [[3, 5]]
    f, d = load(tmp_path / "c.fil")
    assert np.all(d[:, 3:6] == 0)
    live = np.delete(d, [3, 4, 5], axis=1)
    assert abs(live.mean()) < 0.05 and abs(live.std() - 1) < 0.05


def test_chan_parsing(tmp_path):
    assert parse_chan_list("110, 121-123,185") == [110, 121, 122, 123, 185]
    p = tmp_path / "f.txt"
    p.write_text("# commento\n121 127\n185 185\n")
    assert read_flag_file(p) == list(range(121, 128)) + [185]
