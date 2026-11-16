import csv, json, os, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "injection"))
sys.path.insert(0, str(ROOT / "tests"))
import plan_injections as pi  # noqa: E402
from test_fil_cut import make_fil  # noqa: E402


def test_dm_delay_lband():
    # DM 500 tra 1200 e 1500 MHz
    assert pi.dm_delay(500, 1200, 1500) == pytest.approx(0.5187, rel=1e-3)


def test_band_edges():
    lo, hi = pi.band_edges({"fch1": 1500.0, "foff": -1.0, "nchans": 64})
    assert (lo, hi) == (1436.5, 1500.5)


def test_sample_toas_constraints():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = pi.sample_toas(50, 0.5, 100.0, 0.2, rng)
        assert t.min() >= 0.5 and t.max() <= 100.0
        assert np.diff(t).min() >= 0.2 - 1e-12


def test_sample_toas_tight_and_impossible():
    rng = np.random.default_rng(0)
    t = pi.sample_toas(11, 0.0, 1.0, 0.1, rng)  # unica configurazione possibile
    np.testing.assert_allclose(t, np.linspace(0, 1, 11), atol=1e-12)
    with pytest.raises(ValueError):
        pi.sample_toas(12, 0.0, 1.0, 0.1, rng)


def test_spec_parse_and_sample():
    rng = np.random.default_rng(1)
    assert pi.Spec.parse("500").sample(3, rng).tolist() == [500, 500, 500]
    x = pi.Spec.parse("log:10:1000").sample(10000, rng)
    assert x.min() >= 10 and x.max() < 1000
    assert np.median(np.log10(x)) == pytest.approx(2.0, abs=0.05)
    assert set(pi.Spec.parse("1,2,3").sample(100, rng)) <= {1, 2, 3}
    with pytest.raises(Exception):
        pi.Spec.parse("10:5")


def test_bins_no_name_collision(tmp_path):
    src = tmp_path / "a.fil"
    make_fil(src, nsamp=200000, tsamp=1e-3)  # 200 s
    fil = pi.Fil(str(src))
    groups, _ = pi.plan(fil, nfrb=5, snr=pi.Spec.parse("4:6"), dm=pi.Spec.parse("500"),
                        width=pi.Spec.parse("0.01"), layout="bins", snr_bin=0.5, toa_mode="random",
                        min_gap=None, start_margin=0.5, end_margin=0.5, rng=np.random.default_rng(0))
    names = [n for n, _ in groups]
    assert len(names) == 4 == len(set(names))
    for n, rows in groups:
        lo = float(n.split("_")[1])
        assert ((rows[:, 1] >= lo) & (rows[:, 1] <= lo + 0.5)).all()


def test_cli_end_to_end_with_fake_injector(tmp_path):
    src = tmp_path / "a.fil"
    make_fil(src, nsamp=100000, tsamp=1e-3)  # 100 s
    fake = tmp_path / "fake_injector.py"
    fake.write_text(
        "import sys, shutil\n"
        "a = sys.argv\n"
        "shutil.copy(a[a.index('--FIL')+1], a[a.index('--outfile')+1] + '.fil')\n")
    out = tmp_path / "camp"
    rc = pi.main(["--fil", str(src), "--outdir", str(out), "--nfrb", "20", "--snr", "6:30",
                  "--dm", "log:20:2000", "--width", "log:0.0005:0.02", "--seed", "7",
                  "--run", "--injector", str(fake)])
    assert rc == 0
    meta = json.loads((out / "campaign.json").read_text())
    assert meta["seed"] == 7
    rows = list(csv.DictReader(open(out / "injections.csv")))
    assert len(rows) == 20
    toa = np.array([float(r["toa_s"]) for r in rows])
    assert toa.max() <= meta["toa_range_s"][1]
    assert (out / "fil" / "INJECTED_SyntheticFRB.fil").exists()
    # stesso seed -> stessa campagna
    out2 = tmp_path / "camp2"
    pi.main(["--fil", str(src), "--outdir", str(out2), "--nfrb", "20", "--snr", "6:30",
             "--dm", "log:20:2000", "--width", "log:0.0005:0.02", "--seed", "7"])
    assert (out / "injections.csv").read_text() == (out2 / "injections.csv").read_text()


def test_refuses_nonempty_outdir(tmp_path):
    src = tmp_path / "a.fil"
    make_fil(src, nsamp=100000, tsamp=1e-3)
    out = tmp_path / "camp"
    out.mkdir(); (out / "old.params").write_text("x")
    assert pi.main(["--fil", str(src), "--outdir", str(out), "--nfrb", "5"]) == 2


def test_cli_run_builtin_injector(tmp_path):
    src = tmp_path / "a.fil"
    make_fil(src, nsamp=30000, tsamp=1e-3)  # 30 s, 64 canali
    out = tmp_path / "camp"
    rc = pi.main(["--fil", str(src), "--outdir", str(out), "--nfrb", "5", "--snr", "10:20",
                  "--dm", "100", "--width", "0.005", "--seed", "3", "--run", "--smear"])
    assert rc == 0
    inj = out / "fil" / "INJECTED_SyntheticFRB.fil"
    assert inj.exists() and inj.stat().st_size == src.stat().st_size
    assert inj.with_suffix(".truth.csv").exists()
