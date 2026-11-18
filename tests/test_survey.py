import csv, json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from frbclean.sigproc import Filterbank, write_header, pack_bits, unpack_bits  # noqa: E402
from frbclean.survey import SurveyConfig, run_survey  # noqa: E402

TS = 64e-6
NCH = 64
PERIOD = 0.5
PARTIAL = (7.3, 13.1, 21.7)


def make_rfi_fil(path, dur=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur / TS)
    sig = 8 * (1 + 0.5 * np.sin(np.linspace(0, 3, NCH)))
    sig[40] *= 3                          # canale rumoroso
    items = [("HEADER_START", None), ("source_name", "RFI"), ("telescope_id", 0), ("machine_id", 0),
             ("data_type", 1), ("fch1", 1446.125), ("foff", -1.0), ("nchans", NCH), ("nbits", 8),
             ("nifs", 1), ("tstart", 60000.0), ("tsamp", TS), ("HEADER_END", None)]
    with open(path, "wb") as f:
        f.write(write_header(items))
        step = 100000
        for s in range(0, n, step):
            k = min(step, n - s)
            x = 100 + rng.normal(size=(k, NCH)) * sig
            t = (s + np.arange(k)) * TS
            # radar: impulsi a banda intera ogni 0.5 s, larghi 0.3 ms
            ph = (t + 0.1) % PERIOD
            x[ph < 3e-4] += 3 * sig
            # impulsi a banda parziale (canali 8-15), larghi 2 ms
            for tp in PARTIAL:
                m = np.abs(t - tp) < 1e-3
                x[np.ix_(m, np.arange(8, 16))] += 6 * sig[8:16]
            # canale impulsivo
            spikes = rng.random(k) < 2e-3
            x[spikes, 50] += 15 * sig[50]
            x[:, 5] = 100                     # canale morto
            f.write(np.clip(np.round(x), 0, 255).astype(np.uint8).tobytes())


@pytest.fixture(scope="module")
def survey_out(tmp_path_factory):
    d = tmp_path_factory.mktemp("sv")
    fil = d / "rfi.fil"
    make_rfi_fil(fil)
    summ = run_survey(fil, d / "out", SurveyConfig(), plot=True, verbose=False)
    return d / "out", summ


def test_radar_pulses_found(survey_out):
    out, _ = survey_out
    ev = list(csv.DictReader(open(out / "events.csv")))
    t = np.array([float(e["t"]) for e in ev])
    expected = np.arange(PERIOD - 0.1, 30, PERIOD)
    expected = expected[(expected > 0.1) & (expected < 29.9)]
    found = [np.min(np.abs(t - te)) < 2e-3 for te in expected]
    assert np.mean(found) > 0.9
    for te in expected:
        j = np.argmin(np.abs(t - te))
        if abs(t[j] - te) < 2e-3:
            assert float(ev[j]["frac_band"]) > 0.75


def test_partial_band_events(survey_out):
    out, _ = survey_out
    ev = list(csv.DictReader(open(out / "events.csv")))
    t = np.array([float(e["t"]) for e in ev])
    for tp in PARTIAL:
        j = np.argmin(np.abs(t - tp))
        assert abs(t[j] - tp) < 3e-3
        e = ev[j]
        assert float(e["frac_band"]) <= 0.25 and e["best"].startswith("sub")
        # canali 8-15 su 64 con foff<0: frequenze alte -> sottobande 12-13 in ordine crescente
        assert e["sub_hits"][12:14] == "11"


def test_few_false_events(survey_out):
    out, summ = survey_out
    ev = list(csv.DictReader(open(out / "events.csv")))
    t = np.array([float(e["t"]) for e in ev])
    ph = (t + 0.1) % PERIOD
    radar = (ph < 3e-3) | (ph > PERIOD - 3e-3)
    partial = np.any([np.abs(t - tp) < 5e-3 for tp in PARTIAL], axis=0)
    # il canale impulsivo 50 (frequenza 1396.1 MHz) cade nella sottobanda 3 in ordine crescente
    chan50 = np.array([e["best"] == "sub03" for e in ev])
    assert chan50.sum() > 50
    other = ~(radar | partial | chan50)
    assert other.sum() < 15, t[other]


def test_periodicity(survey_out):
    _, summ = survey_out
    for key in ("periodicity_event_train_broadband", "periodicity_zerodm"):
        f = summ[key][0]["freq_hz"]
        assert abs(f - 1 / PERIOD) < 0.01, (key, summ[key][:3])


def test_flagged_channels(survey_out):
    out, summ = survey_out
    fl = {int(k) for k in summ["flagged_channels"]}
    assert {5, 40, 50} <= fl
    assert len(fl) <= 6
    assert (out / "flagged_channels.txt").read_text().count("\n") >= 3
    for png in ("channels.png", "dynspec.png", "events.png", "zerodm.png"):
        assert (out / png).stat().st_size > 1000


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_low_bit_reading(tmp_path, nbits):
    rng = np.random.default_rng(1)
    vals = rng.integers(0, 2 ** nbits, size=(100, 16)).astype(np.uint8)
    assert np.array_equal(unpack_bits(pack_bits(vals.ravel(), nbits), nbits).reshape(100, 16), vals)
    items = [("HEADER_START", None), ("nchans", 16), ("nbits", nbits), ("nifs", 1), ("tsamp", 1e-3),
             ("fch1", 400.0), ("foff", -1.0), ("tstart", 60000.0), ("HEADER_END", None)]
    p = tmp_path / "low.fil"
    p.write_bytes(write_header(items) + pack_bits(vals.ravel(), nbits).tobytes())
    fb = Filterbank(p)
    assert fb.nsamp == 100
    assert np.array_equal(fb.read(10, 30), vals[10:40])
