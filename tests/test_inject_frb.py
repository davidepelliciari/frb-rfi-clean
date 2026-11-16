import csv, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "injection"))
sys.path.insert(0, str(ROOT / "tools"))
import inject_frb as ij  # noqa: E402
import fil_cut  # noqa: E402

TSAMP = 128e-6


def make_noise_fil(path, nchan=256, nsamp=160000, nbits=8, fch1=1500.0, foff=-1.0, seed=0, sig=None):
    rng = np.random.default_rng(seed)
    if sig is None:
        sig = 5 + 10 * np.sin(np.linspace(0, np.pi, nchan)) ** 2   # bandpass non piatto
    items = [("HEADER_START", None), ("source_name", "NOISE"), ("telescope_id", 0), ("machine_id", 0),
             ("data_type", 1), ("fch1", fch1), ("foff", foff), ("nchans", nchan), ("nbits", nbits),
             ("nifs", 1), ("tstart", 60000.0), ("tsamp", TSAMP), ("HEADER_END", None)]
    with open(path, "wb") as f:
        f.write(fil_cut.write_header(items))
        step = 20000
        for s in range(0, nsamp, step):
            n = min(step, nsamp - s)
            x = 100 + rng.normal(size=(n, nchan)) * sig
            if nbits == 8:
                x = np.clip(np.round(x), 0, 255).astype(np.uint8)
            else:
                x = x.astype(np.float32)
            f.write(x.tobytes())
    return sig


def load(path):
    fil = fil_cut.Fil(str(path))
    dt = {8: np.uint8, 32: np.float32}[fil.h["nbits"]]
    with open(path, "rb") as f:
        f.seek(fil.hdr_len)
        return fil, np.fromfile(f, dtype=dt).reshape(-1, fil.h["nchans"]).astype(np.float64)


def measure_snr(data, fil, toa, dm, width_samples):
    """Rivelatore indipendente: dedispersione a shift interi, somma, boxcar, rumore robusto."""
    h = fil.h
    freqs = h["fch1"] + h["foff"] * np.arange(h["nchans"])
    shift = np.round(ij.KDM * dm * (freqs ** -2 - freqs.max() ** -2) / h["tsamp"]).astype(int)
    i0 = int(toa / h["tsamp"])
    lo, hi = i0 - 4000, i0 + 4000
    ts = np.zeros(hi - lo)
    for c in range(h["nchans"]):
        ts += data[lo + shift[c]:hi + shift[c], c]
    ts -= np.median(ts)
    noise = 1.4826 * np.median(np.abs(ts[:3000]))  # parte lontana dal burst
    cs = np.concatenate([[0], np.cumsum(ts)])
    W = width_samples
    box = (cs[W:] - cs[:-W])[4000 - 200:4000 + 200]
    return box.max() / (noise * np.sqrt(W))


@pytest.mark.parametrize("nbits", [8, 32])
def test_recovered_snr_matches_requested(tmp_path, nbits):
    src = tmp_path / "noise.fil"
    make_noise_fil(src, nbits=nbits)
    params = tmp_path / "b.params"
    toas = np.arange(1.0, 19.0, 1.5)
    snrs = np.where(np.arange(len(toas)) % 2, 15.0, 30.0)
    params.write_text("".join(f"{t} {s} 0.002 300\n" for t, s in zip(toas, snrs)))
    out = tmp_path / "inj.fil"
    assert ij.main(["--fil", str(src), "--params", str(params), "--out", str(out), "--seed", "1", "-q"]) == 0
    fil, data = load(out)
    truth = list(csv.DictReader(open(out.with_suffix(".truth.csv"))))
    ratios = []
    for r in truth:
        snr = measure_snr(data, fil, float(r["toa_s"]), float(r["dm"]), int(r["best_boxcar_samples"]))
        ratios.append(snr / float(r["snr_requested"]))
        assert int(r["n_clipped"]) == 0
    ratios = np.array(ratios)
    assert abs(ratios.mean() - 1) < 0.06, ratios
    assert np.all(np.abs(ratios - 1) < 0.25), ratios


def test_chunk_size_invariance(tmp_path):
    src = tmp_path / "noise.fil"
    make_noise_fil(src, nsamp=40000)
    params = tmp_path / "b.params"
    params.write_text("1.0 20 0.003 500\n2.5 10 0.001 1500 -1.5\n3.5 12 0.002 200 0 1400 60\n")
    outs = []
    for mb in (0.05, 1000):
        out = tmp_path / f"inj_{mb}.fil"
        ij.main(["--fil", str(src), "--params", str(params), "--out", str(out),
                 "--seed", "3", "--chunk-mb", str(mb), "-q"])
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


def test_dithering_preserves_weak_signal(tmp_path):
    # molti canali, rumore per canale basso: ampiezza per canale << 1 conteggio
    src = tmp_path / "noise.fil"
    sig = np.full(1024, 3.0)
    make_noise_fil(src, nchan=1024, nsamp=30000, sig=sig, fch1=1500.0, foff=-0.25)
    params = tmp_path / "b.params"
    params.write_text("1.5 8 0.005 400\n")
    out = tmp_path / "inj.fil"
    ij.main(["--fil", str(src), "--params", str(params), "--out", str(out), "--seed", "5", "-q"])
    _, a = load(src)
    fil, b = load(out)
    added = (b - a).sum()
    r = next(csv.DictReader(open(out.with_suffix(".truth.csv"))))
    expected = float(r["amplitude"]) * sig.sum()   # area per canale = 1, peso medio = 1
    assert added == pytest.approx(expected, rel=0.03)
    snr = measure_snr(b, fil, 1.5, 400, int(r["best_boxcar_samples"]))
    assert snr == pytest.approx(8, abs=2.0)


def test_partial_band_burst_only_in_band(tmp_path):
    src = tmp_path / "noise.fil"
    make_noise_fil(src, nsamp=30000)
    params = tmp_path / "b.params"
    params.write_text("1.0 40 0.002 100 0 1450 10\n")
    out = tmp_path / "inj.fil"
    ij.main(["--fil", str(src), "--params", str(params), "--out", str(out), "--seed", "2", "-q"])
    _, a = load(src)
    fil, b = load(out)
    freqs = fil.h["fch1"] + fil.h["foff"] * np.arange(fil.h["nchans"])
    per_chan = (b - a).sum(axis=0)
    inband = np.abs(freqs - 1450) < 10
    far = np.abs(freqs - 1450) > 30
    assert per_chan[inband].sum() > 0.9 * per_chan.sum()
    assert per_chan[far].sum() < 0.01 * per_chan.sum()


def test_saturation_is_counted(tmp_path):
    src = tmp_path / "noise.fil"
    make_noise_fil(src, nsamp=30000)
    params = tmp_path / "b.params"
    params.write_text("1.0 5000 0.0005 100\n")
    out = tmp_path / "inj.fil"
    ij.main(["--fil", str(src), "--params", str(params), "--out", str(out), "--seed", "2", "-q"])
    r = next(csv.DictReader(open(out.with_suffix(".truth.csv"))))
    assert int(r["n_clipped"]) > 0


def test_snr_calib_and_noise_tool(tmp_path):
    import subprocess
    src = tmp_path / "noise.fil"
    make_noise_fil(src, nsamp=20000)
    fake = tmp_path / "fake.fil"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_noise_fil.py"), "--like", str(src),
                        "--out", str(fake), "--duration", "3", "--bandpass-from-data"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    f = fil_cut.Fil(str(fake))
    assert f.nsamp == int(round(3 / TSAMP)) and f.h["nchans"] == 256
    params = tmp_path / "b.params"
    params.write_text("1.0 20 0.002 300\n")
    out = tmp_path / "inj.fil"
    ij.main(["--fil", str(fake), "--params", str(params), "--out", str(out), "--seed", "1", "-q",
             "--snr-calib", "0.84,0.52"])
    row = next(csv.DictReader(open(out.with_suffix(".truth.csv"))))
    assert float(row["snr_model"]) == pytest.approx((20 - 0.52) / 0.84, rel=1e-3)
