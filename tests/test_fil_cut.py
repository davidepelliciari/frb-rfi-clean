import os, struct, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fil_cut  # noqa: E402


def make_fil(path, nchans=64, nsamp=5000, tsamp=1e-4, nbits=8, tstart=60000.0):
    items = [("HEADER_START", None), ("source_name", "TEST"), ("telescope_id", 0),
             ("machine_id", 0), ("data_type", 1), ("fch1", 1500.0), ("foff", -1.0),
             ("nchans", nchans), ("nbits", nbits), ("nifs", 1), ("tstart", tstart),
             ("tsamp", tsamp), ("HEADER_END", None)]
    data = (np.arange(nsamp)[:, None] % 251 + np.zeros((1, nchans))).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(fil_cut.write_header(items))
        f.write(data.tobytes())
    return data


def test_roundtrip_and_cut(tmp_path):
    src = tmp_path / "a.fil"
    data = make_fil(src)
    fil = fil_cut.Fil(str(src))
    assert fil.nsamp == 5000 and fil.bytes_per_samp == 64
    out = tmp_path / "b.fil"
    fil.cut(str(out), start_s=0.1, duration_s=0.2)  # campioni 1000..2999
    cut = fil_cut.Fil(str(out))
    assert cut.nsamp == 2000
    assert abs(cut.h["tstart"] - (60000.0 + 0.1 / 86400)) < 1e-12
    with open(out, "rb") as f:
        f.seek(cut.hdr_len)
        got = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 64)
    np.testing.assert_array_equal(got, data[1000:3000])


def test_cli_sample(tmp_path):
    src = tmp_path / "a.fil"
    make_fil(src)
    script = os.path.join(os.path.dirname(__file__), "..", "tools", "fil_cut.py")
    r = subprocess.run([sys.executable, script, "sample", str(src), str(tmp_path / "o"),
                        "--n", "3", "--duration", "0.1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(os.listdir(tmp_path / "o")) == 3
