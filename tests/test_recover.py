import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "injection"))
from frbclean.sigproc import write_header  # noqa: E402
import completeness_plan as cp  # noqa: E402
import recover  # noqa: E402


def _sparse_fil(path, dur, ts=16e-6, nch=256):
    items = [("HEADER_START", None), ("nchans", nch), ("nbits", 8), ("nifs", 1), ("tsamp", ts), ("fch1", 1446.125),
             ("foff", -0.25), ("tstart", 60000.0), ("data_type", 1), ("HEADER_END", None)]
    h = write_header(items)
    with open(path, "wb") as f:
        f.write(h)
        f.truncate(len(h) + int(dur / ts) * nch)


def _fake_search(d: Path, plan_rows, detect, false_c, rng):
    """Output finto di FRB_HF_Finder: burst rivelati secondo `detect`, piu' falsi candidati."""
    d.mkdir(parents=True)
    sel, raw, fetch = [], [], []
    for r, det in zip(plan_rows, detect):
        t, dm, s = float(r["toa_s"]) + rng.normal(0, 1e-3), float(r["dm"]) * (1 + rng.normal(0, 0.02)), float(r["snr_expected"])
        raw.append((s, t, 6, dm))
        if det:
            sel.append((t, dm, s, 6))
            fetch.append((t, dm, s, 0.9))
    for t, dm, s in false_c:
        sel.append((t, dm, s, 3))
        raw.append((s, t, 3, dm))
        fetch.append((t, dm, s, 0.8 if s > 15 else 0.1))
    with open(d / "x_all_candidates.csv", "w") as f:
        f.write("file,snr,width,dm,label,stime,chan_mask_path,num_files\n")
        for t, dm, s, w in sel:
            f.write(f"a.fil,{s:.5f},{w},{dm:.5f},0,{t:.5f}, ,1\n")
    with open(d / "x_all.cand", "w") as f:
        for s, t, w, dm in raw:
            f.write(f"{s:.5f} {int(t / 16e-6)} {t:.6f} {w} 0 {dm:.4f} 1 0 1\n")
    with open(d / "results_a.csv", "w") as f:
        f.write(",candidate,probability,label\n")
        for i, (t, dm, s, p) in enumerate(fetch):
            f.write(f"{i},/x/cand_tstart_1.0_tcand_{t:.7f}_dm_{dm:.5f}_snr_{s:.5f}.h5,{p},{int(p > 0.5)}\n")
    with open(d / "Best_candidates.csv", "w") as f:
        f.write("candidate,probability,label\n")


def test_plan_and_recover(tmp_path):
    _sparse_fil(tmp_path / "f.fil", 300.0)
    assert cp.main(["--fil", str(tmp_path / "f.fil"), "--outdir", str(tmp_path / "c"), "--scans", "A", "B", "D",
                    "--per-point", "4", "--seed", "3"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "c" / "plan.csv")))
    assert len(rows) == (10 + 9 + 9) * 4
    toa = np.array(sorted(float(r["toa_s"]) for r in rows))
    assert np.all(np.diff(toa) >= 2.0 - 1e-6)
    # flusso fisso: S/N decrescente con la larghezza, ~ 1/sqrt(w) per larghezze grandi
    b = {float(r["width_s"]): float(r["snr_expected"]) for r in rows if r["scan"] == "B"}
    ws = sorted(b)
    assert all(b[ws[i]] > b[ws[i + 1]] for i in range(len(ws) - 1))
    assert abs(b[0.001] - 20) < 1e-6 and abs(b[0.016] / b[0.004] - 0.5) < 0.05

    rng = np.random.default_rng(0)
    det1 = [float(r["snr_expected"]) >= 8 for r in rows]
    det2 = [d and rng.random() > 0.3 for d in det1]
    false_c = [(t, 100.0, 9.0) for t in toa[:-1] + 1.0][:5] + [(toa[3] + 0.9, 300.0, 20.0)]
    _fake_search(tmp_path / "s1", rows, det1, false_c, rng)
    _fake_search(tmp_path / "s2", rows, det2, false_c[:2], rng)
    out = tmp_path / "m_r01.csv"
    assert recover.main(["match", "--plan", str(tmp_path / "c" / "plan.csv"), "--realization", "1",
                         "--step", "uno", str(tmp_path / "s1"), "--step", "due", str(tmp_path / "s2"),
                         "--out", str(out)]) == 0
    m = list(csv.DictReader(open(out)))
    for st, det in (("uno", det1), ("due", det2)):
        got = [int(r["detected"]) for r in m if r["step"] == st]
        assert got == [int(x) for x in det]
    f = {r["step"]: r for r in csv.DictReader(open(tmp_path / "m_r01_false.csv"))}
    assert int(f["uno"]["n_false"]) == 6 and int(f["uno"]["n_false_fetch07"]) == 1
    assert int(f["due"]["n_false"]) == 2
    assert recover.main(["report", "--matched", str(tmp_path / "m_r*.csv"), "--out", str(tmp_path / "rep")]) == 0
    t = list(csv.DictReader(open(tmp_path / "rep" / "completeness.csv")))
    a = {r["point"]: float(r["completeness"]) for r in t if r["scan"] == "A" and r["step"] == "uno"}
    assert a["snr=5"] == 0 and a["snr=30"] == 1
    assert (tmp_path / "rep" / "scan_B.png").exists() and (tmp_path / "rep" / "false_candidates.csv").exists()
