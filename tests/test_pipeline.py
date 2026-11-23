import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import synth_impulses as si  # noqa: E402

TOOLS = ROOT / "tools"


def test_pipeline_equals_manual_chain(tmp_path):
    src = tmp_path / "obs" / "x_processed.fil"
    src.parent.mkdir()
    si.make(src, dur=8.0, rfi=True, frbs=[(4.0, 100, 0.002, 0.6)])
    r = subprocess.run([sys.executable, str(TOOLS / "rfi_pipeline.py"), str(src), str(tmp_path / "clean") + "/"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "clean" / "x_processed.fil"
    diag = Path(str(out) + ".pipeline")
    info = json.loads((diag / "pipeline.json").read_text())
    assert info["status"] == "completato" and sorted(info["steps"]) == ["1", "2", "3", "4"]
    assert not (diag / "tmp").exists()                     # intermedi cancellati
    for k in (1, 2, 3, 4):
        assert (diag / f"step{k}.log").exists()
    # stesso risultato della catena manuale degli strumenti
    prev = src
    for k, tool in enumerate(("rfi_clean.py", "rfi_periodic.py", "rfi_impulses.py", "rfi_commonmode.py"), 1):
        dst = tmp_path / f"m{k}.fil"
        subprocess.run([sys.executable, str(TOOLS / tool), str(prev), str(dst), "-q"], check=True,
                       capture_output=True)
        prev = dst
    assert prev.read_bytes() == out.read_bytes()
    # non sovrascrive senza --force
    r = subprocess.run([sys.executable, str(TOOLS / "rfi_pipeline.py"), str(src), str(out)], capture_output=True, text=True)
    assert r.returncode != 0
