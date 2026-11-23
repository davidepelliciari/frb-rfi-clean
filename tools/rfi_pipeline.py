#!/usr/bin/env python3
"""
rfi_pipeline.py - esegue l'intera pipeline di pulizia (step 1 -> 2 -> 3 -> 4) con un solo comando.

  step 1  tools/rfi_clean.py       maschera, normalizzazione per canale, celle anomale
  step 2  tools/rfi_periodic.py    RFI periodico
  step 3  tools/rfi_impulses.py    impulsi a DM 0
  step 4  tools/rfi_commonmode.py  rumore di modo comune

Uso:
  python3 rfi_pipeline.py INPUT.fil OUTPUT.fil
  python3 rfi_pipeline.py INPUT.fil CARTELLA/          (output: CARTELLA/<nome dell'input>)

Prodotti:
  OUTPUT.fil                       il file pulito (stesso header e numero di bit dell'input)
  OUTPUT.fil.pipeline/             diagnostica:
      pipeline.json                riassunto (comando, versione del codice, tempi, numeri principali di ogni step)
      stepN.log                    output a schermo di ogni step
      stepN.<...>                  riassunti e diagnostica di ogni step (.json, .npz, .csv, .png)
      tmp/                         file intermedi (cancellati appena non servono, salvo --keep)

I file intermedi sono grandi quanto l'input: senza --keep ne esistono al massimo due alla volta (oltre
all'output). Il comando si ferma al primo step che fallisce, indicando il log da guardare.

Opzioni per i singoli step: --step1-args="...", --step2-args="...", ecc., passate cosi' come sono allo
strumento corrispondente (es. --step3-args="--thresholds 10 8 6"; usare il segno = perche' il valore
inizia con un trattino). --steps sceglie quali step eseguire.

Nota: FRB_HF_Finder.py analizza solo i file *_processed.fil in una sottocartella di --proc_path.

Esempi:
  python3 rfi_pipeline.py /data/obs/x_processed.fil /data/CLEAN/obs/
  python3 rfi_pipeline.py in.fil out.fil --keep --step1-args="--mask-chans 110"
  python3 rfi_pipeline.py in.fil out.fil --tmax 60          (prova sui primi 60 s)
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {
    1: ("rfi_clean.py", "pulizia di base (maschera, normalizzazione, celle anomale)"),
    2: ("rfi_periodic.py", "RFI periodico"),
    3: ("rfi_impulses.py", "impulsi a DM 0"),
    4: ("rfi_commonmode.py", "rumore di modo comune"),
}
# numeri principali da riportare per ogni step (se presenti nel .json dello step)
KEYS = {
    1: ["masked_channels", "n_masked", "n_changepoints", "frac_cells_replaced", "frac_data_replaced", "runtime_s"],
    2: ["n_lines_before", "n_lines_after", "runtime_s"],
    3: ["events", "removed", "protected", "frac_data_replaced", "runtime_s"],
    4: ["zerodm_excess_before_median", "zerodm_excess_after_median", "split_excess_before_median",
        "split_excess_after_median", "gain_median", "runtime_s"],
}


def git_version():
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        d = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
        return r.stdout.strip() + ("+modifiche locali" if d.stdout.strip() else "") if r.returncode == 0 else None
    except OSError:
        return None


def summarize(step, js):
    out = {k: js[k] for k in KEYS[step] if k in js}
    if step == 2 and "families" in js:
        out["families"] = [{k: f.get(k) for k in ("f0_hz", "n_lines", "subtracted", "protected", "dm") if k in f}
                           for f in js["families"]]
    return out


def fmt(step, s):
    if step == 1:
        mc = s.get("masked_channels")
        nm = len(mc) if isinstance(mc, (dict, list)) else s.get("n_masked")
        parts = [f"canali mascherati {nm}"] if nm is not None else []
        if "n_changepoints" in s:
            parts.append(f"salti di guadagno {s['n_changepoints']}")
        for k, lab in (("frac_cells_replaced", "celle sostituite"), ("frac_data_replaced", "dati sostituiti")):
            if k in s:
                parts.append(f"{lab} {100 * s[k]:.3f}%")
        return ", ".join(parts)
    if step == 3 and "events" in s:
        return (f"eventi {s['events']}, rimossi {s.get('removed')}, protetti {s.get('protected')}, "
                f"dati sostituiti {100 * s.get('frac_data_replaced', 0):.2f}%")
    if step == 4 and "zerodm_excess_before_median" in s:
        return (f"eccesso di rumore a DM 0 {s['zerodm_excess_before_median']:.2f} -> "
                f"{s['zerodm_excess_after_median']:.2f}, guadagno mediano {s.get('gain_median', float('nan')):.2f}")
    if step == 2 and "n_lines_before" in s:
        return f"righe periodiche {s['n_lines_before']} -> {s['n_lines_after']}"
    return ", ".join(f"{k} {v}" for k, v in s.items() if k != "runtime_s")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", help="file di output, o cartella (il nome resta quello dell'input)")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4],
                   help="step da eseguire, nell'ordine 1-4 (default tutti)")
    for k in STEPS:
        p.add_argument(f"--step{k}-args", default="", help=f"opzioni per {STEPS[k][0]} (forma --step{k}-args=\"...\")")
    p.add_argument("--tmpdir", type=Path, default=None, help="cartella dei file intermedi (default OUTPUT.pipeline/tmp)")
    p.add_argument("--keep", action="store_true", help="conserva i file intermedi (in OUTPUT.pipeline/)")
    p.add_argument("--tmax", type=float, default=None, help="elabora solo i primi TMAX secondi (prove)")
    p.add_argument("--force", action="store_true", help="sovrascrive un output esistente")
    a = p.parse_args(argv)

    inp = a.input.resolve()
    if not inp.is_file():
        sys.exit(f"ERRORE: {inp} non esiste")
    out = Path(a.output)
    if out.is_dir() or a.output.endswith(("/", "\\")):
        out = out / inp.name
    out = out.resolve()
    if out == inp:
        sys.exit("ERRORE: l'output coincide con l'input")
    if out.exists() and not a.force:
        sys.exit(f"ERRORE: {out} esiste gia' (usa --force per sovrascriverlo)")
    steps = sorted(set(a.steps))
    diag = Path(str(out) + ".pipeline")
    tmp = (a.tmpdir.resolve() if a.tmpdir else diag / "tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    if not out.name.endswith("_processed.fil"):
        print(f"NOTA: FRB_HF_Finder.py analizza solo i file *_processed.fil; l'output si chiama {out.name}")

    # spazio su disco: senza --keep, al massimo due intermedi oltre all'output
    size = inp.stat().st_size
    if a.tmax is not None:
        sys.path.insert(0, str(HERE.parent))
        from frbclean.sigproc import Filterbank
        size = min(size, int(size * a.tmax / max(Filterbank(inp).duration, 1e-9)))
    n_tmp = (len(steps) - 1) if a.keep else min(2, len(steps) - 1)
    need_tmp = n_tmp * size
    free_tmp = shutil.disk_usage(tmp).free
    free_out = shutil.disk_usage(out.parent).free
    same_fs = tmp.stat().st_dev == out.parent.stat().st_dev
    if same_fs and free_tmp < need_tmp + size or (not same_fs and (free_tmp < need_tmp or free_out < size)):
        print(f"ATTENZIONE: spazio forse insufficiente (servono ~{(need_tmp + size) / 2**30:.0f} GB, liberi "
              f"{free_tmp / 2**30:.0f} GB)", file=sys.stderr)

    version = git_version()
    info = dict(command=" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) if argv is None else argv,
                started=datetime.now().isoformat(timespec="seconds"), code_version=version, input=str(inp),
                output=str(out), steps={}, status="in corso")
    print(f"Pipeline di pulizia: {inp}\n  -> {out}\n  step {steps}, codice {version}, diagnostica in {diag}")
    t_all = time.time()
    src = inp
    produced = []                     # intermedi prodotti, da cancellare quando non servono piu'
    stem = out.name[:-4] if out.name.endswith(".fil") else out.name
    for i, k in enumerate(steps):
        tool, what = STEPS[k]
        last = i == len(steps) - 1
        dst = out if last else tmp / f"{stem}.step{k}.fil"
        cmd = [sys.executable, str(HERE / tool), str(src), str(dst)] + shlex.split(getattr(a, f"step{k}_args"))
        if a.tmax is not None:
            cmd += ["--tmax", str(a.tmax)]
        log = diag / f"step{k}.log"
        print(f"[{datetime.now():%H:%M:%S}] step {k} ({what}) ...", flush=True)
        t0 = time.time()
        with open(log, "w") as fl:
            fl.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            fl.flush()
            r = subprocess.run(cmd, stdout=fl, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if r.returncode != 0:
            info["status"] = f"fallito allo step {k}"
            info["steps"][k] = dict(tool=tool, returncode=r.returncode, log=str(log), runtime_s=dt)
            (diag / "pipeline.json").write_text(json.dumps(info, indent=2, default=str))
            sys.exit(f"ERRORE: lo step {k} e' fallito (codice {r.returncode}); vedi {log}")
        # diagnostica dello step: i file OUTPUT.<...> accanto al suo output, copiati in diag/ come stepN.<...>
        js = {}
        for f in sorted(dst.parent.glob(dst.name + ".*")):
            suffix = f.name[len(dst.name) + 1:]
            target = diag / f"step{k}.{suffix}"
            if f.is_file():
                shutil.copy2(f, target) if last else shutil.move(str(f), str(target))
                if suffix.endswith(".json") and not js:
                    try:
                        js = json.loads(target.read_text())
                    except ValueError:
                        js = {}
        s = summarize(k, js)
        info["steps"][k] = dict(tool=tool, args=getattr(a, f"step{k}_args"), runtime_s=round(dt, 1), log=str(log),
                                summary=s)
        print(f"           fatto in {dt / 60:.1f} min. {fmt(k, s)}", flush=True)
        # l'input di questo step non serve piu' (se era un intermedio)
        if src != inp and not a.keep:
            try:
                src.unlink()
            except FileNotFoundError:
                pass
        elif src != inp and a.keep:
            produced.append(src)
        src = dst
        (diag / "pipeline.json").write_text(json.dumps(info, indent=2, default=str))
    if a.keep:
        for f in produced:
            shutil.move(str(f), str(diag / f.name))
    try:
        tmp.rmdir()
    except OSError:
        pass
    info["status"] = "completato"
    info["runtime_s"] = round(time.time() - t_all, 1)
    (diag / "pipeline.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"Completato in {(time.time() - t_all) / 60:.1f} min: {out}\nRiassunto: {diag / 'pipeline.json'}")


if __name__ == "__main__":
    main()
