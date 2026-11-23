#!/usr/bin/env python3
"""
recover.py - associa i candidati della ricerca (Heimdall + FETCH) ai burst iniettati e misura
completezza, S/N recuperato e falsi candidati, per ogni step della pulizia.

Due comandi:

  match   una realizzazione: per ogni step (cartella di output della ricerca, con
          *_all_candidates.csv, results_a.csv, *_all.cand) associa candidati e burst.
          Scrive MATCHED.csv (un burst per step per riga) e MATCHED_false.csv (falsi candidati
          per step: quelli selezionati dalla pipeline non associati ad alcun burst).

  report  una o piu' realizzazioni: completezza per punto di ogni scansione (intervallo di Wilson
          al 68%), S/N recuperato / atteso, frazione classificata da FETCH, falsi candidati
          per file; tabelle CSV e grafici.

Associazione: il ToA iniettato e i tempi di Heimdall sono riferiti al canale piu' alto della banda.
Un candidato e' associato a un burst se |t_cand - ToA| <= dt_tol + 2 x larghezza e
|DM_cand - DM| <= max(dm_frac x DM, dm_abs). Tra piu' candidati vale quello a S/N maggiore.
  rivelato     = associato a un candidato selezionato dalla pipeline (*_all_candidates.csv,
                 cioe' S/N > 8 e DM > 28 in produzione)
  snr_raw      = S/N del miglior candidato associato in *_all.cand (anche sotto soglia)
  fetch_prob   = probabilita' FETCH del candidato selezionato associato
Le tolleranze vanno verificate sul pilota: `match` stampa gli scarti in tempo e DM.

Esempi:
  python3 recover.py match --plan camp/plan.csv --realization 1 \\
      --step orig search/r01_orig --step step1 search/r01_step1 \\
      --step step2 search/r01_step2 --step step3 search/r01_step3 --out camp/matched_r01.csv
  python3 recover.py report --matched camp/matched_r*.csv --out camp/report
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np

KDM = 4.148808e3


# --------------------------------------------------------------------------- lettura
def _newest(paths):
    paths = [Path(p) for p in paths]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def find_files(d: Path):
    sel = _newest(glob.glob(str(d / "**" / "*_all_candidates.csv"), recursive=True))
    fetch = _newest(glob.glob(str(d / "**" / "results_*.csv"), recursive=True))
    raw = _newest(glob.glob(str(d / "**" / "*_all.cand"), recursive=True))
    best = _newest(glob.glob(str(d / "**" / "Best_candidates.csv"), recursive=True))
    return sel, fetch, raw, best


def load_selected(path):
    """Candidati selezionati: array (t, dm, snr, width_log2)."""
    if path is None:
        return np.zeros((0, 4))
    rows = list(csv.DictReader(open(path)))
    return np.array([[float(r["stime"]), float(r["dm"]), float(r["snr"]), float(r["width"])] for r in rows]).reshape(-1, 4)


def load_raw(path):
    """Tutti i candidati di Heimdall (.cand): array (t, dm, snr, width_log2)."""
    if path is None:
        return np.zeros((0, 4))
    a = np.loadtxt(path, ndmin=2)
    if a.size == 0:
        return np.zeros((0, 4))
    return np.column_stack([a[:, 2], a[:, 5], a[:, 0], a[:, 3]])


_RX = re.compile(r"tcand_(\d+\.?\d*)_dm_(\d+\.?\d*)_snr_(\d+\.?\d*)")


def load_fetch(path):
    """(t, dm, snr, prob) dai nomi dei candidati in results_*.csv."""
    if path is None:
        return np.zeros((0, 4))
    out = []
    for r in csv.DictReader(open(path)):
        m = _RX.search(r.get("candidate", ""))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), float(r["probability"])))
    return np.array(out).reshape(-1, 4)


def fetch_prob_for(sel, fetch):
    """Probabilita' FETCH per ogni candidato selezionato (nan se assente)."""
    pr = np.full(len(sel), np.nan)
    if len(fetch) == 0:
        return pr
    used = np.zeros(len(fetch), bool)
    for i, (t, dm, s, _) in enumerate(sel):
        j = np.flatnonzero((np.abs(fetch[:, 0] - t) < 1e-3) & (np.abs(fetch[:, 1] - dm) < 1e-2)
                           & (np.abs(fetch[:, 2] - s) < 1e-2) & ~used)
        if j.size:
            used[j[0]] = True
            pr[i] = fetch[j[0], 3]
    return pr


def read_plan(path, realization):
    rows = [r for r in csv.DictReader(open(path)) if int(r["realization"]) == realization]
    if not rows:
        raise SystemExit(f"nessun burst della realizzazione {realization} in {path}")
    return rows


# --------------------------------------------------------------------------- associazione
def match_one(toa, dm, width, cands, dt_tol, dm_frac, dm_abs):
    """Indice del candidato a S/N maggiore compatibile con il burst, o -1."""
    if len(cands) == 0:
        return -1
    ok = (np.abs(cands[:, 0] - toa) <= dt_tol + 2 * width) & \
         (np.abs(cands[:, 1] - dm) <= max(dm_frac * dm, dm_abs))
    j = np.flatnonzero(ok)
    return int(j[np.argmax(cands[j, 2])]) if j.size else -1


def removed_fraction(toa, dm, width, ev, f_lo, f_hi):
    """Frazione del tempo di arrivo del burst (ToA .. ToA + sweep, +- larghezza) coperta da eventi rimossi."""
    if ev is None or len(ev) == 0:
        return np.nan
    t0, t1 = toa - width, toa + KDM * dm * (f_lo ** -2 - f_hi ** -2) + width
    m = (ev[:, 0] < t1) & (ev[:, 0] + ev[:, 1] > t0)
    if not m.any():
        return 0.0
    iv = sorted((max(t0, a), min(t1, a + w)) for a, w in ev[m])
    cov, cur_a, cur_b = 0.0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a > cur_b:
            cov += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    cov += cur_b - cur_a
    return cov / (t1 - t0)


def cmd_match(a):
    plan = read_plan(a.plan, a.realization)
    toa = np.array([float(r["toa_s"]) for r in plan])
    dm = np.array([float(r["dm"]) for r in plan])
    wid = np.array([float(r["width_s"]) for r in plan])
    imp = {}
    for lab, path in (a.impulses or []):
        e = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        imp[lab] = np.column_stack([np.atleast_1d(e["t_s"]), np.atleast_1d(e["width_s"])])
    out_rows, false_rows = [], []
    for lab, d in a.step:
        sel_p, fetch_p, raw_p, best_p = find_files(Path(d))
        if sel_p is None:
            print(f"ATTENZIONE: {lab}: nessun *_all_candidates.csv in {d}", file=sys.stderr)
        sel, raw = load_selected(sel_p), load_raw(raw_p)
        pr = fetch_prob_for(sel, load_fetch(fetch_p))
        used = np.zeros(len(sel), bool)
        dts, ddms = [], []
        for i, r in enumerate(plan):
            j = match_one(toa[i], dm[i], wid[i], sel, a.dt_tol, a.dm_frac, a.dm_abs)
            k = match_one(toa[i], dm[i], wid[i], raw, a.dt_tol, a.dm_frac, a.dm_abs)
            if j >= 0:
                used[j] = True
                dts.append(sel[j, 0] - toa[i])
                ddms.append(sel[j, 1] - dm[i])
            out_rows.append(dict(
                realization=a.realization, toa_s=r["toa_s"], scan=r["scan"], point=r["point"],
                snr_expected=r["snr_expected"], snr_1ms=r["snr_1ms"], width_s=r["width_s"], dm=r["dm"],
                step=lab, detected=int(j >= 0),
                snr_found=f"{sel[j, 2]:.3f}" if j >= 0 else "", dm_found=f"{sel[j, 1]:.3f}" if j >= 0 else "",
                dt_found=f"{sel[j, 0] - toa[i]:.5f}" if j >= 0 else "",
                width_found=int(sel[j, 3]) if j >= 0 else "",
                fetch_prob=f"{pr[j]:.4f}" if j >= 0 and np.isfinite(pr[j]) else "",
                snr_raw=f"{raw[k, 2]:.3f}" if k >= 0 else "",
                removed_frac=(f"{removed_fraction(toa[i], dm[i], wid[i], imp[lab], a.f_lo, a.f_hi):.3f}"
                              if lab in imp else "")))
        # falsi candidati: selezionati e non associati ad alcun burst
        fp = ~used
        n_best = 0
        if best_p is not None:
            n_best = sum(1 for ln in open(best_p) if ln.strip().endswith((",1", ",1.0")))
        false_rows.append(dict(realization=a.realization, step=lab, n_selected=len(sel),
                               n_matched=int(used.sum()), n_false=int(fp.sum()),
                               n_false_fetch05=int(np.sum(fp & (pr >= 0.5))),
                               n_false_fetch07=int(np.sum(fp & (pr >= 0.7))),
                               n_fetch07=int(np.sum(pr >= 0.7)), n_best_file=n_best,
                               n_raw=len(raw)))
        det = sum(1 for x in out_rows if x["step"] == lab and x["detected"])
        msg = f"{lab:>8s}: rivelati {det}/{len(plan)}, falsi {int(fp.sum())} (FETCH>=0.7: {int(np.sum(fp & (pr >= 0.7)))})"
        if dts:
            msg += (f"; scarti dt mediana {np.median(dts) * 1e3:+.2f} ms (p5..p95 {np.percentile(dts, 5) * 1e3:+.1f}.."
                    f"{np.percentile(dts, 95) * 1e3:+.1f}), dDM mediana {np.median(ddms):+.2f} "
                    f"(p5..p95 {np.percentile(ddms, 5):+.1f}..{np.percentile(ddms, 95):+.1f})")
        print(msg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    fpath = a.out.with_name(a.out.stem + "_false.csv")
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(false_rows[0]))
        w.writeheader()
        w.writerows(false_rows)
    print(f"Scritti {a.out} e {fpath}")
    return 0


# --------------------------------------------------------------------------- resoconto
def wilson(k, n, z=1.0):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def _xval(scan, r):
    if scan == "A":
        return float(r["snr_expected"])
    if scan in ("B", "C"):
        return float(r["width_s"]) * 1e3
    if scan == "D":
        return float(r["dm"])
    return None


def cmd_report(a):
    rows = []
    for pat in a.matched:
        for p in sorted(glob.glob(pat)):
            if p.endswith("_false.csv"):
                continue
            rows += list(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit("nessun file matched")
    steps = a.steps or list(dict.fromkeys(r["step"] for r in rows))
    a.out.mkdir(parents=True, exist_ok=True)
    table = []
    for scan in sorted({r["scan"] for r in rows}):
        for st in steps:
            sub = [r for r in rows if r["scan"] == scan and r["step"] == st]
            for pt in dict.fromkeys(r["point"] for r in sub):
                g = [r for r in sub if r["point"] == pt]
                n, k = len(g), sum(int(r["detected"]) for r in g)
                p, lo, hi = wilson(k, n)
                ratio = [float(r["snr_raw"]) / float(r["snr_expected"]) for r in g if r["snr_raw"]]
                fp = [float(r["fetch_prob"]) for r in g if r["fetch_prob"]]
                rem = [float(r["removed_frac"]) for r in g if r["removed_frac"]]
                table.append(dict(scan=scan, step=st, point=pt, x=_xval(scan, g[0]),
                                  snr_expected=float(g[0]["snr_expected"]), width_ms=float(g[0]["width_s"]) * 1e3,
                                  dm=float(g[0]["dm"]), n=n, detected=k, completeness=p, c_lo=lo, c_hi=hi,
                                  snr_ratio_median=float(np.median(ratio)) if ratio else np.nan,
                                  n_snr_raw=len(ratio),
                                  fetch07_frac=float(np.mean(np.array(fp) >= 0.7)) if fp else np.nan,
                                  removed_frac_median=float(np.median(rem)) if rem else np.nan))
    with open(a.out / "completeness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        for t in table:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in t.items()})
    # falsi candidati
    frows = []
    for pat in a.matched:
        for p in sorted(glob.glob(pat)):
            q = p if p.endswith("_false.csv") else p[:-4] + "_false.csv"
            if Path(q).exists() and q not in [x.get("_src") for x in frows]:
                for r in csv.DictReader(open(q)):
                    r["_src"] = q
                    frows.append(r)
    seen, fl = set(), []
    for r in frows:
        key = (r["realization"], r["step"])
        if key not in seen:
            seen.add(key)
            fl.append(r)
    if fl:
        keys = [k for k in fl[0] if k != "_src"]
        with open(a.out / "false_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(fl)
        print("Falsi candidati per file (selezionati dalla pipeline e non associati a un burst):")
        print(f"  {'realizz.':>8s} {'step':>8s} {'selez.':>7s} {'associati':>9s} {'falsi':>6s} "
              f"{'falsi F>=0.5':>12s} {'falsi F>=0.7':>12s}")
        for r in fl:
            print(f"  {r['realization']:>8s} {r['step']:>8s} {r['n_selected']:>7s} {r['n_matched']:>9s} "
                  f"{r['n_false']:>6s} {r['n_false_fetch05']:>12s} {r['n_false_fetch07']:>12s}")
        for st in steps:
            v = [int(r["n_false"]) for r in fl if r["step"] == st]
            if v:
                print(f"  {st:>8s}: falsi per file, media {np.mean(v):.1f} (min {min(v)}, max {max(v)})")
    _plots(table, steps, a.out)
    print(f"Scritti {a.out / 'completeness.csv'}" + (", false_candidates.csv" if fl else "") + " e i grafici")
    return 0


def _plots(table, steps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = dict(A=("S/N atteso (w = 1 ms, DM 250)", False), B=("larghezza [ms] (flusso fisso, S/N$_{1ms}$ = 20, DM 250)", True),
                  C=("larghezza [ms] (S/N = 20, DM 250)", True), D=("DM [pc cm$^{-3}$] (w = 1 ms, S/N = 20)", True))
    for scan, (xl, logx) in labels.items():
        t = [r for r in table if r["scan"] == scan]
        if not t:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for st in steps:
            s = sorted([r for r in t if r["step"] == st], key=lambda r: r["x"])
            if not s:
                continue
            x = np.array([r["x"] for r in s])
            c = np.array([r["completeness"] for r in s])
            e = np.array([[r["completeness"] - r["c_lo"] for r in s], [r["c_hi"] - r["completeness"] for r in s]])
            ax[0].errorbar(x, c, yerr=e, marker="o", ms=4, capsize=2, label=st)
            ax[1].plot(x, [r["snr_ratio_median"] for r in s], marker="o", ms=4, label=st)
        if scan == "B":
            for r in sorted([r for r in t if r["step"] == steps[0]], key=lambda r: r["x"]):
                ax[0].annotate(f"{r['snr_expected']:.0f}", (r["x"], 1.04), fontsize=7, ha="center")
            ax[0].set_title("numeri in alto: S/N atteso", fontsize=8)
        for k in (0, 1):
            ax[k].set_xlabel(xl)
            if logx:
                ax[k].set_xscale("log")
        ax[0].set_ylabel("completezza (S/N > 8, DM > 28)")
        ax[0].set_ylim(-0.03, 1.1)
        ax[1].set_ylabel("S/N Heimdall / S/N atteso (mediana)")
        ax[1].axhline(1, color="k", lw=0.5)
        ax[0].legend(fontsize=8)
        fig.suptitle(f"Scansione {scan}")
        fig.tight_layout()
        fig.savefig(out / f"scan_{scan}.png", dpi=120)
        plt.close(fig)
    t = [r for r in table if r["scan"] == "E"]
    if t:
        dms = sorted({r["dm"] for r in t})
        ws = sorted({r["width_ms"] for r in t})
        fig, ax = plt.subplots(1, len(steps), figsize=(3.6 * len(steps), 3.4), squeeze=False)
        for k, st in enumerate(steps):
            M = np.full((len(dms), len(ws)), np.nan)
            for r in t:
                if r["step"] == st:
                    M[dms.index(r["dm"]), ws.index(r["width_ms"])] = r["completeness"]
            im = ax[0, k].imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
            for i in range(len(dms)):
                for j in range(len(ws)):
                    if np.isfinite(M[i, j]):
                        ax[0, k].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                                      color="w" if M[i, j] < 0.6 else "k")
            ax[0, k].set_xticks(range(len(ws)), [f"{w:g}" for w in ws])
            ax[0, k].set_yticks(range(len(dms)), [f"{d:g}" for d in dms])
            ax[0, k].set_xlabel("larghezza [ms]")
            ax[0, k].set_title(st, fontsize=9)
        ax[0, 0].set_ylabel("DM")
        fig.colorbar(im, ax=ax[0, -1], label="completezza")
        fig.suptitle("Scansione E (S/N = 20)")
        fig.tight_layout()
        fig.savefig(out / "scan_E.png", dpi=120)
        plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("match", help="associa candidati e burst di una realizzazione")
    m.add_argument("--plan", required=True, type=Path)
    m.add_argument("--realization", required=True, type=int)
    m.add_argument("--step", nargs=2, action="append", required=True, metavar=("ETICHETTA", "CARTELLA"),
                   help="step e cartella di output della ricerca (ripetibile, nell'ordine della pipeline)")
    m.add_argument("--impulses", nargs=2, action="append", metavar=("ETICHETTA", "CSV"),
                   help="impulses_events.csv dello step 3 per lo step con questa etichetta (diagnostica)")
    m.add_argument("--dt-tol", type=float, default=0.02, help="tolleranza in tempo [s] oltre 2 x larghezza (default 0.02)")
    m.add_argument("--dm-frac", type=float, default=0.2, help="tolleranza relativa in DM (default 0.2)")
    m.add_argument("--dm-abs", type=float, default=10.0, help="tolleranza minima in DM (default 10)")
    m.add_argument("--f-lo", type=float, default=1382.25, help="bordo inferiore della banda [MHz] (diagnostica)")
    m.add_argument("--f-hi", type=float, default=1446.25, help="bordo superiore della banda [MHz] (diagnostica)")
    m.add_argument("--out", required=True, type=Path)
    r = sp.add_parser("report", help="completezza e falsi candidati da uno o piu' file matched")
    r.add_argument("--matched", nargs="+", required=True, help="file o pattern (es. 'camp/matched_r*.csv')")
    r.add_argument("--steps", nargs="+", default=None, help="ordine degli step nei grafici")
    r.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    return cmd_match(a) if a.cmd == "match" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
