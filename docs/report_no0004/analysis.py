#!/usr/bin/env python3
"""
Analisi riproducibile per il report su ag278m_mc_no0004_processed.fil.

Input (cartella data/ accanto a questo script, piu' survey.npz dal repo frb-rfi-data):
  summary.json, events.csv, periodicity.csv, flagged_channels.txt   -> output di rfi_survey.py (file completo)
  heimdall_all_candidates.csv                                        -> candidati Heimdall (dopo csv_from_cand.py)
  fetch_results_a.csv                                                -> output FETCH, modello a

Output:
  fig/*.png        figure del report
  numbers.json     tutti i numeri citati nel testo
  tab_*.tex        tabelle LaTeX incluse nel report

Uso:
  python3 analysis.py --npz /percorso/frb-rfi-data/surveys/ag278m_mc_no0004_full/survey.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from frbclean.survey import Event  # noqa: E402

KDM = 4.148808e3
NUM: dict = {}


def put(key, val):
    if isinstance(val, (np.floating, np.integer)):
        val = val.item()
    NUM[key] = val
    return val


# --------------------------------------------------------------------------- caricamento
def load_events(path):
    ev = list(csv.DictReader(open(path)))
    return dict(
        t=np.array([float(e["t"]) for e in ev]),
        w=np.array([float(e["width"]) for e in ev]),
        snr=np.array([float(e["snr_max"]) for e in ev]),
        snr_full=np.array([float(e["snr_full"]) for e in ev]),
        fb=np.array([float(e["frac_band"]) for e in ev]),
        best=np.array([e["best"] for e in ev]),
        hits=np.array([e["sub_hits"] for e in ev]),
        raw=ev)


def load_cands(cpath, fpath):
    rows = list(csv.DictReader(open(cpath)))
    c = dict(
        t=np.array([float(r["stime"]) for r in rows]),
        dm=np.array([float(r["dm"]) for r in rows]),
        snr=np.array([float(r["snr"]) for r in rows]),
        wexp=np.array([int(r["width"]) for r in rows]))
    # abbinamento uno-a-uno su (tempo, DM, S/N): la sola coppia (tempo, DM) non e' univoca
    ft = []
    for r in csv.DictReader(open(fpath)):
        m = re.search(r"tcand_([\d.]+)_dm_([\d.]+)_snr_(\d+\.\d+)", r["candidate"])
        ft.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), float(r["probability"])))
    ft = np.array(ft)
    used = np.zeros(len(ft), bool)
    c["prob"] = np.full(c["t"].size, np.nan)
    for i, (t, d, s) in enumerate(zip(c["t"], c["dm"], c["snr"])):
        j = np.flatnonzero((np.abs(ft[:, 0] - t) < 1e-4) & (np.abs(ft[:, 1] - d) < 1e-3)
                           & (np.abs(ft[:, 2] - s) < 1e-3) & ~used)
        if j.size:
            used[j[0]] = True
            c["prob"][i] = ft[j[0], 3]
    prob = ft
    c["n_fetch_rows"] = len(prob)
    return c


# --------------------------------------------------------------------------- analisi
def setup(summ):
    f_lo, f_hi = summ["band_mhz"]
    put("nchans", summ["nchans"]); put("tsamp_us", summ["tsamp"] * 1e6); put("nbits", summ["nbits"])
    put("duration_s", summ["duration_s"]); put("f_lo", f_lo); put("f_hi", f_hi)
    put("bw", f_hi - f_lo); put("foff", (f_hi - f_lo) / summ["nchans"])
    k = KDM * (f_lo ** -2 - f_hi ** -2)
    put("sweep_ms_per_dm", k * 1e3)
    fc = 0.5 * (f_lo + f_hi)
    put("smear_us_per_dm_center", 8.3e3 * NUM["foff"] / fc ** 3 * 1e6)
    put("dm_eq_width", {f"{w}": w * 1e-3 / k for w in (0.5, 1, 5, 10)})
    put("sweep_ms", {str(d): k * d * 1e3 for d in (25, 28, 50, 100, 300, 500, 700)})
    put("data_rate_MBs", summ["nchans"] * summ["nbits"] / 8 / summ["tsamp"] / 1e6)


def zerodm_noise(d):
    Z = d["zerodm"].astype(np.float64)
    dt = float(d["zerodm_dt"])
    nper = int(round(dt / NUM["tsamp_us"] * 1e6))
    full = Z[:, 0]
    sig_diff = 1.4826 * np.median(np.abs(np.diff(full))) / np.sqrt(2)
    chan_std = np.median(d["chan_std"], 0)
    sig_exp = np.sqrt(np.sum(chan_std ** 2)) / np.sqrt(nper)
    put("zerodm_bin_ms", dt * 1e3); put("zerodm_samples_per_bin", nper)
    put("zerodm_sigma_measured", sig_diff); put("zerodm_sigma_expected_indep", sig_exp)
    put("zerodm_noise_excess", sig_diff / sig_exp)
    put("zerodm_median_level", float(np.median(full)))
    # serie con baseline mobile di 2 s (mediana mobile, come la baseline di Heimdall di default)
    from scipy.ndimage import median_filter
    n2 = int(round(2.0 / dt)) | 1
    base = median_filter(full, size=n2, mode="nearest")
    r = full - base
    out = {}
    for wms in (1, 8, 32, 128):
        k = max(1, int(round(wms * 1e-3 / dt)))
        b = np.convolve(r, np.ones(k), "valid")
        s_white = b / (sig_diff * np.sqrt(k))
        s_rob = b / (1.4826 * np.median(np.abs(b - np.median(b))))
        out[str(wms)] = dict(frac_white_gt8=float((np.abs(s_white) > 8).mean()),
                             frac_robust_gt8=float((np.abs(s_rob) > 8).mean()),
                             max_white=float(np.abs(s_white).max()), max_robust=float(np.abs(s_rob).max()),
                             robust_over_white=float(1.4826 * np.median(np.abs(b - np.median(b))) / (sig_diff * np.sqrt(k))))
    put("baseline2s_residual", out)
    return r


def channel_summary(d, summ):
    fl = summ["flagged_channels"]
    freqs = d["freqs"]
    rows = []
    for ch, reasons in fl.items():
        c = int(ch)
        rows.append((c, freqs[c], "; ".join(reasons)))
    put("flagged", [dict(ch=r[0], f=float(r[1]), why=r[2]) for r in rows])
    kmed = np.median(d["chan_kurt"], 0)
    put("kurt_median_all", float(np.median(kmed)))
    put("kurt_range_p10_p90", [float(np.percentile(kmed, 10)), float(np.percentile(kmed, 90))])
    with open(HERE / "tab_channels.tex", "w") as f:
        for c, fr, why in rows:
            why = why.replace("%", r"\%").replace("rumore", "rumore").replace("z=", "$z=$")
            f.write(f"{c} & {fr:.3f} & {why} \\\\\n")


def radar(ev, nsub=16):
    t, s, fb, hits = ev["t"], ev["snr"], ev["fb"], ev["hits"]
    lower = np.array([h[: nsub // 2].count("1") >= 7 and h[nsub // 2:].count("1") <= 2 for h in hits])
    classes = {"A": (s >= 30) & (fb >= 0.9), "B": (s >= 30) & lower}
    res = {}
    for name, sel in classes.items():
        tt = np.sort(t[sel])
        # passaggi: gruppi di eventi separati da < 1 s
        g = np.split(tt, np.flatnonzero(np.diff(tt) > 1.0) + 1)
        first = np.array([x[0] for x in g])
        # scansione di Rayleigh sui primi impulsi di ogni passaggio, 5-15 s
        P = np.linspace(5, 15, 20001)
        Z = np.abs(np.exp(2j * np.pi * np.outer(1 / P, first)).sum(1)) ** 2 / first.size
        P0 = P[np.argmax(Z)]
        # fit lineare dei tempi dei passaggi coerenti con il periodo
        ph = ((first - first[0]) / P0)
        n = np.round(ph)
        resid = (ph - n) * P0
        good = np.abs(resid - np.median(resid)) < 0.3
        A = np.vstack([np.ones(good.sum()), n[good]]).T
        coef, *_ = np.linalg.lstsq(A, first[good], rcond=None)
        rr = first[good] - A @ coef
        sig = rr.std(ddof=2) if good.sum() > 2 else np.nan
        cov = sig ** 2 * np.linalg.inv(A.T @ A)
        ncyc = int(n.max() - n.min() + 1)
        res[name] = dict(n_events=int(sel.sum()), n_passages=int(first.size), n_coherent=int(good.sum()),
                         period_scan=float(P0), rayleigh_z=float(Z.max()),
                         period_fit=float(coef[1]), period_err=float(np.sqrt(cov[1, 1])),
                         t0=float(coef[0]), resid_rms_s=float(sig), cycles_spanned=ncyc,
                         phase_s=float(coef[0] % coef[1]),
                         pulses_per_passage_median=float(np.median([len(x) for x in g])),
                         intra_passage_spacing_ms=float(np.median(np.concatenate([np.diff(x) for x in g if len(x) > 1])) * 1e3)
                         if any(len(x) > 1 for x in g) else None)
        res[name]["_times"] = tt
        res[name]["_first"] = first
    put("radar", {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()})
    return res


def candidates(c):
    t, dm, snr, wexp, pr = c["t"], c["dm"], c["snr"], c["wexp"], c["prob"]
    put("cand_n", int(t.size)); put("cand_t_range", [float(t.min()), float(t.max())])
    put("cand_snr", dict(min=float(snr.min()), median=float(np.median(snr)), max=float(snr.max())))
    put("cand_dm_percentiles", dict(zip(["p5", "p25", "p50", "p75", "p95"], np.percentile(dm, [5, 25, 50, 75, 95]).tolist())))
    u, n = np.unique(wexp, return_counts=True)
    put("cand_width_counts", {int(a): int(b) for a, b in zip(u, n)})
    put("cand_frac_width_le3", float((wexp <= 3).mean()))
    put("fetch_rows", c["n_fetch_rows"]); put("fetch_missing", int(np.isnan(pr).sum()))
    put("fetch_gt05", int((pr > 0.5).sum())); put("fetch_ge07", int((pr >= 0.7).sum()))
    put("fetch_max", float(np.nanmax(pr)))
    put("fetch_p99", float(np.nanpercentile(pr, 99))); put("fetch_median", float(np.nanmedian(pr)))
    top = np.argsort(-np.nan_to_num(pr, nan=-1))[:5]
    put("fetch_top", [dict(t=float(t[i]), dm=float(dm[i]), snr=float(snr[i]), wexp=int(wexp[i]),
                           w_ms=float(2 ** wexp[i] * NUM["tsamp_us"] * 1e-3), p=float(pr[i])) for i in top])
    # episodi
    o = np.argsort(t)
    ts = t[o]
    groups = np.split(o, np.flatnonzero(np.diff(ts) > 0.5) + 1)
    big = [g for g in groups if g.size >= 20]
    eps = []
    for g in big:
        eps.append(dict(t0=float(t[g].min()), t1=float(t[g].max()), n=int(g.size),
                        dm_min=float(dm[g].min()), dm_max=float(dm[g].max()), dm_med=float(np.median(dm[g])),
                        wexp_med=float(np.median(wexp[g])), snr_max=float(snr[g].max()),
                        p_max=float(np.nanmax(pr[g])) if np.isfinite(pr[g]).any() else None))
    put("episodes", eps)
    put("episodes_n_cand", int(sum(e["n"] for e in eps)))
    rest = np.setdiff1d(np.arange(t.size), np.concatenate(big)) if big else np.arange(t.size)
    put("rest_n", int(rest.size)); put("rest_rate_per_min", rest.size / (NUM["duration_s"] / 60))
    put("rest_wexp_median", float(np.median(wexp[rest]))); put("rest_dm_median", float(np.median(dm[rest])))
    return eps


def episode_physics(d, eps):
    """Livello e rumore per sottobanda attorno agli episodi."""
    Z = d["zerodm"].astype(np.float64)
    dt = float(d["zerodm_dt"])
    sf = d["sub_freq"]
    o = np.argsort(sf)
    S = Z[:, 1:][:, o]
    med = np.median(S, 0)
    k = int(round(0.25 / dt))
    nb = S.shape[0] // k
    X = S[: nb * k].reshape(nb, k, -1)
    sd = X.std(1)
    lv = X.mean(1) / med
    ref_sd = np.median(sd, 0)
    tb = (np.arange(nb) + 0.5) * k * dt
    rows = []
    for e in eps:
        a, b = e["t0"], e["t1"]
        win = (tb > a - 1.5) & (tb < b + 1.5)
        pre = (tb > a - 6) & (tb < a - 2)
        edge = [0, 1, 14, 15]
        lv_rel = lv[win] / lv[pre].mean(0)
        dlev_edge = lv_rel[:, edge].mean(1)
        dlev_mid = lv_rel[:, 2:14].mean(1)
        r1416 = sd[:, 8] / ref_sd[8]
        noise1416 = r1416[win].max()
        # probabilita' empirica che una finestra casuale della stessa lunghezza abbia un massimo >= osservato
        L = int(win.sum())
        from numpy.lib.stride_tricks import sliding_window_view as swv
        wmax = swv(r1416, L).max(1)
        edge_lv = lv[:, edge].mean(1)
        jump = np.abs(np.diff(edge_lv, prepend=edge_lv[0]))
        jmax = swv(jump, L).max(1)
        rows.append(dict(t0=a, edge_min=float(dlev_edge.min()), edge_max=float(dlev_edge.max()),
                         mid_min=float(dlev_mid.min()), mid_max=float(dlev_mid.max()),
                         noise1416_max=float(noise1416), p_noise=float((wmax >= noise1416).mean()),
                         jump_max=float(jump[win].max()), p_jump=float((jmax >= jump[win].max()).mean())))
    put("episode_physics", rows)
    put("sub_freq_sorted", sf[o].tolist())
    return dict(S=S, lv=lv, sd=sd, ref_sd=ref_sd, tb=tb, k=k, dt=dt)


def correlation(c, P):
    lv, sd, ref_sd, tb, k, dt = P["lv"], P["sd"], P["ref_sd"], P["tb"], P["k"], P["dt"]
    nb = tb.size
    cnt = np.histogram(c["t"], bins=np.arange(nb + 1) * k * dt)[0]
    r1416 = sd[:, 8] / ref_sd[8]
    edge = lv[:, [0, 1, 14, 15]].mean(1)
    jump = np.abs(np.diff(edge, prepend=edge[0]))
    out = []
    for thr_n, thr_j in ((2, 0.01), (3, 0.02), (5, 0.05)):
        A = r1416 > thr_n
        B = jump > thr_j
        for lab, m in (("A", A), ("B", B), ("AoB", A | B), ("none", ~(A | B))):
            out.append(dict(thr_noise=thr_n, thr_jump=thr_j, cond=lab, frac_time=float(m.mean()),
                            n_cand=int(cnt[m].sum()), frac_cand=float(cnt[m].sum() / cnt.sum()),
                            density_ratio=float(cnt[m].mean() / cnt.mean()) if m.any() else None))
    # stessa analisi escludendo gli episodi a 78 s e 197 s (dominano i conteggi)
    excl = np.zeros(nb, bool)
    for a, b in ((77.5, 79.0), (196.5, 197.5)):
        excl |= (tb > a) & (tb < b)
    out_x = []
    for thr_n, thr_j in ((3, 0.02),):
        A = (r1416 > thr_n) & ~excl
        B = (jump > thr_j) & ~excl
        keep = ~excl
        for lab, m in (("A", A), ("B", B), ("AoB", A | B), ("none", ~(A | B) & keep)):
            out_x.append(dict(cond=lab, frac_time=float(m.sum() / keep.sum()), n_cand=int(cnt[m].sum()),
                              frac_cand=float(cnt[m].sum() / cnt[keep].sum()),
                              density_ratio=float(cnt[m].mean() / cnt[keep].mean())))
    put("correlation_excl_78_197", out_x)
    put("correlation", out)
    with open(HERE / "tab_correlation.tex", "w") as f:
        names = {"A": r"rumore 1414--1418\,MHz $>X\sigma_{\rm rif}$", "B": r"salto di livello ai bordi $>Y$",
                 "AoB": "A oppure B", "none": "n\\'e A n\\'e B"}
        for r in out:
            f.write(f"{r['thr_noise']} & {100 * r['thr_jump']:.0f}\\% & {names[r['cond']]} & "
                    f"{100 * r['frac_time']:.1f}\\% & {r['n_cand']} & {100 * r['frac_cand']:.1f}\\% & "
                    f"{r['density_ratio']:.2f} \\\\\n")
            if r["cond"] == "none" and r is not out[-1]:
                f.write("\\midrule\n")
    return cnt, r1416, jump


# --------------------------------------------------------------------------- figure
def figures(d, ev, c, P, rad, eps, cnt, r1416, jump, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frbclean import survey_plots

    figdir = HERE / "fig"
    # (1) figure standard del survey, rigenerate dai dati
    events = [Event(t=float(e["t"]), width=float(e["width"]), snr_full=float(e["snr_full"]),
                    snr_max=float(e["snr_max"]), best=e["best"], n_sub_hit=int(e["n_sub_hit"]),
                    frac_band=float(e["frac_band"]), sub_hits=e["sub_hits"]) for e in ev["raw"]]

    class FB:
        path = summ["file"]
        foff = -NUM["foff"]
    tmp = figdir / "survey"
    tmp.mkdir(exist_ok=True)
    np.savez(tmp / "survey.npz", **{k: d[k] for k in d.files})
    survey_plots.make_all(tmp, summ, events, FB)
    (tmp / "survey.npz").unlink()

    # (2) livelli per sottobanda attorno a 4 episodi
    S, dt = P["S"], P["dt"]
    med = np.median(S, 0)
    sf = np.array(NUM["sub_freq_sorted"])
    sel = [e for e in eps if round(e["t0"]) in (78, 197, 604, 1027)]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    kb = int(round(0.02 / dt))
    cmap = plt.get_cmap("viridis")
    for ax, e in zip(axs.ravel(), sel):
        a, b = e["t0"] - 3, e["t1"] + 3
        i, j = int(a / dt), int(b / dt)
        x = S[i:j]
        n = x.shape[0] // kb
        x = x[: n * kb].reshape(n, kb, -1).mean(1) / med
        tt = a + (np.arange(n) + 0.5) * kb * dt
        for s in range(16):
            ax.plot(tt, x[:, s], lw=0.8, color=cmap(s / 15), label=f"{sf[s]:.0f}" if s in (0, 5, 8, 10, 15) else None)
        ct = c["t"][(c["t"] > a) & (c["t"] < b)]
        ax.plot(ct, np.full(ct.size, 0.72), "|", color="red", ms=8, alpha=0.5)
        ax.set_title(f"episodio {e['t0']:.2f}-{e['t1']:.2f} s ({e['n']} candidati)", fontsize=9)
        ax.set_xlabel("tempo [s]")
        ax.axvspan(e["t0"], e["t1"], color="red", alpha=0.08)
    axs[0, 0].set_ylabel("livello / mediana")
    axs[1, 0].set_ylabel("livello / mediana")
    axs[0, 0].legend(title="MHz", fontsize=7, ncol=2)
    fig.suptitle("Livello delle 16 sottobande (bin 20 ms); trattini rossi = candidati Heimdall", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "episodes_subbands.png", dpi=130)
    plt.close(fig)

    # (3) candidati
    t, dm, snr, wexp, pr = c["t"], c["dm"], c["snr"], c["wexp"], c["prob"]
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(3, 6)
    ax = fig.add_subplot(gs[0, :])
    sc = ax.scatter(t, dm, c=wexp, s=4, cmap="plasma")
    ax.set_yscale("log")
    ax.set_xlabel("tempo [s]"); ax.set_ylabel("DM [pc cm$^{-3}$]")
    fig.colorbar(sc, ax=ax, label=r"log$_2$(larghezza/campioni)")
    for e in eps:
        ax.axvspan(e["t0"] - 0.5, e["t1"] + 0.5, color="red", alpha=0.15)
    ax.set_title(f"{t.size} candidati Heimdall (S/N > 8, DM > 28); in rosso gli episodi con $\\geq$20 candidati", fontsize=9)
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.bar(list(NUM["cand_width_counts"].keys()), list(NUM["cand_width_counts"].values()))
    ax2.set_xlabel(r"log$_2$(larghezza/campioni), 1 campione = 16 $\mu$s"); ax2.set_ylabel("candidati")
    ax3 = fig.add_subplot(gs[1, 2:4])
    ax3.hist(dm, bins=np.geomspace(25, 750, 30)); ax3.set_xscale("log")
    ax3.set_xticks([30, 50, 100, 200, 500]); ax3.set_xticklabels(["30", "50", "100", "200", "500"])
    ax3.minorticks_off()
    ax3.set_xlabel("DM [pc cm$^{-3}$]")
    ax4 = fig.add_subplot(gs[1, 4:6])
    ax4.hist(snr, bins=np.arange(8, 26, 0.5)); ax4.set_yscale("log"); ax4.set_xlabel("S/N Heimdall")
    ax5 = fig.add_subplot(gs[2, 0:3])
    p = pr[np.isfinite(pr)]
    ax5.hist(np.log10(np.clip(p, 1e-12, 1)), bins=60)
    ax5.axvline(np.log10(0.7), color="r", ls="--", label="soglia 0.7")
    ax5.set_xlabel("log$_{10}$ probabilita' FETCH (modello a)"); ax5.set_ylabel("candidati")
    ax5.set_yscale("log"); ax5.legend(fontsize=8)
    ax6 = fig.add_subplot(gs[2, 3:6])
    ax6.plot(np.sort(t), np.arange(1, t.size + 1), lw=1)
    ax6.set_xlabel("tempo [s]"); ax6.set_ylabel("candidati cumulativi")
    fig.savefig(figdir / "candidates.png", dpi=130)
    plt.close(fig)

    # (4) correlazione: candidati per bin vs condizioni
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    tb = P["tb"]
    ax[0].plot(tb, r1416, lw=0.6, label="rumore sottobanda 1414-1418 MHz / riferimento")
    ax[0].plot(tb, 1 + 100 * jump, lw=0.6, label="1 + salto di livello ai bordi [%]")
    ax[0].axhline(3, color="k", ls=":", lw=0.8)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_ylabel("indicatori (bin 0.25 s)")
    ax[1].plot(tb, cnt, drawstyle="steps-mid", color="tab:red", lw=0.8)
    ax[1].set_yscale("symlog"); ax[1].set_ylabel("candidati per bin"); ax[1].set_xlabel("tempo [s]")
    fig.tight_layout()
    fig.savefig(figdir / "correlation.png", dpi=130)
    plt.close(fig)

    # (5) radar: fase dei passaggi
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axs, ("A", "B")):
        r = rad[name]
        Pf = NUM["radar"][name]["period_fit"]
        t0 = NUM["radar"][name]["t0"]
        ph = ((r["_times"] - t0) / Pf + 0.5) % 1 - 0.5
        ax.scatter(ph * Pf, r["_times"], s=5)
        ax.set_xlabel(f"tempo dal passaggio previsto [s] (P = {Pf:.4f} s)")
        ax.set_ylabel("tempo [s]")
        ax.set_xlim(-1, 1)
        ax.set_title(f"famiglia {name}: {r['_times'].size} eventi S/N$\\geq$30, {r['_first'].size} passaggi", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "radar.png", dpi=130)
    plt.close(fig)


def fp(p):
    return f"{p:.2f}" if p >= 0.01 else "$<$0.01"


def tables(eps):
    with open(HERE / "tab_episodes.tex", "w") as f:
        for e, p in zip(eps, NUM["episode_physics"]):
            pm = "--" if e["p_max"] is None else f"{e['p_max']:.3f}"
            f.write(f"{e['t0']:.2f}--{e['t1']:.2f} & {e['n']} & {e['dm_min']:.0f}--{e['dm_max']:.0f} & "
                    f"{e['wexp_med']:.0f} & {e['snr_max']:.1f} & {pm} & "
                    f"{100 * (p['edge_min'] - 1):+.0f}/{100 * (p['edge_max'] - 1):+.0f} & "
                    f"{100 * (p['mid_min'] - 1):+.0f}/{100 * (p['mid_max'] - 1):+.0f} & {p['noise1416_max']:.1f} ({fp(p['p_noise'])}) & "
                    f"{100 * p['jump_max']:.1f} ({fp(p['p_jump'])}) \\\\\n")
    with open(HERE / "tab_periodicity.tex", "w") as f:
        rows = list(csv.DictReader(open(HERE / "data" / "periodicity.csv")))
        names = {"zerodm": "DM 0", "event_train": "eventi", "event_train_broadband": "eventi banda larga"}
        for r in rows:
            if float(r["sigma"]) < 300 and r["series"] != "event_train_broadband":
                continue
            z = "" if r["rayleigh_z"] == "nan" else f"{float(r['rayleigh_z']):.0f}"
            f.write(f"{names[r['series']]} & {float(r['freq_hz']):.6f} & {float(r['period_s']):.6f} & "
                    f"{float(r['strongest_harmonic_hz']):.3f} & {float(r['sigma']):.0f} & {z} \\\\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    d = np.load(a.npz)
    summ = json.load(open(HERE / "data" / "summary.json"))
    ev = load_events(HERE / "data" / "events.csv")
    c = load_cands(HERE / "data" / "heimdall_all_candidates.csv", HERE / "data" / "fetch_results_a.csv")
    setup(summ)
    put("n_events", summ["n_events"]); put("events_rate_per_s", summ["n_events"] / summ["duration_s"])
    put("events_per_hour_above", summ["events_per_hour_above"])
    put("fraction_time_in_events", summ["fraction_time_in_events"])
    put("events_by_band_fraction", summ["events_by_band_fraction"])
    put("survey_runtime_s", summ["runtime_s"])
    zerodm_noise(d)
    channel_summary(d, summ)
    rad = radar(ev)
    eps = candidates(c)
    P = episode_physics(d, eps)
    cnt, r1416, jump = correlation(c, P)
    tables(eps)
    figures(d, ev, c, P, rad, eps, cnt, r1416, jump, summ)
    (HERE / "numbers.json").write_text(json.dumps(NUM, indent=2))
    print(json.dumps({k: NUM[k] for k in ("zerodm_noise_excess", "radar", "correlation", "episode_physics")}, indent=1))


if __name__ == "__main__":
    main()
