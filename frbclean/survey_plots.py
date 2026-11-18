"""Grafici diagnostici per run_survey (richiede matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _shade_flagged(ax, freqs, flagged, df):
    for c in flagged:
        ax.axvspan(freqs[int(c)] - df / 2, freqs[int(c)] + df / 2, color="tab:red", alpha=0.15, lw=0)


def make_all(out: Path, summary: dict, events, fb):
    d = np.load(out / "survey.npz")
    freqs = d["freqs"]
    df = abs(fb.foff)
    flagged = list(summary["flagged_channels"].keys())
    title = f"{Path(fb.path).name}  ({summary['duration_s']:.0f} s)"

    # --- canali
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for a, key, lab in zip(ax, ("median", "std", "kurt", "outfrac"),
                           ("mediana (bandpass)", "deviazione standard", "kurtosis in eccesso", "frazione outlier >6σ")):
        v = d[f"chan_{key}"]
        a.plot(freqs, np.median(v, 0), lw=1, label="mediana nel tempo")
        a.fill_between(freqs, np.percentile(v, 10, 0), np.percentile(v, 90, 0), alpha=0.3, lw=0, label="10-90%")
        if key in ("kurt", "outfrac"):
            a.plot(freqs, v.max(0), lw=0.6, color="k", alpha=0.6, label="massimo")
        if key == "kurt":
            a.set_yscale("symlog", linthresh=0.1)
        if key == "outfrac":
            a.set_ylim(bottom=0)
            a.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        _shade_flagged(a, freqs, flagged, df)
        a.set_ylabel(lab)
    ax[0].legend(fontsize=8)
    ax[0].set_title(title + f" — canali sospetti (rosso): {len(flagged)}")
    ax[-1].set_xlabel("frequenza [MHz]")
    fig.tight_layout()
    fig.savefig(out / "channels.png", dpi=110)
    plt.close(fig)

    # --- spettri dinamici ridotti
    med = np.median(d["chan_median"], 0)
    sig = np.median(d["chan_std"], 0)
    sig = np.where(sig > 0, sig, 1)
    dt = float(d["dyn_dt"])
    mean = (d["dyn_mean"] - med) / sig
    mx = (d["dyn_max"] - med) / sig
    order = np.argsort(freqs)
    ext = [0, mean.shape[0] * dt, freqs.min() - df / 2, freqs.max() + df / 2]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for a, img, lab in ((ax[0], mean, f"media per bin ({dt:.3g} s), in σ del canale"),
                        (ax[1], mx, "massimo per bin (impulsi), in σ del canale")):
        z = img[:, order].T
        lo, hi = np.nanpercentile(z, [1, 99.5])
        im = a.imshow(z, aspect="auto", origin="lower", extent=ext, vmin=lo, vmax=hi, cmap="viridis",
                      interpolation="nearest")
        fig.colorbar(im, ax=a, pad=0.01)
        a.set_ylabel("frequenza [MHz]")
        a.set_title(lab, fontsize=9)
    ax[0].set_title(title + "\n" + ax[0].get_title(), fontsize=9)
    ax[1].set_xlabel("tempo [s]")
    fig.tight_layout()
    fig.savefig(out / "dynspec.png", dpi=110)
    plt.close(fig)

    # --- serie a DM 0
    z = d["zerodm"][:, 0]
    tres = float(d["zerodm_dt"])
    nb = max(1, z.size // 4000)
    zz = z[: z.size // nb * nb].reshape(-1, nb)
    tt = np.arange(zz.shape[0]) * nb * tres
    fig, a = plt.subplots(figsize=(12, 3.5))
    a.fill_between(tt, zz.min(1), zz.max(1), lw=0, alpha=0.5, label="min-max")
    a.plot(tt, np.median(zz, 1), lw=0.6, color="k", label="mediana")
    a.set_xlabel("tempo [s]")
    a.set_ylabel(f"banda intera a DM 0 ({tres * 1e3:.3g} ms)")
    a.set_title(title, fontsize=9)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "zerodm.png", dpi=110)
    plt.close(fig)

    # --- eventi
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    if events:
        t = np.array([e.t for e in events])
        w = np.array([e.width for e in events]) * 1e3
        s = np.array([e.snr_max for e in events])
        fb_ = np.array([e.frac_band for e in events])
        sc = ax[0, 0].scatter(t, w * np.exp(np.random.default_rng(0).uniform(-0.15, 0.15, w.size)),
                              c=np.log10(s), s=4, cmap="plasma")
        ax[0, 0].set_yscale("log")
        fig.colorbar(sc, ax=ax[0, 0], label="log10 S/N")
        ax[0, 0].set_xlabel("tempo [s]")
        ax[0, 0].set_ylabel("larghezza [ms]")
        ss = np.sort(s)[::-1]
        ax[0, 1].loglog(ss, np.arange(1, ss.size + 1) / (summary["duration_s"] / 3600))
        ax[0, 1].set_xlabel("S/N a DM 0")
        ax[0, 1].set_ylabel("eventi/ora con S/N maggiore")
        ax[0, 1].grid(alpha=0.3, which="both")
        ax[1, 0].hist(fb_, bins=np.linspace(0, 1, 21))
        ax[1, 0].set_xlabel("frazione di banda colpita")
        ax[1, 0].set_ylabel("eventi")
        dur = summary["duration_s"]
        bw = 60.0 if dur >= 1800 else max(1.0, dur / 60)
        bins = np.arange(0, dur + bw, bw)
        ax[1, 1].hist(t, bins=bins, weights=np.full(t.size, 60 / bw), histtype="step", lw=1.5, label="tutti")
        ax[1, 1].hist(t[s >= 10], bins=bins, weights=np.full(int((s >= 10).sum()), 60 / bw), alpha=0.5,
                      label="S/N ≥ 10")
        ax[1, 1].set_xlabel(f"tempo [s] (bin da {bw:.3g} s)")
        ax[1, 1].set_ylabel("eventi al minuto")
        ax[1, 1].legend(fontsize=8)
    fig.suptitle(title + f" — {len(events)} eventi a DM 0", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "events.png", dpi=110)
    plt.close(fig)
