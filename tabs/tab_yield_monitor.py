"""Tab — 良率監控分析（Trend/I-MR/EWMA/CUSUM/SPM/Group Comparison）"""
import re
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────
_LAMBDA      = 0.2
_K_FAC       = 0.5
_H_FAC       = 4.0
_N_BASE      = 15
_MIN_CP      = 8
_Z_THRESH    = 2.0
_D4          = 3.2665
_d2          = 1.1284

_COLORS = [
    "#2E86AB", "#6C3B8C", "#E84855", "#F4A261", "#27AE60",
    "#1A5276", "#E67E22", "#8E44AD", "#16A085", "#C0392B",
]

C1 = "#3498DB"   # Group 1 blue
C2 = "#E74C3C"   # Group 2 red
CPc = "#F39C12"  # Change Point orange


# ── Utility functions ─────────────────────────────────────────────────────────

def _batch_order(bid):
    m = re.search(r"(\d{4,6})", str(bid))
    return int(m.group(1)) if m else 0


def _batch_year(bid):
    m = re.search(r"(\d{2})(\d{2})", str(bid))
    return f"20{m.group(1)}" if m else "?"


def _make_short(col):
    param = col.split(":", 1)[1] if ":" in col else col
    m = re.search(r"Yield[\s_]+Rate[\s_]+([^(%]+)", param, re.IGNORECASE)
    if m:
        suf = m.group(1).strip("_ ").strip()
        if suf:
            return suf.title()
    prefix = re.sub(r"[\s_]*Yield[\s_]+Rate.*", "", param, flags=re.IGNORECASE).strip("_ ").strip()
    return prefix.title() if prefix else col.split(":")[0].strip().title()


def _detect_yield_cols(raw_df):
    return [c for c in raw_df.columns if "Yield Rate" in c and ":" in c]


def _prepare_df(raw_df):
    df = raw_df.copy()
    df["_order"] = df["BatchID"].apply(_batch_order)
    df = df.sort_values("_order").reset_index(drop=True)
    df["_seq"]  = np.arange(1, len(df) + 1)
    df["_year"] = df["BatchID"].apply(_batch_year)
    return df


def _ewma(y, lam):
    e = np.empty(len(y))
    e[0] = y[0]
    for t in range(1, len(y)):
        e[t] = lam * y[t] + (1 - lam) * e[t - 1]
    return e


def _cusum(y, n_base, h_fac, min_cp):
    base = y[:n_base]
    mu0  = np.mean(base)
    sig0 = np.std(base, ddof=1)
    if sig0 < 1e-9:
        sig0 = np.std(y, ddof=1)
    k = _K_FAC * sig0
    h = h_fac * sig0
    n = len(y)
    C_pos = np.zeros(n)
    C_neg = np.zeros(n)
    for t in range(1, n):
        C_pos[t] = max(0.0, C_pos[t - 1] + y[t] - mu0 - k)
        C_neg[t] = min(0.0, C_neg[t - 1] + y[t] - mu0 + k)
    alarm_neg = np.where((C_neg < -h) & (np.arange(n) >= min_cp))[0]
    alarm_pos = np.where( C_pos >  h)[0]
    cp_idx = None
    if len(alarm_neg) > 0:
        fa = int(alarm_neg[0])
        cp_start = fa
        for t in range(fa - 1, min_cp - 1, -1):
            if C_neg[t] == 0.0:
                cp_start = t + 1
                break
        cp_idx = cp_start
    return C_pos, C_neg, k, h, mu0, sig0, alarm_neg, alarm_pos, cp_idx


def _imr(y):
    mr     = np.abs(np.diff(y))
    mr_bar = np.nanmean(mr)
    xbar   = np.nanmean(y)
    ucl_i  = xbar + 3 * mr_bar / _d2
    lcl_i  = xbar - 3 * mr_bar / _d2
    ucl_mr = _D4 * mr_bar
    return mr, xbar, ucl_i, lcl_i, ucl_mr, mr_bar


def _cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / sp if sp > 0 else np.nan


def _rank_biserial(U, n1, n2):
    return 1 - (2 * U) / (n1 * n2)


def _effect_d(d):
    d = abs(d)
    if d >= 0.8: return "Large"
    if d >= 0.5: return "Medium"
    if d >= 0.2: return "Small"
    return "Negligible"


def _effect_r(r):
    r = abs(r)
    if r >= 0.5: return "Large"
    if r >= 0.3: return "Medium"
    if r >= 0.1: return "Small"
    return "Negligible"


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _xticks(ax, x_idx, yr_labels, xlim_n):
    tick_pos, tick_lab = [], []
    prev = None
    for xi, yr in zip(x_idx, yr_labels):
        if yr != prev:
            tick_pos.append(xi); tick_lab.append(yr); prev = yr
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=0)
    ax.set_xlim(0.5, xlim_n + 0.5)
    ax.grid(axis="y", ls="--", alpha=0.35, color="#AAAAAA")
    ax.spines[["top", "right"]].set_visible(False)


# ── Compute all step analytics ────────────────────────────────────────────────

def _compute_steps(df, sel_cols, short_map, lam, n_base, h_fac, min_cp, z_thresh):
    step_data = []
    for i, col in enumerate(sel_cols):
        short = short_map[col]
        color = _COLORS[i % len(_COLORS)]

        sub = df[["_seq", "_year", "BatchID", col]].dropna(subset=[col]).copy()
        sub = sub.reset_index(drop=True)
        y   = sub[col].values.astype(float)
        n   = len(y)
        bid = sub["BatchID"].values
        yr  = sub["_year"].values
        xi  = np.arange(1, n + 1)

        if n < max(6, n_base + 2):
            continue

        # EWMA + CUSUM
        ew = _ewma(y, lam)
        C_pos, C_neg, k, h, mu0, sig0, alarm_neg, alarm_pos, cp_idx = _cusum(y, n_base, h_fac, min_cp)

        # Z-Score
        mu_all  = np.mean(y)
        sig_all = np.std(y, ddof=1)
        z = (y - mu_all) / sig_all if sig_all > 0 else np.zeros(n)
        outlier = np.abs(z) > z_thresh

        # Groups
        has_cp = cp_idx is not None and (n - cp_idx) >= 3
        if has_cp:
            g1 = y[:cp_idx]; g2 = y[cp_idx:]
        else:
            g1 = y; g2 = np.array([])
        g1_mask = np.zeros(n, bool); g2_mask = np.zeros(n, bool)
        if has_cp:
            g1_mask[:cp_idx] = True; g2_mask[cp_idx:] = True
        else:
            g1_mask[:] = True
        g1_mean = np.mean(g1) if len(g1) else np.nan
        g2_mean = np.mean(g2) if len(g2) else np.nan
        cp_batch = bid[cp_idx] if has_cp else "N/A"

        # Group stats
        if has_cp:
            U_stat, p_mw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            d   = _cohens_d(g1, g2)
            rrb = _rank_biserial(U_stat, len(g1), len(g2))
            t1  = np.arange(len(g1)) + 1
            t2  = np.arange(len(g2)) + 1
            r1, p_r1 = stats.pearsonr(t1, g1) if len(g1) >= 3 else (np.nan, np.nan)
            r2, p_r2 = stats.pearsonr(t2, g2) if len(g2) >= 3 else (np.nan, np.nan)
        else:
            U_stat = p_mw = d = rrb = np.nan
            r1 = p_r1 = r2 = p_r2 = np.nan

        # I-MR
        mr, xbar, ucl_i, lcl_i, ucl_mr, mr_bar = _imr(y)
        oor_i  = (y > ucl_i) | (y < lcl_i)
        oor_mr = mr > ucl_mr

        # Linear regression
        slope, intercept, r_lr, p_lr, _ = stats.linregress(xi, y)
        r2_lr = r_lr ** 2
        y_fit = slope * xi + intercept
        t_crit = stats.t.ppf(0.975, df=n - 2)
        x_mean = xi.mean()
        s_res  = np.sqrt(np.sum((y - y_fit) ** 2) / (n - 2))
        se_fit = s_res * np.sqrt(1 / n + (xi - x_mean) ** 2 / np.sum((xi - x_mean) ** 2))
        ci_hi  = y_fit + t_crit * se_fit
        ci_lo  = y_fit - t_crit * se_fit

        step_data.append(dict(
            col=col, short=short, color=color,
            y=y, n=n, bid=bid, xi=xi, yr=yr,
            ew=ew, C_pos=C_pos, C_neg=C_neg, k=k, h=h, mu0=mu0, sig0=sig0,
            alarm_neg=alarm_neg, alarm_pos=alarm_pos,
            cp_idx=cp_idx, cp_batch=cp_batch, has_cp=has_cp,
            z=z, outlier=outlier, mu_all=mu_all, sig_all=sig_all,
            g1=g1, g2=g2, g1_mask=g1_mask, g2_mask=g2_mask,
            g1_mean=g1_mean, g2_mean=g2_mean,
            U_stat=U_stat, p_mw=p_mw, d=d, rrb=rrb,
            r1=r1, p_r1=p_r1, r2=r2, p_r2=p_r2,
            mr=mr, xbar=xbar, ucl_i=ucl_i, lcl_i=lcl_i,
            ucl_mr=ucl_mr, mr_bar=mr_bar, oor_i=oor_i, oor_mr=oor_mr,
            slope=slope, intercept=intercept, r2_lr=r2_lr, p_lr=p_lr,
            y_fit=y_fit, ci_hi=ci_hi, ci_lo=ci_lo,
        ))
    return step_data


# ── Sub-tab renderers ─────────────────────────────────────────────────────────

def _tab_summary(sds):
    st.subheader("各步驟良率分析摘要")
    rows = []
    for sd in sds:
        delta = sd["g2_mean"] - sd["g1_mean"] if sd["has_cp"] else np.nan
        rows.append({
            "步驟":            sd["short"],
            "n":               sd["n"],
            "整體均值 (%)":     round(sd["mu_all"], 2),
            "Z 異常點":         int(sd["outlier"].sum()),
            "CUSUM 下行警報":   len(sd["alarm_neg"]),
            "Change Point":    sd["cp_batch"],
            "G1 n":            int(sd["g1_mask"].sum()),
            "G1 均值 (%)":      round(sd["g1_mean"], 2) if not np.isnan(sd["g1_mean"]) else "-",
            "G2 n":            int(sd["g2_mask"].sum()),
            "G2 均值 (%)":      round(sd["g2_mean"], 2) if sd["has_cp"] else "-",
            "Δ G2−G1 (%)":     f"{delta:+.2f}" if not np.isnan(delta) else "-",
            "MWU p":           f"{sd['p_mw']:.4f} {_sig(sd['p_mw'])}" if sd["has_cp"] else "-",
            "Cohen's d":       f"{sd['d']:+.3f} [{_effect_d(sd['d'])}]" if sd["has_cp"] else "-",
            "Trend slope":     f"{sd['slope']:+.4f} {_sig(sd['p_lr'])}" ,
        })
    df_sum = pd.DataFrame(rows)

    def _color_delta(v):
        s = str(v)
        if s.startswith("-") and s != "-":
            return "color: #C0392B; font-weight:bold"
        if s.startswith("+"):
            return "color: #27AE60; font-weight:bold"
        return ""

    st.dataframe(
        df_sum.style.map(_color_delta, subset=["Δ G2−G1 (%)"]),
        use_container_width=True, hide_index=True,
    )
    csv = df_sum.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ 下載摘要 CSV", csv, "yield_monitor_summary.csv", "text/csv",
                       key="ym_dl_summary")


def _tab_trend_imr(sds):
    st.subheader("良率趨勢（Linear Regression）+ I-MR 管制圖")
    N = len(sds)
    fig = plt.figure(figsize=(18, N * 3.8))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Yield Rate — Trend & I-MR Control Charts",
                 fontsize=13, fontweight="bold", y=1.001, color="#1A1A2E")
    gs = GridSpec(N, 3, figure=fig, hspace=0.58, wspace=0.32,
                  left=0.07, right=0.97, top=0.975, bottom=0.03)

    for ri, sd in enumerate(sds):
        y   = sd["y"];  n = sd["n"];  xi = sd["xi"]
        yr  = sd["yr"]; color = sd["color"]; short = sd["short"]

        # (A) Trend
        ax = fig.add_subplot(gs[ri, 0]);  ax.set_facecolor("white")
        sc = "#E84855" if sd["slope"] < 0 else "#27AE60"
        ax.scatter(xi, y, color=color, s=35, zorder=4, edgecolors="white", lw=0.5, label="Batch")
        ax.plot(xi, sd["y_fit"], color=sc, lw=2, zorder=3,
                label=f"slope={sd['slope']:+.4f}/batch")
        ax.fill_between(xi, sd["ci_lo"], sd["ci_hi"], alpha=0.15, color=sc, label="95% CI")
        ax.axhline(sd["mu_all"], color="#555", lw=1, ls="--", alpha=0.6,
                   label=f"Mean={sd['mu_all']:.2f}%")
        if sd["slope"] < 0 and sd["p_lr"] < 0.05:
            ax.set_facecolor("#FFF5F5")
        _xticks(ax, xi, yr, n)
        ax.set_title(
            f"{short}  |  Linear Regression\n"
            f"slope={sd['slope']:+.4f}  R²={sd['r2_lr']:.3f}  "
            f"p={sd['p_lr']:.4f} {_sig(sd['p_lr'])}",
            loc="left", fontsize=9)
        ax.set_ylabel("Yield Rate (%)")
        ax.legend(loc="best", fontsize=7.5, framealpha=0.8)

        # (B) I-Chart
        ax = fig.add_subplot(gs[ri, 1]);  ax.set_facecolor("white")
        ax.plot(xi, y, color=color, lw=1.2, marker="o", ms=4, zorder=3, label="Individual")
        ax.axhline(sd["xbar"],  color="#2E86AB", lw=1.5, ls="-",  label=f"CL={sd['xbar']:.2f}")
        ax.axhline(sd["ucl_i"], color="#E84855", lw=1.5, ls="--", label=f"UCL={sd['ucl_i']:.2f}")
        ax.axhline(sd["lcl_i"], color="#E84855", lw=1.5, ls="--", label=f"LCL={sd['lcl_i']:.2f}")
        if sd["oor_i"].any():
            ax.scatter(xi[sd["oor_i"]], y[sd["oor_i"]], color="#E84855",
                       zorder=5, s=60, marker="D", label=f"OOC ({sd['oor_i'].sum()})")
        _xticks(ax, xi, yr, n)
        ax.set_title(
            f"{short}  |  I-Chart\nUCL={sd['ucl_i']:.2f}  CL={sd['xbar']:.2f}  LCL={sd['lcl_i']:.2f}",
            loc="left", fontsize=9)
        ax.set_ylabel("Yield Rate (%)")
        ax.legend(loc="best", fontsize=7, framealpha=0.8)

        # (C) MR-Chart
        ax = fig.add_subplot(gs[ri, 2]);  ax.set_facecolor("white")
        mr_x = xi[1:]
        ax.bar(mr_x, sd["mr"], color=color, alpha=0.7, width=0.6, zorder=3, label="MR")
        ax.axhline(sd["mr_bar"], color="#2E86AB", lw=1.5, ls="-",  label=f"MR̄={sd['mr_bar']:.2f}")
        ax.axhline(sd["ucl_mr"], color="#E84855", lw=1.5, ls="--", label=f"UCL={sd['ucl_mr']:.2f}")
        ax.axhline(0, color="#888", lw=0.8)
        if sd["oor_mr"].any():
            ax.bar(mr_x[sd["oor_mr"]], sd["mr"][sd["oor_mr"]],
                   color="#E84855", alpha=0.9, width=0.6, zorder=5,
                   label=f"OOC ({sd['oor_mr'].sum()})")
        _xticks(ax, mr_x, yr[1:], n)
        ax.set_title(
            f"{short}  |  MR-Chart\nUCL={sd['ucl_mr']:.2f}  MR̄={sd['mr_bar']:.2f}",
            loc="left", fontsize=9)
        ax.set_ylabel("Moving Range")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _tab_ewma_cusum(sds, lam, h_fac):
    st.subheader("EWMA 平滑 + CUSUM Change Point 偵測")
    N = len(sds)
    fig = plt.figure(figsize=(16, N * 4.0))
    fig.patch.set_facecolor("#F0F2F5")
    fig.suptitle(
        f"Yield Rate — EWMA (λ={lam}) + CUSUM Change Point Detection",
        fontsize=13, fontweight="bold", y=1.001, color="#1A1A2E")
    gs = GridSpec(N, 2, figure=fig, hspace=0.62, wspace=0.30,
                  left=0.06, right=0.97, top=0.976, bottom=0.025)

    for ri, sd in enumerate(sds):
        y   = sd["y"];  n = sd["n"];  xi = sd["xi"]
        yr  = sd["yr"]; color = sd["color"]; short = sd["short"]
        cp  = sd["cp_idx"];  bid = sd["bid"]
        g1m = sd["g1_mask"]; g2m = sd["g2_mask"]
        has = sd["has_cp"]

        # ── Left: EWMA ──────────────────────────────────────
        ax = fig.add_subplot(gs[ri, 0]);  ax.set_facecolor("white")
        if has:
            ax.axvspan(0.5,          xi[cp] - 0.5, alpha=0.06, color=C1)
            ax.axvspan(xi[cp] - 0.5, n + 0.5,      alpha=0.06, color=C2)
        ax.scatter(xi[g1m], y[g1m], color=C1, s=30, alpha=0.75, zorder=3,
                   label=f"G1 n={g1m.sum()} μ={sd['g1_mean']:.2f}%")
        if has:
            ax.scatter(xi[g2m], y[g2m], color=C2, s=30, alpha=0.75, zorder=3,
                       label=f"G2 n={g2m.sum()} μ={sd['g2_mean']:.2f}%")
        ax.plot(xi, sd["ew"], color="#1A1A2E", lw=2.2, zorder=4,
                label=f"EWMA λ={lam}")
        ax.axhline(sd["mu0"], color="#888", lw=1.3, ls="--", alpha=0.75,
                   label=f"μ₀={sd['mu0']:.2f}%")
        if g1m.any(): ax.axhline(sd["g1_mean"], color=C1, lw=1.2, ls=":", alpha=0.7)
        if has:       ax.axhline(sd["g2_mean"], color=C2, lw=1.2, ls=":", alpha=0.7)
        if has:
            ax.axvline(xi[cp], color=CPc, lw=2.5, alpha=0.9, zorder=5,
                       label=f"CP: {sd['cp_batch'][-8:]}")
            ann_y = (sd["g1_mean"] + sd["g2_mean"]) / 2
            delta = sd["g2_mean"] - sd["g1_mean"]
            ax.annotate(f"Δ={delta:+.2f}%", xy=(xi[cp], ann_y),
                        xytext=(14, 0), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#C0392B", va="center",
                        arrowprops=dict(arrowstyle="->", color=CPc, lw=1.4))
        _xticks(ax, xi, yr, n)
        ax.set_title(
            f"{short}  |  EWMA (λ={lam})\n"
            f"Phase-I baseline: first {sd['mu0']:.2f}%  "
            f"{'|  CP detected' if has else '|  No downward CP'}",
            loc="left", fontsize=9.5)
        ax.set_ylabel("Yield Rate (%)")
        ax.legend(loc="lower left", fontsize=7.8, framealpha=0.88)

        # ── Right: CUSUM ─────────────────────────────────────
        ax = fig.add_subplot(gs[ri, 1]);  ax.set_facecolor("white")
        ax.fill_between(xi, sd["C_neg"], 0, where=sd["C_neg"] < 0, alpha=0.13, color=C2)
        ax.fill_between(xi, sd["C_pos"], 0, where=sd["C_pos"] > 0, alpha=0.13, color="#27AE60")
        ax.plot(xi, sd["C_pos"], color="#27AE60", lw=2.0, zorder=3, label="C⁺ (rise)")
        ax.plot(xi, sd["C_neg"], color=C2,        lw=2.0, zorder=3, label="C⁻ (drop)")
        ax.axhline( sd["h"], color="#27AE60", lw=1.3, ls="--", alpha=0.85,
                    label=f"+h={sd['h']:.2f} ({h_fac}σ₀)")
        ax.axhline(-sd["h"], color=C2,        lw=1.3, ls="--", alpha=0.85,
                    label=f"-h={-sd['h']:.2f}")
        ax.axhline(0, color="#666", lw=0.8)
        if len(sd["alarm_neg"]) > 0:
            fa = sd["alarm_neg"][0]
            ax.axvline(xi[fa], color="#C0392B", lw=1.5, ls=":", alpha=0.9,
                       label=f"1st alarm↓ (#{fa+1})")
            ax.scatter([xi[fa]], [sd["C_neg"][fa]], color="#C0392B", s=80, zorder=6, marker="v")
        if has:
            ax.axvline(xi[cp], color=CPc, lw=2.5, alpha=0.9, zorder=5,
                       label=f"CP={sd['cp_batch'][-8:]}")
            ax.scatter([xi[cp]], [sd["C_neg"][cp]], color=CPc, s=100, zorder=7,
                       marker="D", edgecolors="white", lw=0.8)
            delta = sd["g2_mean"] - sd["g1_mean"]
            ax.annotate(
                f"CP:{sd['cp_batch'][-8:]}\nΔ={delta:+.2f}%",
                xy=(xi[cp], -sd["h"] * 0.55),
                xytext=(12, 0), textcoords="offset points",
                fontsize=8.5, color="#8E44AD", fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=CPc, lw=1.3))
        _xticks(ax, xi, yr, n)
        ax.set_title(
            f"{short}  |  CUSUM\n"
            f"μ₀={sd['mu0']:.2f}%  σ₀={sd['sig0']:.2f}  k={sd['k']:.2f}  "
            f"h=±{sd['h']:.2f}  alarm↓={len(sd['alarm_neg'])}",
            loc="left", fontsize=9.5)
        ax.set_ylabel("Cumulative Sum")
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.88)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _tab_spm(sds, lam, z_thresh):
    st.subheader("SPM 全覽：Z-Score | EWMA | CUSUM | Change Point Groups")
    N = len(sds)
    fig = plt.figure(figsize=(22, N * 3.8))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"Yield Rate — SPM: Z-Score (|Z|>{z_thresh}) | EWMA (λ={lam}) | CUSUM | Change Point",
        fontsize=13, fontweight="bold", y=1.001, color="#1A1A2E")
    gs = GridSpec(N, 4, figure=fig, hspace=0.60, wspace=0.33,
                  left=0.05, right=0.98, top=0.976, bottom=0.02)

    for ri, sd in enumerate(sds):
        y   = sd["y"];  n = sd["n"];  xi = sd["xi"]
        yr  = sd["yr"]; color = sd["color"]; short = sd["short"]
        cp  = sd["cp_idx"]; has = sd["has_cp"]
        g1m = sd["g1_mask"]; g2m = sd["g2_mask"]

        # ── Z-Score ────────────────────────────────────────
        ax0 = fig.add_subplot(gs[ri, 0]);  ax0.set_facecolor("white")
        bar_colors = np.where(sd["outlier"], "#E84855", color)
        for xv, zv, c in zip(xi, sd["z"], bar_colors):
            ax0.bar(xv, zv, color=c, alpha=0.75, width=0.7, zorder=2)
        ax0.axhline( z_thresh, color="#E84855", lw=1.5, ls="--", label=f"+{z_thresh}σ")
        ax0.axhline(-z_thresh, color="#E84855", lw=1.5, ls="--", label=f"-{z_thresh}σ")
        ax0.axhline(0, color="#555", lw=0.8)
        for xv, zv in zip(xi[sd["outlier"]], sd["z"][sd["outlier"]]):
            ax0.annotate(f"{zv:.1f}", (xv, zv),
                         textcoords="offset points", xytext=(0, 5 if zv > 0 else -12),
                         fontsize=6.5, color="#C0392B", ha="center", fontweight="bold")
        if has:
            ax0.axvline(xi[cp], color=CPc, lw=2, ls="--", alpha=0.8)
        _xticks(ax0, xi, yr, n)
        ax0.set_title(
            f"{short} — Z-Score\nμ={sd['mu_all']:.2f}%  σ={sd['sig_all']:.2f}  "
            f"Outliers={sd['outlier'].sum()}",
            loc="left", fontsize=9)
        ax0.set_ylabel("Z-Score")
        ax0.legend(loc="upper right", framealpha=0.8, fontsize=7)

        # ── EWMA ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[ri, 1]);  ax1.set_facecolor("white")
        ax1.scatter(xi[g1m], y[g1m], color=C1, s=24, alpha=0.65, zorder=2, label="G1")
        if has:
            ax1.scatter(xi[g2m], y[g2m], color=C2, s=24, alpha=0.65, zorder=2, label="G2")
        ax1.scatter(xi[sd["outlier"]], y[sd["outlier"]], color="#F4A261",
                    s=60, marker="D", zorder=5, label=f"Z-Outlier ({sd['outlier'].sum()})")
        ax1.plot(xi, sd["ew"], color="#1A1A2E", lw=2.0, zorder=3, label=f"EWMA λ={lam}")
        ax1.axhline(sd["mu0"], color="#888", lw=1.2, ls="--", alpha=0.7,
                    label=f"μ₀={sd['mu0']:.2f}%")
        if has:
            ax1.axvline(xi[cp], color=CPc, lw=2.5, alpha=0.9, label=f"CP:{sd['cp_batch'][-6:]}")
            ax1.axvspan(0, xi[cp], alpha=0.05, color=C1)
            ax1.axvspan(xi[cp], n + 1, alpha=0.05, color=C2)
        _xticks(ax1, xi, yr, n)
        ax1.set_title(f"{short} — EWMA\nPhase-I μ₀={sd['mu0']:.2f}%", loc="left", fontsize=9)
        ax1.set_ylabel("Yield Rate (%)")
        ax1.legend(loc="lower left", framealpha=0.8, fontsize=7, ncol=2)

        # ── CUSUM ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[ri, 2]);  ax2.set_facecolor("white")
        ax2.plot(xi, sd["C_pos"], color="#27AE60", lw=1.8, label="C⁺")
        ax2.plot(xi, sd["C_neg"], color=C2,        lw=1.8, label="C⁻")
        ax2.axhline( sd["h"], color="#27AE60", lw=1.2, ls="--", label=f"+h={sd['h']:.2f}")
        ax2.axhline(-sd["h"], color=C2,        lw=1.2, ls="--", label=f"-h={-sd['h']:.2f}")
        ax2.axhline(0, color="#555", lw=0.8)
        if len(sd["alarm_neg"]) > 0:
            fa = sd["alarm_neg"][0]
            ax2.axvline(xi[fa], color="#C0392B", lw=1.5, ls=":", alpha=0.9,
                        label=f"alarm↓(#{fa+1})")
        if has:
            ax2.axvline(xi[cp], color=CPc, lw=2.5, alpha=0.9, label=f"CP:{sd['cp_batch'][-6:]}")
        ax2.fill_between(xi, sd["C_neg"], 0, where=sd["C_neg"] < 0, alpha=0.12, color=C2)
        ax2.fill_between(xi, sd["C_pos"], 0, where=sd["C_pos"] > 0, alpha=0.12, color="#27AE60")
        _xticks(ax2, xi, yr, n)
        ax2.set_title(
            f"{short} — CUSUM\nμ₀={sd['mu0']:.2f}%  k={sd['k']:.2f}  h=±{sd['h']:.2f}  "
            f"alarm↓={len(sd['alarm_neg'])}",
            loc="left", fontsize=9)
        ax2.set_ylabel("Cumulative Sum")
        ax2.legend(loc="lower left", fontsize=7, framealpha=0.8)

        # ── Group ──────────────────────────────────────────
        ax3 = fig.add_subplot(gs[ri, 3]);  ax3.set_facecolor("white")
        ax3.scatter(xi[g1m], y[g1m], color=C1, s=28, zorder=3,
                    label=f"G1 n={g1m.sum()} μ={sd['g1_mean']:.2f}%")
        if has:
            ax3.scatter(xi[g2m], y[g2m], color=C2, s=28, zorder=3,
                        label=f"G2 n={g2m.sum()} μ={sd['g2_mean']:.2f}%")
        ax3.scatter(xi[sd["outlier"]], y[sd["outlier"]], color="#F4A261",
                    s=60, marker="D", zorder=5, label="Z-Outlier")
        ax3.plot(xi, sd["ew"], color="#1A1A2E", lw=1.8, alpha=0.7, zorder=2)
        if g1m.any(): ax3.axhline(sd["g1_mean"], color=C1, lw=1.5, ls="--", alpha=0.7)
        if has:
            ax3.axhline(sd["g2_mean"], color=C2, lw=1.5, ls="--", alpha=0.7)
            ax3.axvspan(0, xi[cp], alpha=0.07, color=C1)
            ax3.axvspan(xi[cp], n + 1, alpha=0.07, color=C2)
            ax3.axvline(xi[cp], color=CPc, lw=2.5, alpha=0.9, zorder=5)
            delta = sd["g2_mean"] - sd["g1_mean"]
            ax3.annotate(
                f"CP:{sd['cp_batch'][-8:]}\nΔ={delta:+.2f}%",
                xy=(xi[cp], (sd["g1_mean"] + sd["g2_mean"]) / 2),
                xytext=(10, 0), textcoords="offset points",
                fontsize=8, color="#D35400", fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=CPc, lw=1.2))
        _xticks(ax3, xi, yr, n)
        ax3.set_title(f"{short} — Change Point Groups", loc="left", fontsize=9)
        ax3.set_ylabel("Yield Rate (%)")
        ax3.legend(loc="lower left", fontsize=7.5, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _tab_group_comparison(sds):
    st.subheader("組間比較：散佈圖 + 分佈直方圖 + 統計檢定")
    N = len(sds)
    fig = plt.figure(figsize=(16, N * 4.2))
    fig.patch.set_facecolor("#F0F2F5")
    fig.suptitle(
        "Yield Rate — Group Comparison (CUSUM CP | Mann-Whitney U | Cohen's d | rank-biserial r)",
        fontsize=13, fontweight="bold", y=1.001, color="#1A1A2E")
    gs = GridSpec(N, 2, figure=fig, hspace=0.72, wspace=0.30,
                  left=0.06, right=0.97, top=0.972, bottom=0.025)

    for ri, sd in enumerate(sds):
        y   = sd["y"];  n = sd["n"];  xi = sd["xi"]
        yr  = sd["yr"]; short = sd["short"]
        cp  = sd["cp_idx"]; has = sd["has_cp"]
        g1m = sd["g1_mask"]; g2m = sd["g2_mask"]

        # within-group correlation text
        if has:
            r1_txt = (f"G1: r={sd['r1']:+.3f}, p={sd['p_r1']:.3f}"
                      if not np.isnan(sd["r1"]) else "G1: r=N/A")
            r2_txt = (f"G2: r={sd['r2']:+.3f}, p={sd['p_r2']:.3f}"
                      if not np.isnan(sd["r2"]) else "G2: r=N/A")
            corr_note = f"Within-group correlation\n  {r1_txt}\n  {r2_txt}"
        else:
            corr_note = "No change point detected\n(entire series = Group 1)"

        # ── Scatter ───────────────────────────────────────
        ax = fig.add_subplot(gs[ri, 0]);  ax.set_facecolor("white")
        if has:
            ax.axvspan(0.5, xi[cp] - 0.5, alpha=0.07, color=C1)
            ax.axvspan(xi[cp] - 0.5, n + 0.5, alpha=0.07, color=C2)
        g1x = xi[:cp] if has else xi
        g2x = xi[cp:] if has else np.array([])
        ax.scatter(g1x, sd["g1"], color=C1, s=32, alpha=0.80, zorder=3,
                   label=f"G1 n={len(sd['g1'])} μ={sd['g1_mean']:.2f}%")
        if has:
            ax.scatter(g2x, sd["g2"], color=C2, s=32, alpha=0.80, zorder=3,
                       label=f"G2 n={len(sd['g2'])} μ={sd['g2_mean']:.2f}%")
        # trend lines
        t1 = np.arange(len(sd["g1"])) + 1
        if len(sd["g1"]) >= 3:
            sl1, ic1, *_ = stats.linregress(t1, sd["g1"])
            ax.plot(g1x, sl1 * t1 + ic1, color=C1, lw=1.4, ls="--", alpha=0.75)
        if has and len(sd["g2"]) >= 3:
            t2 = np.arange(len(sd["g2"])) + 1
            sl2, ic2, *_ = stats.linregress(t2, sd["g2"])
            ax.plot(g2x, sl2 * t2 + ic2, color=C2, lw=1.4, ls="--", alpha=0.75)
        ax.plot(xi, sd["ew"], color="#1A1A2E", lw=2.0, zorder=4, alpha=0.85,
                label=f"EWMA")
        ax.axhline(sd["g1_mean"], color=C1, lw=1.2, ls=":", alpha=0.8)
        if has:
            ax.axhline(sd["g2_mean"], color=C2, lw=1.2, ls=":", alpha=0.8)
            ax.axvline(xi[cp], color=CPc, lw=2.5, alpha=0.9, zorder=5,
                       label=f"CP:{sd['cp_batch'][-8:]}")
            delta = sd["g2_mean"] - sd["g1_mean"]
            ax.annotate(f"Δ={delta:+.2f}%",
                        xy=(xi[cp], (sd["g1_mean"] + sd["g2_mean"]) / 2),
                        xytext=(12, 0), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color="#C0392B", va="center",
                        arrowprops=dict(arrowstyle="->", color=CPc, lw=1.4))
        ax.text(0.02, 0.03, corr_note, transform=ax.transAxes,
                fontsize=7.5, va="bottom",
                bbox=dict(fc="white", ec="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.3"))
        _xticks(ax, xi, yr, n)
        ax.set_title(f"{short}  |  Scatter + EWMA (G1 vs G2)", loc="left", fontsize=9.5)
        ax.set_ylabel("Yield Rate (%)")
        ax.legend(loc="lower left", fontsize=7.8, framealpha=0.88, ncol=2)

        # ── Histogram + KDE ───────────────────────────────
        ax = fig.add_subplot(gs[ri, 1]);  ax.set_facecolor("white")
        bins = np.linspace(y.min() - 2, y.max() + 2, 20)
        ax.hist(sd["g1"], bins=bins, color=C1, alpha=0.45, density=True,
                label=f"G1 μ={sd['g1_mean']:.2f}%", zorder=2)
        if has:
            ax.hist(sd["g2"], bins=bins, color=C2, alpha=0.45, density=True,
                    label=f"G2 μ={sd['g2_mean']:.2f}%", zorder=2)
        kde_x = np.linspace(y.min() - 3, y.max() + 3, 300)
        if len(sd["g1"]) >= 5:
            ax.plot(kde_x, stats.gaussian_kde(sd["g1"])(kde_x), color=C1, lw=2.2, zorder=4)
        if has and len(sd["g2"]) >= 5:
            ax.plot(kde_x, stats.gaussian_kde(sd["g2"])(kde_x), color=C2, lw=2.2, zorder=4)
        ax.axvline(sd["g1_mean"], color=C1, lw=1.8, ls="--", alpha=0.85)
        if has:
            ax.axvline(sd["g2_mean"], color=C2, lw=1.8, ls="--", alpha=0.85)

        if has:
            d_abs = abs(sd["d"]); r_abs = abs(sd["rrb"])
            delta = sd["g2_mean"] - sd["g1_mean"]
            stat_txt = (
                f"Mann-Whitney U Test\n"
                f"  U={sd['U_stat']:.0f}  p={sd['p_mw']:.6f} {_sig(sd['p_mw'])}\n"
                f"\nEffect Size\n"
                f"  Cohen's d={sd['d']:+.3f}  [{_effect_d(sd['d'])}]\n"
                f"  rank-biserial r={sd['rrb']:+.3f}  [{_effect_r(sd['rrb'])}]\n"
                f"\nGroup Means\n"
                f"  G1: {sd['g1_mean']:.2f}%\n"
                f"  G2: {sd['g2_mean']:.2f}%\n"
                f"  Δ:  {delta:+.2f}%"
            )
        else:
            stat_txt = (
                "No CUSUM change point detected\n"
                "(Mann-Whitney U not applicable)\n\n"
                f"Overall n={n}\n"
                f"Mean={sd['g1_mean']:.2f}%\n"
                f"SD={np.std(sd['g1'], ddof=1):.2f}%"
            )
        ax.text(0.97, 0.97, stat_txt, transform=ax.transAxes,
                fontsize=8.0, va="top", ha="right", family="monospace",
                bbox=dict(fc="#FAFAFA", ec="#BBBBBB", alpha=0.90, boxstyle="round,pad=0.45"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Yield Rate (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"{short}  |  Distribution (Histogram + KDE)", loc="left", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.88)
        ax.grid(axis="y", ls="--", alpha=0.35, color="#AAAAAA")

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Main render ───────────────────────────────────────────────────────────────

def render(raw_df: pd.DataFrame):
    st.header("📉 良率監控分析")

    if raw_df is None:
        st.info("請先上傳資料。"); return

    if "BatchID" not in raw_df.columns:
        st.warning("資料中需要 BatchID 欄位。"); return

    yield_cols = _detect_yield_cols(raw_df)
    if not yield_cols:
        st.warning(
            "找不到良率欄位。\n"
            "欄位格式需同時滿足：含 **'Yield Rate'** 且含 **':'** 分隔符，\n"
            "例如 `CM Chromatography:CM chromatography_Yield Rate (%)`。"
        ); return

    df = _prepare_df(raw_df)
    short_map = {c: _make_short(c) for c in yield_cols}

    # ── 參數設定 ──────────────────────────────────────────────
    with st.expander("⚙️ 分析參數設定", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            lam     = st.slider("EWMA λ", 0.05, 0.5, _LAMBDA, 0.05, key="ym_lam")
            n_base  = st.slider("Phase-I 批次數", 5, 30, _N_BASE, 1, key="ym_nbase")
        with c2:
            h_fac   = st.slider("CUSUM h 係數（h = h_fac × σ₀）", 2.0, 6.0, _H_FAC, 0.5,
                                 key="ym_hfac")
            min_cp  = st.slider("CP 最小批次索引", 3, 20, _MIN_CP, 1, key="ym_mincp")
        with c3:
            z_thresh = st.slider("Z-Score 閾值", 1.5, 3.5, _Z_THRESH, 0.1, key="ym_zthr")

    # ── 欄位選擇 ──────────────────────────────────────────────
    avail = [f"{short_map[c]}  [{c.split(':')[0]}]" for c in yield_cols]
    d2c   = dict(zip(avail, yield_cols))
    sel_d = st.multiselect("選擇良率欄位", avail, default=avail, key="ym_cols")
    sel_cols = [d2c[d] for d in sel_d]
    if not sel_cols:
        st.warning("請至少選擇一個良率欄位。"); return

    # ── 計算 ──────────────────────────────────────────────────
    with st.spinner("計算分析中…"):
        sds = _compute_steps(df, sel_cols, short_map, lam, n_base, h_fac, min_cp, z_thresh)

    if not sds:
        st.warning("所選欄位資料不足（需 ≥ n_base+2 筆），無法分析。"); return

    # ── Sub-tabs ──────────────────────────────────────────────
    subtabs = st.tabs([
        "📋 概覽摘要",
        "📈 趨勢 + I-MR",
        "📊 EWMA + CUSUM",
        "🔬 SPM 全覽",
        "⚖️ 組間比較",
    ])

    with subtabs[0]:
        _tab_summary(sds)

    with subtabs[1]:
        _tab_trend_imr(sds)

    with subtabs[2]:
        _tab_ewma_cusum(sds, lam, h_fac)

    with subtabs[3]:
        _tab_spm(sds, lam, z_thresh)

    with subtabs[4]:
        _tab_group_comparison(sds)
