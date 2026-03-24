"""Tab — 統計檢定分析（基於分類趨勢的分群）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from scipy import stats
from itertools import combinations


# ══════════════════════════════════════════════════════════════════════════════
# ── 假設檢查 ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _check_normality(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Shapiro-Wilk 常態性檢定（n ≥ 3 才執行）。"""
    rows = []
    for label, arr in groups.items():
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n < 3:
            rows.append({"組別": label, "n": n, "W 統計量": None,
                         "p-value": None, "常態？": "⚠️ 樣本不足"})
            continue
        if n > 50:
            # 大樣本改用 K-S test
            stat, p = stats.kstest(arr, "norm", args=(arr.mean(), arr.std(ddof=1)))
            rows.append({"組別": label, "n": n, "W 統計量": round(stat, 4),
                         "p-value": round(p, 4),
                         "常態？": "✅ 是" if p > 0.05 else "❌ 否",
                         "方法": "K-S (n>50)"})
        else:
            stat, p = stats.shapiro(arr)
            rows.append({"組別": label, "n": n, "W 統計量": round(stat, 4),
                         "p-value": round(p, 4),
                         "常態？": "✅ 是" if p > 0.05 else "❌ 否",
                         "方法": "Shapiro-Wilk"})
    return pd.DataFrame(rows)


def _check_variance_homogeneity(groups: dict[str, np.ndarray]) -> dict:
    """Levene 變異數同質性檢定（需 ≥ 2 組）。"""
    arrays = [arr[~np.isnan(arr)] for arr in groups.values()]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        return {"stat": None, "p": None, "homogeneous": None, "note": "樣本不足"}
    stat, p = stats.levene(*arrays)
    return {
        "stat": round(stat, 4),
        "p": round(p, 4),
        "homogeneous": p > 0.05,
        "note": f"Levene W={stat:.4f}, p={p:.4f}",
    }


def _check_sample_sizes(groups: dict[str, np.ndarray]) -> dict[str, int]:
    return {k: int(np.sum(~np.isnan(v))) for k, v in groups.items()}


# ══════════════════════════════════════════════════════════════════════════════
# ── Effect Size ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two groups."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan


def _eta_squared(groups: dict[str, np.ndarray]) -> float:
    """η² for one-way ANOVA effect size."""
    all_vals = np.concatenate([v[~np.isnan(v)] for v in groups.values()])
    grand_mean = all_vals.mean()
    ss_between = sum(len(v[~np.isnan(v)]) * (v[~np.isnan(v)].mean() - grand_mean)**2
                     for v in groups.values())
    ss_total = np.sum((all_vals - grand_mean)**2)
    return ss_between / ss_total if ss_total > 0 else np.nan


def _epsilon_squared(H: float, n: int, k: int) -> float:
    """ε² for Kruskal-Wallis effect size."""
    return (H - k + 1) / (n - k) if (n - k) > 0 else np.nan


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation r for Mann-Whitney U."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    return 1 - (2 * U) / (n1 * n2)


def _interpret_d(d: float) -> str:
    d = abs(d)
    if np.isnan(d): return "—"
    if d < 0.2:  return "Negligible"
    if d < 0.5:  return "Small"
    if d < 0.8:  return "Medium"
    return "Large"


def _interpret_eta(eta: float) -> str:
    if np.isnan(eta): return "—"
    if eta < 0.01: return "Negligible"
    if eta < 0.06: return "Small"
    if eta < 0.14: return "Medium"
    return "Large"


def _interpret_r(r: float) -> str:
    r = abs(r)
    if np.isnan(r): return "—"
    if r < 0.1: return "Negligible"
    if r < 0.3: return "Small"
    if r < 0.5: return "Medium"
    return "Large"


# ══════════════════════════════════════════════════════════════════════════════
# ── 主檢定函式 ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _run_parametric(groups: dict[str, np.ndarray]) -> dict:
    """One-way ANOVA + post-hoc Tukey HSD（≥3 組）或 t-test（2 組）。"""
    arrays = [v[~np.isnan(v)] for v in groups.values()]
    labels = list(groups.keys())

    if len(arrays) == 2:
        a, b = arrays[0], arrays[1]
        lev = _check_variance_homogeneity(groups)
        equal_var = lev["homogeneous"] if lev["homogeneous"] is not None else True
        t_stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
        d = _cohens_d(a, b)
        return {
            "test_name": "Welch's t-test" if not equal_var else "Independent t-test",
            "statistic": round(t_stat, 4),
            "p_value": round(p, 4),
            "significant": p < 0.05,
            "effect_size": {"Cohen's d": round(d, 4), "解釋": _interpret_d(d)},
            "posthoc": None,
        }

    # ANOVA
    f_stat, p = stats.f_oneway(*arrays)
    eta2 = _eta_squared(groups)
    # Tukey HSD post-hoc
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    all_vals = np.concatenate(arrays)
    all_labels = np.concatenate([[lbl] * len(arr) for lbl, arr in zip(labels, arrays)])
    tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                             columns=tukey._results_table.data[0])
    # add Cohen's d to posthoc
    d_list = []
    for _, row in tukey_df.iterrows():
        g1 = groups.get(row["group1"], np.array([]))
        g2 = groups.get(row["group2"], np.array([]))
        d = _cohens_d(g1[~np.isnan(g1)], g2[~np.isnan(g2)])
        d_list.append(round(d, 4))
    tukey_df["Cohen's d"] = d_list
    tukey_df["效果量"] = [_interpret_d(d) for d in d_list]

    return {
        "test_name": "One-way ANOVA",
        "statistic": round(f_stat, 4),
        "p_value": round(p, 4),
        "significant": p < 0.05,
        "effect_size": {"η² (eta²)": round(eta2, 4), "解釋": _interpret_eta(eta2)},
        "posthoc": tukey_df,
    }


def _run_nonparametric(groups: dict[str, np.ndarray]) -> dict:
    """Kruskal-Wallis（≥3 組）或 Mann-Whitney U（2 組）+ post-hoc Dunn's。"""
    arrays = [v[~np.isnan(v)] for v in groups.values()]
    labels = list(groups.keys())

    if len(arrays) == 2:
        a, b = arrays[0], arrays[1]
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = _rank_biserial(a, b)
        return {
            "test_name": "Mann-Whitney U",
            "statistic": round(U, 4),
            "p_value": round(p, 4),
            "significant": p < 0.05,
            "effect_size": {"r (rank-biserial)": round(r, 4), "解釋": _interpret_r(r)},
            "posthoc": None,
        }

    # Kruskal-Wallis
    H, p = stats.kruskal(*arrays)
    n_total = sum(len(a) for a in arrays)
    eps2 = _epsilon_squared(H, n_total, len(arrays))
    # Dunn's test (manual Bonferroni)
    posthoc_rows = []
    pairs = list(combinations(labels, 2))
    n_pairs = len(pairs)
    for g1, g2 in pairs:
        a1 = groups[g1][~np.isnan(groups[g1])]
        a2 = groups[g2][~np.isnan(groups[g2])]
        try:
            _, p_pair = stats.mannwhitneyu(a1, a2, alternative="two-sided")
            p_adj = min(p_pair * n_pairs, 1.0)  # Bonferroni
        except Exception:
            p_pair, p_adj = np.nan, np.nan
        r = _rank_biserial(a1, a2)
        posthoc_rows.append({
            "group1": g1, "group2": g2,
            "p (unadj.)": round(p_pair, 4) if not np.isnan(p_pair) else None,
            "p (Bonferroni)": round(p_adj, 4) if not np.isnan(p_adj) else None,
            "significant*": "✅" if (not np.isnan(p_adj) and p_adj < 0.05) else "❌",
            "r (rank-biserial)": round(r, 4) if not np.isnan(r) else None,
            "效果量": _interpret_r(r),
        })

    return {
        "test_name": "Kruskal-Wallis H",
        "statistic": round(H, 4),
        "p_value": round(p, 4),
        "significant": p < 0.05,
        "effect_size": {"ε² (epsilon²)": round(eps2, 4), "解釋": _interpret_eta(eps2)},
        "posthoc": pd.DataFrame(posthoc_rows),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── 視覺化 ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _plot_assumption_checks(groups: dict[str, np.ndarray], target_col: str):
    """Q-Q plot per group for normality visual check."""
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), squeeze=False)
    axes = axes.flatten()
    for i, (label, arr) in enumerate(groups.items()):
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 3:
            stats.probplot(arr, dist="norm", plot=axes[i])
            axes[i].set_title(f"Q-Q: {label[:30]}", fontsize=9)
            axes[i].get_lines()[0].set(color="#2e86ab", markersize=4, alpha=0.8)
            axes[i].get_lines()[1].set(color="#e84855", linewidth=1.5)
        else:
            axes[i].text(0.5, 0.5, f"{label}\n樣本不足", ha="center", va="center",
                         fontsize=10, transform=axes[i].transAxes)
            axes[i].axis("off")
    plt.suptitle(f"Normal Q-Q Plots — {target_col[:50]}", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def _plot_boxplot(groups: dict[str, np.ndarray], zone_colors: dict[str, str],
                  target_col: str, result: dict):
    """Box + strip + significance annotation."""
    records = []
    for label, arr in groups.items():
        for v in arr[~np.isnan(arr)]:
            records.append({"Group": label, "Value": v})
    if not records:
        return None
    plot_df = pd.DataFrame(records)
    order = list(groups.keys())

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 2.5), 6))
    palette = {k: zone_colors.get(k, "#aaaaaa") for k in order}

    sns.boxplot(data=plot_df, x="Group", y="Value", order=order,
                palette=palette, width=0.5, fliersize=0,
                boxprops=dict(alpha=0.6), ax=ax)
    sns.stripplot(data=plot_df, x="Group", y="Value", order=order,
                  palette=palette, size=7, jitter=True,
                  edgecolor="black", linewidth=0.6, alpha=0.85, ax=ax)

    # ── 顯著性標注（pairwise）────────────────────────────────
    posthoc = result.get("posthoc")
    if posthoc is not None and len(posthoc) > 0:
        y_max = plot_df["Value"].max()
        y_range = plot_df["Value"].max() - plot_df["Value"].min()
        step = y_range * 0.10
        tick_h = y_range * 0.02
        sig_col = "p (Bonferroni)" if "p (Bonferroni)" in posthoc.columns else "p-adj"
        raw_p_col = "p-value" if "p-value" in posthoc.columns else None

        # determine which p-col to use
        if sig_col not in posthoc.columns:
            # Tukey uses "p-adj"
            sig_col = [c for c in posthoc.columns if "p" in c.lower() and "adj" in c.lower()]
            sig_col = sig_col[0] if sig_col else None

        if sig_col:
            drawn = 0
            for _, row in posthoc.iterrows():
                try:
                    p_val = float(row[sig_col])
                except Exception:
                    continue
                if p_val >= 0.05:
                    continue
                g1, g2 = str(row["group1"]), str(row["group2"])
                if g1 not in order or g2 not in order:
                    continue
                x1, x2 = order.index(g1), order.index(g2)
                y = y_max + step * (drawn + 1)
                ax.plot([x1, x1, x2, x2], [y - tick_h, y, y, y - tick_h],
                        lw=1.2, color="black")
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                ax.text((x1 + x2) / 2, y + tick_h * 0.5,
                        f"{stars} p={p_val:.3f}", ha="center", va="bottom", fontsize=8)
                drawn += 1

    ax.set_xlabel("分類區間", fontsize=11)
    ax.set_ylabel(target_col[:60], fontsize=11)
    ax.set_title(
        f"Box Plot — {target_col[:45]}\n"
        f"({result['test_name']}: {_sig_label(result['p_value'])})",
        fontsize=12,
    )
    ax.set_xticklabels([textwrap.fill(o, 18) for o in order], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _plot_effect_size_chart(posthoc: pd.DataFrame, metric: str, title: str):
    """橫條圖顯示各 pair 的 effect size。"""
    if posthoc is None or metric not in posthoc.columns:
        return None
    df = posthoc.dropna(subset=[metric]).copy()
    if df.empty:
        return None
    df["Pair"] = df["group1"].astype(str) + " vs " + df["group2"].astype(str)
    df[metric] = df[metric].astype(float)
    df = df.sort_values(metric, key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.7)))
    colors = ["#e84855" if abs(v) >= 0.8 else "#f4a261" if abs(v) >= 0.5
              else "#e9c46a" if abs(v) >= 0.2 else "#90e0ef"
              for v in df[metric]]
    ax.barh(df["Pair"], df[metric].abs(), color=colors, alpha=0.85, edgecolor="white")
    for thresh, lbl, col in [(0.2, "Small", "#e9c46a"), (0.5, "Medium", "#f4a261"),
                              (0.8, "Large", "#e84855")]:
        ax.axvline(thresh, color=col, linestyle="--", lw=1.2, label=lbl)
    ax.set_xlabel(f"|{metric}|")
    ax.set_title(title, fontsize=11)
    ax.legend(title="Effect Size Threshold", fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _sig_label(p: float) -> str:
    if p < 0.001: return f"p={p:.4f} ***"
    if p < 0.01:  return f"p={p:.4f} **"
    if p < 0.05:  return f"p={p:.4f} *"
    return f"p={p:.4f} ns"


# ══════════════════════════════════════════════════════════════════════════════
# ── 主 render ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    st.header("📐 統計檢定分析")

    # ── 資料來源說明 ─────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
    本功能基於「<b>分類趨勢分析</b>」產生的分群，對 Y 欄位進行組間統計檢定。<br>
    若尚未執行分類趨勢分析，請先至 <b>趨勢圖 → 📊 分類趨勢分析</b> 完成分群。
    </div>
    """, unsafe_allow_html=True)

    # ── 檢查是否有分群結果 ───────────────────────────────────
    ct_result_df = st.session_state.get("ct_result_df")
    ct_zones     = st.session_state.get("ct_zones_ct")
    ct_cls_col   = st.session_state.get("ct_classifier_col")
    ct_tgt_col   = st.session_state.get("ct_target_col")

    has_existing = (
        ct_result_df is not None
        and ct_zones is not None
        and "_class" in ct_result_df.columns
    )

    # ── 讓使用者選擇資料來源 ─────────────────────────────────
    numeric_cols = (
        selected_process_df.select_dtypes(include=["number"]).columns.tolist()
        if selected_process_df is not None else []
    )
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    source_mode = st.radio(
        "資料來源",
        ["📊 使用分類趨勢分析的分群結果", "🔧 在此直接設定分群"],
        horizontal=True,
        key="stat_source_mode",
    )

    # ── Mode A: 使用既有分群 ─────────────────────────────────
    if source_mode == "📊 使用分類趨勢分析的分群結果":
        if not has_existing:
            st.warning("⚠️ 尚未在「分類趨勢分析」中執行分群，請先完成分群或改用下方「直接設定分群」。")
            return

        st.success(
            f"✅ 已讀取分群：依 **{ct_cls_col}** 分為 {len(ct_zones)} 組，"
            f"目標變數：**{ct_tgt_col}**"
        )
        result_df  = ct_result_df.copy()
        zones_info = ct_zones
        target_col = ct_tgt_col

        # 允許使用者更換 Y 欄位
        if numeric_cols:
            swap_y = st.checkbox("更換目標 Y 欄位", key="stat_swap_y")
            if swap_y:
                target_col = st.selectbox("選擇新的目標 Y 欄位", numeric_cols, key="stat_new_y")
                # merge Y into result_df from work_df
                if work_df is not None and target_col in work_df.columns:
                    result_df = result_df.copy()
                    result_df[target_col] = work_df[target_col].values

    # ── Mode B: 直接設定分群 ─────────────────────────────────
    else:
        if work_df is None or len(numeric_cols) < 2:
            st.warning("請先在側欄選擇製程步驟並確保有數值欄位。")
            return

        st.markdown("#### ⚙️ 直接設定分群")
        bc1, bc2 = st.columns(2)
        classifier_col = bc1.selectbox("🔑 分類依據欄位", numeric_cols, key="stat_cls")
        target_col     = bc2.selectbox(
            "🎯 目標 Y 欄位",
            [c for c in numeric_cols if c != classifier_col],
            key="stat_tgt",
        )

        cls_s = work_df[classifier_col].dropna()
        st.caption(
            f"**{classifier_col}** 範圍：{cls_s.min():.3f} ～ {cls_s.max():.3f}"
            f"（平均 {cls_s.mean():.3f}）"
        )

        if st.button("⚡ 自動三等分", key="stat_auto3"):
            q33 = round(float(cls_s.quantile(0.33)), 3)
            q67 = round(float(cls_s.quantile(0.67)), 3)
            st.session_state.update({
                "stat_nz": 3,
                "stat_z0_label": f"Low (<{q33})", "stat_z0_min": float(cls_s.min()), "stat_z0_max": q33,
                "stat_z1_label": f"Standard ({q33}–{q67})", "stat_z1_min": q33, "stat_z1_max": q67,
                "stat_z2_label": f"High (>{q67})", "stat_z2_min": q67, "stat_z2_max": float(cls_s.max()),
            })

        n_zones_st = st.number_input("區間數量", 2, 8, st.session_state.get("stat_nz", 3), key="stat_nz")
        ST_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

        zones_info = []
        for i in range(int(n_zones_st)):
            default_min = round(float(cls_s.min()) + i * (float(cls_s.max()) - float(cls_s.min())) / n_zones_st, 3)
            default_max = round(float(cls_s.min()) + (i+1) * (float(cls_s.max()) - float(cls_s.min())) / n_zones_st, 3)
            with st.expander(f"區間 {i+1}", expanded=True):
                zc1, zc2, zc3, zc4 = st.columns([3, 1, 1, 1])
                zl = zc1.text_input("名稱", value=st.session_state.get(f"stat_z{i}_label", f"Zone {i+1}"), key=f"stat_zl_{i}")
                zm = zc2.number_input("最小值", value=float(st.session_state.get(f"stat_z{i}_min", default_min)), key=f"stat_zm_{i}", format="%.3f")
                zx = zc3.number_input("最大值", value=float(st.session_state.get(f"stat_z{i}_max", default_max)), key=f"stat_zx_{i}", format="%.3f")
                zc = zc4.color_picker("顏色", value=ST_COLORS[i % len(ST_COLORS)], key=f"stat_zc_{i}")
                zones_info.append({"label": zl, "min": zm, "max": zx, "color": zc})

        # 建立 result_df
        def _assign(val):
            if pd.isna(val): return "No Data"
            for z in zones_info:
                if z["min"] <= val <= z["max"]: return z["label"]
            return "Outside Zones"

        result_df = work_df.copy()
        result_df["_class"] = result_df[classifier_col].apply(_assign)

    # ══════════════════════════════════════════════════════════
    # ── 檢定設定 ──────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔬 檢定設定")

    test_col1, test_col2, test_col3 = st.columns(3)
    test_type = test_col1.radio(
        "檢定類型",
        ["🔵 參數檢定", "🟠 非參數檢定", "🤖 自動建議"],
        key="stat_test_type",
    )
    alpha_level = test_col2.select_slider(
        "顯著水準 α", [0.01, 0.05, 0.10], value=0.05, key="stat_alpha"
    )
    exclude_outside = test_col3.checkbox(
        "排除 Outside Zones", value=True, key="stat_excl_outside"
    )

    # ── 執行按鈕 ─────────────────────────────────────────────
    if st.button("🚀 執行統計檢定", type="primary", key="run_stat_test"):
        # 整理 groups
        valid_labels = [z["label"] for z in zones_info]
        if exclude_outside:
            df_test = result_df[result_df["_class"].isin(valid_labels)].copy()
        else:
            df_test = result_df.copy()

        if target_col not in df_test.columns:
            st.error(f"找不到目標欄位：{target_col}")
            return

        groups: dict[str, np.ndarray] = {}
        for label in ([z["label"] for z in zones_info] +
                      ([] if exclude_outside else ["Outside Zones"])):
            arr = df_test.loc[df_test["_class"] == label, target_col].dropna().values
            if len(arr) >= 1:
                groups[label] = arr

        if len(groups) < 2:
            st.error("有效分組數不足 2 組（各組至少需要 1 筆有效資料）。")
            return

        # ── 假設檢查 ──────────────────────────────────────────
        norm_df   = _check_normality(groups)
        lev_res   = _check_variance_homogeneity(groups)
        all_normal = all("✅" in str(r) for r in norm_df.get("常態？", []))
        homovar   = lev_res.get("homogeneous", False)

        # 自動建議
        if test_type == "🤖 自動建議":
            recommended = "parametric" if all_normal and homovar else "nonparametric"
        elif test_type == "🔵 參數檢定":
            recommended = "parametric"
        else:
            recommended = "nonparametric"

        # 執行
        with st.spinner("計算中..."):
            try:
                if recommended == "parametric":
                    result = _run_parametric(groups)
                else:
                    result = _run_nonparametric(groups)
            except Exception as e:
                st.error(f"檢定失敗：{e}")
                import traceback; st.code(traceback.format_exc())
                return

        zone_colors = {z["label"]: z["color"] for z in zones_info}

        st.session_state.update({
            "stat_groups":      groups,
            "stat_norm_df":     norm_df,
            "stat_lev_res":     lev_res,
            "stat_result":      result,
            "stat_zone_colors": zone_colors,
            "stat_target_col":  target_col,
            "stat_recommended": recommended,
            "stat_alpha":       alpha_level,
            "stat_all_normal":  all_normal,
            "stat_homovar":     homovar,
        })
        st.success("✅ 分析完成！")

    # ══════════════════════════════════════════════════════════
    # ── 顯示結果 ──────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    if st.session_state.get("stat_result") is None:
        return

    groups      = st.session_state["stat_groups"]
    norm_df     = st.session_state["stat_norm_df"]
    lev_res     = st.session_state["stat_lev_res"]
    result      = st.session_state["stat_result"]
    zone_colors = st.session_state["stat_zone_colors"]
    target_col  = st.session_state["stat_target_col"]
    recommended = st.session_state["stat_recommended"]
    all_normal  = st.session_state["stat_all_normal"]
    homovar     = st.session_state["stat_homovar"]

    res_tabs = st.tabs([
        "📋 假設檢查", "📊 Box Plot", "🧪 主檢定結果",
        "🔍 Post-hoc 分析", "📏 Effect Size"
    ])

    # ── Tab 0: 假設檢查 ───────────────────────────────────────
    with res_tabs[0]:
        st.markdown("#### 📋 資料假設檢查")

        # 常態性
        st.markdown("**① 常態性檢定（Shapiro-Wilk / K-S）**")
        norm_style = norm_df.style.apply(
            lambda col: ["background-color: #d4edda" if "✅" in str(v)
                         else "background-color: #f8d7da" if "❌" in str(v)
                         else "" for v in col],
            axis=0,
        )
        st.dataframe(norm_style, width="stretch", hide_index=True)

        if all_normal:
            st.success("✅ 所有組別通過常態性檢定（p > 0.05）")
        else:
            st.warning("⚠️ 部分組別不符合常態分佈，建議使用非參數檢定")

        # Q-Q plot
        with st.expander("🔍 Q-Q Plot（常態性視覺檢查）"):
            fig_qq = _plot_assumption_checks(groups, target_col)
            st.pyplot(fig_qq); plt.close()

        st.markdown("---")
        # 變異數同質性
        st.markdown("**② 變異數同質性（Levene's Test）**")
        lev_cols = st.columns(3)
        lev_cols[0].metric("Levene W 統計量", f"{lev_res.get('stat', '—')}")
        lev_cols[1].metric("p-value", f"{lev_res.get('p', '—')}")
        lev_cols[2].metric(
            "同質？",
            "✅ 是" if lev_res.get("homogeneous") else
            ("❌ 否" if lev_res.get("homogeneous") is False else "⚠️ 無法判斷")
        )
        if lev_res.get("homogeneous"):
            st.success("✅ 各組變異數無顯著差異（Levene p > 0.05）")
        elif lev_res.get("homogeneous") is False:
            st.warning("⚠️ 各組變異數存在顯著差異（Levene p ≤ 0.05）")

        st.markdown("---")
        # 自動建議
        st.markdown("**③ 檢定方法建議**")
        if all_normal and homovar:
            st.info("💡 **建議：參數檢定**（ANOVA / t-test）— 常態✅ + 變異數同質✅")
        elif all_normal and not homovar:
            st.info("💡 **建議：Welch's ANOVA / Welch's t-test**（常態✅ 但變異數不同質❌）")
        else:
            st.info("💡 **建議：非參數檢定**（Kruskal-Wallis / Mann-Whitney）— 不符常態❌")

        st.markdown(f"**本次執行：** `{result['test_name']}` ({'參數' if recommended == 'parametric' else '非參數'})")

    # ── Tab 1: Box Plot ───────────────────────────────────────
    with res_tabs[1]:
        st.markdown("#### 📊 Box Plot + 顯著性標注")
        fig_box = _plot_boxplot(groups, zone_colors, target_col, result)
        if fig_box:
            st.pyplot(fig_box); plt.close()
        else:
            st.warning("資料不足，無法繪製 Box Plot。")

        # Violin plot option
        if st.checkbox("同時顯示 Violin Plot", key="stat_show_violin"):
            records = []
            for label, arr in groups.items():
                for v in arr[~np.isnan(arr)]:
                    records.append({"Group": label, "Value": v})
            vdf = pd.DataFrame(records)
            if not vdf.empty:
                order = list(groups.keys())
                palette_v = {k: zone_colors.get(k, "#aaaaaa") for k in order}
                fig_v, ax_v = plt.subplots(figsize=(max(7, len(groups) * 2.5), 5))
                sns.violinplot(data=vdf, x="Group", y="Value", order=order,
                               palette=palette_v, inner="box", alpha=0.7, ax=ax_v)
                ax_v.set_xlabel("分類區間", fontsize=11)
                ax_v.set_ylabel(target_col[:60], fontsize=11)
                ax_v.set_title(f"Violin Plot — {target_col[:50]}", fontsize=12)
                ax_v.set_xticklabels([textwrap.fill(o, 18) for o in order], fontsize=9)
                ax_v.grid(axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig_v); plt.close()

    # ── Tab 2: 主檢定結果 ─────────────────────────────────────
    with res_tabs[2]:
        st.markdown(f"#### 🧪 {result['test_name']} 結果")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("檢定統計量", f"{result['statistic']:.4f}")
        rc2.metric("p-value", f"{result['p_value']:.4f}")
        rc3.metric(
            "結論",
            "✅ 顯著差異" if result["significant"] else "❌ 無顯著差異",
        )

        if result["significant"]:
            st.success(f"✅ 各組之間存在統計顯著差異（p = {result['p_value']:.4f} < α = {alpha_level}）")
        else:
            st.info(f"ℹ️ 各組之間無統計顯著差異（p = {result['p_value']:.4f} ≥ α = {alpha_level}）")

        # Effect size
        st.markdown("**整體 Effect Size**")
        es = result["effect_size"]
        es_metric, es_val = list(es.items())[0]
        es_interp = es.get("解釋", "—")
        ec1, ec2 = st.columns(2)
        ec1.metric(es_metric, f"{es_val:.4f}" if isinstance(es_val, float) else str(es_val))
        ec2.metric("效果量大小", es_interp)

        # 各組描述統計
        st.markdown("**各組描述統計**")
        desc_rows = []
        for label, arr in groups.items():
            a = arr[~np.isnan(arr)]
            desc_rows.append({
                "組別": label, "n": len(a),
                "Mean": round(a.mean(), 4) if len(a) else None,
                "Median": round(np.median(a), 4) if len(a) else None,
                "SD": round(a.std(ddof=1), 4) if len(a) > 1 else None,
                "Min": round(a.min(), 4) if len(a) else None,
                "Max": round(a.max(), 4) if len(a) else None,
                "IQR": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 4) if len(a) else None,
            })
        desc_df = pd.DataFrame(desc_rows)
        st.dataframe(
            desc_df.style.background_gradient(cmap="Blues", subset=["Mean"]),
            width="stretch", hide_index=True,
        )

    # ── Tab 3: Post-hoc ───────────────────────────────────────
    with res_tabs[3]:
        st.markdown("#### 🔍 Post-hoc 成對比較")
        posthoc = result.get("posthoc")

        if posthoc is None:
            st.info("兩組比較無需 post-hoc 分析（主檢定即為成對比較）。")
            # 顯示唯一一對的詳細資訊
            arr_list = list(groups.values())
            lbl_list = list(groups.keys())
            if len(arr_list) == 2:
                a0, a1 = arr_list[0][~np.isnan(arr_list[0])], arr_list[1][~np.isnan(arr_list[1])]
                mean_diff = a0.mean() - a1.mean() if len(a0) and len(a1) else np.nan
                ci = stats.t.interval(
                    0.95, df=len(a0)+len(a1)-2,
                    loc=mean_diff,
                    scale=stats.sem(np.concatenate([a0, a1]))
                ) if len(a0) + len(a1) > 2 else (np.nan, np.nan)
                st.markdown(f"""
| 項目 | 值 |
|---|---|
| {lbl_list[0]} Mean | {a0.mean():.4f} |
| {lbl_list[1]} Mean | {a1.mean():.4f} |
| Mean Diff | {mean_diff:.4f} |
| 95% CI | [{ci[0]:.4f}, {ci[1]:.4f}] |
""")
        else:
            method_name = (
                "Tukey HSD" if recommended == "parametric"
                else "Dunn's test (Bonferroni adjusted)"
            )
            st.caption(f"方法：{method_name}")

            # 顏色標示顯著列
            def _style_posthoc(df):
                sig_col = next(
                    (c for c in ["significant*", "reject"] if c in df.columns), None
                )
                if sig_col is None:
                    return df
                return df.style.apply(
                    lambda row: [
                        "background-color: #d4edda" if "✅" in str(row.get(sig_col, "")) else ""
                    ] * len(row),
                    axis=1,
                )

            st.dataframe(_style_posthoc(posthoc), width="stretch", hide_index=True)

    # ── Tab 4: Effect Size ────────────────────────────────────
    with res_tabs[4]:
        st.markdown("#### 📏 Effect Size 視覺化")
        posthoc = result.get("posthoc")

        # 整體 effect size 說明
        es = result["effect_size"]
        es_metric, es_val = list(es.items())[0]
        es_interp = es.get("解釋", "—")

        st.markdown(f"""
| 指標 | 數值 | 解釋 |
|---|---|---|
| **{es_metric}** | {es_val:.4f} | **{es_interp}** |
""")
        st.markdown("""
> **Effect Size 解讀標準**
> - Cohen's d / r：Negligible < 0.2 ≤ Small < 0.5 ≤ Medium < 0.8 ≤ Large
> - η² / ε²：Negligible < 0.01 ≤ Small < 0.06 ≤ Medium < 0.14 ≤ Large
""")

        if posthoc is not None and len(groups) >= 3:
            # 找 effect size 欄位
            d_col = next(
                (c for c in ["Cohen's d", "r (rank-biserial)"] if c in posthoc.columns),
                None,
            )
            if d_col:
                fig_es = _plot_effect_size_chart(
                    posthoc, d_col,
                    f"Pairwise Effect Size ({d_col})"
                )
                if fig_es:
                    st.pyplot(fig_es); plt.close()

        # Heatmap of pairwise effect sizes
        if posthoc is not None:
            d_col = next(
                (c for c in ["Cohen's d", "r (rank-biserial)"] if c in posthoc.columns),
                None,
            )
            if d_col:
                st.markdown("**Pairwise Effect Size Heatmap**")
                labels = list(groups.keys())
                mat = pd.DataFrame(np.zeros((len(labels), len(labels))),
                                   index=labels, columns=labels)
                for _, row in posthoc.iterrows():
                    g1, g2 = str(row["group1"]), str(row["group2"])
                    try:
                        v = abs(float(row[d_col]))
                    except Exception:
                        v = 0
                    if g1 in mat.index and g2 in mat.columns:
                        mat.loc[g1, g2] = v
                        mat.loc[g2, g1] = v

                fig_hm, ax_hm = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
                sns.heatmap(mat.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
                            vmin=0, vmax=1, ax=ax_hm,
                            linewidths=0.5, linecolor="white",
                            cbar_kws={"label": f"|{d_col}|"})
                ax_hm.set_title(f"Pairwise |{d_col}| Heatmap", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_hm); plt.close()
