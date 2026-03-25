"""Tab 1 — 趨勢圖 + 特徵比較（色帶區間）+ 分類趨勢分析"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st
from scipy import stats as _scipy_stats

from utils import filt_specific_name, smooth_process_data, plot_indexed_lineplots, extract_number


# ══════════════════════════════════════════════════════════════════════════════
# ── 相關係數工具 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _corr_stats(x: np.ndarray, y: np.ndarray, method: str = "pearson"):
    """
    回傳 (r, p, n, 方法名稱) — 自動濾掉 NaN pair。
    method: 'pearson' | 'spearman'
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return np.nan, np.nan, n, method
    if method == "spearman":
        r, p = _scipy_stats.spearmanr(xv, yv)
    else:
        r, p = _scipy_stats.pearsonr(xv, yv)
    return float(r), float(p), n, method


def _corr_label(r, p, n, method):
    """格式化相關係數文字。"""
    if np.isnan(r):
        return f"n={n}，樣本不足"
    sym = "ρ" if method == "spearman" else "r"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return f"{sym}={r:.3f}  p={p:.4f} {stars}  (n={n})"


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 2: 特徵比較 + 區間（散佈圖 / 雙軸折線）────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _zone_assign(val, zones):
    if pd.isna(val):
        return None
    for z in zones:
        if z["min"] <= val <= z["max"]:
            return z["label"]
    return "Outside"


def _plot_feature_comparison(df, feat_x, feat_y, zones, batch_col="BatchID",
                              plot_type="scatter", smooth_method="none", frac=0.3,
                              color_mode="zone",       # "zone" | "gradient"
                              cmap_name="viridis",
                              corr_method="pearson",   # "pearson" | "spearman"
                              show_corr_per_zone=True,
                              show_trendline=False):
    plot_df = df.copy()
    if batch_col in plot_df.columns:
        plot_df["_sort"] = plot_df[batch_col].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    x_vals = plot_df[feat_x].values.astype(float)
    y_vals = plot_df[feat_y].values.astype(float)

    if smooth_method != "none":
        tmp = smooth_process_data(plot_df[[feat_x, feat_y]], [feat_x, feat_y],
                                  id_cols=[], method=smooth_method, frac=frac)
        x_vals_s = tmp[feat_x].values if feat_x in tmp else x_vals
        y_vals_s = tmp[feat_y].values if feat_y in tmp else y_vals
    else:
        x_vals_s, y_vals_s = x_vals, y_vals

    # ── 顏色準備 ─────────────────────────────────────────────
    if color_mode == "gradient":
        # 連續漸層：依 x_vals 大小對映顏色
        cmap = mcm.get_cmap(cmap_name)
        x_norm = x_vals.copy()
        finite_mask = np.isfinite(x_norm)
        if finite_mask.sum() > 1:
            xmin, xmax = x_norm[finite_mask].min(), x_norm[finite_mask].max()
            x_norm = (x_norm - xmin) / (xmax - xmin + 1e-12)
        else:
            x_norm = np.zeros_like(x_norm)
        point_colors = cmap(x_norm)
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(
                vmin=x_vals[finite_mask].min() if finite_mask.sum() > 0 else 0,
                vmax=x_vals[finite_mask].max() if finite_mask.sum() > 0 else 1,
            ),
        )
        sm.set_array([])
    else:
        # 分區顏色
        def _point_color(val):
            for z in zones:
                if np.isfinite(val) and z["min"] <= val <= z["max"]:
                    return z["color"]
            return "#aaaaaa"
        point_colors = [_point_color(v) for v in x_vals]

    # ══════════════════════════════════════════════════════════
    # ── scatter / scatter+line ────────────────────────────────
    if plot_type in ("scatter", "scatter+line"):
        fig, ax = plt.subplots(figsize=(11, 6))

        # 背景色帶
        if color_mode == "zone":
            for z in zones:
                ax.axvspan(z["min"], z["max"], alpha=0.08, color=z["color"], zorder=0)

        if plot_type == "scatter+line":
            ax.plot(x_vals_s, y_vals_s, color="#bbbbbb", linewidth=0.9,
                    alpha=0.5, zorder=1)

        sc = ax.scatter(x_vals, y_vals, c=point_colors, s=70,
                        edgecolors="white", linewidths=0.6, zorder=3, alpha=0.88)

        # 顏色條（漸層模式）
        if color_mode == "gradient":
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(feat_x[:40], fontsize=9)

        # Batch 標籤
        if batch_col in plot_df.columns:
            for xi, yi, bid in zip(x_vals, y_vals, plot_df[batch_col]):
                if np.isfinite(xi) and np.isfinite(yi):
                    ax.annotate(str(bid)[-6:], (xi, yi), fontsize=5.5, alpha=0.6,
                                xytext=(3, 3), textcoords="offset points")

        # ── 全域相關係數 ──────────────────────────────────────
        r_all, p_all, n_all, _ = _corr_stats(x_vals, y_vals, corr_method)
        ax.set_title(
            f"{feat_y}  vs  {feat_x}\n全域 {_corr_label(r_all, p_all, n_all, corr_method)}",
            fontsize=11, pad=10,
        )

        # ── 全域趨勢線 ────────────────────────────────────────
        if show_trendline:
            valid = np.isfinite(x_vals) & np.isfinite(y_vals)
            if valid.sum() >= 2:
                m, b = np.polyfit(x_vals[valid], y_vals[valid], 1)
                xfit = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 200)
                ax.plot(xfit, m * xfit + b, color="#333333", lw=1.5,
                        linestyle="--", alpha=0.7, label="Overall fit", zorder=4)

        # ── 各 Zone 相關係數標注 ──────────────────────────────
        if show_corr_per_zone and color_mode == "zone":
            corr_lines = []
            for z in zones:
                mask_z = np.array([_zone_assign(v, zones) == z["label"] for v in x_vals])
                if mask_z.sum() < 3:
                    continue
                xz, yz = x_vals[mask_z], y_vals[mask_z]
                r_z, p_z, n_z, _ = _corr_stats(xz, yz, corr_method)

                # Zone 趨勢線
                if show_trendline and np.isfinite(r_z):
                    valid_z = np.isfinite(xz) & np.isfinite(yz)
                    if valid_z.sum() >= 2:
                        m_z, b_z = np.polyfit(xz[valid_z], yz[valid_z], 1)
                        xfit_z = np.linspace(xz[valid_z].min(), xz[valid_z].max(), 100)
                        ax.plot(xfit_z, m_z * xfit_z + b_z, color=z["color"],
                                lw=1.8, linestyle="-", alpha=0.75, zorder=5)

                corr_lines.append(
                    mpatches.Patch(
                        color=z["color"], alpha=0.8,
                        label=f"{z['label']}: {_corr_label(r_z, p_z, n_z, corr_method)}",
                    )
                )

            if corr_lines:
                # 加上 Outside
                out_mask = np.array([_zone_assign(v, zones) == "Outside" for v in x_vals])
                if out_mask.sum() >= 3:
                    r_o, p_o, n_o, _ = _corr_stats(x_vals[out_mask], y_vals[out_mask], corr_method)
                    corr_lines.append(
                        mpatches.Patch(color="#aaaaaa", alpha=0.8,
                                       label=f"Outside: {_corr_label(r_o, p_o, n_o, corr_method)}")
                    )
                ax.legend(handles=corr_lines, loc="best", fontsize=7.5,
                          title=f"Zone 相關係數（{corr_method}）", title_fontsize=8,
                          framealpha=0.85)
        else:
            # 分區圖例（無相關係數）
            if color_mode == "zone":
                handles = [mpatches.Patch(color=z["color"], alpha=0.7, label=z["label"])
                           for z in zones]
                handles.append(mpatches.Patch(color="#aaaaaa", alpha=0.7, label="Outside zones"))
                ax.legend(handles=handles, loc="best", fontsize=8)

        ax.set_xlabel(feat_x[:60], fontsize=10)
        ax.set_ylabel(feat_y[:60], fontsize=10)
        ax.grid(linestyle="--", alpha=0.35)
        plt.tight_layout()
        return fig, plot_df

    # ══════════════════════════════════════════════════════════
    # ── dual_line ─────────────────────────────────────────────
    elif plot_type == "dual_line":
        fig, ax1 = plt.subplots(figsize=(13, 5))
        seq = plot_df["_seq"].values
        ax1.set_xlabel("Batch Sequence", fontsize=10)
        ax1.set_ylabel(feat_x[:50], color="#2e86ab", fontsize=10)
        ax1.plot(seq, x_vals_s, color="#2e86ab", marker="o", ms=4,
                 linewidth=1.5, label=feat_x[:30])
        ax1.tick_params(axis="y", labelcolor="#2e86ab")
        for z in zones:
            ax1.axhspan(z["min"], z["max"], alpha=0.10, color=z["color"], zorder=0)
        ax2 = ax1.twinx()
        ax2.set_ylabel(feat_y[:50], color="#e84855", fontsize=10)
        ax2.plot(seq, y_vals_s, color="#e84855", marker="s", ms=4,
                 linewidth=1.5, linestyle="--", label=feat_y[:30])
        ax2.tick_params(axis="y", labelcolor="#e84855")
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        zone_handles = [mpatches.Patch(color=z["color"], alpha=0.5, label=z["label"]) for z in zones]
        ax1.legend(l1 + l2 + zone_handles, lb1 + lb2 + [z["label"] for z in zones],
                   loc="best", fontsize=8)
        r_all, p_all, n_all, _ = _corr_stats(x_vals, y_vals, corr_method)
        ax1.set_title(
            f"{feat_x[:35]}  &  {feat_y[:35]}  over Batch Sequence\n"
            f"{_corr_label(r_all, p_all, n_all, corr_method)}",
            fontsize=11,
        )
        ax1.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        return fig, plot_df

    return None, plot_df


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 3: 分類趨勢分析 ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _plot_classified_trend(df, classifier_col, target_col, zones, batch_col="BatchID",
                            show_avg_line=True, show_batch_label=False, point_size=120,
                            color_mode="zone", cmap_name="plasma"):
    plot_df = df.copy()
    if batch_col in plot_df.columns:
        plot_df["_sort"] = plot_df[batch_col].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    def _assign(val):
        if pd.isna(val):
            return "No Data"
        for z in zones:
            if z["min"] <= val <= z["max"]:
                return z["label"]
        return "Outside Zones"

    plot_df["_class"] = plot_df[classifier_col].apply(_assign)

    palette = {z["label"]: z["color"] for z in zones}
    palette["Outside Zones"] = "#aaaaaa"
    palette["No Data"] = "#dddddd"
    order = [z["label"] for z in zones] + ["Outside Zones"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.set_style("whitegrid")

    # 背景折線
    ax.plot(plot_df["_seq"], plot_df[target_col],
            color="silver", alpha=0.35, linestyle="-", linewidth=1.2, zorder=1)

    # 平均線
    if show_avg_line:
        avg_val = plot_df[target_col].mean()
        ax.axhline(avg_val, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Avg: {avg_val:.2f}", zorder=2)

    # ── 漸層模式 ──────────────────────────────────────────────
    if color_mode == "gradient":
        cmap = mcm.get_cmap(cmap_name)
        cls_vals = plot_df[classifier_col].values.astype(float)
        finite_m = np.isfinite(cls_vals)
        if finite_m.sum() > 1:
            vmin, vmax = cls_vals[finite_m].min(), cls_vals[finite_m].max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)
        c_mapped = [cmap(norm(v)) if np.isfinite(v) else "#dddddd" for v in cls_vals]
        y_vals = plot_df[target_col].values
        sc = ax.scatter(plot_df["_seq"], y_vals, c=c_mapped,
                        s=point_size, edgecolors="black", linewidths=0.7,
                        alpha=0.9, zorder=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(classifier_col[:40], fontsize=9)

    else:
        # 分區顏色散點
        present_classes = [o for o in order if o in plot_df["_class"].values]
        for cls in present_classes:
            mask = plot_df["_class"] == cls
            ax.scatter(
                plot_df.loc[mask, "_seq"],
                plot_df.loc[mask, target_col],
                color=palette.get(cls, "#aaaaaa"),
                s=point_size, edgecolors="black", linewidths=0.7,
                alpha=0.9, label=cls, zorder=3,
            )

    # Batch 標籤
    if show_batch_label and batch_col in plot_df.columns:
        for _, row in plot_df.iterrows():
            if pd.notna(row[target_col]):
                ax.annotate(str(row[batch_col])[-6:], (row["_seq"], row[target_col]),
                            fontsize=6, alpha=0.65, xytext=(0, 6),
                            textcoords="offset points", ha="center")

    # X 軸
    if batch_col in plot_df.columns and len(plot_df) <= 80:
        ax.set_xticks(plot_df["_seq"])
        ax.set_xticklabels([str(b)[-6:] for b in plot_df[batch_col]],
                           rotation=90, fontsize=7)
    else:
        ax.set_xlabel("Batch Sequence", fontsize=11)

    ax.set_ylabel(target_col[:60], fontsize=11)
    ax.set_title(f"{target_col[:50]}  —  colored by  {classifier_col[:50]}",
                 fontsize=13, pad=14)
    if color_mode == "zone":
        ax.legend(title=classifier_col[:40], bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig, plot_df


# ══════════════════════════════════════════════════════════════════════════════
# ── Box plot (各 Zone Y 值分布) ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _plot_zone_boxplot(plot_df, zones, target_col, classifier_col):
    """為每個 Zone 畫 Box + strip，比較 Y 值分布。"""
    records = []
    for z in zones:
        mask = plot_df["_class"] == z["label"]
        for v in plot_df.loc[mask, target_col].dropna():
            records.append({"Zone": z["label"], "Value": float(v)})
    out_mask = plot_df["_class"] == "Outside Zones"
    for v in plot_df.loc[out_mask, target_col].dropna():
        records.append({"Zone": "Outside Zones", "Value": float(v)})

    if not records:
        return None

    box_df = pd.DataFrame(records)
    zone_labels = [z["label"] for z in zones]
    if "Outside Zones" in box_df["Zone"].values:
        zone_labels.append("Outside Zones")
    palette = {z["label"]: z["color"] for z in zones}
    palette["Outside Zones"] = "#aaaaaa"

    fig, ax = plt.subplots(figsize=(max(7, len(zone_labels) * 2.2), 5))
    sns.boxplot(data=box_df, x="Zone", y="Value", order=zone_labels,
                palette=palette, width=0.5, fliersize=0,
                boxprops=dict(alpha=0.55), ax=ax)
    sns.stripplot(data=box_df, x="Zone", y="Value", order=zone_labels,
                  palette=palette, size=6, jitter=True,
                  edgecolor="black", linewidth=0.5, alpha=0.8, ax=ax)
    ax.set_xlabel(f"Zone（依 {classifier_col[:30]}）", fontsize=10)
    ax.set_ylabel(target_col[:60], fontsize=10)
    ax.set_title(f"{target_col[:50]} 各區間分布比較", fontsize=12)
    ax.set_xticklabels([textwrap.fill(l, 18) for l in zone_labels], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ── ANOVA / Kruskal 快速檢定 ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _quick_stat_test(plot_df, zones, target_col):
    """回傳 dict with test_name, stat, p, recommendation。"""
    groups = {}
    for z in zones:
        mask = plot_df["_class"] == z["label"]
        arr = plot_df.loc[mask, target_col].dropna().values.astype(float)
        if len(arr) >= 2:
            groups[z["label"]] = arr

    if len(groups) < 2:
        return None

    arrays = list(groups.values())

    # 常態性
    all_normal = True
    for arr in arrays:
        if len(arr) < 3:
            all_normal = False; break
        if len(arr) <= 50:
            _, p_sw = _scipy_stats.shapiro(arr)
            if p_sw <= 0.05:
                all_normal = False; break
        else:
            _, p_ks = _scipy_stats.kstest(arr, "norm", args=(arr.mean(), arr.std(ddof=1)))
            if p_ks <= 0.05:
                all_normal = False; break

    if all_normal:
        stat, p = _scipy_stats.f_oneway(*arrays)
        test_name = "One-way ANOVA"
    else:
        stat, p = _scipy_stats.kruskal(*arrays)
        test_name = "Kruskal-Wallis H"

    return {
        "test_name": test_name,
        "stat": round(float(stat), 4),
        "p": round(float(p), 4),
        "significant": p < 0.05,
        "all_normal": all_normal,
        "groups": groups,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── render ────────────────────────────────────────────────────────════════════
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    subtabs = st.tabs(["📈 全局趨勢圖", "🎨 特徵比較 + 區間顏色", "📊 分類趨勢分析"])

    # ══════════════════════════════════════════════════════════
    # ── Tab 0: 全局趨勢圖 ─────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    with subtabs[0]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        keyword       = col_a.text_input("欄位關鍵字篩選（留空 = 全部）", "")
        smooth_method = col_b.selectbox("平滑方法", ["loess", "ewma", "none"], key="trend_smooth")
        cols_per_row  = col_c.slider("每列圖數", 1, 5, 3, key="trend_cols")

        if keyword:
            display_df = filt_specific_name(selected_process_df, keyword)
            if "BatchID" not in display_df.columns and "BatchID" in selected_process_df.columns:
                display_df.insert(0, "BatchID", selected_process_df["BatchID"])
        else:
            display_df = selected_process_df.copy()

        if st.button("🖼️ 繪製趨勢圖", key="plot_trend"):
            if smooth_method != "none":
                num_cols = display_df.select_dtypes(include=["number"]).columns.tolist()
                plot_df  = smooth_process_data(display_df, num_cols, method=smooth_method)
                if "BatchID" in display_df.columns:
                    plot_df["BatchID"] = display_df["BatchID"].values
            else:
                plot_df = display_df.copy()
            with st.spinner("繪圖中..."):
                fig = plot_indexed_lineplots(plot_df, cols_per_row=cols_per_row)
                if fig:
                    st.pyplot(fig); plt.close()

    # ══════════════════════════════════════════════════════════
    # ── Tab 1: 特徵比較 + 區間顏色 ────────────────────────────
    # ══════════════════════════════════════════════════════════
    with subtabs[1]:
        st.markdown("#### 🎨 特徵比較圖（自訂區間顏色）")
        numeric_cols = selected_process_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("需要至少 2 個數值欄位。")
            return

        ca, cb, cc = st.columns(3)
        feat_x    = ca.selectbox("X 軸 / 主特徵", numeric_cols, key="cz_feat_x")
        feat_y    = cb.selectbox("Y 軸 / 比較特徵",
                                  [c for c in numeric_cols if c != feat_x], key="cz_feat_y")
        plot_type = cc.selectbox(
            "圖表類型", ["scatter", "scatter+line", "dual_line"],
            key="cz_plot_type",
            format_func=lambda x: {"scatter": "散佈圖", "scatter+line": "散佈圖+連線",
                                   "dual_line": "雙軸折線"}[x],
        )

        cd, ce = st.columns(2)
        smooth_cz = cd.selectbox("平滑（雙軸折線）", ["none", "loess", "ewma"], key="cz_smooth")
        frac_cz   = ce.slider("LOESS frac", 0.1, 0.8, 0.3, 0.05, key="cz_frac",
                               disabled=(smooth_cz != "loess"))

        # ── 新增選項 ──────────────────────────────────────────
        opt1, opt2, opt3, opt4 = st.columns(4)
        color_mode_cz  = opt1.radio("著色模式", ["zone", "gradient"], key="cz_color_mode",
                                     format_func=lambda x: {"zone": "🎨 分區顏色",
                                                            "gradient": "🌈 連續漸層"}[x])
        cmap_cz        = opt2.selectbox("漸層色板", ["viridis", "plasma", "coolwarm",
                                                     "RdYlGn", "RdBu", "YlOrRd"],
                                        key="cz_cmap",
                                        disabled=(color_mode_cz != "gradient"))
        corr_method_cz = opt3.radio("相關係數", ["pearson", "spearman"], key="cz_corr_method",
                                     format_func=lambda x: {"pearson": "Pearson r",
                                                            "spearman": "Spearman ρ"}[x])
        show_corr_cz   = opt4.checkbox("顯示各 Zone 相關係數", value=True, key="cz_show_corr",
                                        disabled=(color_mode_cz != "zone"))
        show_fit_cz    = opt4.checkbox("顯示趨勢線", value=False, key="cz_show_fit")

        x_series = selected_process_df[feat_x].dropna()
        st.caption(
            f"**{feat_x}** 範圍：{x_series.min():.3f} ～ {x_series.max():.3f}"
            f"（平均 {x_series.mean():.3f}）"
        )

        st.markdown("#### 🎯 設定數值區間")
        n_zones = st.number_input("區間數量", min_value=1, max_value=8, value=3,
                                   step=1, key="cz_n_zones")
        COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db",
                  "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
        zones = []
        for i in range(int(n_zones)):
            with st.expander(f"區間 {i+1}", expanded=(i < 3)):
                zc1, zc2, zc3, zc4 = st.columns([2, 1, 1, 1])
                label = zc1.text_input("名稱", value=f"Zone {i+1}", key=f"cz_label_{i}")
                zmin  = zc2.number_input(
                    "最小值",
                    value=float(round(x_series.min() + i * (x_series.max() - x_series.min()) / n_zones, 3)),
                    key=f"cz_min_{i}", format="%.3f",
                )
                zmax  = zc3.number_input(
                    "最大值",
                    value=float(round(x_series.min() + (i+1) * (x_series.max() - x_series.min()) / n_zones, 3)),
                    key=f"cz_max_{i}", format="%.3f",
                )
                color = zc4.color_picker("顏色", value=COLORS[i % len(COLORS)], key=f"cz_color_{i}")
                zones.append({"label": label, "min": zmin, "max": zmax, "color": color})

        if st.button("🎨 繪製比較圖", type="primary", key="plot_comparison"):
            if not all(z["min"] < z["max"] for z in zones):
                st.error("每個區間的最小值必須小於最大值。")
            else:
                with st.spinner("繪圖中..."):
                    try:
                        result = _plot_feature_comparison(
                            selected_process_df, feat_x, feat_y, zones,
                            plot_type=plot_type,
                            smooth_method=smooth_cz, frac=frac_cz,
                            color_mode=color_mode_cz, cmap_name=cmap_cz,
                            corr_method=corr_method_cz,
                            show_corr_per_zone=show_corr_cz,
                            show_trendline=show_fit_cz,
                        )
                        fig = result[0] if isinstance(result, tuple) else result
                        if fig:
                            st.pyplot(fig); plt.close()
                    except Exception as e:
                        st.error(f"繪圖失敗：{e}")
                        import traceback; st.code(traceback.format_exc())

        # ── 批次 Zone 表格 ────────────────────────────────────
        if st.checkbox("顯示各批次所在區間", key="cz_show_table"):
            def assign_zone(val):
                for z in zones:
                    if pd.notna(val) and z["min"] <= val <= z["max"]:
                        return z["label"]
                return "Outside"
            cols_needed = (["BatchID", feat_x, feat_y]
                           if "BatchID" in selected_process_df.columns
                           else [feat_x, feat_y])
            summary = selected_process_df[cols_needed].copy()
            summary["Zone"] = selected_process_df[feat_x].apply(assign_zone)
            st.dataframe(summary.sort_values(feat_x).reset_index(drop=True),
                         width="stretch", hide_index=True)
            zone_stats = []
            for z in zones:
                mask = summary["Zone"] == z["label"]
                n = mask.sum()
                if n > 0:
                    zone_stats.append({
                        "Zone": z["label"],
                        f"{feat_x} 範圍": f"{z['min']:.3f}–{z['max']:.3f}",
                        "批次數": n,
                        f"{feat_y} 平均": summary.loc[mask, feat_y].mean().round(3),
                        f"{feat_y} 標準差": summary.loc[mask, feat_y].std().round(3),
                    })
            if zone_stats:
                st.dataframe(pd.DataFrame(zone_stats), width="stretch", hide_index=True)

    # ══════════════════════════════════════════════════════════
    # ── Tab 2: 分類趨勢分析 ───────────────────────────────────
    # ══════════════════════════════════════════════════════════
    with subtabs[2]:
        st.markdown("#### 📊 分類趨勢分析")
        st.markdown(
            "選擇一個**分類依據欄位**（如緩衝液溫度）與**目標 Y 欄位**（如 Yield Rate），"
            "設定數值區間後，散點將依分類著色顯示在 Y 的時序趨勢上。"
        )

        numeric_cols = selected_process_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("需要至少 2 個數值欄位。")
        else:
            ct_col1, ct_col2 = st.columns(2)
            classifier_col = ct_col1.selectbox(
                "🔑 分類依據欄位（X）", numeric_cols, key="ct_classifier",
                help="用來分群的欄位，例如：最高緩衝液溫度",
            )
            target_col_ct = ct_col2.selectbox(
                "🎯 目標 Y 欄位",
                [c for c in numeric_cols if c != classifier_col],
                key="ct_target",
                help="顯示在 Y 軸的欄位，例如：Yield Rate",
            )

            cls_series = selected_process_df[classifier_col].dropna()
            cls_min, cls_max = cls_series.min(), cls_series.max()
            st.caption(
                f"**{classifier_col}** 範圍：{cls_min:.3f} ～ {cls_max:.3f}"
                f"（平均 {cls_series.mean():.3f}，中位數 {cls_series.median():.3f}）"
            )

            # 顯示選項
            opt1, opt2, opt3, opt4 = st.columns(4)
            show_avg       = opt1.checkbox("顯示平均線", value=True, key="ct_avg")
            show_labels    = opt2.checkbox("顯示 Batch 標籤", value=False, key="ct_labels")
            point_sz       = opt3.slider("散點大小", 40, 300, 120, 20, key="ct_ptsize")
            color_mode_ct  = opt4.radio(
                "著色模式", ["zone", "gradient"], key="ct_color_mode",
                format_func=lambda x: {"zone": "🎨 分區", "gradient": "🌈 漸層"}[x],
            )
            cmap_ct = opt4.selectbox(
                "漸層色板", ["plasma", "viridis", "coolwarm", "RdYlGn", "YlOrRd"],
                key="ct_cmap", disabled=(color_mode_ct != "gradient"),
            )

            # 區間設定
            st.markdown("#### 🎯 設定分類區間")

            if st.button("⚡ 自動填入三等分區間", key="ct_auto_zones"):
                q33 = round(float(cls_series.quantile(0.33)), 3)
                q67 = round(float(cls_series.quantile(0.67)), 3)
                st.session_state["ct_n_zones"] = 3
                st.session_state["ct_z0_min"]   = round(float(cls_min), 3)
                st.session_state["ct_z0_max"]   = q33
                st.session_state["ct_z0_label"] = f"Low (< {q33})"
                st.session_state["ct_z1_min"]   = q33
                st.session_state["ct_z1_max"]   = q67
                st.session_state["ct_z1_label"] = f"Standard ({q33}–{q67})"
                st.session_state["ct_z2_min"]   = q67
                st.session_state["ct_z2_max"]   = round(float(cls_max), 3)
                st.session_state["ct_z2_label"] = f"High (> {q67})"

            n_zones_ct = st.number_input(
                "區間數量", min_value=2, max_value=8,
                value=st.session_state.get("ct_n_zones", 3),
                key="ct_n_zones",
            )
            CT_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e",
                         "#9467bd", "#8c564b", "#e377c2", "#17becf"]

            zones_ct = []
            for i in range(int(n_zones_ct)):
                default_min = round(cls_min + i * (cls_max - cls_min) / n_zones_ct, 3)
                default_max = round(cls_min + (i + 1) * (cls_max - cls_min) / n_zones_ct, 3)
                with st.expander(f"區間 {i + 1}", expanded=True):
                    zc1, zc2, zc3, zc4 = st.columns([3, 1, 1, 1])
                    z_label = zc1.text_input(
                        "名稱",
                        value=st.session_state.get(f"ct_z{i}_label", f"Zone {i + 1}"),
                        key=f"ct_zlabel_{i}",
                    )
                    z_min = zc2.number_input(
                        "最小值",
                        value=float(st.session_state.get(f"ct_z{i}_min", default_min)),
                        key=f"ct_zmin_{i}", format="%.3f",
                    )
                    z_max = zc3.number_input(
                        "最大值",
                        value=float(st.session_state.get(f"ct_z{i}_max", default_max)),
                        key=f"ct_zmax_{i}", format="%.3f",
                    )
                    z_color = zc4.color_picker(
                        "顏色", value=CT_COLORS[i % len(CT_COLORS)],
                        key=f"ct_zcolor_{i}",
                    )
                    zones_ct.append({"label": z_label, "min": z_min,
                                     "max": z_max, "color": z_color})

            # 繪圖
            if st.button("📊 繪製分類趨勢圖", type="primary", key="ct_plot"):
                if not all(z["min"] < z["max"] for z in zones_ct):
                    st.error("每個區間的最小值必須小於最大值。")
                elif len(set(z["label"] for z in zones_ct)) < len(zones_ct):
                    st.error("區間名稱不可重複。")
                else:
                    with st.spinner("繪圖中..."):
                        try:
                            fig_ct, result_df = _plot_classified_trend(
                                selected_process_df,
                                classifier_col=classifier_col,
                                target_col=target_col_ct,
                                zones=zones_ct,
                                show_avg_line=show_avg,
                                show_batch_label=show_labels,
                                point_size=point_sz,
                                color_mode=color_mode_ct,
                                cmap_name=cmap_ct,
                            )
                            st.pyplot(fig_ct); plt.close()
                            st.session_state["ct_result_df"]      = result_df
                            st.session_state["ct_zones_ct"]       = zones_ct
                            st.session_state["ct_classifier_col"] = classifier_col
                            st.session_state["ct_target_col"]     = target_col_ct
                        except Exception as e:
                            st.error(f"繪圖失敗：{e}")
                            import traceback; st.code(traceback.format_exc())

            # ── 統計摘要 & Box + 快速檢定 ────────────────────────
            if st.session_state.get("ct_result_df") is not None:
                result_df   = st.session_state["ct_result_df"]
                saved_zones = st.session_state.get("ct_zones_ct", [])
                saved_cls   = st.session_state.get("ct_classifier_col", classifier_col)
                saved_tgt   = st.session_state.get("ct_target_col", target_col_ct)

                st.markdown("---")

                # ── 各區間統計摘要 ────────────────────────────────
                st.markdown("#### 📋 各區間統計摘要")
                stats_rows = []
                for z in saved_zones:
                    mask = result_df["_class"] == z["label"]
                    n = mask.sum()
                    if n == 0:
                        continue
                    y_vals_z = result_df.loc[mask, saved_tgt].dropna()
                    x_vals_z = result_df.loc[mask, saved_cls].dropna()
                    stats_rows.append({
                        "區間": z["label"],
                        f"{saved_cls[:25]} 範圍": f"{z['min']:.3f} – {z['max']:.3f}",
                        "批次數": int(n),
                        f"{saved_tgt[:25]} Mean": round(y_vals_z.mean(), 3) if len(y_vals_z) else None,
                        f"{saved_tgt[:25]} Median": round(y_vals_z.median(), 3) if len(y_vals_z) else None,
                        "SD": round(y_vals_z.std(), 3) if len(y_vals_z) > 1 else None,
                        "Min": round(y_vals_z.min(), 3) if len(y_vals_z) else None,
                        "Max": round(y_vals_z.max(), 3) if len(y_vals_z) else None,
                    })
                outside_mask = result_df["_class"] == "Outside Zones"
                if outside_mask.sum() > 0:
                    y_out = result_df.loc[outside_mask, saved_tgt].dropna()
                    stats_rows.append({
                        "區間": "Outside Zones",
                        f"{saved_cls[:25]} 範圍": "區間外",
                        "批次數": int(outside_mask.sum()),
                        f"{saved_tgt[:25]} Mean": round(y_out.mean(), 3) if len(y_out) else None,
                        f"{saved_tgt[:25]} Median": round(y_out.median(), 3) if len(y_out) else None,
                        "SD": round(y_out.std(), 3) if len(y_out) > 1 else None,
                        "Min": round(y_out.min(), 3) if len(y_out) else None,
                        "Max": round(y_out.max(), 3) if len(y_out) else None,
                    })
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    mean_col = f"{saved_tgt[:25]} Mean"
                    st.dataframe(
                        stats_df.style.background_gradient(cmap="RdYlGn", subset=[mean_col])
                        if mean_col in stats_df.columns else stats_df,
                        width="stretch", hide_index=True,
                    )

                # ── Box Plot ──────────────────────────────────────
                st.markdown("#### 📦 各區間 Y 值分布（Box Plot）")
                fig_box = _plot_zone_boxplot(result_df, saved_zones, saved_tgt, saved_cls)
                if fig_box:
                    st.pyplot(fig_box); plt.close()

                # ── 快速統計檢定 ──────────────────────────────────
                st.markdown("#### 🧪 快速組間顯著性檢定")
                test_res = _quick_stat_test(result_df, saved_zones, saved_tgt)
                if test_res:
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("檢定方法", test_res["test_name"])
                    tc2.metric("統計量", f"{test_res['stat']:.4f}")
                    tc3.metric("p-value", f"{test_res['p']:.4f}")
                    tc4.metric("結論",
                               "✅ 顯著差異" if test_res["significant"] else "❌ 無顯著差異")
                    if test_res["significant"]:
                        st.success(
                            f"✅ 各區間之間存在統計顯著差異（{test_res['test_name']}，"
                            f"p = {test_res['p']:.4f} < 0.05）"
                        )
                        st.caption(
                            "💡 前往「📐 統計檢定」Tab 可進行完整假設檢查、Post-hoc 成對比較與 Effect Size 分析。"
                        )
                    else:
                        st.info(
                            f"ℹ️ 各區間之間無統計顯著差異（{test_res['test_name']}，"
                            f"p = {test_res['p']:.4f}）"
                        )
                    if not test_res["all_normal"]:
                        st.caption("⚠️ 部分組別不符合常態分佈，已自動使用 Kruskal-Wallis 非參數檢定。")
                else:
                    st.warning("有效分組不足（每組至少需要 2 筆資料）。")

                # ── 批次明細 ──────────────────────────────────────
                if st.checkbox("顯示各批次明細", key="ct_show_detail"):
                    detail_cols = (
                        ["BatchID", saved_cls, saved_tgt, "_class"]
                        if "BatchID" in result_df.columns
                        else [saved_cls, saved_tgt, "_class"]
                    )
                    detail_df = result_df[
                        [c for c in detail_cols if c in result_df.columns]
                    ].copy().rename(columns={"_class": "分類"})
                    st.dataframe(detail_df.reset_index(drop=True),
                                 width="stretch", hide_index=True)
