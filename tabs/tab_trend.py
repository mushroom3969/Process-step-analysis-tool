"""Tab — 趨勢圖分析（折線圖 + 分類趨勢）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from scipy import stats as sp_stats

from utils import extract_number


def _sort_by_batch(df, batch_col="BatchID"):
    out = df.copy()
    if batch_col in out.columns:
        out["_sort"] = out[batch_col].apply(extract_number)
        out = out.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    return out


def _add_mean_line(ax, vals, color, show_mean: bool, show_sigma: bool):
    """Helper: draw mean ± sigma on axis if requested."""
    mu = np.nanmean(vals)
    sigma = np.nanstd(vals, ddof=1)
    seq = np.arange(1, len(vals) + 1)
    if show_sigma:
        ax.fill_between(seq, mu - sigma, mu + sigma,
                        alpha=0.10, color=color, zorder=0, label="±1σ")
    if show_mean:
        ax.axhline(mu, color=color, linewidth=1.2, linestyle="--",
                   alpha=0.80, label=f"Mean={mu:.4f}", zorder=1)


def render(selected_process_df):
    st.header("📈 趨勢圖分析")

    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    non_batch = [c for c in numeric_cols if c != "BatchID"]

    if not non_batch:
        st.warning("無數值型欄位可繪圖。")
        return

    subtabs = st.tabs(["📈 時序趨勢圖", "📊 分類趨勢分析", "🔀 多欄位對比"])

    # ══════════════════════════════════════════════════════════
    # Subtab 0: 時序趨勢圖
    # ══════════════════════════════════════════════════════════
    with subtabs[0]:
        st.markdown("選擇欄位繪製 batch 時序折線圖。")

        tr1, tr2, tr3 = st.columns(3)
        sel_cols = tr1.multiselect("選擇欄位", non_batch,
                                    default=non_batch[:min(3, len(non_batch))],
                                    key="trend_cols")
        cols_per_row = tr2.slider("每列圖數", 1, 4, 2, key="trend_cpr")
        show_mean    = tr3.checkbox("顯示平均線", value=True, key="trend_mean")
        show_sigma   = tr3.checkbox("顯示 ±1σ 色帶", value=True, key="trend_sigma")

        if not sel_cols:
            st.warning("請選擇至少一個欄位。")
        elif st.button("📈 繪製趨勢圖", key="run_trend"):
            plot_df = _sort_by_batch(work_df, "BatchID") if "BatchID" in work_df.columns else work_df.copy()
            plot_df["_seq"] = range(1, len(plot_df) + 1)

            n_rows = math.ceil(len(sel_cols) / cols_per_row)
            fig, axes = plt.subplots(n_rows, cols_per_row,
                                      figsize=(cols_per_row * 6, n_rows * 4),
                                      squeeze=False)
            axes = axes.flatten()
            COLORS = plt.cm.tab10.colors

            for i, col in enumerate(sel_cols):
                color = COLORS[i % 10]
                y = plot_df[col].values.astype(float)
                x = plot_df["_seq"].values

                # mean / sigma band
                mu = np.nanmean(y)
                sigma = np.nanstd(y, ddof=1)
                if show_sigma:
                    axes[i].fill_between(x, mu - sigma, mu + sigma,
                                         alpha=0.10, color=color, zorder=0)
                if show_mean:
                    axes[i].axhline(mu, color=color, linewidth=1.2, linestyle="--",
                                    alpha=0.80, label=f"Mean={mu:.3f}", zorder=1)

                axes[i].plot(x, y, color=color, linewidth=1.5, zorder=2)
                axes[i].scatter(x, y, color=color, s=22, zorder=3,
                                edgecolors="white", linewidths=0.4)

                # missing value shading
                for j, val in enumerate(y):
                    if np.isnan(val):
                        axes[i].axvspan(j + 0.5, j + 1.5, alpha=0.15, color="#e84855")

                if "BatchID" in plot_df.columns and len(plot_df) <= 80:
                    axes[i].set_xticks(x)
                    axes[i].set_xticklabels([str(b)[-6:] for b in plot_df["BatchID"]],
                                             rotation=90, fontsize=6)

                axes[i].set_title("\n".join(textwrap.wrap(col[:60], 35)),
                                   fontsize=8.5, color=color)
                axes[i].set_ylabel("Value", fontsize=8)
                axes[i].grid(linestyle="--", alpha=0.35)
                if show_mean:
                    axes[i].legend(fontsize=7, loc="upper right", framealpha=0.7)

            for j in range(len(sel_cols), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ══════════════════════════════════════════════════════════
    # Subtab 1: 分類趨勢分析
    # ══════════════════════════════════════════════════════════
    with subtabs[1]:
        st.markdown("依分類欄位的數值區間對 Y 欄位進行分類，並繪製趨勢圖。")

        ct1, ct2 = st.columns(2)
        classifier_col = ct1.selectbox("🔑 分類依據欄位（X）", non_batch, key="ct_cls")
        target_col_ct  = ct2.selectbox(
            "🎯 目標欄位（Y）",
            [c for c in non_batch if c != classifier_col],
            key="ct_tgt",
        )

        cls_s = work_df[classifier_col].dropna()
        st.caption(
            f"**{classifier_col}** 範圍：{cls_s.min():.3f} ～ {cls_s.max():.3f}"
            f"（平均 {cls_s.mean():.3f}）"
        )

        # Auto 3-split button
        if st.button("⚡ 自動三等分", key="ct_auto3"):
            q33 = round(float(cls_s.quantile(0.33)), 3)
            q67 = round(float(cls_s.quantile(0.67)), 3)
            st.session_state.update({
                "ct_nz": 3,
                "ct_z0_label": f"Low (<{q33})", "ct_z0_min": float(cls_s.min()), "ct_z0_max": q33,
                "ct_z1_label": f"Standard ({q33}–{q67})", "ct_z1_min": q33, "ct_z1_max": q67,
                "ct_z2_label": f"High (>{q67})", "ct_z2_min": q67, "ct_z2_max": float(cls_s.max()),
            })

        n_zones = st.number_input("區間數量", 2, 8,
                                   st.session_state.get("ct_nz", 3), key="ct_nz")
        CT_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e",
                     "#9467bd", "#8c564b", "#e377c2", "#17becf"]

        zones = []
        for i in range(int(n_zones)):
            default_min = round(float(cls_s.min()) + i * (float(cls_s.max()) - float(cls_s.min())) / n_zones, 3)
            default_max = round(float(cls_s.min()) + (i+1) * (float(cls_s.max()) - float(cls_s.min())) / n_zones, 3)
            with st.expander(f"區間 {i+1}", expanded=(i < 3)):
                zc1, zc2, zc3, zc4 = st.columns([3, 1, 1, 1])
                zl = zc1.text_input("名稱", value=st.session_state.get(f"ct_z{i}_label", f"Zone {i+1}"), key=f"ct_zl_{i}")
                zm = zc2.number_input("最小值", value=float(st.session_state.get(f"ct_z{i}_min", default_min)), key=f"ct_zm_{i}", format="%.3f")
                zx = zc3.number_input("最大值", value=float(st.session_state.get(f"ct_z{i}_max", default_max)), key=f"ct_zx_{i}", format="%.3f")
                zc = zc4.color_picker("顏色", value=CT_COLORS[i % len(CT_COLORS)], key=f"ct_zc_{i}")
                zones.append({"label": zl, "min": zm, "max": zx, "color": zc})

        ct_mean    = st.checkbox("顯示各區間平均線", value=True, key="ct_show_mean")
        ct_sigma   = st.checkbox("顯示 ±1σ 色帶", value=True, key="ct_show_sigma")

        if st.button("📊 執行分類趨勢分析", key="run_ct", type="primary"):
            def _assign(val):
                if pd.isna(val):
                    return "No Data"
                for z in zones:
                    if z["min"] <= val <= z["max"]:
                        return z["label"]
                return "Outside Zones"

            result_df = _sort_by_batch(work_df.copy())
            result_df["_class"] = result_df[classifier_col].apply(_assign)
            result_df["_seq"]   = range(1, len(result_df) + 1)

            # Store for stat_test tab
            st.session_state.update({
                "ct_result_df":     result_df,
                "ct_zones_ct":      zones,
                "ct_classifier_col": classifier_col,
                "ct_target_col":    target_col_ct,
            })

            # Plot
            fig, ax = plt.subplots(figsize=(14, 5))
            y_vals = result_df[target_col_ct].values.astype(float)
            x_vals = result_df["_seq"].values

            # Background zone shading on classifier
            cls_vals = result_df[classifier_col].values.astype(float)
            for j, (cls_v, row_class) in enumerate(zip(cls_vals, result_df["_class"])):
                zone_match = next((z for z in zones if z["label"] == row_class), None)
                if zone_match:
                    ax.axvspan(j + 0.5, j + 1.5, alpha=0.12, color=zone_match["color"], zorder=0)

            # Per-zone mean lines
            if ct_mean:
                for z in zones:
                    mask = result_df["_class"] == z["label"]
                    if mask.sum() > 0:
                        mu_z = np.nanmean(y_vals[mask.values])
                        ax.axhline(mu_z, color=z["color"], linewidth=1.2,
                                   linestyle="--", alpha=0.85,
                                   label=f"{z['label']} mean={mu_z:.3f}")

            if ct_sigma:
                for z in zones:
                    mask = result_df["_class"] == z["label"]
                    if mask.sum() > 1:
                        mu_z = np.nanmean(y_vals[mask.values])
                        sg_z = np.nanstd(y_vals[mask.values], ddof=1)
                        ax.fill_between(
                            x_vals[mask.values],
                            mu_z - sg_z, mu_z + sg_z,
                            alpha=0.08, color=z["color"]
                        )

            # Main line with per-point coloring
            for j in range(len(x_vals) - 1):
                row_class = result_df["_class"].iloc[j]
                zone_match = next((z for z in zones if z["label"] == row_class), None)
                c = zone_match["color"] if zone_match else "#aaaaaa"
                ax.plot(x_vals[j:j+2], y_vals[j:j+2], color=c, linewidth=1.5, zorder=2)

            scatter_colors = [
                next((z["color"] for z in zones if z["label"] == rc), "#aaaaaa")
                for rc in result_df["_class"]
            ]
            ax.scatter(x_vals, y_vals, c=scatter_colors, s=40, zorder=3,
                       edgecolors="white", linewidths=0.4)

            # Batch labels
            if "BatchID" in result_df.columns and len(result_df) <= 80:
                ax.set_xticks(x_vals)
                ax.set_xticklabels([str(b)[-6:] for b in result_df["BatchID"]],
                                    rotation=90, fontsize=6)

            ax.set_ylabel(target_col_ct[:60], fontsize=10)
            ax.set_title(f"分類趨勢：{target_col_ct} 依 {classifier_col} 分類", fontsize=12)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8,
                      ncol=min(len(zones) + 1, 4))
            ax.grid(linestyle="--", alpha=0.35)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Summary table
            summary_rows = []
            for z in zones:
                mask = result_df["_class"] == z["label"]
                arr = y_vals[mask.values]
                arr = arr[~np.isnan(arr)]
                summary_rows.append({
                    "區間": z["label"],
                    "n": len(arr),
                    "Mean": round(np.mean(arr), 4) if len(arr) else None,
                    "SD":   round(np.std(arr, ddof=1), 4) if len(arr) > 1 else None,
                    "Min":  round(np.min(arr), 4) if len(arr) else None,
                    "Max":  round(np.max(arr), 4) if len(arr) else None,
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.success("✅ 分類完成！可至「📐 統計檢定分析」頁面進行組間檢定。")

    # ══════════════════════════════════════════════════════════
    # Subtab 2: 多欄位對比
    # ══════════════════════════════════════════════════════════
    with subtabs[2]:
        st.markdown("選擇多個欄位，在同一張圖上標準化後對比趨勢。")

        mv_cols = st.multiselect("選擇欄位（2–8 個）", non_batch,
                                  default=non_batch[:min(3, len(non_batch))],
                                  key="mv_cols")
        mv_mean  = st.checkbox("顯示平均線", value=True, key="mv_mean")
        mv_norm  = st.checkbox("Z-score 標準化（方便比較量綱不同的欄位）",
                                value=True, key="mv_norm")

        if len(mv_cols) >= 2 and st.button("🔀 繪製對比圖", key="run_mv"):
            plot_df = _sort_by_batch(work_df.copy())
            plot_df["_seq"] = range(1, len(plot_df) + 1)

            fig, ax = plt.subplots(figsize=(14, 5))
            COLORS = plt.cm.tab10.colors

            for i, col in enumerate(mv_cols):
                y = plot_df[col].values.astype(float)
                x = plot_df["_seq"].values
                color = COLORS[i % 10]

                if mv_norm:
                    mu, sd = np.nanmean(y), np.nanstd(y, ddof=1)
                    y_plot = (y - mu) / sd if sd > 0 else y - mu
                else:
                    y_plot = y

                ax.plot(x, y_plot, color=color, linewidth=1.5, label=col[:35], zorder=2)
                ax.scatter(x, y_plot, color=color, s=18, zorder=3,
                           edgecolors="white", linewidths=0.3, alpha=0.85)

                if mv_mean:
                    mu_p = np.nanmean(y_plot)
                    ax.axhline(mu_p, color=color, linewidth=0.9, linestyle=":",
                               alpha=0.70)

            if "BatchID" in plot_df.columns and len(plot_df) <= 80:
                ax.set_xticks(plot_df["_seq"])
                ax.set_xticklabels([str(b)[-6:] for b in plot_df["BatchID"]],
                                    rotation=90, fontsize=6)

            ax.set_ylabel("Z-score" if mv_norm else "Value", fontsize=10)
            ax.set_title("多欄位趨勢對比" + ("（Z-score 標準化）" if mv_norm else ""), fontsize=12)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8,
                      ncol=min(len(mv_cols), 4))
            ax.grid(linestyle="--", alpha=0.35)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="-", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
