"""Tab — 跨製程步驟參數監控 + 跨步驟特徵比較"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils import extract_number


def _sort_by_batch(df, batch_col="BatchID"):
    out = df.copy()
    if batch_col in out.columns:
        out["_sort"] = out[batch_col].apply(extract_number)
        out = out.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    return out


def render(raw_df):
    st.header("🔭 跨製程步驟監控")

    if raw_df is None:
        st.info("請先上傳資料。"); return

    # ── 識別所有製程步驟和欄位 ─────────────────────────────────
    # 欄位格式: "ProcessStep:ParameterName (unit)"
    process_cols = [c for c in raw_df.columns if ":" in c]
    if not process_cols:
        st.warning("沒有找到跨製程欄位（欄位需包含 ':' 分隔符）"); return

    all_steps = sorted(set(c.split(":")[0] for c in process_cols))
    batch_col = "BatchID" if "BatchID" in raw_df.columns else None

    subtabs = st.tabs(["📈 跨步驟趨勢監控", "🔗 跨步驟特徵比較"])

    # ══════════════════════════════════════════════════════════
    # Subtab 0: 跨步驟趨勢監控
    # ══════════════════════════════════════════════════════════
    with subtabs[0]:
        st.markdown("選擇多個製程步驟的參數，繪製在同一時序圖上對比。")

        sel_steps = st.multiselect("選擇製程步驟", all_steps, default=all_steps[:2],
                                    key="cp_steps")
        if not sel_steps:
            st.warning("請至少選擇一個步驟"); return

        # 列出所選步驟的所有欄位（保留完整名稱）
        avail_cols = [c for c in process_cols if c.split(":")[0] in sel_steps]
        if not avail_cols:
            st.warning("所選步驟無欄位"); return

        sel_cols = st.multiselect("選擇要監控的欄位（保留完整 ProcessStep:ParameterName 格式）",
                                   avail_cols, default=avail_cols[:3], key="cp_cols")
        if not sel_cols:
            st.warning("請選擇至少一個欄位"); return

        cols_per_row = st.slider("每列圖數", 1, 4, 2, key="cp_cols_per_row")

        if st.button("📊 繪製跨步驟趨勢圖", key="cp_plot_trend"):
            plot_df = raw_df[[batch_col] + sel_cols].copy() if batch_col else raw_df[sel_cols].copy()
            plot_df = _sort_by_batch(plot_df, batch_col) if batch_col else plot_df
            plot_df["_seq"] = range(1, len(plot_df)+1)

            import math
            n_cols_plot = len(sel_cols)
            n_rows = math.ceil(n_cols_plot / cols_per_row)
            fig, axes = plt.subplots(n_rows, cols_per_row,
                                      figsize=(cols_per_row*6, n_rows*4),
                                      squeeze=False)
            axes = axes.flatten()

            STEP_COLORS = plt.cm.tab10.colors
            step_color_map = {s: STEP_COLORS[i % 10] for i, s in enumerate(all_steps)}

            for i, col in enumerate(sel_cols):
                step = col.split(":")[0]
                param = col.split(":")[-1]
                color = step_color_map[step]

                y = plot_df[col].values.astype(float)
                x = plot_df["_seq"].values

                axes[i].plot(x, y, marker="o", color=color, linewidth=1.5, ms=4)
                axes[i].fill_between(x, y, alpha=0.08, color=color)

                # 缺失值陰影
                for j, val in enumerate(y):
                    if np.isnan(val):
                        axes[i].axvspan(j+0.5, j+1.5, alpha=0.15, color="#e84855")

                # X 軸標籤
                if batch_col and batch_col in plot_df.columns:
                    axes[i].set_xticks(x)
                    axes[i].set_xticklabels(
                        [str(b)[-6:] for b in plot_df[batch_col]],
                        rotation=90, fontsize=6)

                import textwrap
                title = f"[{step}]\n{textwrap.fill(param, 35)}"
                axes[i].set_title(title, fontsize=8, color=color)
                axes[i].set_ylabel("Value", fontsize=8)
                axes[i].grid(linestyle="--", alpha=0.4)

            for j in range(n_cols_plot, len(axes)):
                axes[j].axis("off")

            # Legend for steps
            handles = [mpatches.Patch(color=step_color_map[s], label=s) for s in sel_steps]
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(len(sel_steps), 5), fontsize=9,
                       bbox_to_anchor=(0.5, 0))
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            st.pyplot(fig); plt.close()

    # ══════════════════════════════════════════════════════════
    # Subtab 1: 跨步驟特徵比較（散佈圖 + 相關係數）
    # ══════════════════════════════════════════════════════════
    with subtabs[1]:
        st.markdown("選擇兩個不同製程步驟的欄位，分析它們之間的關係。")
        st.caption("⚠️ 欄位名稱保留完整格式（`ProcessStep:ParameterName`），不去除步驟前綴。")

        ca, cb = st.columns(2)
        step_x = ca.selectbox("X 軸所在步驟", all_steps, key="cp_step_x")
        step_y = cb.selectbox("Y 軸所在步驟", all_steps,
                               index=min(1, len(all_steps)-1), key="cp_step_y")

        cols_x = [c for c in process_cols if c.split(":")[0] == step_x]
        cols_y = [c for c in process_cols if c.split(":")[0] == step_y]

        cc, cd = st.columns(2)
        feat_x = cc.selectbox("X 軸特徵（含步驟前綴）", cols_x, key="cp_feat_x")
        feat_y = cd.selectbox("Y 軸特徵（含步驟前綴）", cols_y, key="cp_feat_y")

        alpha_level = st.select_slider("顯著水準 α", [0.001,0.01,0.05,0.10],
                                        value=0.05, key="cp_alpha")
        show_reg = st.checkbox("顯示迴歸線", value=True, key="cp_show_reg")

        if st.button("🔗 繪製跨步驟散佈圖", key="cp_plot_scatter"):
            plot_df = raw_df[[batch_col, feat_x, feat_y]].copy() if batch_col \
                      else raw_df[[feat_x, feat_y]].copy()
            plot_df = _sort_by_batch(plot_df, batch_col) if batch_col else plot_df

            x_vals = plot_df[feat_x].values.astype(float)
            y_vals = plot_df[feat_y].values.astype(float)
            valid  = ~(np.isnan(x_vals) | np.isnan(y_vals))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x_vals[valid], y_vals[valid], color="#2e86ab",
                       edgecolors="white", s=65, linewidths=0.5, zorder=2)
            if batch_col in plot_df.columns:
                for xi, yi, bid in zip(x_vals, y_vals, plot_df[batch_col]):
                    if np.isfinite(xi) and np.isfinite(yi):
                        ax.annotate(str(bid)[-6:], (xi, yi), fontsize=5.5, alpha=0.65,
                                    xytext=(3,3), textcoords="offset points")

            # Pearson r
            if valid.sum() >= 3:
                r, p = sp_stats.pearsonr(x_vals[valid], y_vals[valid])
                p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                sig = " ✓" if p < alpha_level else ""
                stat_text = f"n={valid.sum()}, r={r:.3f}, {p_str} {stars}{sig}"

                if show_reg:
                    m, b, *_ = sp_stats.linregress(x_vals[valid], y_vals[valid])
                    xr = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
                    ax.plot(xr, m*xr+b, color="#e84855", lw=1.5, linestyle="--", alpha=0.8)

                ax.text(0.98, 0.02, stat_text, transform=ax.transAxes,
                        fontsize=9, va="bottom", ha="right",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.92))

                color_box = "#d4edda" if p < alpha_level else "#f8f9fa"
                st.markdown(
                    f"<div style='padding:8px;border-radius:6px;background:{color_box};"
                    f"border:1px solid #ccc;font-size:14px'>📊 {stat_text}</div>",
                    unsafe_allow_html=True)

            ax.set_xlabel(feat_x, fontsize=9)
            ax.set_ylabel(feat_y, fontsize=9)
            ax.set_title(f"跨步驟分析\n{feat_x.split(':')[-1]} vs {feat_y.split(':')[-1]}", fontsize=11)
            ax.grid(linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # 多欄位相關性矩陣
        st.markdown("---")
        st.markdown("#### 🔢 跨步驟多欄位相關矩陣")
        sel_for_corr = st.multiselect(
            "選擇欄位（可跨步驟）", process_cols,
            default=process_cols[:min(6, len(process_cols))],
            key="cp_corr_cols")

        if len(sel_for_corr) >= 2 and st.button("計算相關矩陣", key="cp_corr_btn"):
            corr_df = raw_df[sel_for_corr].apply(pd.to_numeric, errors="coerce").corr()
            # 短標籤: Step_Param
            short_labels = [f"{c.split(':')[0][:4]}…{c.split(':')[-1][:20]}" for c in sel_for_corr]
            corr_df.index = short_labels; corr_df.columns = short_labels

            import seaborn as sns
            n = len(short_labels)
            fig, ax = plt.subplots(figsize=(max(6, n*0.8), max(5, n*0.7)))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
                        vmin=-1, vmax=1, ax=ax, square=True,
                        linewidths=0.3, annot_kws={"size":8})
            ax.set_title("跨步驟相關係數矩陣", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
