"""Tab 1 — 趨勢圖 + 特徵比較（色帶區間）"""
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
from utils import filt_specific_name, smooth_process_data, plot_indexed_lineplots, extract_number


def _plot_feature_comparison(df, feat_x, feat_y, zones, batch_col="BatchID",
                               plot_type="scatter", smooth_method="none", frac=0.3):
    plot_df = df.copy()
    if batch_col in plot_df.columns:
        plot_df["_sort"] = plot_df[batch_col].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    x_vals = plot_df[feat_x].values
    y_vals = plot_df[feat_y].values

    if smooth_method != "none":
        tmp = smooth_process_data(plot_df[[feat_x, feat_y]], [feat_x, feat_y],
                                  id_cols=[], method=smooth_method, frac=frac)
        x_vals_s = tmp[feat_x].values if feat_x in tmp else x_vals
        y_vals_s = tmp[feat_y].values if feat_y in tmp else y_vals
    else:
        x_vals_s, y_vals_s = x_vals, y_vals

    def point_color(val):
        for z in zones:
            if z["min"] <= val <= z["max"]: return z["color"]
        return "#aaaaaa"

    point_colors = [point_color(v) for v in x_vals]

    if plot_type in ("scatter", "scatter+line"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for z in zones:
            ax.axvspan(z["min"], z["max"], alpha=0.10, color=z["color"], zorder=0)
        if plot_type == "scatter+line":
            ax.plot(x_vals_s, y_vals_s, color="#999999", linewidth=0.8, alpha=0.5, zorder=1)
        ax.scatter(x_vals, y_vals, c=point_colors, s=60, edgecolors="white", linewidths=0.5, zorder=2)
        if batch_col in plot_df.columns:
            for xi, yi, bid in zip(x_vals, y_vals, plot_df[batch_col]):
                ax.annotate(str(bid)[-6:], (xi, yi), fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(feat_x, fontsize=10); ax.set_ylabel(feat_y, fontsize=10)
        ax.set_title(f"{feat_y}  vs  {feat_x}", fontsize=12)
        ax.grid(linestyle="--", alpha=0.4)
        handles = [mpatches.Patch(color=z["color"], alpha=0.7, label=z["label"]) for z in zones]
        handles.append(mpatches.Patch(color="#aaaaaa", alpha=0.7, label="Outside zones"))
        ax.legend(handles=handles, loc="best", fontsize=8)
        plt.tight_layout(); return fig

    elif plot_type == "dual_line":
        fig, ax1 = plt.subplots(figsize=(12, 5))
        seq = plot_df["_seq"].values
        ax1.set_xlabel("Batch Sequence"); ax1.set_ylabel(feat_x, color="#2e86ab")
        ax1.plot(seq, x_vals_s, color="#2e86ab", marker="o", ms=4, linewidth=1.5, label=feat_x)
        ax1.tick_params(axis="y", labelcolor="#2e86ab")
        for z in zones:
            ax1.axhspan(z["min"], z["max"], alpha=0.12, color=z["color"], zorder=0)
        ax2 = ax1.twinx()
        ax2.set_ylabel(feat_y, color="#e84855")
        ax2.plot(seq, y_vals_s, color="#e84855", marker="s", ms=4, linewidth=1.5,
                 linestyle="--", label=feat_y)
        ax2.tick_params(axis="y", labelcolor="#e84855")
        l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        zone_handles = [mpatches.Patch(color=z["color"], alpha=0.5, label=z["label"]) for z in zones]
        ax1.legend(l1+l2+zone_handles, lb1+lb2+[z["label"] for z in zones], loc="best", fontsize=8)
        ax1.set_title(f"{feat_x}  &  {feat_y}  over Batch Sequence", fontsize=12)
        ax1.grid(linestyle="--", alpha=0.3); plt.tight_layout(); return fig
    return None


def render(selected_process_df):
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    subtabs = st.tabs(["📈 全局趨勢圖", "🎨 特徵比較 + 區間顏色"])

    with subtabs[0]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        keyword      = col_a.text_input("欄位關鍵字篩選（留空 = 全部）", "")
        smooth_method= col_b.selectbox("平滑方法", ["loess", "ewma", "none"], key="trend_smooth")
        cols_per_row = col_c.slider("每列圖數", 1, 5, 3, key="trend_cols")

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
                if fig: st.pyplot(fig); plt.close()

    with subtabs[1]:
        st.markdown("#### 🎨 特徵比較圖（自訂區間顏色）")
        numeric_cols = selected_process_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("需要至少 2 個數值欄位。"); return

        ca, cb, cc = st.columns(3)
        feat_x    = ca.selectbox("X 軸 / 主特徵", numeric_cols, key="cz_feat_x")
        feat_y    = cb.selectbox("Y 軸 / 比較特徵",
                                  [c for c in numeric_cols if c != feat_x], key="cz_feat_y")
        plot_type = cc.selectbox("圖表類型", ["scatter","scatter+line","dual_line"],
                                  key="cz_plot_type",
                                  format_func=lambda x: {"scatter":"散佈圖","scatter+line":"散佈圖+連線","dual_line":"雙軸折線"}[x])

        cd, ce = st.columns(2)
        smooth_cz = cd.selectbox("平滑（雙軸折線）", ["none","loess","ewma"], key="cz_smooth")
        frac_cz   = ce.slider("LOESS frac", 0.1, 0.8, 0.3, 0.05, key="cz_frac",
                               disabled=(smooth_cz != "loess"))

        x_series = selected_process_df[feat_x].dropna()
        st.caption(f"**{feat_x}** 範圍：{x_series.min():.3f} ～ {x_series.max():.3f}（平均 {x_series.mean():.3f}）")

        st.markdown("#### 🎯 設定數值區間")
        n_zones = st.number_input("區間數量", min_value=1, max_value=8, value=3, step=1, key="cz_n_zones")
        COLORS = ["#2ecc71","#f39c12","#e74c3c","#3498db","#9b59b6","#1abc9c","#e67e22","#95a5a6"]
        zones = []
        for i in range(int(n_zones)):
            with st.expander(f"區間 {i+1}", expanded=(i < 3)):
                zc1, zc2, zc3, zc4 = st.columns([2,1,1,1])
                label = zc1.text_input("名稱", value=f"Zone {i+1}", key=f"cz_label_{i}")
                zmin  = zc2.number_input("最小值", value=float(round(
                    x_series.min() + i*(x_series.max()-x_series.min())/n_zones, 3)),
                    key=f"cz_min_{i}", format="%.3f")
                zmax  = zc3.number_input("最大值", value=float(round(
                    x_series.min() + (i+1)*(x_series.max()-x_series.min())/n_zones, 3)),
                    key=f"cz_max_{i}", format="%.3f")
                color = zc4.color_picker("顏色", value=COLORS[i % len(COLORS)], key=f"cz_color_{i}")
                zones.append({"label": label, "min": zmin, "max": zmax, "color": color})

        if st.button("🎨 繪製比較圖", type="primary", key="plot_comparison"):
            if not all(z["min"] < z["max"] for z in zones):
                st.error("每個區間的最小值必須小於最大值。")
            else:
                with st.spinner("繪圖中..."):
                    try:
                        fig = _plot_feature_comparison(selected_process_df, feat_x, feat_y,
                                                        zones, plot_type=plot_type,
                                                        smooth_method=smooth_cz, frac=frac_cz)
                        if fig: st.pyplot(fig); plt.close()
                    except Exception as e:
                        st.error(f"繪圖失敗：{e}")

        if st.checkbox("顯示各批次所在區間", key="cz_show_table"):
            def assign_zone(val):
                for z in zones:
                    if pd.notna(val) and z["min"] <= val <= z["max"]: return z["label"]
                return "Outside"
            cols_needed = ["BatchID", feat_x, feat_y] if "BatchID" in selected_process_df.columns else [feat_x, feat_y]
            summary = selected_process_df[cols_needed].copy()
            summary["Zone"] = selected_process_df[feat_x].apply(assign_zone)
            st.dataframe(summary.sort_values(feat_x).reset_index(drop=True), width="stretch", hide_index=True)
            zone_stats = []
            for z in zones:
                mask = summary["Zone"] == z["label"]
                n = mask.sum()
                if n > 0:
                    zone_stats.append({"Zone": z["label"],
                        f"{feat_x} 範圍": f"{z['min']:.3f}–{z['max']:.3f}",
                        "批次數": n,
                        f"{feat_y} 平均": summary.loc[mask, feat_y].mean().round(3),
                        f"{feat_y} 標準差": summary.loc[mask, feat_y].std().round(3)})
            if zone_stats:
                st.dataframe(pd.DataFrame(zone_stats), width="stretch", hide_index=True)
