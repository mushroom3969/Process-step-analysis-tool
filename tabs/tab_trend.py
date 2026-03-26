"""Tab 1 — 趨勢圖 + 特徵比較（色帶區間 + 漸層 + 統計）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils import filt_specific_name, smooth_process_data, plot_indexed_lineplots, extract_number


def _sort_df(df, batch_col="BatchID"):
    out = df.copy()
    if batch_col in out.columns:
        out["_sort"] = out[batch_col].apply(extract_number)
        out = out.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    out["_seq"] = range(1, len(out) + 1)
    return out


def _apply_smooth(df, feat_x, feat_y, method, frac):
    if method == "none":
        return df[feat_x].values, df[feat_y].values
    tmp = smooth_process_data(df[[feat_x, feat_y]], [feat_x, feat_y],
                              id_cols=[], method=method, frac=frac)
    return (tmp[feat_x].values if feat_x in tmp else df[feat_x].values,
            tmp[feat_y].values if feat_y in tmp else df[feat_y].values)


def _zone_color(val, zones):
    for z in zones:
        if z["min"] <= val <= z["max"]:
            return z["color"]
    return "#aaaaaa"


def _pearson_stat(xs, ys):
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 3:
        return None
    r, p = sp_stats.pearsonr(xs, ys)
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return {"n": len(xs), "r": r, "p": p, "p_str": p_str, "stars": stars}


def _add_regression_line(ax, xs, ys, color, alpha=0.7):
    valid = ~(np.isnan(xs) | np.isnan(ys))
    if valid.sum() < 3:
        return
    m, b, *_ = sp_stats.linregress(xs[valid], ys[valid])
    xr = np.linspace(xs[valid].min(), xs[valid].max(), 100)
    ax.plot(xr, m * xr + b, color=color, lw=1.2, linestyle="--", alpha=alpha, zorder=3)


def plot_feature_comparison(df, feat_x, feat_y, zones, batch_col="BatchID",
                             plot_type="scatter", smooth_method="none", frac=0.3,
                             color_mode="zone", gradient_cmap="viridis",
                             show_regression=True, alpha_level=0.05):
    plot_df = _sort_df(df, batch_col)
    x_vals = plot_df[feat_x].values.astype(float)
    y_vals = plot_df[feat_y].values.astype(float)
    x_vals_s, y_vals_s = _apply_smooth(plot_df, feat_x, feat_y, smooth_method, frac)

    if color_mode == "gradient":
        x_fin = x_vals[np.isfinite(x_vals)]
        vmin, vmax = (x_fin.min(), x_fin.max()) if len(x_fin) else (0, 1)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mcm.get_cmap(gradient_cmap)
        point_colors = [cmap(norm(v)) if np.isfinite(v) else (0.7, 0.7, 0.7, 1) for v in x_vals]
    else:
        point_colors = [_zone_color(v, zones) for v in x_vals]

    if plot_type in ("scatter", "scatter+line"):
        fig, ax = plt.subplots(figsize=(11, 6))
        for z in zones:
            ax.axvspan(z["min"], z["max"], alpha=0.08, color=z["color"], zorder=0)
        if plot_type == "scatter+line":
            ax.plot(x_vals_s, y_vals_s, color="#cccccc", lw=0.8, alpha=0.6, zorder=1)
        ax.scatter(x_vals, y_vals, c=point_colors, s=65,
                   edgecolors="white", linewidths=0.5, zorder=2)
        if color_mode == "gradient":
            sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=feat_x, fraction=0.03, pad=0.01)
        if batch_col in plot_df.columns:
            for xi, yi, bid in zip(x_vals, y_vals, plot_df[batch_col]):
                if np.isfinite(xi) and np.isfinite(yi):
                    ax.annotate(str(bid)[-6:], (xi, yi), fontsize=5.5, alpha=0.65,
                                xytext=(3, 3), textcoords="offset points")

        stat_rows = []
        res_all = _pearson_stat(x_vals, y_vals)
        if res_all:
            res_all["label"] = "All data"; res_all["color"] = "black"
            stat_rows.append(res_all)
            if show_regression:
                _add_regression_line(ax, x_vals, y_vals, "black", 0.5)
        for z in zones:
            mask = np.array([z["min"] <= v <= z["max"] for v in x_vals])
            if mask.sum() < 3:
                continue
            res = _pearson_stat(x_vals[mask], y_vals[mask])
            if res:
                res["label"] = z["label"]; res["color"] = z["color"]
                stat_rows.append(res)
                if show_regression:
                    _add_regression_line(ax, x_vals[mask], y_vals[mask], z["color"], 0.8)

        if stat_rows:
            lines = []
            for r in stat_rows:
                sig = " ✓" if r["p"] < alpha_level else ""
                lines.append(f"{r['label']} (n={r['n']}): r={r['r']:.3f}, {r['p_str']} {r['stars']}{sig}")
            ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes,
                    fontsize=7.5, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#cccccc", alpha=0.92))

        handles = [mpatches.Patch(color=z["color"], alpha=0.7, label=z["label"]) for z in zones]
        handles.append(mpatches.Patch(color="#aaaaaa", alpha=0.7, label="Outside zones"))
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        ax.set_xlabel(feat_x, fontsize=10); ax.set_ylabel(feat_y, fontsize=10)
        ax.set_title(f"{feat_y}  vs  {feat_x}", fontsize=12)
        ax.grid(linestyle="--", alpha=0.35)
        plt.tight_layout()
        return fig, stat_rows

    elif plot_type == "dual_line":
        fig, ax1 = plt.subplots(figsize=(12, 5))
        seq = plot_df["_seq"].values
        ax1.set_xlabel("Batch Sequence"); ax1.set_ylabel(feat_x, color="#2e86ab")
        ax1.plot(seq, x_vals_s, color="#2e86ab", marker="o", ms=4, lw=1.5, label=feat_x)
        ax1.tick_params(axis="y", labelcolor="#2e86ab")
        for z in zones:
            ax1.axhspan(z["min"], z["max"], alpha=0.12, color=z["color"], zorder=0)
        ax2 = ax1.twinx()
        ax2.set_ylabel(feat_y, color="#e84855")
        ax2.plot(seq, y_vals_s, color="#e84855", marker="s", ms=4, lw=1.5, linestyle="--", label=feat_y)
        ax2.tick_params(axis="y", labelcolor="#e84855")
        res_all = _pearson_stat(x_vals, y_vals)
        if res_all:
            sig = " ✓" if res_all["p"] < alpha_level else ""
            ax1.set_title(f"{feat_x}  &  {feat_y}\nr={res_all['r']:.3f}, {res_all['p_str']} {res_all['stars']}{sig}", fontsize=11)
        else:
            ax1.set_title(f"{feat_x}  &  {feat_y}", fontsize=12)
        l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        zone_h = [mpatches.Patch(color=z["color"], alpha=0.5, label=z["label"]) for z in zones]
        ax1.legend(l1+l2+zone_h, lb1+lb2+[z["label"] for z in zones], loc="best", fontsize=8)
        ax1.grid(linestyle="--", alpha=0.3); plt.tight_layout()
        return fig, [res_all] if res_all else []
    return None, []


def render(selected_process_df):
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    # FIX 1: 使用 clean_df（特徵工程後）或 selected_process_df
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if _cd is not None:
        st.info("📌 目前使用**特徵工程後**的資料（含新增欄位）。如需原始資料請清除 clean_df。")

    subtabs = st.tabs(["📈 全局趨勢圖", "🎨 特徵比較 + 區間分析"])

    with subtabs[0]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        keyword       = col_a.text_input("欄位關鍵字篩選（留空 = 全部）", "")
        smooth_method = col_b.selectbox("平滑方法", ["loess", "ewma", "none"], key="trend_smooth")
        cols_per_row  = col_c.slider("每列圖數", 1, 5, 3, key="trend_cols")

        display_df = filt_specific_name(work_df, keyword) if keyword else work_df.copy()
        if "BatchID" not in display_df.columns and "BatchID" in work_df.columns:
            display_df.insert(0, "BatchID", work_df["BatchID"])

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

    with subtabs[1]:
        st.markdown("#### 🎨 特徵比較圖（自訂區間 + 統計）")

        numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("需要至少 2 個數值欄位。"); return

        c1, c2, c3 = st.columns(3)
        feat_x    = c1.selectbox("X 軸 / 主特徵", numeric_cols, key="cz_feat_x")
        feat_y    = c2.selectbox("Y 軸 / 比較特徵",
                                  [c for c in numeric_cols if c != feat_x], key="cz_feat_y")
        plot_type = c3.selectbox("圖表類型", ["scatter","scatter+line","dual_line"],
                                  key="cz_plot_type",
                                  format_func=lambda x: {"scatter":"散佈圖","scatter+line":"散佈+連線","dual_line":"雙軸折線"}[x])

        c4, c5, c6 = st.columns(3)
        color_mode = c4.radio("著色模式", ["zone（區間）","gradient（漸層）"],
                               horizontal=True, key="cz_color_mode")
        use_gradient = color_mode.startswith("gradient")
        gradient_cmap = c5.selectbox("漸層色板",
                                      ["viridis","plasma","coolwarm","RdYlGn","Blues","YlOrRd"],
                                      key="cz_cmap", disabled=not use_gradient)
        smooth_cz = c6.selectbox("平滑", ["none","loess","ewma"], key="cz_smooth")

        c7, c8 = st.columns(2)
        frac_cz         = c7.slider("LOESS frac", 0.1, 0.8, 0.3, 0.05, key="cz_frac",
                                     disabled=(smooth_cz != "loess"))
        show_regression = c8.checkbox("顯示迴歸線", value=True, key="cz_show_reg")

        with st.expander("📐 統計設定", expanded=False):
            alpha_level = st.select_slider("顯著水準 α",
                                            options=[0.001, 0.01, 0.05, 0.10],
                                            value=0.05, key="cz_alpha_level")

        x_series = work_df[feat_x].dropna()
        st.caption(f"**{feat_x}** 範圍：{x_series.min():.3f} ～ {x_series.max():.3f}（平均 {x_series.mean():.3f}）")

        st.markdown("#### 🎯 設定數值區間")
        n_zones = st.number_input("區間數量", min_value=1, max_value=8, value=3, step=1, key="cz_n_zones")
        PRESET = ["#2ecc71","#f39c12","#e74c3c","#3498db","#9b59b6","#1abc9c","#e67e22","#95a5a6"]
        zones = []
        for i in range(int(n_zones)):
            with st.expander(f"區間 {i+1}", expanded=(i < 3)):
                zc1, zc2, zc3, zc4 = st.columns([2,1,1,1])
                label = zc1.text_input("名稱", value=f"Zone {i+1}", key=f"cz_label_{i}")
                zmin  = zc2.number_input("最小值", key=f"cz_min_{i}", format="%.3f",
                    value=float(round(x_series.min() + i*(x_series.max()-x_series.min())/n_zones, 3)))
                zmax  = zc3.number_input("最大值", key=f"cz_max_{i}", format="%.3f",
                    value=float(round(x_series.min() + (i+1)*(x_series.max()-x_series.min())/n_zones, 3)))
                color = zc4.color_picker("顏色", value=PRESET[i%len(PRESET)], key=f"cz_color_{i}")
                zones.append({"label": label, "min": zmin, "max": zmax, "color": color})

        if st.button("🎨 繪製比較圖", type="primary", key="plot_comparison"):
            if not all(z["min"] < z["max"] for z in zones):
                st.error("每個區間最小值必須 < 最大值")
            else:
                with st.spinner("繪圖中..."):
                    try:
                        fig, stat_rows = plot_feature_comparison(
                            work_df, feat_x=feat_x, feat_y=feat_y, zones=zones,
                            plot_type=plot_type, smooth_method=smooth_cz, frac=frac_cz,
                            color_mode="gradient" if use_gradient else "zone",
                            gradient_cmap=gradient_cmap,
                            show_regression=show_regression, alpha_level=alpha_level)
                        if fig:
                            st.pyplot(fig); plt.close()
                        if stat_rows:
                            st.markdown("#### 📊 各區間 Pearson 統計")
                            stat_df = pd.DataFrame([{
                                "區間": r["label"], "n": r["n"],
                                "r": round(r["r"],4), "p-value": round(r["p"],4),
                                "significance": r["stars"],
                                "達顯著": "✓" if r["p"] < alpha_level else ""
                            } for r in stat_rows if r])
                            st.dataframe(stat_df.style.apply(
                                lambda row: ["background-color:#d4edda"]*len(row)
                                if row["達顯著"]=="✓"
                                else ["background-color:#f8f9fa"]*len(row), axis=1),
                                width="stretch", hide_index=True)
                    except Exception as e:
                        import traceback
                        st.error(f"繪圖失敗：{e}"); st.code(traceback.format_exc())

        if st.checkbox("顯示各批次所在區間", key="cz_show_table"):
            def assign_zone(val):
                for z in zones:
                    if pd.notna(val) and z["min"] <= val <= z["max"]: return z["label"]
                return "Outside"
            cols_needed = (["BatchID",feat_x,feat_y] if "BatchID" in work_df.columns else [feat_x,feat_y])
            summary = work_df[cols_needed].copy()
            summary["Zone"] = work_df[feat_x].apply(assign_zone)
            st.dataframe(summary.sort_values(feat_x).reset_index(drop=True), width="stretch", hide_index=True)
