"""Tab — 趨勢圖分析（折線圖 + 分類趨勢 + XYZ 散佈圖）"""
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
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
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

    subtabs = st.tabs(["📈 時序趨勢圖", "📊 分類趨勢分析", "🔀 多欄位對比", "🎨 XYZ 散佈圖"])

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

        ct_mean  = st.checkbox("顯示各區間平均線", value=True, key="ct_show_mean")
        ct_sigma = st.checkbox("顯示 ±1σ 色帶", value=True, key="ct_show_sigma")

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

            st.session_state.update({
                "ct_result_df":      result_df,
                "ct_zones_ct":       zones,
                "ct_classifier_col": classifier_col,
                "ct_target_col":     target_col_ct,
            })

            fig, ax = plt.subplots(figsize=(14, 5))
            y_vals = result_df[target_col_ct].values.astype(float)
            x_vals = result_df["_seq"].values

            cls_vals = result_df[classifier_col].values.astype(float)
            for j, (cls_v, row_class) in enumerate(zip(cls_vals, result_df["_class"])):
                zone_match = next((z for z in zones if z["label"] == row_class), None)
                if zone_match:
                    ax.axvspan(j + 0.5, j + 1.5, alpha=0.12, color=zone_match["color"], zorder=0)

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
                        ax.fill_between(x_vals[mask.values], mu_z - sg_z, mu_z + sg_z,
                                        alpha=0.08, color=z["color"])

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

            summary_rows = []
            for z in zones:
                mask = result_df["_class"] == z["label"]
                arr = y_vals[mask.values]
                arr = arr[~np.isnan(arr)]
                summary_rows.append({
                    "區間": z["label"], "n": len(arr),
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
        mv_mean = st.checkbox("顯示平均線", value=True, key="mv_mean")
        mv_norm = st.checkbox("Z-score 標準化（方便比較量綱不同的欄位）",
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

    # ══════════════════════════════════════════════════════════
    # Subtab 3: XYZ 散佈圖（新功能）
    # ══════════════════════════════════════════════════════════
    with subtabs[3]:
        _render_xyz_scatter(work_df, non_batch)


# ──────────────────────────────────────────────────────────────────────────────
# XYZ 散佈圖渲染函式
# ──────────────────────────────────────────────────────────────────────────────

# 支援的漸層色圖
_GRADIENT_CMAPS = {
    "🔵→🔴 Coolwarm":   "coolwarm",
    "🟣→🟡 Plasma":     "plasma",
    "🟢→🟡 Viridis":    "viridis",
    "🟠 Magma":         "magma",
    "🌊 Turbo":         "turbo",
    "🌈 RdYlGn":        "RdYlGn",
    "🔵→🟠 Spectral":   "Spectral_r",
}

_ZONE_PALETTES = {
    "tab10（預設）":     "tab10",
    "Set1（鮮豔）":      "Set1",
    "Set2（柔和）":      "Set2",
    "Dark2（深色）":     "Dark2",
    "Paired（成對）":    "Paired",
}


def _render_xyz_scatter(work_df: pd.DataFrame, non_batch: list):
    """
    XYZ 散佈圖：
    - X 軸：數值欄位 或 BatchID 排序序號
    - Y 軸：數值欄位
    - 顏色（Z）：數值欄位，支援「分區著色」與「漸層著色」兩種模式
    - 支援顯示連線（依 Batch 排序）
    """
    st.markdown(
        "自訂 **X / Y / 顏色（Z）** 三個維度的散佈圖，支援 **分區著色** 與 **漸層著色** 兩種模式。"
    )

    has_batch = "BatchID" in work_df.columns

    # ── 軸欄位選擇 ────────────────────────────────────────────
    st.markdown("#### ① 選擇軸欄位")
    ax_c1, ax_c2, ax_c3 = st.columns(3)

    # X 軸：可選 BatchID 序號 或 數值欄位
    x_options = (["[BatchID 排序序號]"] if has_batch else []) + non_batch
    x_sel = ax_c1.selectbox("X 軸", x_options, key="xyz_x")

    y_sel = ax_c2.selectbox("Y 軸", non_batch,
                             index=min(1, len(non_batch) - 1), key="xyz_y")

    z_options = ["（不使用顏色維度）"] + non_batch
    z_sel = ax_c3.selectbox("顏色（Z）", z_options, key="xyz_z")

    use_z = z_sel != "（不使用顏色維度）"

    # ── 顏色模式設定 ──────────────────────────────────────────
    color_mode = None
    zones_xyz  = []
    cmap_name  = "coolwarm"
    palette_name = "tab10"

    if use_z:
        st.markdown("#### ② 顏色模式")
        color_mode = st.radio(
            "著色方式",
            ["🎨 分區著色（手動設定區間）", "🌈 漸層著色（連續色圖）"],
            horizontal=True,
            key="xyz_color_mode",
        )

        if color_mode == "🌈 漸層著色（連續色圖）":
            cm_c1, cm_c2 = st.columns(2)
            cmap_label = cm_c1.selectbox(
                "色圖", list(_GRADIENT_CMAPS.keys()), key="xyz_cmap"
            )
            cmap_name = _GRADIENT_CMAPS[cmap_label]
            reverse_cmap = cm_c2.checkbox("反轉色圖", value=False, key="xyz_cmap_rev")
            if reverse_cmap and not cmap_name.endswith("_r"):
                cmap_name = cmap_name + "_r"

        else:  # 分區著色
            z_series = work_df[z_sel].dropna()
            z_min, z_max = float(z_series.min()), float(z_series.max())

            st.caption(
                f"**{z_sel}** 範圍：{z_min:.3f} ～ {z_max:.3f}（平均 {z_series.mean():.3f}）"
            )

            zo_c1, zo_c2 = st.columns(2)
            n_zones_xyz = zo_c1.number_input(
                "分區數量", 2, 8,
                st.session_state.get("xyz_nz", 3), key="xyz_nz"
            )
            pal_label = zo_c2.selectbox(
                "配色方案", list(_ZONE_PALETTES.keys()), key="xyz_palette"
            )
            palette_name = _ZONE_PALETTES[pal_label]

            if st.button("⚡ 自動等分", key="xyz_auto_split"):
                step = (z_max - z_min) / int(n_zones_xyz)
                for i in range(int(n_zones_xyz)):
                    lo = round(z_min + i * step, 3)
                    hi = round(z_min + (i + 1) * step, 3)
                    st.session_state[f"xyz_z{i}_label"] = f"Z{i+1} ({lo}~{hi})"
                    st.session_state[f"xyz_z{i}_min"]   = lo
                    st.session_state[f"xyz_z{i}_max"]   = hi

            # 取得調色盤顏色
            try:
                cmap_obj = mcm.get_cmap(palette_name)
                palette_colors = [
                    mcolors.to_hex(cmap_obj(i / max(int(n_zones_xyz) - 1, 1)))
                    for i in range(int(n_zones_xyz))
                ]
            except Exception:
                palette_colors = [
                    "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e",
                    "#9467bd", "#8c564b", "#e377c2", "#17becf"
                ]

            step_default = (z_max - z_min) / int(n_zones_xyz)
            for i in range(int(n_zones_xyz)):
                lo_def = round(z_min + i * step_default, 3)
                hi_def = round(z_min + (i + 1) * step_default, 3)
                with st.expander(f"分區 {i+1}", expanded=(i < 3)):
                    zc1, zc2, zc3, zc4 = st.columns([3, 1, 1, 1])
                    zl = zc1.text_input(
                        "名稱",
                        value=st.session_state.get(f"xyz_z{i}_label", f"Z{i+1}"),
                        key=f"xyz_zl_{i}",
                    )
                    zm = zc2.number_input(
                        "最小值",
                        value=float(st.session_state.get(f"xyz_z{i}_min", lo_def)),
                        key=f"xyz_zm_{i}", format="%.3f",
                    )
                    zx = zc3.number_input(
                        "最大值",
                        value=float(st.session_state.get(f"xyz_z{i}_max", hi_def)),
                        key=f"xyz_zx_{i}", format="%.3f",
                    )
                    default_color = palette_colors[i] if i < len(palette_colors) else "#2e86ab"
                    zc = zc4.color_picker(
                        "顏色",
                        value=st.session_state.get(f"xyz_zc_{i}", default_color),
                        key=f"xyz_zc_{i}",
                    )
                    zones_xyz.append({"label": zl, "min": zm, "max": zx, "color": zc})

    # ── 圖表選項 ──────────────────────────────────────────────
    st.markdown("#### ③ 圖表選項")
    opt_c1, opt_c2, opt_c3, opt_c4 = st.columns(4)
    show_line      = opt_c1.checkbox("連線（依 Batch 排序）", value=False, key="xyz_line")
    show_batch_lbl = opt_c2.checkbox("顯示 Batch 標籤", value=True, key="xyz_batch_lbl")
    show_colorbar  = opt_c3.checkbox("顯示色條（漸層模式）", value=True, key="xyz_colorbar")
    dot_size       = opt_c4.slider("點大小", 20, 200, 60, 10, key="xyz_dotsize")

    stat_c1, stat_c2 = st.columns(2)
    show_mean_xyz  = stat_c1.checkbox("顯示 Mean 線（Y 軸）", value=True, key="xyz_show_mean")
    show_sigma_xyz = stat_c2.checkbox("顯示 ±1σ 色帶（Y 軸）", value=True, key="xyz_show_sigma")

    # ── 繪圖按鈕 ──────────────────────────────────────────────
    if st.button("🎨 繪製 XYZ 散佈圖", key="run_xyz", type="primary"):

        # ── 準備資料 ──────────────────────────────────────────
        plot_df = _sort_by_batch(work_df.copy()) if has_batch else work_df.copy()
        plot_df["_seq"] = range(1, len(plot_df) + 1)

        # X 值
        if x_sel == "[BatchID 排序序號]":
            x_vals = plot_df["_seq"].values.astype(float)
            x_label = "Batch 序號"
            x_is_batch = True
        else:
            x_vals = plot_df[x_sel].values.astype(float)
            x_label = x_sel
            x_is_batch = False

        y_vals = plot_df[y_sel].values.astype(float)

        # 有效行
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        if use_z:
            z_vals_raw = plot_df[z_sel].values.astype(float)
            valid_mask &= ~np.isnan(z_vals_raw)
        else:
            z_vals_raw = None

        if valid_mask.sum() < 2:
            st.error("有效資料點不足 2 筆，無法繪圖。")
            return

        x_plot  = x_vals[valid_mask]
        y_plot  = y_vals[valid_mask]
        z_plot  = z_vals_raw[valid_mask] if use_z else None
        seq_plot = plot_df["_seq"].values[valid_mask]
        batch_labels = (
            plot_df["BatchID"].values[valid_mask]
            if has_batch else seq_plot.astype(str)
        )

        # ── 繪圖 ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 7))

        # 連線（依排序序號）
        if show_line:
            order = np.argsort(seq_plot)
            ax.plot(
                x_plot[order], y_plot[order],
                color="#bbbbbb", linewidth=0.9, zorder=1, alpha=0.55,
            )

        # 散點
        if not use_z:
            sc = ax.scatter(
                x_plot, y_plot,
                c="#2e86ab", s=dot_size, zorder=3,
                edgecolors="white", linewidths=0.5, alpha=0.88,
            )

        elif color_mode == "🌈 漸層著色（連續色圖）":
            sc = ax.scatter(
                x_plot, y_plot,
                c=z_plot, cmap=cmap_name,
                s=dot_size, zorder=3,
                edgecolors="white", linewidths=0.5, alpha=0.88,
            )
            if show_colorbar:
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label(z_sel[:50], fontsize=9)

        else:  # 分區著色
            def _zone_color(val):
                for z in zones_xyz:
                    if z["min"] <= val <= z["max"]:
                        return z["color"]
                return "#aaaaaa"

            point_colors = [_zone_color(v) for v in z_plot]
            sc = ax.scatter(
                x_plot, y_plot,
                c=point_colors, s=dot_size, zorder=3,
                edgecolors="white", linewidths=0.5, alpha=0.88,
            )

            # 圖例 patches
            legend_handles = [
                mpatches.Patch(color=z["color"], label=z["label"])
                for z in zones_xyz
            ]
            legend_handles.append(
                mpatches.Patch(color="#aaaaaa", label="Outside Zones")
            )
            # legend drawn after mean lines below

        # ── Mean 線 & ±1σ 色帶（Y 軸）────────────────────────────
        if use_z and color_mode == "🎨 分區著色（手動設定區間）" and zones_xyz:
            # 每個分區各自的 mean / sigma
            for z in zones_xyz:
                mask_z = np.array([z["min"] <= v <= z["max"] for v in z_plot])
                y_z = y_plot[mask_z]
                y_z = y_z[~np.isnan(y_z)]
                if len(y_z) < 1:
                    continue
                mu_z  = float(np.mean(y_z))
                sig_z = float(np.std(y_z, ddof=1)) if len(y_z) > 1 else 0.0
                if show_sigma_xyz and sig_z > 0:
                    ax.axhspan(mu_z - sig_z, mu_z + sig_z,
                               alpha=0.07, color=z["color"], zorder=0)
                if show_mean_xyz:
                    ax.axhline(mu_z, color=z["color"], linewidth=1.3,
                               linestyle="--", alpha=0.85, zorder=2,
                               label=f"{z['label']} μ={mu_z:.3f}")
            # Zone legend: patches + mean line labels combined
            all_handles = legend_handles + ([
                mpatches.Patch(color="#aaaaaa", label="Outside Zones")
            ] if any(
                not any(z["min"] <= v <= z["max"] for z in zones_xyz)
                for v in z_plot if not np.isnan(v)
            ) else [])
            if show_mean_xyz:
                ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85,
                          title=f"Z: {z_sel[:25]}", title_fontsize=8,
                          ncol=max(1, len(zones_xyz) // 5 + 1))
            else:
                ax.legend(handles=all_handles, title=f"Z: {z_sel[:25]}",
                          fontsize=7.5, title_fontsize=8,
                          loc="upper left", framealpha=0.85,
                          ncol=max(1, len(zones_xyz) // 5 + 1))
        else:
            # 全體 Y 的 mean / sigma
            y_clean = y_plot[~np.isnan(y_plot)]
            if len(y_clean) >= 1:
                mu_all  = float(np.mean(y_clean))
                sig_all = float(np.std(y_clean, ddof=1)) if len(y_clean) > 1 else 0.0
                if show_sigma_xyz and sig_all > 0:
                    ax.axhspan(mu_all - sig_all, mu_all + sig_all,
                               alpha=0.08, color="#2e86ab", zorder=0,
                               label=f"±1σ ({mu_all - sig_all:.3f} ~ {mu_all + sig_all:.3f})")
                if show_mean_xyz:
                    ax.axhline(mu_all, color="#2e86ab", linewidth=1.3,
                               linestyle="--", alpha=0.85, zorder=2,
                               label=f"μ={mu_all:.4f}")
            if show_mean_xyz or show_sigma_xyz:
                ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

        # Batch 標籤
        if show_batch_lbl and has_batch:
            for xi, yi, bid in zip(x_plot, y_plot, batch_labels):
                ax.annotate(
                    str(bid)[-6:],
                    (xi, yi),
                    fontsize=5.5, alpha=0.65,
                    xytext=(3, 3), textcoords="offset points",
                )

        # X 軸標籤：若 X 是 Batch 序號，用 BatchID 替換刻度
        if x_is_batch and has_batch and len(plot_df) <= 80:
            xtick_pos  = seq_plot
            xtick_labs = [str(b)[-6:] for b in batch_labels]
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labs, rotation=90, fontsize=6)

        # 標題 & 軸標籤
        z_title = f"  |  顏色：{z_sel[:30]}" if use_z else ""
        ax.set_title(
            f"XYZ 散佈圖\nX：{x_label[:35]}  |  Y：{y_sel[:35]}{z_title}",
            fontsize=11,
        )
        ax.set_xlabel(x_label[:60], fontsize=10)
        ax.set_ylabel(y_sel[:60], fontsize=10)
        ax.grid(linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── 統計摘要 ──────────────────────────────────────────
        st.markdown("#### 📋 統計摘要")

        if use_z and color_mode == "🎨 分區著色（手動設定區間）" and zones_xyz:
            rows = []
            for z in zones_xyz:
                mask_z = np.array([z["min"] <= v <= z["max"] for v in z_plot])
                y_z = y_plot[mask_z]
                y_z = y_z[~np.isnan(y_z)]
                rows.append({
                    "分區": z["label"],
                    "n": len(y_z),
                    f"Y Mean ({y_sel[:20]})": round(float(np.mean(y_z)), 4) if len(y_z) else None,
                    "Y SD":  round(float(np.std(y_z, ddof=1)), 4) if len(y_z) > 1 else None,
                    "Y Min": round(float(np.min(y_z)), 4) if len(y_z) else None,
                    "Y Max": round(float(np.max(y_z)), 4) if len(y_z) else None,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            # 全體統計
            cols_s = st.columns(5)
            cols_s[0].metric("有效點數", int(valid_mask.sum()))
            cols_s[1].metric(f"Y 平均", f"{np.nanmean(y_plot):.4f}")
            cols_s[2].metric(f"Y SD",   f"{np.nanstd(y_plot, ddof=1):.4f}")
            cols_s[3].metric(f"Y Min",  f"{np.nanmin(y_plot):.4f}")
            cols_s[4].metric(f"Y Max",  f"{np.nanmax(y_plot):.4f}")

        # Pearson r（若 X 是數值）
        if not x_is_batch and valid_mask.sum() >= 3:
            r_val, p_val = sp_stats.pearsonr(x_plot, y_plot)
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            color_box = "#d4edda" if p_val < 0.05 else "#f8f9fa"
            st.markdown(
                f"<div style='padding:8px;border-radius:6px;background:{color_box};"
                f"border:1px solid #ccc;font-size:14px'>"
                f"📊 Pearson r = <b>{r_val:.4f}</b>，p = {p_val:.4f} {stars}</div>",
                unsafe_allow_html=True,
            )
