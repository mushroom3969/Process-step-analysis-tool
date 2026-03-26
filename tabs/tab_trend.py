"""Tab 1 — 趨勢圖 + 特徵比較（色帶區間 + 漸層 + 第三參數 Hue + 統計）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import textwrap
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils import filt_specific_name, smooth_process_data, plot_indexed_lineplots, extract_number


# ══════════════════════════════════════════════════════════════════════════════
# ── 共用工具函式 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _sort_df(df, batch_col="BatchID"):
    out = df.copy()
    if batch_col in out.columns:
        out["_sort"] = out[batch_col].apply(extract_number)
        out = out.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    out["_seq"] = range(1, len(out) + 1)
    return out


def _apply_smooth(df, feat_x, feat_y, method, frac):
    # 如果不平滑，直接回傳數值
    if method == "none":
        # 這裡也要支援 index 取值
        x = df.index.values.astype(float) if feat_x == "index" else df[feat_x].values
        y = df.index.values.astype(float) if feat_y == "index" else df[feat_y].values
        return x, y

    # ── 處理 index 轉欄位的邏輯 ──────────────────────────
    temp_df = df.copy()
    actual_x = feat_x
    actual_y = feat_y

    # 如果 feat_x 是 index，建立暫時欄位
    if feat_x == "index":
        actual_x = "__index_temp__"
        temp_df[actual_x] = temp_df.index.values.astype(float)
    
    # 如果 feat_y 是 index，建立暫時欄位
    if feat_y == "index":
        actual_y = "__index_temp__"
        temp_df[actual_y] = temp_df.index.values.astype(float)

    # ── 執行平滑化 ───────────────────────────────────
    # 使用實際的欄位名稱 (actual_x, actual_y) 進行計算
    tmp = smooth_process_data(
        temp_df[[actual_x, actual_y]], 
        [actual_x, actual_y],
        id_cols=[], 
        method=method, 
        frac=frac
    )

    # ── 取得結果 ─────────────────────────────────────
    # 從結果中取出平滑後的數值，若找不到則回傳原始值
    res_x = tmp[actual_x].values if actual_x in tmp else temp_df[actual_x].values
    res_y = tmp[actual_y].values if actual_y in tmp else temp_df[actual_y].values

    return res_x, res_y


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
    ax.plot(xr, m * xr + b, color=color, lw=1.2, linestyle="--", alpha=alpha, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# ── Hue 著色引擎 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _build_hue_colors(hue_vals, hue_mode, zones, cmap_name):
    """
    依 hue_vals、模式和區間設定回傳：
      colors_list  : 每個點的 RGBA 或 hex 顏色
      legend_items : list of (label, color) for legend / colorbar
      sm           : ScalarMappable（漸層模式才有）
      zone_labels  : list of str（區間模式才有）
    """
    finite_mask = np.isfinite(hue_vals)

    if hue_mode == "gradient":
        cmap = mcm.get_cmap(cmap_name)
        vmin = hue_vals[finite_mask].min() if finite_mask.sum() > 0 else 0
        vmax = hue_vals[finite_mask].max() if finite_mask.sum() > 0 else 1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors_list = [cmap(norm(v)) if np.isfinite(v) else (0.75, 0.75, 0.75, 1.0)
                       for v in hue_vals]
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        return colors_list, [], sm, []

    else:  # zone
        def _zone_color(v):
            if not np.isfinite(v):
                return "#cccccc", "No Data"
            for z in zones:
                if z["min"] <= v <= z["max"]:
                    return z["color"], z["label"]
            return "#aaaaaa", "Outside Zones"

        colors_list  = [_zone_color(v)[0] for v in hue_vals]
        zone_labels  = [_zone_color(v)[1] for v in hue_vals]
        legend_items = [(z["label"], z["color"]) for z in zones]
        legend_items.append(("Outside Zones", "#aaaaaa"))
        return colors_list, legend_items, None, zone_labels


def _add_hue_legend(ax, fig, hue_feat, hue_mode, legend_items, sm,
                    zones, zone_labels, hue_vals):
    """漸層 → colorbar；區間 → patch legend。"""
    if hue_mode == "gradient" and sm is not None:
        cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, aspect=25)
        cbar.set_label(textwrap.shorten(hue_feat, 35), fontsize=8)
    elif hue_mode == "zone" and legend_items:
        present = set(zone_labels)
        handles = [mpatches.Patch(color=c, alpha=0.85, label=l)
                   for l, c in legend_items if l in present]
        ax.legend(handles=handles, title=textwrap.shorten(hue_feat, 30),
                  fontsize=7.5, title_fontsize=8, loc="upper left",
                  framealpha=0.85)


# ══════════════════════════════════════════════════════════════════════════════
# ── 主繪圖函式 ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_comparison(
    df, feat_x, feat_y,
    # ── hue 設定 ─────────────────────────────
    hue_feat=None,          # None → 使用 feat_x 著色（原始行為）
    hue_mode="zone",        # "zone" | "gradient"
    zones=None,             # list of {label, min, max, color}
    gradient_cmap="viridis",
    # ── 其他設定 ─────────────────────────────
    batch_col="BatchID",
    plot_type="scatter",
    smooth_method="none",
    frac=0.3,
    show_regression=True,
    alpha_level=0.05,
):
    """
    通用特徵比較散佈圖，支援：
      - hue_feat = None：用 feat_x 值著色（舊行為）
      - hue_feat = 某欄位：用第三變數著色，X/Y 軸是 feat_x/feat_y
    """
    if zones is None:
        zones = []

    plot_df  = _sort_df(df, batch_col)

    # ── 改動點 1：建立一個統一的取值輔助函式 ──────────────────────
    def _get_val(data, key):
        if key == "index":
            # 如果 index 是數值型態直接轉 float；若不是則轉序號 (0, 1, 2...)
            if np.issubdtype(data.index.dtype, np.number):
                return data.index.values.astype(float)
            else:
                return np.arange(len(data)).astype(float)
        return data[key].values.astype(float)

    # ── 改動點 2：使用輔助函式獲取 X, Y 數值 ─────────────────────
    x_vals = _get_val(plot_df, feat_x)
    y_vals = _get_val(plot_df, feat_y)
    
    # 注意：_apply_smooth 內部也必須支援 index，建議直接傳入 x_vals/y_vals 
    # 或者在 _apply_smooth 內部實作相同的 get_val 邏輯
    x_vals_s, y_vals_s = _apply_smooth(plot_df, feat_x, feat_y, smooth_method, frac)

    # 決定 hue 數值來源
    if hue_feat is None or hue_feat == feat_x:
        hue_vals = x_vals.copy()
        hue_label = feat_x
    else:
        # ── 改動點 3：支援 hue 使用 index ──────────────────────
        if hue_feat == "index":
            hue_vals = _get_val(plot_df, "index")
        else:
            hue_vals = plot_df[hue_feat].values.astype(float) if hue_feat in plot_df.columns else x_vals.copy()
        hue_label = hue_feat

    colors_list, legend_items, sm, zone_labels = _build_hue_colors(
        hue_vals, hue_mode, zones, gradient_cmap
    )

    # ── scatter / scatter+line ────────────────────────────────
    if plot_type in ("scatter", "scatter+line"):
        fig, ax = plt.subplots(figsize=(11, 6))

        # 背景 zone 帶（只在 hue=X 時有意義）
        if hue_feat is None or hue_feat == feat_x:
            for z in zones:
                ax.axvspan(z["min"], z["max"], alpha=0.07, color=z["color"], zorder=0)

        if plot_type == "scatter+line":
            ax.plot(x_vals_s, y_vals_s, color="#cccccc", lw=0.8, alpha=0.6, zorder=1)

        ax.scatter(x_vals, y_vals, c=colors_list, s=65,
                   edgecolors="white", linewidths=0.5, zorder=2, alpha=0.9)

        _add_hue_legend(ax, fig, hue_label, hue_mode, legend_items, sm,
                        zones, zone_labels, hue_vals)

        # Batch 標籤
        if batch_col in plot_df.columns:
            for xi, yi, bid in zip(x_vals, y_vals, plot_df[batch_col]):
                if np.isfinite(xi) and np.isfinite(yi):
                    ax.annotate(str(bid)[-6:], (xi, yi), fontsize=5.5, alpha=0.6,
                                xytext=(3, 3), textcoords="offset points")

        # 相關係數文字框
        stat_rows = []
        res_all = _pearson_stat(x_vals, y_vals)
        if res_all:
            res_all["label"] = "All data"; res_all["color"] = "black"
            stat_rows.append(res_all)
            if show_regression:
                _add_regression_line(ax, x_vals, y_vals, "black", 0.45)

        # 各 zone 相關（按 hue 欄位分群）
        if hue_mode == "zone" and zones:
            for z in zones:
                zone_mask = np.array([l == z["label"] for l in zone_labels])
                if zone_mask.sum() < 3:
                    continue
                res = _pearson_stat(x_vals[zone_mask], y_vals[zone_mask])
                if res:
                    res["label"] = z["label"]; res["color"] = z["color"]
                    stat_rows.append(res)
                    if show_regression:
                        _add_regression_line(ax, x_vals[zone_mask], y_vals[zone_mask],
                                             z["color"], 0.8)

        if stat_rows:
            lines = []
            for r in stat_rows:
                sig = " ✓" if r["p"] < alpha_level else ""
                lines.append(f"{r['label']} (n={r['n']}): r={r['r']:.3f}, {r['p_str']} {r['stars']}{sig}")
            ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes,
                    fontsize=7.5, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#cccccc", alpha=0.92))

        # ─── 插入改動點 4 ───
        display_x = "Sample Index" if feat_x == "index" else feat_x
        display_y = feat_y  # 如果 feat_y 不太可能是 index，可以直接用原變數，或同樣判斷

        # 修改原本的 set_xlabel / set_ylabel
        ax.set_xlabel(textwrap.shorten(display_x, 55), fontsize=10)
        ax.set_ylabel(textwrap.shorten(display_y, 55), fontsize=10)
        
        # Title 也要同步修改，才不會顯示 "index vs Feature"
        hue_suffix = f"  [hue: {textwrap.shorten(hue_label, 30)}]" \
                     if hue_feat and hue_feat != feat_x else ""
        ax.set_title(f"{textwrap.shorten(display_y, 40)}  vs  {textwrap.shorten(display_x, 40)}"
                     + hue_suffix, fontsize=11)
        
        ax.grid(linestyle="--", alpha=0.35)
        plt.tight_layout()
        return fig, stat_rows

    # ── dual_line ─────────────────────────────────────────────
    elif plot_type == "dual_line":
        fig, ax1 = plt.subplots(figsize=(13, 5))
        seq = plot_df["_seq"].values

        ax1.set_xlabel("Batch Sequence", fontsize=10)
        ax1.set_ylabel(textwrap.shorten(feat_x, 45), color="#2e86ab", fontsize=10)
        ax1.plot(seq, x_vals_s, color="#2e86ab", marker="o", ms=4, lw=1.5, label=feat_x[:30])
        ax1.tick_params(axis="y", labelcolor="#2e86ab")

        # hue zone 帶（只有 zone 模式）
        if hue_mode == "zone" and zones and hue_feat in (None, feat_x):
            for z in zones:
                ax1.axhspan(z["min"], z["max"], alpha=0.10, color=z["color"], zorder=0)

        ax2 = ax1.twinx()
        ax2.set_ylabel(textwrap.shorten(feat_y, 45), color="#e84855", fontsize=10)
        ax2.plot(seq, y_vals_s, color="#e84855", marker="s", ms=4, lw=1.5,
                 linestyle="--", label=feat_y[:30])
        ax2.tick_params(axis="y", labelcolor="#e84855")

        # 若有第三參數，在 X 軸下方畫 hue bar
        if hue_feat and hue_feat != feat_x and hue_feat in plot_df.columns:
            ax_hue = ax1.twinx()
            ax_hue.spines["right"].set_position(("outward", 55))
            ax_hue.set_ylabel(textwrap.shorten(hue_feat, 30), fontsize=8, color="#666")
            ax_hue.plot(seq, hue_vals, color="#999", lw=1.0, linestyle=":", alpha=0.6)
            ax_hue.tick_params(axis="y", labelcolor="#888", labelsize=7)

        res_all = _pearson_stat(x_vals, y_vals)
        sig = (" ✓" if res_all["p"] < alpha_level else "") if res_all else ""
        title = (f"{textwrap.shorten(feat_x, 35)}  &  {textwrap.shorten(feat_y, 35)}\n"
                 + (f"r={res_all['r']:.3f}, {res_all['p_str']} {res_all['stars']}{sig}" if res_all else ""))
        ax1.set_title(title, fontsize=11)

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        if hue_mode == "zone" and zones and hue_feat in (None, feat_x):
            zone_h = [mpatches.Patch(color=z["color"], alpha=0.5, label=z["label"]) for z in zones]
            ax1.legend(l1+l2+zone_h, lb1+lb2+[z["label"] for z in zones], loc="best", fontsize=8)
        else:
            ax1.legend(l1+l2, lb1+lb2, loc="best", fontsize=8)

        ax1.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        return fig, [res_all] if res_all else []

    return None, []


# ══════════════════════════════════════════════════════════════════════════════
# ── Hue 設定 UI 共用元件 ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

PRESET_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db",
                 "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
CMAPS = ["viridis", "plasma", "coolwarm", "RdYlGn", "Blues", "YlOrRd", "RdBu_r"]


def _hue_ui(numeric_cols, feat_x, feat_y, key_prefix):
    """
    渲染「第三參數 Hue」設定區塊，回傳設定 dict。
    """
    st.markdown("#### 🎨 著色設定（Hue）")
    h1, h2, h3 = st.columns([2, 1, 1])

    use_hue = h1.checkbox("啟用第三參數著色（Hue）",
                          value=False, key=f"{key_prefix}_use_hue")

    if not use_hue:
        # 舊行為：用 X 軸著色
        hue_mode = h2.radio("X 軸著色模式",
                            ["zone（區間）", "gradient（漸層）"],
                            horizontal=True, key=f"{key_prefix}_x_mode")
        use_gradient = hue_mode.startswith("gradient")
        gradient_cmap = h3.selectbox("漸層色板", CMAPS,
                                     key=f"{key_prefix}_x_cmap",
                                     disabled=not use_gradient)
        return {
            "hue_feat": None,
            "hue_mode": "gradient" if use_gradient else "zone",
            "gradient_cmap": gradient_cmap,
        }

    # 第三參數模式
    available_hue = [c for c in numeric_cols if c not in (feat_x, feat_y)]
    if not available_hue:
        st.warning("沒有其他數值欄位可作為 Hue。")
        return {"hue_feat": None, "hue_mode": "zone", "gradient_cmap": "viridis"}

    hue_feat = h1.selectbox("Hue 欄位（第三參數）", available_hue,
                            key=f"{key_prefix}_hue_feat")
    hue_mode_raw = h2.radio("Hue 著色模式",
                            ["zone（區間）", "gradient（漸層）"],
                            horizontal=True, key=f"{key_prefix}_hue_mode")
    use_gradient  = hue_mode_raw.startswith("gradient")
    gradient_cmap = h3.selectbox("漸層色板", CMAPS,
                                 key=f"{key_prefix}_hue_cmap",
                                 disabled=not use_gradient)

    hue_series = None
    for df_candidate in [st.session_state.get("clean_df"),
                          st.session_state.get("selected_process_df")]:
        if df_candidate is not None and hue_feat in df_candidate.columns:
            hue_series = df_candidate[hue_feat].dropna()
            break

    if hue_series is not None:
        st.caption(
            f"**{textwrap.shorten(hue_feat, 40)}** 範圍："
            f"{hue_series.min():.3f} ～ {hue_series.max():.3f}"
            f"（平均 {hue_series.mean():.3f}）"
        )

    return {
        "hue_feat":     hue_feat,
        "hue_mode":     "gradient" if use_gradient else "zone",
        "gradient_cmap": gradient_cmap,
        "hue_series":   hue_series,
    }


def _zone_ui(hue_cfg, numeric_cols, work_df, key_prefix):
    """
    渲染 Zone 設定 UI，回傳 zones list。
    只在 zone 模式下顯示。
    """
    if hue_cfg["hue_mode"] != "zone":
        return []

    # 決定 zone 依據的欄位
    zone_ref_feat = hue_cfg.get("hue_feat") or numeric_cols[0]
    ref_series    = work_df[zone_ref_feat].dropna() if zone_ref_feat in work_df.columns \
                    else pd.Series([0.0, 1.0])

    st.markdown("#### 🎯 設定數值區間（Zone）")
    n_zones = st.number_input("區間數量", min_value=1, max_value=8,
                              value=3, step=1, key=f"{key_prefix}_n_zones")
    zones = []
    for i in range(int(n_zones)):
        with st.expander(f"區間 {i+1}", expanded=(i < 3)):
            zc1, zc2, zc3, zc4 = st.columns([2, 1, 1, 1])
            label = zc1.text_input("名稱", value=f"Zone {i+1}",
                                   key=f"{key_prefix}_zlabel_{i}")
            default_min = float(round(
                ref_series.min() + i * (ref_series.max() - ref_series.min()) / n_zones, 3))
            default_max = float(round(
                ref_series.min() + (i+1) * (ref_series.max() - ref_series.min()) / n_zones, 3))
            zmin  = zc2.number_input("最小值", value=default_min,
                                     key=f"{key_prefix}_zmin_{i}", format="%.3f")
            zmax  = zc3.number_input("最大值", value=default_max,
                                     key=f"{key_prefix}_zmax_{i}", format="%.3f")
            color = zc4.color_picker("顏色", value=PRESET_COLORS[i % len(PRESET_COLORS)],
                                     key=f"{key_prefix}_zcolor_{i}")
            zones.append({"label": label, "min": zmin, "max": zmax, "color": color})
    return zones


# ══════════════════════════════════════════════════════════════════════════════
# ── render ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if _cd is not None:
        st.info("📌 目前使用**特徵工程後**的資料。")

    subtabs = st.tabs(["📈 全局趨勢圖", "🎨 特徵比較 + Hue 著色"])

    # ══════════════════════════════════════════════════════════
    # ── Tab 0: 全局趨勢圖 ─────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    with subtabs[0]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        keyword       = col_a.text_input("欄位關鍵字篩選（留空 = 全部）", "")
        smooth_method = col_b.selectbox("平滑方法", ["loess", "ewma", "none"],
                                        key="trend_smooth")
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
                    st.pyplot(fig)
                    plt.close()

    # ══════════════════════════════════════════════════════════
    # ── Tab 1: 特徵比較 + Hue ────────────────────────────────
    # ══════════════════════════════════════════════════════════
    with subtabs[1]:
        st.markdown("#### 🎨 特徵比較圖（X vs Y，可加入第三參數 Hue 著色）")

        numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
        display_options = ["index"] + numeric_cols
        
        if len(numeric_cols) < 2:
            st.warning("需要至少 2 個數值欄位。")
            return

        # ── X / Y / 圖表類型 ──────────────────────────────────
        c1, c2, c3 = st.columns(3)
        # 修改點：使用 plot_options 替代 numeric_cols
        feat_x = c1.selectbox("X 軸特徵", plot_options, index=0, key="cz_feat_x")
        
        # 修改點：Y 軸排除掉目前選中的 feat_x
        y_options = [c for c in plot_options if c != feat_x]
        feat_y = c2.selectbox("Y 軸特徵", y_options, index=0, key="cz_feat_y")
        
        plot_type = c3.selectbox(
            "圖表類型", ["scatter", "scatter+line", "dual_line"],
            key="cz_plot_type",
            format_func=lambda x: {"scatter": "散佈圖",
                                   "scatter+line": "散佈+連線",
                                   "dual_line": "雙軸折線"}[x],
        )

        # ── 平滑 ──────────────────────────────────────────────
        sm1, sm2 = st.columns(2)
        smooth_cz = sm1.selectbox("平滑", ["none", "loess", "ewma"], key="cz_smooth")
        frac_cz   = sm2.slider("LOESS frac", 0.1, 0.8, 0.3, 0.05, key="cz_frac",
                                disabled=(smooth_cz != "loess"))

        # ── Hue 設定 UI ───────────────────────────────────────
        st.markdown("---")
        hue_cfg = _hue_ui(plot_options, feat_x, feat_y, key_prefix="cz")

        # ── Zone 設定（zone 模式才顯示）────────────────────────
        zones = _zone_ui(hue_cfg, numeric_cols, work_df, key_prefix="cz")

        # ── 其他選項 ──────────────────────────────────────────
        st.markdown("---")
        opt1, opt2 = st.columns(2)
        show_regression = opt1.checkbox("顯示迴歸線", value=True, key="cz_show_reg")
        with opt2.expander("📐 統計設定"):
            alpha_level = st.select_slider("顯著水準 α",
                                           options=[0.001, 0.01, 0.05, 0.10],
                                           value=0.05, key="cz_alpha_level")

        # ── 繪圖 ──────────────────────────────────────────────
        if st.button("🎨 繪製比較圖", type="primary", key="plot_comparison"):
            if hue_cfg["hue_mode"] == "zone" and zones:
                if not all(z["min"] < z["max"] for z in zones):
                    st.error("每個區間最小值必須 < 最大值。")
                    st.stop()

            with st.spinner("繪圖中..."):
                try:
                    fig, stat_rows = plot_feature_comparison(
                        work_df,
                        feat_x=feat_x,
                        feat_y=feat_y,
                        hue_feat=hue_cfg.get("hue_feat"),
                        hue_mode=hue_cfg["hue_mode"],
                        zones=zones,
                        gradient_cmap=hue_cfg["gradient_cmap"],
                        plot_type=plot_type,
                        smooth_method=smooth_cz,
                        frac=frac_cz,
                        show_regression=show_regression,
                        alpha_level=alpha_level,
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close()

                    # 相關係數表
                    if stat_rows:
                        st.markdown("#### 📊 各組 Pearson 相關係數")
                        stat_df = pd.DataFrame([{
                            "組別": r["label"], "n": r["n"],
                            "r": round(r["r"], 4),
                            "p-value": round(r["p"], 4),
                            "significance": r["stars"],
                            "達顯著": "✓" if r["p"] < alpha_level else "",
                        } for r in stat_rows if r])
                        st.dataframe(
                            stat_df.style.apply(
                                lambda row: ["background-color:#d4edda"] * len(row)
                                if row.get("達顯著") == "✓"
                                else ["background-color:#f8f9fa"] * len(row),
                                axis=1,
                            ),
                            width="stretch", hide_index=True,
                        )

                except Exception as e:
                    import traceback
                    st.error(f"繪圖失敗：{e}")
                    st.code(traceback.format_exc())

        # ── 批次 zone 表格 ────────────────────────────────────
        if hue_cfg["hue_mode"] == "zone" and zones:
            if st.checkbox("顯示各批次所在區間", key="cz_show_table"):
                ref_feat = hue_cfg.get("hue_feat") or feat_x
                
                def _assign(val):
                    for z in zones:
                        if pd.notna(val) and z["min"] <= val <= z["max"]:
                            return z["label"]
                    return "Outside"

                # 建立顯示用的 DataFrame
                summary = work_df.copy()
                
                # ─── 新增：如果參考特徵是 index，先把它轉成真正的欄位方便處理 ───
                if ref_feat == "index":
                    summary["index"] = summary.index.values.astype(float)
                if feat_x == "index" and "index" not in summary.columns:
                    summary["index"] = summary.index.values.astype(float)
                if feat_y == "index" and "index" not in summary.columns:
                    summary["index"] = summary.index.values.astype(float)

                cols_needed = ["BatchID", ref_feat, feat_x, feat_y]
                cols_needed = list(dict.fromkeys(c for c in cols_needed if c in summary.columns))
                
                summary = summary[cols_needed].copy()
                summary["Zone"] = summary[ref_feat].apply(_assign)
                
                st.dataframe(
                    summary.sort_values(ref_feat).reset_index(drop=True),
                    width="stretch", hide_index=True,
                )
