"""Tab — 跨製程步驟參數監控（Cross-Process Trend Monitor）"""
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
import matplotlib.ticker as mticker
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

from utils import extract_number


# ══════════════════════════════════════════════════════════════════════════════
# ── 資料準備：從 raw_df 建立跨製程欄位目錄 ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _build_param_catalog(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    解析 raw_df 所有 'ProcessStep:ParamName' 欄位，
    回傳 DataFrame：[process_step, param_name, full_col]
    """
    rows = []
    for col in raw_df.columns:
        if ":" in col:
            step, param = col.split(":", 1)
            rows.append({"製程步驟": step.strip(), "參數名稱": param.strip(), "原始欄位": col})
    return pd.DataFrame(rows)


def _sort_batches(raw_df: pd.DataFrame) -> pd.DataFrame:
    """依 BatchID 數字排序。"""
    df = raw_df.copy()
    if "BatchID" in df.columns:
        df["_sort"] = df["BatchID"].apply(extract_number)
        df = df.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    df["_seq"] = range(1, len(df) + 1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ── 平滑 ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(series: pd.Series, method: str, frac: float = 0.3, span: int = 10) -> np.ndarray:
    y = series.values.astype(float)
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if method == "loess":
        if mask.sum() > 10:
            res = _lowess(y[mask], x[mask], frac=frac)
            out = np.full(len(y), np.nan)
            out[mask] = res[:, 1]
            return pd.Series(out).interpolate(limit_direction="both").values
        return y
    elif method == "ewma":
        return pd.Series(y).ewm(span=span, adjust=False).mean().values
    return y


# ══════════════════════════════════════════════════════════════════════════════
# ── 核心繪圖 ──────────────────────────────────────────────════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def _plot_cross_trend(
    plot_df: pd.DataFrame,          # 已排序，含 _seq, BatchID, 所選欄位
    selected_params: list[dict],    # [{"full_col": ..., "display": ..., "color": ...}, ...]
    smooth_method: str,
    frac: float,
    ewma_span: int,
    show_mean: bool,
    show_band: bool,
    band_mode: str,                 # "±σ" | "±2σ" | "±3σ" | "percentile"
    band_pct_lo: float,
    band_pct_hi: float,
    cols_per_row: int,
    show_batch_label: bool,
    limit_configs: dict,            # {full_col: {"ucl": float|None, "lcl": float|None, "target": float|None}}
) -> plt.Figure | None:

    if not selected_params:
        return None

    n_params = len(selected_params)
    n_rows = math.ceil(n_params / cols_per_row)
    fig, axes = plt.subplots(
        n_rows, cols_per_row,
        figsize=(cols_per_row * 6, n_rows * 4),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    seq = plot_df["_seq"].values
    batch_labels = plot_df["BatchID"].values if "BatchID" in plot_df.columns else seq

    for i, param in enumerate(selected_params):
        ax = axes_flat[i]
        col = param["full_col"]
        color = param.get("color", "#2e86ab")
        display_name = param.get("display", col)

        if col not in plot_df.columns:
            ax.text(0.5, 0.5, f"欄位不存在\n{col[:40]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.axis("off")
            continue

        raw_vals = plot_df[col].values.astype(float)

        # 平滑
        if smooth_method != "none":
            smooth_vals = _smooth(plot_df[col], smooth_method, frac=frac, span=ewma_span)
        else:
            smooth_vals = raw_vals

        valid_mask = ~np.isnan(raw_vals)
        valid_vals = raw_vals[valid_mask]

        # ── 管制帶 ─────────────────────────────────────────────
        if show_band and len(valid_vals) >= 3:
            if band_mode.startswith("±"):
                n_sigma = int(band_mode[1])
                mu = np.nanmean(raw_vals)
                sigma = np.nanstd(raw_vals, ddof=1)
                lo, hi = mu - n_sigma * sigma, mu + n_sigma * sigma
            else:  # percentile
                lo = np.nanpercentile(raw_vals, band_pct_lo)
                hi = np.nanpercentile(raw_vals, band_pct_hi)
            ax.fill_between(seq, lo, hi, alpha=0.10, color=color, zorder=0,
                            label=f"Band ({band_mode})")
            ax.axhline(lo, color=color, lw=0.8, linestyle=":", alpha=0.6)
            ax.axhline(hi, color=color, lw=0.8, linestyle=":", alpha=0.6)

        # ── 平均線 ─────────────────────────────────────────────
        if show_mean and len(valid_vals) > 0:
            mu = np.nanmean(raw_vals)
            ax.axhline(mu, color=color, lw=1.2, linestyle="--", alpha=0.75,
                       label=f"Mean: {mu:.3f}")

        # ── 使用者自訂管制線 ────────────────────────────────────
        lim = limit_configs.get(col, {})
        if lim.get("ucl") is not None:
            ax.axhline(lim["ucl"], color="#e84855", lw=1.5, linestyle="-.",
                       label=f"UCL: {lim['ucl']:.3f}")
        if lim.get("lcl") is not None:
            ax.axhline(lim["lcl"], color="#e84855", lw=1.5, linestyle="-.",
                       label=f"LCL: {lim['lcl']:.3f}")
        if lim.get("target") is not None:
            ax.axhline(lim["target"], color="#2ca02c", lw=1.5, linestyle="-",
                       label=f"Target: {lim['target']:.3f}")

        # ── 原始資料點 ─────────────────────────────────────────
        # 違規點著紅色
        ucl = lim.get("ucl")
        lcl = lim.get("lcl")
        point_colors = []
        for v in raw_vals:
            if np.isnan(v):
                point_colors.append("#cccccc")
            elif (ucl is not None and v > ucl) or (lcl is not None and v < lcl):
                point_colors.append("#e84855")
            else:
                point_colors.append(color)

        ax.scatter(seq, raw_vals, c=point_colors, s=30, zorder=4,
                   edgecolors="white", linewidths=0.4, alpha=0.85)

        # ── 折線（原始或平滑） ──────────────────────────────────
        if smooth_method != "none":
            # 原始：淡灰底線
            ax.plot(seq, raw_vals, color="#cccccc", lw=0.8, alpha=0.5, zorder=2)
            # 平滑：主線
            ax.plot(seq, smooth_vals, color=color, lw=2.0, zorder=3,
                    label=f"{smooth_method.upper()}")
        else:
            ax.plot(seq, raw_vals, color=color, lw=1.5, zorder=3)

        # ── Batch 標籤 ─────────────────────────────────────────
        if show_batch_label and len(seq) <= 80:
            for xi, yi, bl in zip(seq, raw_vals, batch_labels):
                if not np.isnan(yi):
                    ax.annotate(str(bl)[-6:], (xi, yi),
                                fontsize=5.5, alpha=0.65,
                                xytext=(0, 5), textcoords="offset points",
                                ha="center")

        # ── X 軸標籤 ───────────────────────────────────────────
        if len(seq) <= 60:
            ax.set_xticks(seq)
            ax.set_xticklabels(
                [str(b)[-6:] for b in batch_labels],
                rotation=90, fontsize=6,
            )
        else:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=15))
            ax.set_xlabel("Batch Sequence", fontsize=8)

        # ── 裝飾 ───────────────────────────────────────────────
        title_text = "\n".join(textwrap.wrap(display_name, width=42))
        ax.set_title(title_text, fontsize=8.5, pad=6)
        ax.set_ylabel("Value", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.2)

        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, lbls, fontsize=6.5, loc="best",
                      framealpha=0.7, ncol=1)

    # 關掉多餘子圖
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle("跨製程參數趨勢監控", fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ── render ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(raw_df: pd.DataFrame):
    st.header("🔭 跨製程參數趨勢監控")
    st.markdown("""
    <div class="info-box">
    跨越所有製程步驟自由選取參數，在同一個畫面中觀察各批次（按時間排序）的數值變化。<br>
    可疊加 <b>平均線、管制帶（σ / 百分位）、UCL / LCL / Target</b>，並支援 LOESS / EWMA 平滑。
    </div>
    """, unsafe_allow_html=True)

    if raw_df is None:
        st.info("請先在側欄上傳資料。")
        return

    # ── 建立欄位目錄 ─────────────────────────────────────────
    catalog = _build_param_catalog(raw_df)
    if catalog.empty:
        st.warning("資料中未找到『製程步驟:參數名稱』格式的欄位。")
        return

    all_steps = sorted(catalog["製程步驟"].unique().tolist())

    # ══════════════════════════════════════════════════════════
    # ── Step 1：選擇製程步驟與參數 ────────────────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown("### Step 1：選擇參數")

    # 關鍵字快速篩選
    kw_col, step_col = st.columns([2, 3])
    keyword = kw_col.text_input(
        "🔍 參數關鍵字篩選（留空 = 全部）",
        key="cp_keyword",
        help="在所有製程的參數名稱中搜尋，不分大小寫",
    )
    selected_steps = step_col.multiselect(
        "限制製程步驟範圍（留空 = 全部）",
        options=all_steps,
        key="cp_steps",
    )

    # 篩選目錄
    filtered_catalog = catalog.copy()
    if selected_steps:
        filtered_catalog = filtered_catalog[filtered_catalog["製程步驟"].isin(selected_steps)]
    if keyword.strip():
        filtered_catalog = filtered_catalog[
            filtered_catalog["參數名稱"].str.contains(keyword.strip(), case=False, na=False)
        ]

    if filtered_catalog.empty:
        st.warning("沒有符合條件的參數，請調整篩選條件。")
        return

    # 顯示為「製程步驟 > 參數名稱」供多選
    filtered_catalog["選項標籤"] = (
        filtered_catalog["製程步驟"] + "  ›  " + filtered_catalog["參數名稱"]
    )
    label_to_col = dict(zip(filtered_catalog["選項標籤"], filtered_catalog["原始欄位"]))
    col_to_step  = dict(zip(filtered_catalog["原始欄位"], filtered_catalog["製程步驟"]))
    col_to_param = dict(zip(filtered_catalog["原始欄位"], filtered_catalog["參數名稱"]))

    st.caption(f"符合條件的參數：{len(filtered_catalog)} 個，來自 {filtered_catalog['製程步驟'].nunique()} 個製程步驟")

    selected_labels = st.multiselect(
        "✅ 選擇要監控的參數（可多選）",
        options=sorted(label_to_col.keys()),
        key="cp_selected_labels",
        help="格式：製程步驟  ›  參數名稱",
    )

    if not selected_labels:
        st.info("請選擇至少一個參數以繼續。")
        return

    selected_full_cols = [label_to_col[l] for l in selected_labels]

    # ── 每個參數的顯示顏色 ───────────────────────────────────
    PALETTE = [
        "#2e86ab", "#e84855", "#2ca02c", "#ff7f0e", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#1f77b4",
    ]
    st.markdown("##### 🎨 各參數顏色設定（可選）")
    param_configs: list[dict] = []
    color_cols = st.columns(min(len(selected_full_cols), 4))
    for i, full_col in enumerate(selected_full_cols):
        step  = col_to_step[full_col]
        param = col_to_param[full_col]
        display = f"[{step}] {param}"
        with color_cols[i % len(color_cols)]:
            col_color = st.color_picker(
                label=textwrap.shorten(display, width=30),
                value=PALETTE[i % len(PALETTE)],
                key=f"cp_color_{i}",
            )
        param_configs.append({
            "full_col": full_col,
            "display": display,
            "color": col_color,
        })

    # ══════════════════════════════════════════════════════════
    # ── Step 2：圖形設定 ──────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Step 2：圖形設定")

    with st.expander("⚙️ 展開設定", expanded=True):
        cfg1, cfg2, cfg3 = st.columns(3)

        # 平滑
        smooth_method = cfg1.selectbox(
            "平滑方法", ["none", "loess", "ewma"],
            format_func=lambda x: {"none": "不平滑", "loess": "LOESS", "ewma": "EWMA"}[x],
            key="cp_smooth",
        )
        loess_frac = cfg1.slider(
            "LOESS frac", 0.1, 0.8, 0.3, 0.05,
            key="cp_frac",
            disabled=(smooth_method != "loess"),
        )
        ewma_span = cfg1.slider(
            "EWMA span", 3, 30, 10,
            key="cp_ewma_span",
            disabled=(smooth_method != "ewma"),
        )

        # 管制帶
        show_mean = cfg2.checkbox("顯示平均線", value=True, key="cp_mean")
        show_band = cfg2.checkbox("顯示管制帶", value=True, key="cp_band")
        band_mode = cfg2.radio(
            "管制帶類型",
            ["±1σ", "±2σ", "±3σ", "percentile"],
            index=1,
            key="cp_band_mode",
            disabled=not show_band,
        )
        band_pct_lo, band_pct_hi = 10.0, 90.0
        if band_mode == "percentile" and show_band:
            band_pct_lo = cfg2.number_input("下百分位", 0.0, 49.0, 10.0, 1.0, key="cp_pct_lo")
            band_pct_hi = cfg2.number_input("上百分位", 51.0, 100.0, 90.0, 1.0, key="cp_pct_hi")

        # 版面
        cols_per_row = cfg3.slider("每列圖數", 1, 4, 2, key="cp_cols_per_row")
        show_batch_label = cfg3.checkbox("顯示 Batch ID 標籤", value=False, key="cp_batch_label")

    # ══════════════════════════════════════════════════════════
    # ── Step 3：UCL / LCL / Target 管制線設定 ─────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Step 3：管制線設定（選填）")
    st.caption("為各參數設定 UCL（上限）、LCL（下限）、Target（目標值），違規點將以紅色標示。")

    limit_configs: dict[str, dict] = {}
    with st.expander("📏 設定管制線", expanded=False):
        for param in param_configs:
            col = param["full_col"]
            display = param["display"]
            raw_series = raw_df[col].dropna() if col in raw_df.columns else pd.Series(dtype=float)
            col_mean = raw_series.mean() if len(raw_series) > 0 else 0.0
            col_std  = raw_series.std(ddof=1) if len(raw_series) > 1 else 1.0

            st.markdown(f"**{textwrap.shorten(display, width=60)}**")
            lc1, lc2, lc3, lc4 = st.columns([1, 1, 1, 2])
            use_lim = lc4.checkbox(
                "啟用", value=False, key=f"cp_use_lim_{col}"
            )
            if use_lim:
                ucl_val = lc1.number_input(
                    "UCL", value=round(col_mean + 3 * col_std, 3),
                    key=f"cp_ucl_{col}", format="%.3f",
                )
                lcl_val = lc2.number_input(
                    "LCL", value=round(col_mean - 3 * col_std, 3),
                    key=f"cp_lcl_{col}", format="%.3f",
                )
                tgt_val = lc3.number_input(
                    "Target", value=round(col_mean, 3),
                    key=f"cp_tgt_{col}", format="%.3f",
                )
                limit_configs[col] = {"ucl": ucl_val, "lcl": lcl_val, "target": tgt_val}
            else:
                limit_configs[col] = {"ucl": None, "lcl": None, "target": None}
            st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # ── 繪圖 ──────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    if st.button("📊 繪製跨製程趨勢圖", type="primary", key="cp_plot"):
        with st.spinner("資料整理與繪圖中..."):
            try:
                # 只抽出需要的欄位 + BatchID
                needed_cols = (
                    ["BatchID"] + selected_full_cols
                    if "BatchID" in raw_df.columns
                    else selected_full_cols
                )
                plot_source = raw_df[[c for c in needed_cols if c in raw_df.columns]].copy()

                # 轉數值
                for c in selected_full_cols:
                    if c in plot_source.columns:
                        plot_source[c] = pd.to_numeric(plot_source[c], errors="coerce")

                plot_df = _sort_batches(plot_source)

                fig = _plot_cross_trend(
                    plot_df=plot_df,
                    selected_params=param_configs,
                    smooth_method=smooth_method,
                    frac=loess_frac,
                    ewma_span=ewma_span,
                    show_mean=show_mean,
                    show_band=show_band,
                    band_mode=band_mode,
                    band_pct_lo=band_pct_lo,
                    band_pct_hi=band_pct_hi,
                    cols_per_row=cols_per_row,
                    show_batch_label=show_batch_label,
                    limit_configs=limit_configs,
                )

                if fig:
                    st.session_state["cp_fig"] = fig
                    st.session_state["cp_plot_df"] = plot_df
                    st.session_state["cp_param_configs"] = param_configs
                    st.session_state["cp_limit_configs"] = limit_configs

            except Exception as e:
                import traceback
                st.error(f"繪圖失敗：{e}")
                st.code(traceback.format_exc())

    # ── 顯示圖表 ─────────────────────────────────────────────
    if st.session_state.get("cp_fig") is not None:
        st.pyplot(st.session_state["cp_fig"])
        plt.close()

        # ── 數據摘要表 ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 各參數描述統計 & 違規批次")

        plot_df      = st.session_state["cp_plot_df"]
        param_cfgs   = st.session_state["cp_param_configs"]
        limit_cfgs   = st.session_state["cp_limit_configs"]

        summary_rows = []
        for param in param_cfgs:
            col = param["full_col"]
            if col not in plot_df.columns:
                continue
            s = plot_df[col].dropna()
            lim = limit_cfgs.get(col, {})
            ucl = lim.get("ucl")
            lcl = lim.get("lcl")
            n_ucl = int((s > ucl).sum()) if ucl is not None else "—"
            n_lcl = int((s < lcl).sum()) if lcl is not None else "—"
            summary_rows.append({
                "參數": textwrap.shorten(param["display"], width=55),
                "n": len(s),
                "Mean": round(s.mean(), 4),
                "Median": round(s.median(), 4),
                "SD": round(s.std(ddof=1), 4) if len(s) > 1 else None,
                "Min": round(s.min(), 4),
                "Max": round(s.max(), 4),
                "UCL 違規數": n_ucl,
                "LCL 違規數": n_lcl,
            })

        if summary_rows:
            sum_df = pd.DataFrame(summary_rows)
            st.dataframe(
                sum_df.style.background_gradient(cmap="Blues", subset=["Mean"]),
                width="stretch", hide_index=True,
            )

        # ── 違規批次明細 ─────────────────────────────────────
        viol_frames = []
        for param in param_cfgs:
            col = param["full_col"]
            if col not in plot_df.columns:
                continue
            lim = limit_cfgs.get(col, {})
            ucl, lcl = lim.get("ucl"), lim.get("lcl")
            if ucl is None and lcl is None:
                continue
            mask_viol = pd.Series(False, index=plot_df.index)
            if ucl is not None:
                mask_viol |= plot_df[col] > ucl
            if lcl is not None:
                mask_viol |= plot_df[col] < lcl
            if mask_viol.any():
                vdf = plot_df.loc[mask_viol, ["BatchID", col]].copy() if "BatchID" in plot_df.columns \
                    else plot_df.loc[mask_viol, [col]].copy()
                vdf["參數"] = textwrap.shorten(param["display"], width=50)
                vdf["違規方向"] = plot_df.loc[mask_viol, col].apply(
                    lambda v: "↑ > UCL" if (ucl is not None and v > ucl) else "↓ < LCL"
                )
                viol_frames.append(vdf)

        if viol_frames:
            viol_df = pd.concat(viol_frames, ignore_index=True)
            st.markdown("#### 🚨 違規批次明細")
            st.dataframe(
                viol_df.style.apply(
                    lambda row: ["background-color: #f8d7da"] * len(row), axis=1
                ),
                width="stretch", hide_index=True,
            )
        elif any(limit_cfgs.get(p["full_col"], {}).get("ucl") or
                 limit_cfgs.get(p["full_col"], {}).get("lcl")
                 for p in param_cfgs):
            st.success("✅ 所有批次均在管制限內，無違規。")
