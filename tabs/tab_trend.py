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


def _plot_classified_trend(df, classifier_col, target_col, zones, batch_col="BatchID",
                            show_avg_line=True, show_batch_label=False, point_size=120):
    """
    繪製「分類著色趨勢圖」：
    - X 軸 = Batch 序列
    - Y 軸 = target_col
    - 散點顏色 = classifier_col 所屬分類區間
    - 背景折線 = target_col 趨勢
    """
    import seaborn as sns

    plot_df = df.copy()
    if batch_col in plot_df.columns:
        plot_df["_sort"] = plot_df[batch_col].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    # 分類標籤
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
                   label=f"Avg {target_col[:25]}: {avg_val:.2f}", zorder=2)

    # 散點
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
                ax.annotate(
                    str(row[batch_col])[-6:],
                    (row["_seq"], row[target_col]),
                    fontsize=6, alpha=0.65,
                    xytext=(0, 6), textcoords="offset points",
                    ha="center",
                )

    # X 軸 tick（BatchID 或 seq）
    if batch_col in plot_df.columns and len(plot_df) <= 80:
        ax.set_xticks(plot_df["_seq"])
        ax.set_xticklabels(
            [str(b)[-6:] for b in plot_df[batch_col]],
            rotation=90, fontsize=7,
        )
    else:
        ax.set_xlabel("Batch Sequence", fontsize=11)

    ax.set_ylabel(target_col[:60], fontsize=11)
    ax.set_title(
        f"{target_col[:50]}  —  colored by  {classifier_col[:50]}",
        fontsize=13, pad=14,
    )
    ax.legend(title=classifier_col[:40], bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig, plot_df


def render(selected_process_df):
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    subtabs = st.tabs(["📈 全局趨勢圖", "🎨 特徵比較 + 區間顏色", "📊 分類趨勢分析"])

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

    # ── Tab 3: 分類趨勢分析 ────────────────────────────────────
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
            # ── 欄位選擇 ──────────────────────────────────────
            ct_col1, ct_col2 = st.columns(2)
            classifier_col = ct_col1.selectbox(
                "🔑 分類依據欄位（X）",
                numeric_cols,
                key="ct_classifier",
                help="用來分群的欄位，例如：最高緩衝液溫度",
            )
            target_col_ct = ct_col2.selectbox(
                "🎯 目標 Y 欄位",
                [c for c in numeric_cols if c != classifier_col],
                key="ct_target",
                help="顯示在 Y 軸的欄位，例如：Yield Rate",
            )

            cls_series = selected_process_df[classifier_col].dropna()
            cls_min, cls_max, cls_mean = cls_series.min(), cls_series.max(), cls_series.mean()
            st.caption(
                f"**{classifier_col}** 範圍：{cls_min:.3f} ～ {cls_max:.3f}"
                f"（平均 {cls_mean:.3f}，中位數 {cls_series.median():.3f}）"
            )

            # ── 顯示選項 ──────────────────────────────────────
            opt1, opt2, opt3 = st.columns(3)
            show_avg    = opt1.checkbox("顯示平均線", value=True, key="ct_avg")
            show_labels = opt2.checkbox("顯示 Batch 標籤", value=False, key="ct_labels")
            point_sz    = opt3.slider("散點大小", 40, 300, 120, 20, key="ct_ptsize")

            # ── 區間設定 ──────────────────────────────────────
            st.markdown("#### 🎯 設定分類區間")

            # 快速預設：三分位
            if st.button("⚡ 自動填入三等分區間", key="ct_auto_zones"):
                q33 = round(float(cls_series.quantile(0.33)), 3)
                q67 = round(float(cls_series.quantile(0.67)), 3)
                st.session_state["ct_n_zones"] = 3
                st.session_state["ct_z0_min"]  = round(float(cls_min), 3)
                st.session_state["ct_z0_max"]  = q33
                st.session_state["ct_z0_label"] = f"Low (< {q33})"
                st.session_state["ct_z1_min"]  = q33
                st.session_state["ct_z1_max"]  = q67
                st.session_state["ct_z1_label"] = f"Standard ({q33}–{q67})"
                st.session_state["ct_z2_min"]  = q67
                st.session_state["ct_z2_max"]  = round(float(cls_max), 3)
                st.session_state["ct_z2_label"] = f"High (> {q67})"

            n_zones_ct = st.number_input(
                "區間數量", min_value=2, max_value=8, value=3, step=1, key="ct_n_zones"
            )
            CT_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd",
                         "#8c564b", "#e377c2", "#17becf"]

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
                        "顏色", value=CT_COLORS[i % len(CT_COLORS)], key=f"ct_zcolor_{i}"
                    )
                    zones_ct.append({"label": z_label, "min": z_min, "max": z_max, "color": z_color})

            # ── 繪圖 ──────────────────────────────────────────
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
                            )
                            st.pyplot(fig_ct)
                            plt.close()
                            st.session_state["ct_result_df"] = result_df
                            st.session_state["ct_zones_ct"]  = zones_ct
                            st.session_state["ct_classifier_col"] = classifier_col
                            st.session_state["ct_target_col"] = target_col_ct
                        except Exception as e:
                            st.error(f"繪圖失敗：{e}")

            # ── 統計摘要 ──────────────────────────────────────
            if st.session_state.get("ct_result_df") is not None:
                result_df   = st.session_state["ct_result_df"]
                saved_zones = st.session_state.get("ct_zones_ct", [])
                saved_cls   = st.session_state.get("ct_classifier_col", classifier_col)
                saved_tgt   = st.session_state.get("ct_target_col", target_col_ct)

                st.markdown("---")
                st.markdown("#### 📋 各區間統計摘要")

                stats_rows = []
                for z in saved_zones:
                    mask = result_df["_class"] == z["label"]
                    n = mask.sum()
                    if n == 0:
                        continue
                    y_vals = result_df.loc[mask, saved_tgt].dropna()
                    x_vals = result_df.loc[mask, saved_cls].dropna()
                    stats_rows.append({
                        "區間": z["label"],
                        f"{saved_cls[:30]} 範圍": f"{z['min']:.3f} – {z['max']:.3f}",
                        "批次數": int(n),
                        f"{saved_tgt[:30]} 平均": round(y_vals.mean(), 3) if len(y_vals) else None,
                        f"{saved_tgt[:30]} 標準差": round(y_vals.std(), 3) if len(y_vals) else None,
                        f"{saved_tgt[:30]} 最大": round(y_vals.max(), 3) if len(y_vals) else None,
                        f"{saved_tgt[:30]} 最小": round(y_vals.min(), 3) if len(y_vals) else None,
                        f"{saved_cls[:30]} 平均": round(x_vals.mean(), 3) if len(x_vals) else None,
                    })

                outside_mask = result_df["_class"] == "Outside Zones"
                if outside_mask.sum() > 0:
                    y_out = result_df.loc[outside_mask, saved_tgt].dropna()
                    x_out = result_df.loc[outside_mask, saved_cls].dropna()
                    stats_rows.append({
                        "區間": "Outside Zones",
                        f"{saved_cls[:30]} 範圍": "區間外",
                        "批次數": int(outside_mask.sum()),
                        f"{saved_tgt[:30]} 平均": round(y_out.mean(), 3) if len(y_out) else None,
                        f"{saved_tgt[:30]} 標準差": round(y_out.std(), 3) if len(y_out) else None,
                        f"{saved_tgt[:30]} 最大": round(y_out.max(), 3) if len(y_out) else None,
                        f"{saved_tgt[:30]} 最小": round(y_out.min(), 3) if len(y_out) else None,
                        f"{saved_cls[:30]} 平均": round(x_out.mean(), 3) if len(x_out) else None,
                    })

                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    tgt_avg_col = f"{saved_tgt[:30]} 平均"
                    st.dataframe(
                        stats_df.style.background_gradient(
                            cmap="RdYlGn", subset=[tgt_avg_col]
                        ) if tgt_avg_col in stats_df.columns else stats_df,
                        width="stretch", hide_index=True,
                    )

                if st.checkbox("顯示各批次明細", key="ct_show_detail"):
                    detail_cols = (
                        ["BatchID", saved_cls, saved_tgt, "_class"]
                        if "BatchID" in result_df.columns
                        else [saved_cls, saved_tgt, "_class"]
                    )
                    detail_df = result_df[[c for c in detail_cols if c in result_df.columns]].copy()
                    detail_df = detail_df.rename(columns={"_class": "分類"})
                    st.dataframe(detail_df.reset_index(drop=True), width="stretch", hide_index=True)
