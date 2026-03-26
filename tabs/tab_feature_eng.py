"""Tab 2 — 特徵工程 & 清理（含逐欄預覽 + 反悔功能）"""
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
import seaborn as sns
import streamlit as st

from utils import (
    clean_process_features_with_log,
    filter_columns_by_stats,
    extract_number,
)

# ══════════════════════════════════════════════════════════════════════════════
# ── 趨勢迷你圖 ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _mini_trend(df: pd.DataFrame, col: str, color: str = "#2e86ab",
                title_prefix: str = "", height: float = 2.8) -> plt.Figure:
    plot_df = df.copy()
    if "BatchID" in plot_df.columns:
        plot_df["_sort"] = plot_df["BatchID"].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True)
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    vals = plot_df[col].values.astype(float)
    seq  = plot_df["_seq"].values

    fig, ax = plt.subplots(figsize=(10, height))
    sns.set_style("whitegrid")

    mu    = np.nanmean(vals)
    sigma = np.nanstd(vals, ddof=1)
    ax.fill_between(seq, mu - sigma, mu + sigma, alpha=0.12, color=color, zorder=0, label="±1σ")
    ax.axhline(mu, color=color, linewidth=1.2, linestyle="--", alpha=0.75, label=f"Mean: {mu:.4f}", zorder=1)
    ax.plot(seq, vals, color=color, linewidth=1.5, zorder=2)
    ax.scatter(seq, vals, color=color, s=25, zorder=3, edgecolors="white", linewidths=0.4)

    if "BatchID" in plot_df.columns and len(plot_df) <= 80:
        ax.set_xticks(seq)
        ax.set_xticklabels([str(b)[-6:] for b in plot_df["BatchID"]], rotation=90, fontsize=6)
    
    title = f"{title_prefix}{col[:60]}"
    ax.set_title("\n".join(textwrap.wrap(title, width=72)), fontsize=8.5, pad=5)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# ── Session-state 工具 ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init_fe_state():
    if "fe_op_log" not in st.session_state:
        st.session_state["fe_op_log"] = []
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = None
    if "fe_base_df" not in st.session_state:
        st.session_state["fe_base_df"] = None

def _push_op(op_type: str, cols_removed: list[str], cols_added: list[str],
             snapshot_before: pd.DataFrame, reason_map: dict = None):
    st.session_state["fe_op_log"].append({
        "op_type":       op_type,
        "cols_removed":  cols_removed,
        "cols_added":    cols_added,
        "reason_map":    reason_map or {},
        "snapshot":      snapshot_before.copy(),
    })

def _undo_col(col: str):
    log = st.session_state["fe_op_log"]
    cdf = st.session_state["clean_df"]
    for entry in reversed(log):
        if col in entry["cols_removed"]:
            snap = entry["snapshot"]
            if col in snap.columns:
                cdf = cdf.copy()
                cdf[col] = snap[col].values
                st.session_state["clean_df"] = cdf
                st.toast(f"✅ 已還原：{col}")
            return
        if col in entry["cols_added"]:
            if col in cdf.columns:
                st.session_state["clean_df"] = cdf.drop(columns=[col])
                st.toast(f"✅ 已移除：{col}")
            return

# ══════════════════════════════════════════════════════════════════════════════
# ── 主界面渲染 ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    _init_fe_state()
    st.header("特徵工程 & 清理")
    
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    main_col, hist_col = st.columns([3, 1])

    with hist_col:
        st.markdown("### 📜 操作歷史")
        log = st.session_state["fe_op_log"]
        if not log:
            st.caption("尚無紀錄")
        for i, entry in enumerate(reversed(log)):
            idx = len(log) - 1 - i
            with st.expander(f"#{idx+1} {entry['op_type']}"):
                if st.button("↩️ 整批還原", key=f"undo_batch_{idx}"):
                    st.session_state["clean_df"] = entry["snapshot"].copy()
                    st.session_state["fe_op_log"] = st.session_state["fe_op_log"][:idx]
                    st.rerun()
        
        if st.button("🔄 完全重置", type="secondary", use_container_width=True):
            st.session_state["clean_df"] = None
            st.session_state["fe_op_log"] = []
            st.rerun()

    with main_col:
        _render_main_logic(selected_process_df)

def _render_main_logic(selected_process_df):
    # ── Step 1：自動特徵工程 ──
    st.markdown("### 🔧 Step 1：自動特徵工程")
    if st.button("🔧 執行特徵工程", key="run_fe", type="primary"):
        snapshot_before = selected_process_df.copy()
        with st.spinner("處理中..."):
            clean_df, _ = clean_process_features_with_log(selected_process_df, id_col="BatchID")
        
        st.session_state["fe_base_df"] = snapshot_before
        st.session_state["clean_df"] = clean_df
        
        removed = sorted(set(snapshot_before.columns) - set(clean_df.columns))
        added = sorted(set(clean_df.columns) - set(snapshot_before.columns))
        _push_op("auto_clean", removed, added, snapshot_before)
        st.rerun()

    # ── Step 2 & 3：判斷 clean_df 是否存在 ──
    current_df = st.session_state.get("clean_df")
    
    if current_df is not None:
        st.markdown("---")
        st.markdown("### 📉 Step 2：統計篩選")
        
        c1, c2, c3 = st.columns(3)
        cv_t = c1.slider("CV 門檻", 0.0, 0.1, 0.01, 0.001, format="%.3f", key="s2_cv")
        jr_t = c2.slider("Jump Ratio 門檻", 0.1, 1.0, 0.5, 0.05, key="s2_jr")
        ac_t = c3.slider("ACF 門檻", 0.0, 0.5, 0.2, 0.05, key="s2_ac")

        if st.button("📉 執行統計篩選", key="do_stat_filter", type="primary"):
            snapshot_before = current_df.copy()
            with st.spinner("篩選中..."):
                filtered_df, dropped_info = filter_columns_by_stats(
                    current_df, cv_threshold=cv_t, 
                    jump_ratio_threshold=jr_t, acf_threshold=ac_t
                )
                if "BatchID" in current_df.columns and "BatchID" not in filtered_df.columns:
                    filtered_df.insert(0, "BatchID", current_df["BatchID"])

            st.session_state["clean_df"] = filtered_df
            removed = sorted(set(snapshot_before.columns) - set(filtered_df.columns))
            _push_op("stat_filter", removed, [], snapshot_before, reason_map=dropped_info)
            st.rerun()

        # ── Step 3：當前特徵總覽 ──
        st.markdown("---")
        st.markdown("### 📊 Step 3：當前特徵總覽 & 個別管理")
        
        numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
        non_batch_num = [c for c in numeric_cols if c != "BatchID"]

        m1, m2, m3 = st.columns(3)
        m1.metric("總欄位數", current_df.shape[1])
        m2.metric("數值欄位", len(non_batch_num))
        m3.metric("樣本數", current_df.shape[0])

        search_q = st.text_input("🔍 搜尋欄位關鍵字", key="fe_search_box")
        display_cols = [c for c in non_batch_num if search_q.lower() in c.lower()] if search_q else non_batch_num

        for col in display_cols:
            with st.expander(f"**{col[:75]}**"):
                el, er = st.columns([5, 1])
                with el:
                    try:
                        fig = _mini_trend(current_df, col)
                        st.pyplot(fig)
                        plt.close()
                    except: st.caption("繪圖失敗")
                with er:
                    st.write("")
                    if st.button("🗑️ 刪除", key=f"drop_{col}"):
                        snap = current_df.copy()
                        st.session_state["clean_df"] = current_df.drop(columns=[col])
                        _push_op("manual_drop", [col], [], snap)
                        st.rerun()

