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
    else:
        ax.set_xlabel("Batch Sequence", fontsize=8)

    title = f"{title_prefix}{col[:60]}"
    ax.set_title("\n".join(textwrap.wrap(title, width=72)), fontsize=8.5, pad=5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# ── Session-state 工具 ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init_fe_state():
    defaults = {
        "fe_base_df":    None,
        "clean_df":      None,
        "fe_op_log":     [],
        "fe_stat_dropped": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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
                st.toast(f"✅ 已還原欄位：{col}", icon="↩️")
            return
        if col in entry["cols_added"]:
            if col in cdf.columns:
                cdf = cdf.drop(columns=[col])
                st.session_state["clean_df"] = cdf
                st.toast(f"✅ 已移除新增欄位：{col}", icon="↩️")
            return
    st.toast(f"⚠️ 找不到 {col} 的操作記錄", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# ── 渲染組件 ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_changed_cols(df_before, df_after, cols_removed, cols_added, source_df_for_removed, section_title):
    if not cols_removed and not cols_added:
        st.info("本次操作無欄位變更。")
        return
    st.markdown(f"#### {section_title}")
    
    if cols_removed:
        with st.expander(f"🗑️ 查看已刪除欄位 ({len(cols_removed)})", expanded=False):
            for col in cols_removed:
                c_l, c_r = st.columns([5, 1])
                with c_l: 
                    st.caption(f"**{col}**")
                with c_r:
                    if st.button("↩️", key=f"undo_rm_{col}_{hash(col)}"):
                        _undo_col(col)
                        st.rerun()

    if cols_added:
        with st.expander(f"➕ 查看新增欄位 ({len(cols_added)})", expanded=False):
            for col in cols_added:
                c_l, c_r = st.columns([5, 1])
                with c_l: st.caption(f"**{col}**")
                with c_r:
                    if st.button("↩️", key=f"undo_add_{col}_{hash(col)}"):
                        _undo_col(col)
                        st.rerun()

def _render_history_panel():
    log = st.session_state.get("fe_op_log", [])
    if not log:
        st.caption("尚無操作記錄。")
        return
    for i, entry in enumerate(reversed(log)):
        idx = len(log) - 1 - i
        with st.expander(f"**#{idx+1}** {entry['op_type']}", expanded=False):
            if st.button(f"↩️ 整批反悔", key=f"undo_batch_{idx}"):
                st.session_state["clean_df"] = entry["snapshot"].copy()
                st.session_state["fe_op_log"] = st.session_state["fe_op_log"][:idx]
                st.rerun()

def render(selected_process_df):
    _init_fe_state()
    st.header("特徵工程 & 清理")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    main_col, hist_col = st.columns([3, 1])
    with hist_col:
        st.markdown("### 📜 操作歷史")
        _render_history_panel()
        if st.button("🔄 完全重置", type="secondary", use_container_width=True):
            st.session_state["clean_df"] = st.session_state["fe_base_df"]
            st.session_state["fe_op_log"] = []
            st.rerun()

    with main_col:
        _render_main(selected_process_df)

# ══════════════════════════════════════════════════════════════════════════════
# ── 主 Render 邏輯 ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_main(selected_process_df):
    # --- Step 1：自動特徵工程 ---
    st.markdown("### 🔧 Step 1：自動特徵工程")
    if st.button("🔧 執行特徵工程", key="run_fe", type="primary"):
        snapshot_before = selected_process_df.copy()
        with st.spinner("處理中..."):
            clean_df, drop_log = clean_process_features_with_log(selected_process_df, id_col="BatchID")
        
        st.session_state["fe_base_df"] = snapshot_before
        st.session_state["clean_df"] = clean_df
        
        removed = sorted(set(snapshot_before.columns) - set(clean_df.columns))
        added = sorted(set(clean_df.columns) - set(snapshot_before.columns))
        _push_op("auto_clean", removed, added, snapshot_before)
        st.rerun() # 🚀 關鍵：執行完立即刷新，進入 Step 2

    # --- 獲取當前狀態 ---
    current_df = st.session_state.get("clean_df")

    if current_df is not None:
        # --- Step 2：統計篩選 ---
        st.markdown("---")
        st.markdown("### 📉 Step 2：統計篩選")
        c1, c2, c3 = st.columns(3)
        cv_thresh = c1.slider("CV 門檻", 0.0, 0.1, 0.01, 0.001, format="%.3f")
        jump_thresh = c2.slider("Jump Ratio 門檻", 0.1, 1.0, 0.5, 0.05)
        acf_thresh = c3.slider("ACF 門檻", 0.0, 0.5, 0.2, 0.05)

        if st.button("📉 執行統計篩選", type="primary"):
            snapshot_before = current_df.copy()
            with st.spinner("篩選中..."):
                filtered_df, dropped_info = filter_columns_by_stats(
                    current_df, cv_threshold=cv_thresh, 
                    jump_ratio_threshold=jump_thresh, acf_threshold=acf_thresh
                )
                if "BatchID" in current_df.columns and "BatchID" not in filtered_df.columns:
                    filtered_df.insert(0, "BatchID", current_df["BatchID"])
            
            st.session_state["clean_df"] = filtered_df
            removed = sorted(set(snapshot_before.columns) - set(filtered_df.columns))
            _push_op("stat_filter", removed, [], snapshot_before, reason_map=dropped_info)
            st.rerun() # 🚀 關鍵：執行完立即刷新，更新 Step 3

        # --- Step 3：當前特徵總覽 ---
        st.markdown("---")
        st.markdown("### 📊 Step 3：當前特徵總覽 & 個別管理")
        
        numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
        non_batch_num = [c for c in numeric_cols if c != "BatchID"]

        m1, m2, m3 = st.columns(3)
        m1.metric("當前欄位數", current_df.shape[1])
        m2.metric("數值欄位數", len(non_batch_num))
        m3.metric("批次數", current_df.shape[0])

        search_q = st.text_input("🔍 搜尋欄位", placeholder="輸入關鍵字...")
        display_cols = [c for c in non_batch_num if search_q.lower() in c.lower()] if search_q else non_batch_num

        for col in display_cols:
            with st.expander(f"**{col[:75]}**", expanded=False):
                el, er = st.columns([5, 1])
                with el:
                    try:
                        fig = _mini_trend(current_df, col)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    except: st.caption("暫無圖表資料")
                with er:
                    st.write("")
                    if st.button("🗑️ 刪除", key=f"drop_{col}"):
                        snap = current_df.copy()
                        st.session_state["clean_df"] = current_df.drop(columns=[col])
                        _push_op("manual_drop", [col], [], snap)
                        st.rerun()
