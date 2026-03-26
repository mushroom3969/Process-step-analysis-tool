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
    if col not in df.columns:
        return None
    
    plot_df = df.copy()
    if "BatchID" in plot_df.columns:
        plot_df["_sort"] = plot_df["BatchID"].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True)
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    vals = pd.to_numeric(plot_df[col], errors='coerce').values
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
    
    ax.set_title(f"{title_prefix}{col[:60]}", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# ── Session-state 初始化 ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init_fe_state():
    if "fe_op_log" not in st.session_state:
        st.session_state["fe_op_log"] = []
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = None

def _push_op(op_type: str, cols_removed: list[str], cols_added: list[str],
             snapshot_before: pd.DataFrame, reason_map: dict = None):
    st.session_state["fe_op_log"].append({
        "op_type":       op_type,
        "cols_removed":  cols_removed,
        "cols_added":    cols_added,
        "reason_map":    reason_map or {},
        "snapshot":      snapshot_before.copy(),
    })

# ══════════════════════════════════════════════════════════════════════════════
# ── Render 邏輯 ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    _init_fe_state()
    st.header("特徵工程 & 清理")
    
    # 防呆檢查
    if selected_process_df is None or (isinstance(selected_process_df, pd.DataFrame) and selected_process_df.empty):
        st.warning("⚠️ 尚未偵測到製程數據，請確認資料來源。")
        return

    main_col, hist_col = st.columns([3, 1])

    with hist_col:
        st.markdown("### 📜 操作歷史")
        log = st.session_state.get("fe_op_log", [])
        if not log:
            st.caption("尚無紀錄")
        for i, entry in enumerate(reversed(log)):
            idx = len(log) - 1 - i
            with st.expander(f"**#{idx+1}** {entry['op_type']}"):
                if st.button("↩️ 整批還原", key=f"undo_batch_{idx}"):
                    st.session_state["clean_df"] = entry["snapshot"].copy()
                    st.session_state["fe_op_log"] = st.session_state["fe_op_log"][:idx]
                    st.rerun()
        
        if st.button("🔄 完全重置", type="secondary", use_container_width=True):
            st.session_state["clean_df"] = None
            st.session_state["fe_op_log"] = []
            st.rerun()

    with main_col:
        # ── Step 1：自動特徵工程 ──
        st.markdown("### 🔧 Step 1：自動特徵工程")
        if st.button("🔧 執行特徵工程", key="run_fe", type="primary"):
            snapshot_before = selected_process_df.copy()
            with st.spinner("清理中..."):
                # 確保傳入的是 DataFrame
                clean_df, _ = clean_process_features_with_log(selected_process_df, id_col="BatchID")
            
            st.session_state["clean_df"] = clean_df
            removed = sorted(set(snapshot_before.columns) - set(clean_df.columns))
            added = sorted(set(clean_df.columns) - set(snapshot_before.columns))
            _push_op("auto_clean", removed, added, snapshot_before)
            st.rerun()

        # ── 獲取最新狀態 ──
        current_df = st.session_state.get("clean_df")
        
        if current_df is not None:
            st.markdown("---")
            st.markdown("### 📉 Step 2：統計篩選")
            
            sc1, sc2, sc3 = st.columns(3)
            cv_t = sc1.slider("CV 門檻", 0.0, 0.1, 0.01, 0.001, format="%.3f", key="s2_cv_val")
            jr_t = sc2.slider("Jump Ratio 門檻", 0.1, 1.0, 0.5, 0.05, key="s2_jr_val")
            ac_t = sc3.slider("ACF 門檻", 0.0, 0.5, 0.2, 0.05, key="s2_ac_val")

            if st.button("📉 執行統計篩選", key="btn_stat_filter", type="primary"):
                snapshot_before = current_df.copy()
                with st.spinner("計算中..."):
                    filtered_df, dropped_info = filter_columns_by_stats(
                        current_df, cv_threshold=cv_t, 
                        jump_ratio_threshold=jr_t, acf_threshold=ac_t
                    )
                    # 確保 ID 存在
                    if "BatchID" in current_df.columns and "BatchID" not in filtered_df.columns:
                        filtered_df.insert(0, "BatchID", current_df["BatchID"])

                st.session_state["clean_df"] = filtered_df
                removed = sorted(set(snapshot_before.columns) - set(filtered_df.columns))
                _push_op("stat_filter", removed, [], snapshot_before, reason_map=dropped_info)
                st.rerun()

            st.markdown("---")
            st.markdown("### 📊 Step 3：特徵總覽")
            
            num_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
            feat_cols = [c for c in num_cols if c != "BatchID"]

            m1, m2, m3 = st.columns(3)
            m1.metric("總欄位", current_df.shape[1])
            m2.metric("特徵數", len(feat_cols))
            m3.metric("批次數", current_df.shape[0])

            search = st.text_input("🔍 搜尋欄位", key="search_feat")
            display_list = [c for c in feat_cols if search.lower() in c.lower()] if search else feat_cols

            for col in display_list:
                with st.expander(f"📈 {col}", expanded=False):
                    el, er = st.columns([5, 1])
                    with el:
                        fig = _mini_trend(current_df, col)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.caption("無效數據")
                    with er:
                        if st.button("🗑️", key=f"del_{col}"):
                            snap = current_df.copy()
                            st.session_state["clean_df"] = current_df.drop(columns=[col])
                            _push_op("manual_drop", [col], [], snap)
                            st.rerun()
