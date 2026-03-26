"""Tab — 特徵工程 & 清理（含逐欄預覽 + 反悔功能）"""
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
                title_prefix: str = "", height: float = 2.8,
                show_mean: bool = True) -> plt.Figure:
    """
    單欄 batch 時序折線圖，含平均線與 ±1σ 色帶。
    X 軸依 BatchID 數字排序（若有）。
    show_mean: 是否顯示平均線與 ±1σ 色帶。
    """
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

    if show_mean:
        # ±1σ 色帶
        ax.fill_between(seq, mu - sigma, mu + sigma,
                        alpha=0.12, color=color, zorder=0, label="±1σ")
        # 平均線
        ax.axhline(mu, color=color, linewidth=1.2, linestyle="--",
                   alpha=0.75, label=f"Mean: {mu:.4f}", zorder=1)

    # 折線 + 散點
    ax.plot(seq, vals, color=color, linewidth=1.5, zorder=2)
    ax.scatter(seq, vals, color=color, s=25, zorder=3,
               edgecolors="white", linewidths=0.4)

    # X 軸標籤
    if "BatchID" in plot_df.columns and len(plot_df) <= 80:
        ax.set_xticks(seq)
        ax.set_xticklabels([str(b)[-6:] for b in plot_df["BatchID"]],
                           rotation=90, fontsize=6)
    else:
        ax.set_xlabel("Batch Sequence", fontsize=8)

    title = f"{title_prefix}{col[:60]}"
    ax.set_title("\n".join(textwrap.wrap(title, width=72)), fontsize=8.5, pad=5)
    ax.set_ylabel("Value", fontsize=8)
    if show_mean:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ── Session-state 工具 ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init_fe_state():
    """確保所有特徵工程相關 session_state 鍵都已初始化。"""
    defaults = {
        "fe_base_df":        None,
        "clean_df":          None,
        "fe_op_log":         [],
        "fe_stat_dropped":   {},
        # BUG FIX: store stat filter results separately so rerun doesn't re-trigger
        "fe_stat_result":    None,
        "fe_auto_result":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _push_op(op_type: str, cols_removed: list, cols_added: list,
             snapshot_before: pd.DataFrame, reason_map: dict = None):
    st.session_state["fe_op_log"].append({
        "op_type":       op_type,
        "cols_removed":  cols_removed,
        "cols_added":    cols_added,
        "reason_map":    reason_map or {},
        "snapshot":      snapshot_before.copy(),
    })


def _undo_col(col: str):
    log   = st.session_state["fe_op_log"]
    cdf   = st.session_state["clean_df"]

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

    st.toast(f"⚠️ 找不到 {col} 的操作記錄", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# ── 欄位變更展示區 ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_changed_cols(df_before, df_after, cols_removed, cols_added,
                          source_df_for_removed, section_title, show_mean=True):
    if not cols_removed and not cols_added:
        st.info("本次操作無欄位變更。")
        return

    import hashlib as _hl
    _sh = _hl.md5(section_title.encode()).hexdigest()[:10]
    def _k(pfx, col, idx):
        return f"fi_{pfx}_{_sh}_{idx}_" + _hl.md5(col.encode()).hexdigest()[:10]

    st.markdown(f"#### {section_title}")
    st.caption(
        f"共 **{len(cols_removed)}** 個欄位被刪除／{len(cols_added)} 個欄位被新增。"
        "　點擊「↩️ 反悔」可逐欄還原。"
    )

    if cols_removed:
        st.markdown("##### 🗑️ 被刪除的欄位")
        for i, col in enumerate(cols_removed):
            with st.expander(f"🗑️  **{col[:70]}**", expanded=False):
                c_left, c_right = st.columns([5, 1])
                with c_left:
                    if col in source_df_for_removed.columns:
                        try:
                            fig = _mini_trend(source_df_for_removed, col,
                                              color="#e84855",
                                              title_prefix="[刪除前] ",
                                              show_mean=show_mean)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        except Exception as e:
                            st.caption(f"無法繪圖：{e}")
                    else:
                        st.caption("（欄位已不存在，無法繪圖）")
                with c_right:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("↩️ 反悔", key=_k("rm", col, i),
                                 help=f"還原欄位：{col}", type="secondary"):
                        _undo_col(col)
                        st.rerun()

    if cols_added:
        st.markdown("##### ➕ 新增的欄位")
        for i, col in enumerate(cols_added):
            with st.expander(f"➕  **{col[:70]}**", expanded=False):
                c_left, c_right = st.columns([5, 1])
                with c_left:
                    cdf = st.session_state.get("clean_df")
                    _src = cdf if cdf is not None and col in cdf.columns else df_after
                    if col in _src.columns:
                        try:
                            fig = _mini_trend(_src, col, color="#2ca02c",
                                              title_prefix="[新增] ",
                                              show_mean=show_mean)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        except Exception as e:
                            st.caption(f"無法繪圖：{e}")
                    else:
                        st.caption("（欄位已不存在）")
                with c_right:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("↩️ 反悔", key=_k("add", col, i),
                                 help=f"移除新增欄位：{col}", type="secondary"):
                        _undo_col(col)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ── 操作歷史面板 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_history_panel():
    log = st.session_state.get("fe_op_log", [])
    if not log:
        st.caption("尚無操作記錄。")
        return

    for i, entry in enumerate(reversed(log)):
        idx = len(log) - 1 - i
        op_label = {
            "auto_clean":   "🔧 自動特徵工程",
            "stat_filter":  "📉 統計篩選",
            "manual_drop":  "🗑️ 手動移除",
        }.get(entry["op_type"], entry["op_type"])

        n_rm = len(entry["cols_removed"])
        n_add = len(entry["cols_added"])
        label_parts = []
        if n_rm:  label_parts.append(f"-{n_rm}")
        if n_add: label_parts.append(f"+{n_add}")
        badge = " | ".join(label_parts)

        with st.expander(f"**#{idx+1}** {op_label}  `{badge}`", expanded=False):
            if entry["cols_removed"]:
                st.markdown("🗑️ " + "、".join(
                    [c[:30] for c in entry["cols_removed"][:8]]
                ) + ("..." if len(entry["cols_removed"]) > 8 else ""))
            if entry["cols_added"]:
                st.markdown("➕ " + "、".join(
                    [c[:30] for c in entry["cols_added"][:8]]
                ) + ("..." if len(entry["cols_added"]) > 8 else ""))

            if st.button(f"↩️ 整批反悔此操作", key=f"undo_batch_{idx}",
                         type="secondary"):
                st.session_state["clean_df"] = entry["snapshot"].copy()
                st.session_state["fe_op_log"] = st.session_state["fe_op_log"][:idx]
                st.session_state["fe_stat_result"] = None
                st.session_state["fe_auto_result"]  = None
                st.toast(f"✅ 已整批還原操作 #{idx+1}", icon="↩️")
                st.rerun()

    if st.button("🔄 完全重置（回到特徵工程前）",
                 key="fe_full_reset", type="secondary"):
        base = st.session_state.get("fe_base_df")
        if base is not None:
            st.session_state["clean_df"] = base.copy()
            st.session_state["fe_op_log"] = []
            st.session_state["fe_stat_dropped"] = {}
            st.session_state["fe_stat_result"] = None
            st.session_state["fe_auto_result"]  = None
            st.toast("✅ 已完全重置", icon="🔄")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ── 主 render ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    _init_fe_state()
    st.header("特徵工程 & 清理")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return
        # 全域平均線顯示控制
    show_mean = st.checkbox("📊 圖表顯示平均線與 ±1σ 色帶", value=True, key="fe_show_mean")

    main_col, hist_col = st.columns([3, 1])

    with hist_col:
        st.markdown("### 📜 操作歷史")
        _render_history_panel()

    with main_col:
        _render_main(selected_process_df, show_mean)


def _render_main(selected_process_df):
    """主操作區"""

    # 1. 初始化檢查：如果換了製程步驟，自動清空舊的清理結果（可選）
    # if st.session_state.get("last_df_id") != id(selected_process_df):
    #     st.session_state["clean_df"] = None
    #     st.session_state["last_df_id"] = id(selected_process_df)

    # ══════════════════════════════════════════════════════════
    # ── Step 1：自動特徵工程
    # ══════════════════════════════════════════════════════════
    st.markdown("### 🔧 Step 1：自動特徵工程")
    
    # 即使已經執行過，也可以重新執行
    if st.button("🔧 執行特徵工程", key="run_fe", type="primary"):
        snapshot_before = selected_process_df.copy()
        with st.spinner("處理中..."):
            clean_df, _ = clean_process_features_with_log(selected_process_df, id_col="BatchID")
        
        # 將結果永久存入 Session State
        st.session_state["fe_base_df"] = snapshot_before
        st.session_state["clean_df"] = clean_df
        st.session_state["fe_op_log"] = [] # 重置日誌
        
        st.success("Step 1 完成！")
        st.rerun() # 強制刷新，讓下方的 Step 2 偵測到 clean_df 已存在

    # ══════════════════════════════════════════════════════════
    # ── Step 2 & 3：只有當 Step 1 的結果存在時才顯示
    # ══════════════════════════════════════════════════════════
    if st.session_state.get("clean_df") is not None:
        current_df = st.session_state["clean_df"]
        
        st.markdown("---")
        st.markdown("### 📉 Step 2：統計篩選")
        
        # 使用 columns 放置滑桿
        c1, c2, c3 = st.columns(3)
        cv_t = c1.slider("CV 門檻", 0.0, 0.1, 0.01, 0.001, format="%.3f", key="s2_cv")
        jr_t = c2.slider("Jump Ratio 門檻", 0.1, 1.0, 0.5, 0.05, key="s2_jr")
        ac_t = c3.slider("ACF 門檻", 0.0, 0.5, 0.2, 0.05, key="s2_ac")

        # 注意：這個按鈕現在與 Step 1 是平級的，不會互相干擾
        if st.button("📉 執行統計篩選", key="do_stat_filter", type="primary"):
            snapshot_before = current_df.copy()
            with st.spinner("統計計算中..."):
                filtered_df, dropped_info = filter_columns_by_stats(
                    current_df, 
                    cv_threshold=cv_t, 
                    jump_ratio_threshold=jr_t, 
                    acf_threshold=ac_t
                )
                
                # 補回 ID 欄位
                if "BatchID" in current_df.columns and "BatchID" not in filtered_df.columns:
                    filtered_df.insert(0, "BatchID", current_df["BatchID"])

            # 更新 Session State 中的資料
            st.session_state["clean_df"] = filtered_df
            
            # 紀錄操作
            removed = sorted(set(snapshot_before.columns) - set(filtered_df.columns))
            _push_op("stat_filter", removed, [], snapshot_before, reason_map=dropped_info)
            
            st.toast("統計篩選完成！", icon="📉")
            st.rerun() # 再次刷新，更新 Step 3 的展示內容

        # ══════════════════════════════════════════════════════════
        # ── Step 3：總覽區
        # ══════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📊 Step 3：當前特徵總覽")

        numeric_cols = clean_df.select_dtypes(include=["number"]).columns.tolist()
        non_batch_num = [c for c in numeric_cols if c != "BatchID"]
        
        # 佈局 A: 數據指標
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("當前總欄位", clean_df.shape[1])
            m2.metric("數值特徵", len(non_batch_num))
            m3.metric("樣本批次數", clean_df.shape[0])

        if not non_batch_num:
            st.warning("目前沒有可顯示的數值欄位。")
            return

        # 佈局 B: 搜尋過濾 (修正縮進)
        search_q = st.text_input("🔍 搜尋欄位名稱", key="fe_search", placeholder="輸入關鍵字...")
        
        display_cols = (
            [c for c in non_batch_num if search_q.lower() in c.lower()]
            if search_q.strip() else non_batch_num
        )
        st.caption(f"顯示 {len(display_cols)} / {len(non_batch_num)} 個欄位")

        # 準備記錄 (用於標籤顯示)
        all_added = {c for entry in op_log for c in entry.get("cols_added", [])}
        all_removed_from_base = {c for entry in op_log for c in entry.get("cols_removed", [])}

        # 佈局 C: 逐欄清單
        import hashlib as _hl_ov
        for col in display_cols:
            _ch = _hl_ov.md5(col.encode()).hexdigest()[:12]
            badge = " :blue-background[🆕 新增]" if col in all_added else ""
            
            with st.expander(f"**{col[:75]}**{badge}", expanded=False):
                exp_l, exp_r = st.columns([5, 1])

                with exp_l:
                    s = clean_df[col].dropna()
                    # 統計小卡
                    stat_vals = {
                        "n": len(s),
                        "Mean": f"{s.mean():.3f}",
                        "SD": f"{s.std():.3f}",
                        "Max": f"{s.max():.2f}"
                    }
                    st.caption(f" 📊 Stats: " + " | ".join([f"**{k}**: {v}" for k, v in stat_vals.items()]))
                    
                    # 繪圖
                    color = "#2ca02c" if col in all_added else "#2e86ab"
                    try:
                        fig = _mini_trend(clean_df, col, color=color, show_mean=show_mean)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.error(f"繪圖失敗: {e}")

                with exp_r:
                    st.write("") # 調整按鈕垂直位置
                    if col in all_added:
                        if st.button("↩️ 撤銷", key=f"fi_ov_undo_{_ch}", use_container_width=True):
                            _undo_col(col)
                            st.rerun()

                    if st.button("🗑️ 刪除", key=f"fi_ov_drop_{_ch}", type="secondary", use_container_width=True):
                        snapshot_before = clean_df.copy()
                        st.session_state["clean_df"] = clean_df.drop(columns=[col])
                        _push_op("manual_drop", [col], [], snapshot_before)
                        st.toast(f"已刪除：{col}")
                        st.rerun()

