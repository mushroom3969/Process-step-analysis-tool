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
    """
    單欄 batch 時序折線圖，含平均線與 ±1σ 色帶。
    X 軸依 BatchID 數字排序（若有）。
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

    # ±1σ 色帶
    mu    = np.nanmean(vals)
    sigma = np.nanstd(vals, ddof=1)
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
        "fe_base_df":        None,   # 特徵工程前的「基底」DataFrame
        "clean_df":          None,   # 當前有效的 DataFrame
        "fe_op_log":         [],     # 操作日誌 list[dict]
        "fe_stat_dropped":   {},     # 統計篩選被剔除的欄位 {col: reason}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _push_op(op_type: str, cols_removed: list[str], cols_added: list[str],
             snapshot_before: pd.DataFrame, reason_map: dict = None):
    """
    記錄一次操作到 fe_op_log。
    snapshot_before：操作前的完整 DataFrame（用於反悔時復原）。
    """
    st.session_state["fe_op_log"].append({
        "op_type":       op_type,
        "cols_removed":  cols_removed,
        "cols_added":    cols_added,
        "reason_map":    reason_map or {},
        "snapshot":      snapshot_before.copy(),
    })


def _undo_col(col: str):
    """
    反悔單一欄位：
    - 若該欄位是「被刪除」的 → 把它加回 clean_df
    - 若該欄位是「被新增」的 → 把它從 clean_df 移除
    掃描 op_log 找到最近一次涉及 col 的操作，做逆操作。
    """
    log   = st.session_state["fe_op_log"]
    cdf   = st.session_state["clean_df"]
    base  = st.session_state["fe_base_df"]

    # 找最近一筆涉及 col 的紀錄
    for entry in reversed(log):
        if col in entry["cols_removed"]:
            # col 被刪除 → 從 snapshot_before 找回原始資料
            snap = entry["snapshot"]
            if col in snap.columns:
                cdf = cdf.copy()
                cdf[col] = snap[col].values
                st.session_state["clean_df"] = cdf
                st.toast(f"✅ 已還原欄位：{col}", icon="↩️")
            return
        if col in entry["cols_added"]:
            # col 是新增的 → 直接刪掉
            if col in cdf.columns:
                cdf = cdf.drop(columns=[col])
                st.session_state["clean_df"] = cdf
                st.toast(f"✅ 已移除新增欄位：{col}", icon="↩️")
            return

    st.toast(f"⚠️ 找不到 {col} 的操作記錄", icon="⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# ── 欄位變更展示區 ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_changed_cols(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cols_removed: list[str],
    cols_added: list[str],
    source_df_for_removed: pd.DataFrame,
    section_title: str,
):
    """
    顯示本次操作的變更欄位清單，每個欄位有：
    - 類型標籤（🗑️ 已刪除 / ➕ 已新增）
    - batch 趨勢迷你圖
    - 反悔按鈕
    """
    if not cols_removed and not cols_added:
        st.info("本次操作無欄位變更。")
        return

    st.markdown(f"#### {section_title}")
    st.caption(
        f"共 **{len(cols_removed)}** 個欄位被刪除／{len(cols_added)} 個欄位被新增。"
        "　點擊「↩️ 反悔」可逐欄還原。"
    )

    # ── 被刪除的欄位 ──────────────────────────────────────────
    if cols_removed:
        st.markdown("##### 🗑️ 被刪除的欄位")
        for col in cols_removed:
            with st.expander(f"🗑️  **{col[:70]}**", expanded=False):
                c_left, c_right = st.columns([5, 1])
                with c_left:
                    # 用 source_df_for_removed（操作前的 df）畫圖
                    if col in source_df_for_removed.columns:
                        try:
                            fig = _mini_trend(source_df_for_removed, col,
                                              color="#e84855",
                                              title_prefix="[刪除前] ")
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        except Exception as e:
                            st.caption(f"無法繪圖：{e}")
                    else:
                        st.caption("（欄位已不存在，無法繪圖）")
                with c_right:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("↩️ 反悔", key=f"undo_{col}_{hash(col)}",
                                 help=f"還原欄位：{col}", type="secondary"):
                        _undo_col(col)
                        st.rerun()

    # ── 被新增的欄位 ──────────────────────────────────────────
    if cols_added:
        st.markdown("##### ➕ 新增的欄位")
        for col in cols_added:
            with st.expander(f"➕  **{col[:70]}**", expanded=False):
                c_left, c_right = st.columns([5, 1])
                with c_left:
                    cdf = st.session_state.get("clean_df")
                    src = cdf if cdf is not None and col in cdf.columns else df_after
                    if col in src.columns:
                        try:
                            fig = _mini_trend(src, col, color="#2ca02c",
                                              title_prefix="[新增] ")
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        except Exception as e:
                            st.caption(f"無法繪圖：{e}")
                    else:
                        st.caption("（欄位已不存在）")
                with c_right:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("↩️ 反悔", key=f"undo_{col}_{hash(col)}",
                                 help=f"移除新增欄位：{col}", type="secondary"):
                        _undo_col(col)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ── 操作歷史面板 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_history_panel():
    """側邊顯示所有操作歷史與整批反悔按鈕。"""
    log = st.session_state.get("fe_op_log", [])
    if not log:
        st.caption("目前無特徵工程操作記錄。")
        return

    # 建議在 log 的 entry 中加入一個 timestamp 或 uuid 作為唯一標記
    for i, entry in enumerate(reversed(log)):
        # 原始索引位置
        idx = len(log) - 1 - i
        # 建立更強健的唯一 Key
        op_time = entry.get("timestamp", idx)
        unique_key_prefix = f"fe_hist_{idx}_{op_time}"

        op_label = {
            "auto_clean":   "🔧 自動特徵工程",
            "stat_filter":  "📉 統計篩選",
            "manual_drop":  "🗑️ 手動移除",
        }.get(entry["op_type"], entry["op_type"])

        n_rm = len(entry["cols_removed"])
        n_add = len(entry["cols_added"])
        badge = " | ".join(filter(None, [f"-{n_rm}" if n_rm else "", f"+{n_add}" if n_add else ""]))

        with st.expander(f"**#{idx+1}** {op_label}  `{badge}`", expanded=False):
            if entry["cols_removed"]:
                st.markdown("🗑️ " + "、".join([c[:30] for c in entry["cols_removed"][:8]]) + 
                            ("..." if len(entry["cols_removed"]) > 8 else ""))
            if entry["cols_added"]:
                st.markdown("➕ " + "、".join([c[:30] for c in entry["cols_added"][:8]]) + 
                            ("..." if len(entry["cols_added"]) > 8 else ""))

            # 修正關鍵：使用 unique_key 並設定新版寬度參數
            if st.button(f"↩️ 復原至此步驟前", 
                         key=f"undo_btn_{unique_key_prefix}",
                         type="secondary"):
                # 還原邏輯
                st.session_state["clean_df"] = entry["snapshot"].copy()
                st.session_state["fe_op_log"] = st.session_state["fe_op_log"][:idx]
                st.toast(f"✅ 已還原至步驟 #{idx+1} 之前的狀態", icon="↩️")
                st.rerun()

    st.divider()
    if st.button("🔄 完全重置（回到初始狀態）",
                 key="fe_full_reset_global", 
                 type="primary", # 改為 primary 提醒這是重大操作
                 width="stretch"): # 修正原本警告的用詞
        base = st.session_state.get("fe_base_df")
        if base is not None:
            st.session_state["clean_df"] = base.copy()
            st.session_state["fe_op_log"] = []
            st.session_state["fe_stat_dropped"] = {}
            st.toast("✅ 數據已回到特徵工程起始點", icon="🔄")
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

    # ── 版面：主區 + 歷史側欄 ─────────────────────────────────
    main_col, hist_col = st.columns([3, 1])

    with hist_col:
        st.markdown("### 📜 操作歷史")
        _render_history_panel()

    with main_col:
        _render_main(selected_process_df)


def _render_main(selected_process_df):
    """主操作區（左側 3/4）。"""

    # ══════════════════════════════════════════════════════════
    # ── 區塊 A：自動特徵工程 ──────────────────────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown("### 🔧 Step 1：自動特徵工程")
    st.markdown("""
    **自動執行以下規則：**
    - 🗑️ 過濾含有 `Verification Result` / `No (na)` 關鍵字的欄位
    - ➕ 配對 Max/Min、After/Before、End/Start → 計算差值欄位
    - 🔢 數字編號欄位（如 `_1`、`_2`）→ 取平均後合併
    """)

    if st.button("🔧 執行特徵工程", key="run_fe", type="primary"):
        snapshot_before = selected_process_df.copy()
        with st.spinner("處理中..."):
            clean_df, drop_log = clean_process_features_with_log(
                selected_process_df, id_col="BatchID"
            )

        cols_before = set(selected_process_df.columns)
        cols_after  = set(clean_df.columns)
        removed = sorted(cols_before - cols_after)
        added   = sorted(cols_after  - cols_before)

        # 儲存「基底」（特徵工程前）與操作記錄
        st.session_state["fe_base_df"] = snapshot_before
        st.session_state["clean_df"]   = clean_df
        st.session_state["fe_op_log"]  = []   # 重新執行則清空舊記錄
        _push_op("auto_clean", removed, added, snapshot_before,
                 reason_map={r["Column"]: r.get("Reason", "") for _, r in drop_log.iterrows()
                              if "Column" in drop_log.columns})

        st.success(
            f"✅ 完成！從 {selected_process_df.shape[1]} 欄 → {clean_df.shape[1]} 欄"
            f"（刪除 {len(removed)} / 新增 {len(added)}）"
        )

        # 顯示欄位變更
        _render_changed_cols(
            df_before=selected_process_df,
            df_after=clean_df,
            cols_removed=removed,
            cols_added=added,
            source_df_for_removed=snapshot_before,
            section_title="📋 本次變更欄位",
        )

    # ══════════════════════════════════════════════════════════
    # ── 區塊 B：統計篩選 ──────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    if st.session_state.get("clean_df") is not None:
        clean_df = st.session_state["clean_df"]

        st.markdown("---")
        st.markdown("### 📉 Step 2：統計篩選（移除低資訊量欄位）")
        st.caption("篩選條件：CV 過低（接近常數）、Jump Ratio 過高（雜訊）、ACF 過低（無自相關）。")

        c1, c2, c3 = st.columns(3)
        cv_thresh   = c1.slider("CV 門檻（低於此值剔除）",
                                 0.0, 0.1, 0.01, 0.001, format="%.3f", key="fe_cv")
        jump_thresh = c2.slider("Jump Ratio 門檻（高於此值剔除）",
                                 0.1, 1.0, 0.5, 0.05, key="fe_jump")
        acf_thresh  = c3.slider("ACF 門檻（低於此值剔除）",
                                 0.0, 0.5, 0.2, 0.05, key="fe_acf")

        if st.button("📉 執行統計篩選", key="run_stat_filter"):
            snapshot_before = clean_df.copy()
            with st.spinner("篩選中..."):
                filtered_df, dropped_info = filter_columns_by_stats(
                    clean_df,
                    cv_threshold=cv_thresh,
                    jump_ratio_threshold=jump_thresh,
                    acf_threshold=acf_thresh,
                )
                if "BatchID" in clean_df.columns and "BatchID" not in filtered_df.columns:
                    filtered_df.insert(0, "BatchID", clean_df["BatchID"])

            cols_before = set(clean_df.columns)
            cols_after  = set(filtered_df.columns)
            removed = sorted(cols_before - cols_after)
            added   = sorted(cols_after  - cols_before)

            st.session_state["clean_df"]       = filtered_df
            st.session_state["fe_stat_dropped"] = dropped_info
            _push_op("stat_filter", removed, added, snapshot_before,
                     reason_map=dropped_info)

            st.success(
                f"✅ 剔除 {len(removed)} 個欄位 → 剩餘 {filtered_df.shape[1]} 欄"
            )

            # 顯示統計篩選原因表
            if dropped_info:
                reason_df = pd.DataFrame(
                    [(k, v) for k, v in dropped_info.items()],
                    columns=["Column", "Reason"],
                )
                with st.expander("📋 被剔除欄位原因", expanded=True):
                    st.dataframe(reason_df, width="stretch", hide_index=True)

            # 顯示欄位變更（含反悔）
            _render_changed_cols(
                df_before=clean_df,
                df_after=filtered_df,
                cols_removed=removed,
                cols_added=added,
                source_df_for_removed=snapshot_before,
                section_title="📋 本次統計篩選變更欄位",
            )

    # ══════════════════════════════════════════════════════════
    # ── 區塊 C：當前特徵總覽（可個別反悔）────────────────────
    # ══════════════════════════════════════════════════════════
    if st.session_state.get("clean_df") is not None:
        clean_df = st.session_state["clean_df"]
        op_log   = st.session_state.get("fe_op_log", [])

        st.markdown("---")
        st.markdown("### 📊 Step 3：當前特徵總覽 & 個別管理")

        numeric_cols = clean_df.select_dtypes(include=["number"]).columns.tolist()
        non_batch_num = [c for c in numeric_cols if c != "BatchID"]

        # 快速 metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("當前欄位數", clean_df.shape[1])
        m2.metric("數值欄位數", len(non_batch_num))
        m3.metric("批次數", clean_df.shape[0])

        if not non_batch_num:
            st.warning("目前沒有可顯示的數值欄位。")
            return

        # ── 欄位搜尋 + 批量刪除 ──────────────────────────────
        search_q = st.text_input("🔍 欄位關鍵字篩選", key="fe_search",
                                  placeholder="留空顯示全部...")
        display_cols = (
            [c for c in non_batch_num if search_q.lower() in c.lower()]
            if search_q.strip() else non_batch_num
        )
        st.caption(f"顯示 {len(display_cols)} / {len(non_batch_num)} 個欄位")

        # ── 欄位卡片 ──────────────────────────────────────────
        # 找出哪些欄位是「新增」的（供標注）
        all_added = set()
        for entry in op_log:
            all_added.update(entry.get("cols_added", []))
        all_removed_from_base = set()
        for entry in op_log:
            all_removed_from_base.update(entry.get("cols_removed", []))

        cols_per_row = 1   # 全寬顯示讓圖夠大
        for col in display_cols:
            badge = " `🆕 新增`" if col in all_added else ""
            with st.expander(f"**{col[:75]}**{badge}", expanded=False):
                exp_l, exp_r = st.columns([5, 1])

                with exp_l:
                    # 描述統計
                    s = clean_df[col].dropna()
                    stat_cols = st.columns(6)
                    stat_cols[0].metric("n",       len(s))
                    stat_cols[1].metric("Mean",    f"{s.mean():.4f}" if len(s) else "—")
                    stat_cols[2].metric("Median",  f"{s.median():.4f}" if len(s) else "—")
                    stat_cols[3].metric("SD",      f"{s.std(ddof=1):.4f}" if len(s) > 1 else "—")
                    stat_cols[4].metric("Min",     f"{s.min():.4f}" if len(s) else "—")
                    stat_cols[5].metric("Max",     f"{s.max():.4f}" if len(s) else "—")

                    # 趨勢圖
                    color = "#2ca02c" if col in all_added else "#2e86ab"
                    try:
                        fig = _mini_trend(clean_df, col, color=color)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.caption(f"繪圖失敗：{e}")

                with exp_r:
                    st.markdown("<br><br><br>", unsafe_allow_html=True)

                    # 反悔按鈕（只有曾被操作過的欄位才顯示）
                    was_touched = col in all_added or col in all_removed_from_base
                    if was_touched:
                        if st.button("↩️ 反悔", key=f"undo_overview_{col}",
                                     help=f"還原此欄位的最近一次操作",
                                     type="secondary"):
                            _undo_col(col)
                            st.rerun()

                    # 手動刪除按鈕
                    if st.button("🗑️ 刪除", key=f"manual_drop_{col}",
                                 help=f"手動刪除此欄位",
                                 type="secondary"):
                        snapshot_before = clean_df.copy()
                        new_clean = clean_df.drop(columns=[col])
                        st.session_state["clean_df"] = new_clean
                        _push_op("manual_drop", [col], [], snapshot_before)
                        st.toast(f"🗑️ 已刪除：{col}", icon="🗑️")
                        st.rerun()
