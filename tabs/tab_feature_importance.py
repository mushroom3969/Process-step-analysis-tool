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
            return

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


def _render_main(selected_process_df, show_mean: bool = True):
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

        st.session_state["fe_base_df"]      = snapshot_before
        st.session_state["clean_df"]        = clean_df
        st.session_state["fe_op_log"]       = []
        st.session_state["fe_stat_result"]  = None
        st.session_state["fe_auto_result"]  = {
            "removed":  removed, "added": added,
            "snap":     snapshot_before, "clean": clean_df,
            "n_before": selected_process_df.shape[1],
            "n_after":  clean_df.shape[1],
        }
        _push_op("auto_clean", removed, added, snapshot_before,
                 reason_map={r["Column"]: r.get("Reason", "") for _, r in drop_log.iterrows()
                              if "Column" in drop_log.columns})

    # ── 顯示 Step1 結果（穩定，從 session_state 讀）──────────────────
    _auto = st.session_state.get("fe_auto_result")
    # 清除格式不符的舊版 session_state（key 名稱不同）
    if _auto is not None and ("snap" not in _auto or "clean" not in _auto):
        st.session_state["fe_auto_result"] = None
        _auto = None
    if _auto is not None:
        st.success(
            f"✅ 完成！從 {_auto['n_before']} 欄 → {_auto['n_after']} 欄"
            f"（刪除 {len(_auto['removed'])} / 新增 {len(_auto['added'])}）"
        )
        _render_changed_cols(
            df_before=_auto["snap"], df_after=_auto["clean"],
            cols_removed=_auto["removed"], cols_added=_auto["added"],
            source_df_for_removed=_auto["snap"],
            section_title="Step1_auto_clean", show_mean=show_mean,
        )

    # ══════════════════════════════════════════════════════════
    # ── 區塊 B：統計篩選 ──────────────────────────────────────
    # BUG FIX: Separate the "run" button from the "display results" section
    # so that clicking the button stores results in session_state, and the
    # display always reads from session_state — never re-executes on rerun.
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
        # BUG FIX: changed default from 0.2 → 0.1 (0.2 was too aggressive,
        # and was silently dropping almost all columns which caused the apparent "jump")
        acf_thresh  = c3.slider("ACF 門檻（低於此值剔除）",
                                 0.0, 0.5, 0.1, 0.05, key="fe_acf")

        source_df = st.session_state.get("df_before_stats", st.session_state["clean_df"])
        
        if st.button("📉 執行統計篩選", key="run_stat_filter"):
            try:
                with st.spinner("篩選中..."):
                    # 1. 備份原始狀態（若尚未備份）
                    if "df_before_stats" not in st.session_state:
                        st.session_state["df_before_stats"] = source_df.copy()
                    
                    # 2. 執行篩選
                    filtered_df, dropped_info = filter_columns_by_stats(
                        st.session_state["df_before_stats"], # 永遠用乾淨的來源跑
                        cv_threshold=cv_thresh,
                        jump_ratio_threshold=jump_thresh,
                        acf_threshold=acf_thresh,
                    )
        
                    # 3. 確保 BatchID 存在 (更安全的寫法)
                    if "BatchID" in st.session_state["df_before_stats"].columns:
                        if "BatchID" not in filtered_df.columns:
                            filtered_df.insert(0, "BatchID", st.session_state["df_before_stats"]["BatchID"])
        
                    # 4. 更新狀態
                    st.session_state["clean_df"] = filtered_df
                    st.session_state["fe_stat_result"] = {
                        "removed": sorted(set(st.session_state["df_before_stats"].columns) - set(filtered_df.columns)),
                        "dropped_info": dropped_info
                    }
                    
                    # 5. 紀錄操作並強制刷新
                    _push_op("stat_filter", ...) 
                    st.rerun() 
        
            except Exception as e:
                st.error(f"統計篩選失敗：{e}")

        # Display results from session_state (stable across reruns)
        stat_res = st.session_state.get("fe_stat_result")
        if stat_res is not None:
            removed      = stat_res["removed"]
            added        = stat_res["added"]
            dropped_info = stat_res["dropped_info"]
            snapshot_bef = stat_res["snapshot"]
            filtered_df  = stat_res["filtered_df"]

            st.success(f"✅ 剔除 {len(removed)} 個欄位 → 剩餘 {filtered_df.shape[1]} 欄")

            if dropped_info:
                reason_df = pd.DataFrame(
                    [(k, v) for k, v in dropped_info.items()],
                    columns=["Column", "Reason"],
                )
                with st.expander("📋 被剔除欄位原因", expanded=True):
                    st.dataframe(reason_df, width="stretch", hide_index=True)

            _render_changed_cols(
                df_before=snapshot_bef,
                df_after=filtered_df,
                cols_removed=removed,
                cols_added=added,
                source_df_for_removed=snapshot_bef,
                section_title="📋 本次統計篩選變更欄位",
                show_mean=show_mean,
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

        m1, m2, m3 = st.columns(3)
        m1.metric("當前欄位數", clean_df.shape[1])
        m2.metric("數值欄位數", len(non_batch_num))
        m3.metric("批次數", clean_df.shape[0])

        if not non_batch_num:
            st.warning("目前沒有可顯示的數值欄位。")
            return

        search_q = st.text_input("🔍 欄位關鍵字篩選", key="fe_search",
                                  placeholder="留空顯示全部...")
        display_cols = (
            [c for c in non_batch_num if search_q.lower() in c.lower()]
            if search_q.strip() else non_batch_num
        )
        st.caption(f"顯示 {len(display_cols)} / {len(non_batch_num)} 個欄位")

        all_added = set()
        for entry in op_log:
            all_added.update(entry.get("cols_added", []))
        all_removed_from_base = set()
        for entry in op_log:
            all_removed_from_base.update(entry.get("cols_removed", []))

        import hashlib as _hl_ov
        for col in display_cols:
            _ch = _hl_ov.md5(col.encode()).hexdigest()[:12]
            badge = " `🆕 新增`" if col in all_added else ""
            with st.expander(f"**{col[:75]}**{badge}", expanded=False):
                exp_l, exp_r = st.columns([5, 1])

                with exp_l:
                    s = clean_df[col].dropna()
                    stat_cols = st.columns(6)
                    stat_cols[0].metric("n",       len(s))
                    stat_cols[1].metric("Mean",    f"{s.mean():.4f}" if len(s) else "—")
                    stat_cols[2].metric("Median",  f"{s.median():.4f}" if len(s) else "—")
                    stat_cols[3].metric("SD",      f"{s.std(ddof=1):.4f}" if len(s) > 1 else "—")
                    stat_cols[4].metric("Min",     f"{s.min():.4f}" if len(s) else "—")
                    stat_cols[5].metric("Max",     f"{s.max():.4f}" if len(s) else "—")

                    color = "#2ca02c" if col in all_added else "#2e86ab"
                    try:
                        fig = _mini_trend(clean_df, col, color=color, show_mean=show_mean)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.caption(f"繪圖失敗：{e}")

                with exp_r:
                    st.markdown("<br><br><br>", unsafe_allow_html=True)

                    was_touched = col in all_added or col in all_removed_from_base
                    if was_touched:
                        if st.button("↩️ 反悔", key=f"fi_ov_undo_{_ch}",
                                     help=f"還原此欄位的最近一次操作",
                                     type="secondary"):
                            _undo_col(col)
                            st.rerun()

                    if st.button("🗑️ 刪除", key=f"fi_ov_drop_{_ch}",
                                 help=f"手動刪除此欄位",
                                 type="secondary"):
                        snapshot_before = clean_df.copy()
                        new_clean = clean_df.drop(columns=[col])
                        st.session_state["clean_df"] = new_clean
                        _push_op("manual_drop", [col], [], snapshot_before)
                        st.toast(f"🗑️ 已刪除：{col}", icon="🗑️")
                        st.rerun()
