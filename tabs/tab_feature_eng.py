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
# ── 共線性偵測 & 自動合併 ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

def _group_key(col: str) -> str:
    """
    將特徵名稱正規化為「分組鍵」，把 Max/Min、編號、Before/After
    等可辨識的差異字詞替換為佔位符，相同 base 的特徵會得到相同的 key。
    """
    s = _re.sub(r'\s*\([^)]*\)\s*$', '', col.strip())          # 去除單位
    s = _re.sub(                                                 # Max / Min
        r'\b(Maximum|Minimum|Maximun|Minimun)\b',
        '__MM__', s, flags=_re.IGNORECASE)
    s = _re.sub(                                                 # max / min 小寫
        r'(?<![A-Za-z])(maximun?|minimun?|max|min)(?![A-Za-z])',
        '__MM__', s, flags=_re.IGNORECASE)
    s = _re.sub(r'[_ ]No\.?\s*\d+\s*$', '__N__', s)            # _No1 / _No2
    s = _re.sub(r'[_ ]\d+\s*$', '__N__', s)                    # _1 / _2 / _4
    s = _re.sub(                                                 # before / after
        r'\b(before|after)\b', '__BA__', s, flags=_re.IGNORECASE)
    return s.strip().rstrip('_').strip()


def _detect_collinear_groups(df: pd.DataFrame,
                              r_threshold: float = 0.85) -> list[dict]:
    """
    1. 對所有數值特徵計算分組鍵
    2. 同鍵 & 組內至少 2 個特徵 → 計算 pairwise |Pearson r|
    3. 組內 max |r| > r_threshold → 標記為「可合併」群組

    回傳 list of dict：
      { 'key': str, 'cols': list, 'max_r': float, 'mergeable': bool,
        'corr_matrix': pd.DataFrame }
    """
    num_cols = [c for c in df.select_dtypes(include='number').columns
                if c.lower() != 'batchid']
    if len(num_cols) < 2:
        return []

    key_map: dict[str, list[str]] = {}
    for col in num_cols:
        k = _group_key(col)
        key_map.setdefault(k, []).append(col)

    groups = []
    for key, cols in key_map.items():
        if len(cols) < 2:
            continue
        sub = df[cols].dropna(how='all')
        if len(sub) < 3:
            continue
        corr = sub.corr(method='pearson').abs()
        # 取上三角排除自身
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        vals = corr.values[mask]
        max_r = float(np.nanmax(vals)) if len(vals) > 0 else 0.0
        groups.append({
            'key':         key,
            'cols':        cols,
            'max_r':       max_r,
            'mergeable':   max_r >= r_threshold,
            'corr_matrix': corr,
        })

    groups.sort(key=lambda g: -g['max_r'])
    return groups


def _render_collinearity_merge(show_mean: bool = True):
    """Step 2.5 UI：共線性偵測 & 自動合併"""
    st.markdown("---")
    st.markdown("### 🔗 Step 2.5：共線性偵測 & 自動合併")
    st.caption(
        "偵測 Max/Min 配對、重複量測（_1/_2）、Before/After 配對等高度相關特徵群組，"
        "若組內 Pearson |r| 超過門檻，可一鍵合併為均值欄位。"
    )

    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        return

    col_thr, col_run = st.columns([1, 1])
    r_thr = col_thr.slider(
        "共線性門檻 |r| ≥", 0.5, 1.0, 0.85, 0.01,
        key="fe_col_thr",
        help="組內任意兩特徵 Pearson |r| 超過此值 → 標示為可合併"
    )

    if not col_run.button("🔍 偵測共線性群組", key="run_col_detect", type="primary"):
        st.info("設定門檻後點擊「🔍 偵測共線性群組」。")
        return

    with st.spinner("計算中..."):
        groups = _detect_collinear_groups(clean_df, r_threshold=r_thr)

    if not groups:
        st.success("✅ 未偵測到共線性群組（所有特徵組內 |r| 均低於門檻）。")
        return

    mergeable   = [g for g in groups if g['mergeable']]
    unmergeable = [g for g in groups if not g['mergeable']]

    st.info(
        f"共偵測到 **{len(groups)}** 個命名相似群組，"
        f"其中 **{len(mergeable)}** 組超過門檻（|r| ≥ {r_thr}）建議合併，"
        f"**{len(unmergeable)}** 組低於門檻保留。"
    )

    # ── 可合併群組 ─────────────────────────────────────────────────
    if mergeable:
        st.markdown(f"#### ✅ 可合併群組（|r| ≥ {r_thr}）")

        merge_selections = {}
        for i, g in enumerate(mergeable):
            label = f"**Group {i+1}** — max |r| = {g['max_r']:.3f}  ({len(g['cols'])} 個特徵)"
            with st.expander(label, expanded=(i < 3)):
                # 相關矩陣熱圖
                fig, ax = plt.subplots(figsize=(max(3, len(g['cols']) * 1.2),
                                                max(2.5, len(g['cols']) * 1.0)))
                import seaborn as _sns_col
                _sns_col.heatmap(
                    g['corr_matrix'], annot=True, fmt=".2f",
                    cmap="RdYlGn", vmin=0, vmax=1, center=0.5,
                    linewidths=0.5, ax=ax, annot_kws={"size": 8}
                )
                ax.set_title(f"Pairwise |r|", fontsize=9)
                plt.xticks(rotation=30, ha='right', fontsize=7)
                plt.yticks(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                # 特徵列表
                for c in g['cols']:
                    st.markdown(f"  - `{c}`")

                # 建議合併名稱（取最短的 col 去掉識別符）
                suggested = min(g['cols'], key=len)
                new_name = st.text_input(
                    "合併後新欄位名稱",
                    value=f"MERGED_{suggested[:50]}",
                    key=f"fe_merge_name_{i}"
                )
                do_merge = st.checkbox(
                    f"✔ 納入合併（→ mean）", value=True, key=f"fe_do_merge_{i}"
                )
                merge_selections[i] = {
                    'cols': g['cols'], 'new_name': new_name, 'do_merge': do_merge
                }

        # ── 執行合併按鈕 ────────────────────────────────────────────
        if st.button("⚡ 執行合併", key="fe_run_merge", type="primary"):
            df_work = clean_df.copy()
            merged_log = []

            for i, sel in merge_selections.items():
                if not sel['do_merge']:
                    continue
                cols_to_merge = [c for c in sel['cols'] if c in df_work.columns]
                if len(cols_to_merge) < 2:
                    continue
                new_col = sel['new_name'].strip() or f"MERGED_{i}"
                df_work[new_col] = df_work[cols_to_merge].mean(axis=1)
                df_work.drop(columns=cols_to_merge, inplace=True)
                merged_log.append({
                    'new_col': new_col,
                    'merged_from': cols_to_merge,
                    'n': len(cols_to_merge)
                })

            if not merged_log:
                st.warning("沒有選擇任何群組合併。")
            else:
                snapshot_before = clean_df.copy()
                st.session_state["clean_df"] = df_work

                cols_removed = [c for m in merged_log for c in m['merged_from']]
                cols_added   = [m['new_col'] for m in merged_log]
                _push_op("collinear_merge", cols_removed, cols_added, snapshot_before)

                st.success(
                    f"✅ 完成！合併 {len(merged_log)} 個群組，"
                    f"移除 {len(cols_removed)} 欄 → 新增 {len(cols_added)} 欄（mean）"
                )
                for m in merged_log:
                    st.caption(f"  `{m['new_col']}` ← {m['merged_from']}")
                st.rerun()

    # ── 低於門檻群組（僅顯示，不合併）────────────────────────────────
    if unmergeable:
        with st.expander(
            f"ℹ️ 低於門檻群組（|r| < {r_thr}，保留原特徵）— {len(unmergeable)} 組",
            expanded=False
        ):
            for g in unmergeable:
                st.markdown(f"- max |r| = **{g['max_r']:.3f}** | " +
                            " / ".join([f"`{c[:50]}`" for c in g['cols']]))


# ══════════════════════════════════════════════════════════════════════════════
# ── Session-state 工具 ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init_fe_state():
    """確保所有特徵工程相關 session_state 鍵都已初始化。"""
    defaults = {
        "fe_base_df":          None,
        "clean_df":            None,
        "fe_op_log":           [],
        "fe_stat_dropped":     {},
        # BUG FIX: store stat filter results separately so rerun doesn't re-trigger
        "fe_stat_result":      None,
        "fe_auto_result":      None,
        "df_before_step2":     None,
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
    log = st.session_state.get("fe_op_log", [])
    cdf = st.session_state.get("clean_df")

    for entry in reversed(log):
        if col in entry["cols_removed"]:
            snap = entry.get("snapshot")
            # Guard: both clean_df and snapshot must exist
            if snap is None:
                st.toast(f"⚠️ 找不到 {col} 的快照，無法還原", icon="⚠️")
                return
            if col not in snap.columns:
                st.toast(f"⚠️ 快照中找不到欄位：{col}", icon="⚠️")
                return
            # Guard: clean_df must exist; if missing, restore from snapshot directly
            if cdf is None:
                cdf = snap[[c for c in snap.columns]].copy()
            else:
                cdf = cdf.copy()
                cdf[col] = snap[col].values
            st.session_state["clean_df"] = cdf
            st.toast(f"✅ 已還原欄位：{col}", icon="↩️")
            return
        if col in entry.get("cols_added", []):
            if cdf is not None and col in cdf.columns:
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
        # 效能優化：欄位多時只顯示列表和反悔按鈕，不自動繪圖
        # 點擊欄位名稱後才繪圖，避免同時建立大量 matplotlib figure
        CHART_LIMIT = 5   # 超過此數量就不自動繪圖
        auto_chart = len(cols_removed) <= CHART_LIMIT

        if not auto_chart:
            st.caption(f"共 {len(cols_removed)} 個欄位被刪除，展開各欄可查看趨勢圖。")

        for i, col in enumerate(cols_removed):
            with st.expander(f"🗑️  **{col[:70]}**", expanded=False):
                c_left, c_right = st.columns([5, 1])
                with c_left:
                    if col in source_df_for_removed.columns:
                        # 超過限制時加一個按鈕才繪圖，避免大量 figure 同時建立
                        draw_key = f"draw_rm_{_sh}_{i}"
                        should_draw = auto_chart or st.button(
                            "📈 顯示趨勢圖", key=draw_key, type="secondary"
                        )
                        if should_draw:
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
            "auto_clean":      "🔧 自動特徵工程",
            "stat_filter":     "📉 統計篩選",
            "manual_drop":     "🗑️ 手動移除",
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
            st.session_state["fe_auto_result"] = None
            st.session_state["df_before_step2"] = None
            st.toast("✅ 已完全重置", icon="🔄")
            st.rerun()

def render(selected_process_df):  # <--- 報錯就是因為找不到這一行
    _init_fe_state()
    st.header("特徵工程 & 清理")
    
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return
        
    # 控制開關
    show_mean = st.checkbox("📊 圖表顯示平均線與 ±1σ 色帶", value=True, key="fe_show_mean")

    main_col, hist_col = st.columns([3, 1])

    with hist_col:
        st.markdown("### 📜 操作歷史")
        _render_history_panel()

    with main_col:
        # 呼叫主要的渲染邏輯
        _render_main(selected_process_df, show_mean)
# ══════════════════════════════════════════════════════════════════════════════
# ── 主 render ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_main(selected_process_df, show_mean: bool = True):
    """主操作區：確保 Step 1 與 Step 2 邏輯解耦，不互相干擾"""

    # ══════════════════════════════════════════════════════════
    # ── 區塊 A：Step 1 自動特徵工程 ───────────────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown("### 🔧 Step 1：自動特徵工程")
    
    if st.button("🔧 執行特徵工程", key="run_fe", type="primary"):
        with st.spinner("處理中..."):
            snapshot_before = selected_process_df.copy()
            clean_df, drop_log = clean_process_features_with_log(
                selected_process_df, id_col="BatchID"
            )

            # 更新關鍵狀態
            st.session_state["fe_base_df"] = snapshot_before
            st.session_state["clean_df"] = clean_df
            # 【關鍵】為 Step 2 建立一個專屬的起點備份，防止連續刪除
            st.session_state["df_before_step2"] = clean_df.copy()
            
            # 存儲 Step 1 結果
            st.session_state["fe_auto_result"] = {
                "removed": sorted(list(set(selected_process_df.columns) - set(clean_df.columns))),
                "added": sorted(list(set(clean_df.columns) - set(selected_process_df.columns))),
                "snap": snapshot_before,
                "clean": clean_df,
                "n_before": selected_process_df.shape[1],
                "n_after": clean_df.shape[1],
            }
            # 清除舊的 Step 2 結果，因為數據源變了
            st.session_state["fe_stat_result"] = None
            st.rerun()

    # 持久化顯示 Step 1 結果
    auto_res = st.session_state.get("fe_auto_result")
    if auto_res:
        st.success(f"✅ Step 1 完成：從 {auto_res['n_before']} 欄 → {auto_res['n_after']} 欄")
        with st.expander("查看 Step 1 變更詳情", expanded=False):
            _render_changed_cols(
                df_before=auto_res["snap"], df_after=auto_res["clean"],
                cols_removed=auto_res["removed"], cols_added=auto_res["added"],
                source_df_for_removed=auto_res["snap"],
                section_title="Step 1 變更", show_mean=show_mean
            )


    # ══════════════════════════════════════════════════════════
    # ── 區塊 B：Step 2 統計篩選 ───────────────────────────────
    # ══════════════════════════════════════════════════════════
    # 只有當 Step 1 有產出時，才顯示 Step 2
    if st.session_state.get("df_before_step2") is not None:
        st.markdown("---")
        st.markdown("### 📉 Step 2：統計篩選")
        
        c1, c2, c3 = st.columns(3)
        cv_t = c1.slider("CV 門檻", 0.0, 0.1, 0.01, 0.001, format="%.3f", key="fe_cv")
        jp_t = c2.slider("Jump Ratio 門檻", 0.1, 1.0, 0.5, 0.05, key="fe_jump")
        af_t = c3.slider("ACF 門檻", 0.0, 0.5, 0.1, 0.05, key="fe_acf")

        if st.button("📉 執行統計篩選", key="run_stat_filter", type="primary"):
            # 永遠使用 Step 1 的穩定產出作為輸入
            source_df = st.session_state["df_before_step2"]
            
            try:
                with st.spinner("篩選計算中..."):
                    filtered_df, dropped_info = filter_columns_by_stats(
                        source_df, cv_threshold=cv_t, 
                        jump_ratio_threshold=jp_t, acf_threshold=af_t
                    )
                    
                    # 安全插入 BatchID
                    if "BatchID" in source_df.columns and "BatchID" not in filtered_df.columns:
                        filtered_df.insert(0, "BatchID", source_df["BatchID"])

                    # 紀錄結果
                    st.session_state["clean_df"] = filtered_df
                    st.session_state["fe_stat_result"] = {
                        "filtered_df": filtered_df,
                        "removed": sorted(list(set(source_df.columns) - set(filtered_df.columns))),
                        "dropped_info": dropped_info,
                        "snapshot": source_df.copy()
                    }
                    # 紀錄歷史 Log
                    _push_op("stat_filter", 
                             sorted(list(set(source_df.columns) - set(filtered_df.columns))), 
                             [], source_df.copy(), reason_map=dropped_info)
                    
                    st.rerun()
            except Exception as e:
                st.error(f"統計篩選失敗：{e}")

        # 持久化顯示 Step 2 結果
        stat_res = st.session_state.get("fe_stat_result")
        if stat_res:
            st.info(f"✅ Step 2 完成：剔除 {len(stat_res['removed'])} 欄，剩餘 {stat_res['filtered_df'].shape[1]} 欄")
            
            if stat_res.get("dropped_info"):
                with st.expander("📋 查看剔除原因", expanded=False):
                    reasons = pd.DataFrame([(k, v) for k, v in stat_res["dropped_info"].items()], 
                                         columns=["Column", "Reason"])
                    st.dataframe(reasons, use_container_width=True, hide_index=True)

            _render_changed_cols(
                df_before=stat_res["snapshot"], df_after=stat_res["filtered_df"],
                cols_removed=stat_res["removed"], cols_added=[],
                source_df_for_removed=stat_res["snapshot"],
                section_title="Step 2 變更詳情", show_mean=show_mean
            )
# ══════════════════════════════════════════════════════════════════════════
    # ── 區塊 B5：Step 2.5 共線性偵測 & 自動合併 ──────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("clean_df") is not None:
        _render_collinearity_merge(show_mean)

    # ── 區塊 C：當前特徵總覽（個別管理與反悔） ────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
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
        else:
            # 🔍 搜尋與篩選邏輯 (確保這裡縮排一致)
            search_q = st.text_input("🔍 欄位關鍵字篩選", key="fe_search", placeholder="留空顯示全部...")
            
            display_cols = (
                [c for c in non_batch_num if search_q.lower() in c.lower()]
                if search_q.strip() else non_batch_num
            )
            
            st.caption(f"顯示 {len(display_cols)} / {len(non_batch_num)} 個欄位")

            # 建立操作記錄快照以便判斷狀態
            all_added = set()
            for entry in op_log:
                all_added.update(entry.get("cols_added", []))
            
            all_removed_from_base = set()
            for entry in op_log:
                all_removed_from_base.update(entry.get("cols_removed", []))

            # 🛠️ 遍歷顯示各個特徵卡片
            import hashlib as _hl_ov
            import matplotlib.pyplot as plt

            for col in display_cols:
                # 使用 Hash 確保 Key 唯一且不包含特殊字元
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
                        # Step 3 欄位多，用按鈕觸發繪圖避免全部同時渲染
                        if st.button("📈 顯示趨勢圖", key=f"fi_ov_chart_{_ch}",
                                     type="secondary"):
                            try:
                                show_mean_val = st.session_state.get("fe_show_mean", True)
                                fig = _mini_trend(clean_df, col, color=color,
                                                  show_mean=show_mean_val)
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            except Exception as e:
                                st.caption(f"繪圖失敗：{e}")

                    with exp_r:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        
                        # 反悔按鈕
                        was_touched = col in all_added or col in all_removed_from_base
                        if was_touched:
                            if st.button("↩️ 反悔", key=f"fi_ov_undo_{_ch}", 
                                         help="還原此欄位的最近一次操作"):
                                _undo_col(col)
                                st.rerun()

                        # 刪除按鈕
                        if st.button("🗑️ 刪除", key=f"fi_ov_drop_{_ch}", 
                                     help="手動刪除此欄位", type="secondary"):
                            snapshot_before = clean_df.copy()
                            new_clean = clean_df.drop(columns=[col])
                            st.session_state["clean_df"] = new_clean
                            # 呼叫記錄函數
                            _push_op("manual_drop", [], [col], snapshot_before)
                            st.toast(f"🗑️ 已刪除：{col}")
                            st.rerun()
