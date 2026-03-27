"""Tab — 缺失值分析"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils import missing_col


def render(selected_process_df):
    st.header("缺失值分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟，或執行特徵工程。")
        return

    # ── 顯示當前欄位數，讓使用者清楚現在的基準 ───────────────
    n_cols_now = work_df.shape[1]
    n_rows_now = work_df.shape[0]
    mc1, mc2 = st.columns(2)
    mc1.metric("當前批次數", n_rows_now)
    mc2.metric("當前欄位數", n_cols_now)

    # ── 缺失值統計 ────────────────────────────────────────────
    summary_df = missing_col(work_df)
    if summary_df.empty:
        st.success("🎉 無缺失值！")
    else:
        st.metric("含缺失值的欄位數", len(summary_df))
        st.dataframe(
            summary_df.style.background_gradient(cmap="Reds", subset=["Missing Ratio (%)"]),
            use_container_width=True,
        )

        st.markdown("#### 缺失值熱圖")
        n_miss_cols = len(summary_df)
        fig_h = max(4, min(40, n_miss_cols * 0.38))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        missing_matrix = work_df[summary_df.index].isnull().T
        font_sz = max(5, min(9, int(200 / max(n_miss_cols, 1))))
        sns.heatmap(
            missing_matrix, cmap="Reds", cbar=False, ax=ax,
            yticklabels=[c[:45] for c in summary_df.index],
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_sz)
        ax.set_xlabel("Sample Index")
        ax.set_title(f"Missing Value Pattern ({n_miss_cols} columns)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── 手動移除 ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗑️ 手動移除")

    # --- 移除批次 (BatchID) ---
    if "BatchID" in work_df.columns:
        drop_batches = st.multiselect(
            "選擇要移除的 BatchID",
            work_df["BatchID"].tolist(),
            key="miss_drop_batches",
        )
    else:
        drop_batches = []

    # --- 移除欄位：改用 multiselect，避免逗號解析出錯 ---
    all_cols = [c for c in work_df.columns if c != "BatchID"]
    drop_cols_sel = st.multiselect(
        "選擇要移除的欄位",
        all_cols,
        key="miss_drop_cols",
        help="直接從清單選擇，無需手動輸入，避免名稱打錯。",
    )

    # 保留原本的文字輸入作為補充（欄位過多時方便貼上）
    with st.expander("📝 或輸入欄位名稱（逗號分隔，補充用）", expanded=False):
        drop_cols_text = st.text_area(
            "要移除的欄位名稱（逗號分隔）",
            value="",
            key="miss_drop_cols_text",
            height=80,
        )
    drop_cols_text = st.session_state.get("miss_drop_cols_text", "")

    if st.button("🗑️ 執行移除", key="drop_rows", type="primary"):
        filtered = work_df.copy()
        n_before_rows = filtered.shape[0]
        n_before_cols = filtered.shape[1]

        # 移除批次
        if drop_batches:
            filtered = filtered[~filtered["BatchID"].isin(drop_batches)]

        # 合併兩種欄位來源（multiselect + 文字輸入）
        cols_to_drop = list(drop_cols_sel)
        if drop_cols_text.strip():
            text_cols = [
                c.strip() for c in drop_cols_text.split(",")
                if c.strip() in filtered.columns
            ]
            # 加入尚未在 multiselect 的欄位
            for c in text_cols:
                if c not in cols_to_drop:
                    cols_to_drop.append(c)

        if cols_to_drop:
            # 只移除確實存在的欄位
            valid_drop = [c for c in cols_to_drop if c in filtered.columns]
            invalid_drop = [c for c in cols_to_drop if c not in filtered.columns]
            if invalid_drop:
                st.warning(f"以下欄位不存在，已跳過：{invalid_drop}")
            filtered = filtered.drop(columns=valid_drop)

        # 存回 session state
        st.session_state["clean_df"] = filtered

        # 顯示前後對比
        removed_rows = n_before_rows - filtered.shape[0]
        removed_cols = n_before_cols - filtered.shape[1]
        st.success(
            f"✅ 移除完成：\n"
            f"- 批次：{n_before_rows} → {filtered.shape[0]}（移除 {removed_rows} 筆）\n"
            f"- 欄位：{n_before_cols} → {filtered.shape[1]}（移除 {removed_cols} 欄）"
        )

        # 顯示操作後的欄位數確認
        rc1, rc2 = st.columns(2)
        rc1.metric("移除後批次數", filtered.shape[0], delta=f"-{removed_rows}" if removed_rows else "0")
        rc2.metric("移除後欄位數", filtered.shape[1], delta=f"-{removed_cols}" if removed_cols else "0")

        st.dataframe(filtered.head(5), use_container_width=True)
        st.rerun()
