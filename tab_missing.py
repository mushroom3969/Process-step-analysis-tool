"""Tab 3 — 缺失值分析"""
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

    summary_df = missing_col(work_df)
    if summary_df.empty:
        st.success("🎉 無缺失值！")
    else:
        st.metric("含缺失值的欄位數", len(summary_df))
        st.dataframe(
            summary_df.style.background_gradient(cmap="Reds", subset=["Missing Ratio (%)"]),
            width="stretch",
        )

        st.markdown("#### 缺失值熱圖")
        fig, ax = plt.subplots(figsize=(14, 4))
        missing_matrix = work_df[summary_df.index].isnull().T
        sns.heatmap(
            missing_matrix, cmap="Reds", cbar=False, ax=ax,
            yticklabels=[c[:40] for c in summary_df.index],
        )
        ax.set_xlabel("Sample Index")
        ax.set_title("Missing Value Pattern")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── 手動移除 ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗑️ 手動移除批次")
    if "BatchID" in work_df.columns:
        drop_batches = st.multiselect("選擇要移除的 BatchID", work_df["BatchID"].tolist())
        drop_cols_ui = st.text_input("要移除的欄位名稱（逗號分隔）", "")

        if st.button("🗑️ 執行移除", key="drop_rows"):
            filtered = work_df.copy()
            if drop_batches:
                filtered = filtered[~filtered["BatchID"].isin(drop_batches)]
            if drop_cols_ui.strip():
                cols_to_drop = [
                    c.strip() for c in drop_cols_ui.split(",")
                    if c.strip() in filtered.columns
                ]
                filtered = filtered.drop(columns=cols_to_drop)
            st.session_state["clean_df"] = filtered
            st.success(f"✅ 移除後：{filtered.shape[0]} 筆 × {filtered.shape[1]} 欄")
            st.dataframe(filtered.head(), width="stretch")
