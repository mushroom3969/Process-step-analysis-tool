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
            st.dataframe(filtered.head(), use_container_width=True)
