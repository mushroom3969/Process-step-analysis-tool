"""Tab — 相關性分析"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import streamlit as st
from utils import analyze_correlation


def render(selected_process_df):
    st.header("相關性分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.warning("無數值型欄位。")
        return

    col1, col2, col3 = st.columns(3)
    target_col = col1.selectbox("目標欄位（Y）", numeric_cols)
    method     = col2.selectbox("相關係數方法", ["pearson", "spearman"])
    top_n      = col3.slider("顯示前 N 個特徵", 5, min(50, len(numeric_cols)), 15)

    if st.button("🔗 計算相關性", key="run_corr"):
        with st.spinner("計算中..."):
            result = analyze_correlation(work_df, target_col, method=method, top_n=top_n)
        if result:
            fig, corr_rank = result
            st.pyplot(fig)
            plt.close()
            st.markdown("#### 相關係數排行")
            st.dataframe(
                corr_rank.style.background_gradient(
                    cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1
                ),
                use_container_width=True, hide_index=True,
            )
            st.session_state["target_col"] = target_col
            st.session_state["corr_rank"]  = corr_rank
