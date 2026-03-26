"""
Bioprocess Data Analysis Tool
一個針對生物製藥製程（rhG-CSF）的互動式數據分析平台
"""
import warnings
warnings.filterwarnings("ignore")
import sys, os as _os

# 修正路徑環境
_dir = _os.path.dirname(_os.path.abspath(__file__))
for _p in [_dir, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st
from utils import split_process_df

# ── Tab imports ─────────────────────────────────────────────────────────────
from tabs import (
    tab_overview, tab_cross_process, tab_feature_eng, 
    tab_feature_importance, tab_missing, tab_correlation, 
    tab_pca, tab_stat_test, tab_trend, tab_literature
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bioprocess Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1f6aa5; }
    h2 { color: #2e86ab; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2e86ab;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .step-badge {
        display: inline-block;
        background: #2e86ab22;
        border: 1px solid #2e86ab66;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        color: #1a5f7a;
        margin-right: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── 多製程步驟合併工具 ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _merge_process_steps(raw_df: pd.DataFrame, dfs_dict: dict, selected_steps: list) -> pd.DataFrame or None:
    if not selected_steps:
        return None
        
    batch_col = "BatchID"
    if len(selected_steps) == 1:
        return dfs_dict[selected_steps[0]].copy()

    has_batch = batch_col in raw_df.columns
    base = raw_df[[batch_col]].copy() if has_batch else pd.DataFrame(index=raw_df.index)
    
    for step in selected_steps:
        step_cols = [c for c in raw_df.columns if c.startswith(f"{step}:")]
        if not step_cols: continue
            
        sub = raw_df[[batch_col] + step_cols].copy() if has_batch else raw_df[step_cols].copy()
        
        if base.empty:
            base = sub
        elif has_batch:
            base = base.merge(sub, on=batch_col, how="outer")
        else:
            base = pd.concat([base.reset_index(drop=True), sub.reset_index(drop=True)], axis=1)

    non_id = [c for c in base.columns if c != batch_col]
    base[non_id] = base[non_id].apply(pd.to_numeric, errors="coerce")
    base = base.dropna(axis=1, how="all")
    return base

# ── Session state init ────────────────────────────────────────────────────────
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
    st.session_state["dfs_dict"] = None
    st.session_state["selected_steps"] = []
    st.session_state["clean_df"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title(" Bioprocess Analytics")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("上傳 CSV 資料檔", type=["csv"])
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            batch_candidates = [c for c in raw_df.columns if "BatchID" in c]
            if batch_candidates:
                raw_df = raw_df.rename(columns={batch_candidates[0]: "BatchID"})
            
            non_batch = raw_df.columns.difference(["BatchID"])
            raw_df[non_batch] = raw_df[non_batch].apply(pd.to_numeric, errors="coerce")
            
            st.session_state["raw_df"] = raw_df
            st.session_state["dfs_dict"] = split_process_df(raw_df)
            st.success("載入成功！")
        except Exception as e:
            st.error(f"載入失敗：{e}")

    # 製程選擇邏輯
    if st.session_state["dfs_dict"]:
        process_list = list(st.session_state["dfs_dict"].keys())
        select_mode = st.radio("選擇模式", [" 單一步驟", " 多步驟合併"], horizontal=True)
        
        if select_mode == " 單一步驟":
            sel = st.selectbox("選擇製程步驟", process_list)
            selected_steps = [sel]
        else:
            selected_steps = st.multiselect("選擇步驟", process_list, default=process_list[:1])
        
        # 步驟變更時，重置清理狀態
        if selected_steps != st.session_state["selected_steps"]:
            st.session_state["selected_steps"] = selected_steps
            st.session_state["clean_df"] = None # 重要：步驟切換，清理結果要歸零
            
        merged_df = _merge_process_steps(st.session_state["raw_df"], st.session_state["dfs_dict"], selected_steps)
        st.session_state["selected_process_df"] = merged_df

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title(" Bioprocess Data Analysis Platform")

if st.session_state["raw_df"] is None:
    st.info("請先在上傳資料以開始分析。")
    st.stop()

# 決定後續 Tab 使用哪份資料：優先使用特徵工程後的 clean_df，若無則用 merged_df
df_to_analyze = st.session_state.get("clean_df")
if df_to_analyze is None:
    df_to_analyze = st.session_state.get("selected_process_df")

# 顯示目前狀態
if st.session_state["selected_steps"]:
    badges = "".join(f'<span class="step-badge">{s}</span>' for s in st.session_state["selected_steps"])
    data_status = " (已執行特徵工程)" if st.session_state.get("clean_df") is not None else " (原始數據)"
    st.markdown(f"**分析範圍：** {badges} {data_status}", unsafe_allow_html=True)

tabs = st.tabs([
    " 跨製程監控", " 資料總覽", " 趨勢圖", " 特徵工程", 
    " 缺失值分析", " 相關性分析", " PCA 分析", 
    " 特徵重要性", " 統計檢定", " 文獻佐證"
])

# 根據你的 tab_xxx.render 函式參數傳入對應資料
with tabs[0]:
    tab_cross_process.render(st.session_state["raw_df"])
with tabs[1]:
    tab_overview.render(st.session_state["raw_df"], st.session_state["dfs_dict"], df_to_analyze, " + ".join(st.session_state["selected_steps"]))
with tabs[2]:
    tab_trend.render(df_to_analyze)
with tabs[3]:
    # 特徵工程必須傳入原始合併後的資料，讓它進行清理
    tab_feature_eng.render(st.session_state["selected_process_df"])
with tabs[4]:
    tab_missing.render(df_to_analyze)
with tabs[5]:
    tab_correlation.render(df_to_analyze)
with tabs[6]:
    tab_pca.render(df_to_analyze)
with tabs[7]:
    tab_feature_importance.render(df_to_analyze)
with tabs[8]:
    tab_stat_test.render(df_to_analyze)
with tabs[9]:
    tab_literature.render()
