"""
Bioprocess Data Analysis Tool
一個針對生物製藥製程（rhG-CSF）的互動式數據分析平台
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
for _p in [_dir, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from utils import split_process_df
from tabs import tab_overview
from tabs import tab_trend
from tabs import tab_feature_eng
from tabs import tab_missing
from tabs import tab_correlation
from tabs import tab_pca
from tabs import tab_feature_importance
from tabs import tab_literature


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
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
_SESSION_DEFAULTS = {
    "raw_df": None, "dfs_dict": None,
    "selected_process_df": None, "clean_df": None,
    "X": None, "y": None, "label": None, "x_scaled": None,
}
for key, default in _SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("🧬 Bioprocess Analytics")
    st.markdown("---")
    st.markdown("### 📁 資料載入")

    uploaded_file = st.file_uploader("上傳 CSV 資料檔", type=["csv"])
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            # 統一 BatchID 欄位名稱
            batch_col_candidates = [c for c in raw_df.columns if "BatchID" in c]
            if batch_col_candidates:
                raw_df = raw_df.rename(columns={batch_col_candidates[0]: "BatchID"})
            # 非 BatchID 欄位全部轉數值
            cols_to_convert = raw_df.columns.difference(["BatchID"])
            raw_df[cols_to_convert] = raw_df[cols_to_convert].apply(
                pd.to_numeric, errors="coerce"
            )
            st.session_state["raw_df"]    = raw_df
            st.session_state["dfs_dict"]  = split_process_df(raw_df)
            st.success(f"✅ 載入成功！{raw_df.shape[0]} 筆 × {raw_df.shape[1]} 欄")
        except Exception as e:
            st.error(f"載入失敗：{e}")

    st.markdown("---")
    if st.session_state["dfs_dict"]:
        st.markdown("### ⚙️ 製程步驟選擇")
        process_list     = list(st.session_state["dfs_dict"].keys())
        selected_process = st.selectbox("選擇製程步驟", process_list)
        st.session_state["selected_process"]    = selected_process
        st.session_state["selected_process_df"] = st.session_state["dfs_dict"][selected_process]

    st.markdown("---")
    st.caption("Bioprocess Analytics v2.0")


# ── Guard: require data ───────────────────────────────────────────────────────
st.title("🧬 Bioprocess Data Analysis Platform")

if st.session_state["raw_df"] is None:
    st.markdown("""
    <div class="info-box">
    👈 請先在左側上傳 <b>CSV 資料檔</b>（raw_data.csv）開始分析。
    </div>
    """, unsafe_allow_html=True)
    st.stop()

raw_df              = st.session_state["raw_df"]
dfs_dict            = st.session_state["dfs_dict"]
selected_process_df = st.session_state.get("selected_process_df")
selected_process    = st.session_state.get("selected_process", "")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 資料總覽",
    "📈 趨勢圖",
    "🔧 特徵工程",
    "🔍 缺失值分析",
    "🔗 相關性分析",
    "🧩 PCA 分析",
    "🌲 特徵重要性",
    "📚 文獻佐證分析",
])

with tabs[0]:
    tab_overview.render(raw_df, dfs_dict, selected_process_df, selected_process)

with tabs[1]:
    tab_trend.render(selected_process_df)

with tabs[2]:
    tab_feature_eng.render(selected_process_df)

with tabs[3]:
    tab_missing.render(selected_process_df)

with tabs[4]:
    tab_correlation.render(selected_process_df)

with tabs[5]:
    tab_pca.render(selected_process_df)

with tabs[6]:
    tab_feature_importance.render(selected_process_df)

with tabs[7]:
    tab_literature.render()
