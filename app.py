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
from tabs import tab_stat_test
from tabs import tab_cross_process


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

def _merge_process_steps(
    raw_df: pd.DataFrame,
    dfs_dict: dict,
    selected_steps: list,
) -> pd.DataFrame | None:
    """
    合併多個製程步驟：
      • 單步驟 → 直接回傳 dfs_dict[step]（欄位已去掉前綴，原始行為不變）
      • 多步驟 → 從 raw_df 取出 'Step:Param' 欄位，保留完整前綴，以 BatchID outer join
    """
    if not selected_steps:
        return None

    batch_col = "BatchID"

    # ── 單步驟：原始行為 ──────────────────────────────────────
    if len(selected_steps) == 1:
        return dfs_dict[selected_steps[0]].copy()

    # ── 多步驟合併 ────────────────────────────────────────────
    has_batch = batch_col in raw_df.columns
    base = raw_df[[batch_col]].copy() if has_batch else pd.DataFrame(index=raw_df.index)

    for step in selected_steps:
        # 取出該 step 的所有欄位（完整 'Step:Param' 格式）
        step_cols = [c for c in raw_df.columns if c.startswith(f"{step}:")]
        if not step_cols:
            continue
        sub = raw_df[[batch_col] + step_cols].copy() if has_batch \
              else raw_df[step_cols].copy()

        if base.empty:
            base = sub
        elif has_batch:
            base = base.merge(sub, on=batch_col, how="outer")
        else:
            base = pd.concat([base.reset_index(drop=True),
                              sub.reset_index(drop=True)], axis=1)

    # 數值欄轉型
    non_id = [c for c in base.columns if c != batch_col]
    base[non_id] = base[non_id].apply(pd.to_numeric, errors="coerce")
    base = base.dropna(axis=1, how="all")
    return base


# ── Session state init ────────────────────────────────────────────────────────
_SESSION_DEFAULTS = {
    "raw_df": None, "dfs_dict": None,
    "selected_process_df": None, "clean_df": None,
    "selected_steps": [],
    "X": None, "y": None, "label": None, "x_scaled": None,
}
for k, v in _SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
            batch_candidates = [c for c in raw_df.columns if "BatchID" in c]
            if batch_candidates:
                raw_df = raw_df.rename(columns={batch_candidates[0]: "BatchID"})
            non_batch = raw_df.columns.difference(["BatchID"])
            raw_df[non_batch] = raw_df[non_batch].apply(pd.to_numeric, errors="coerce")
            st.session_state["raw_df"]          = raw_df
            st.session_state["dfs_dict"]        = split_process_df(raw_df)
            st.session_state["selected_steps"]  = []
            st.session_state["selected_process_df"] = None
            st.session_state["clean_df"]        = None
            st.success(f"✅ 載入成功！{raw_df.shape[0]} 筆 × {raw_df.shape[1]} 欄")
        except Exception as e:
            st.error(f"載入失敗：{e}")

    # ── 製程步驟選擇 ──────────────────────────────────────────
    st.markdown("---")
    if st.session_state["dfs_dict"]:
        st.markdown("### ⚙️ 製程步驟選擇")
        process_list = list(st.session_state["dfs_dict"].keys())

        select_mode = st.radio(
            "選擇模式",
            ["📋 單一步驟", "🔀 多步驟合併"],
            horizontal=True,
            key="sidebar_mode",
        )

        prev_steps = st.session_state.get("selected_steps", [])

        if select_mode == "📋 單一步驟":
            # 若之前是多選，取第一個作為預設
            default_single = prev_steps[0] if prev_steps and prev_steps[0] in process_list \
                             else process_list[0]
            sel = st.selectbox("選擇製程步驟", process_list,
                               index=process_list.index(default_single),
                               key="sidebar_single")
            selected_steps = [sel]
            selected_process = sel

        else:
            default_multi = prev_steps if prev_steps else process_list[:min(2, len(process_list))]
            selected_steps = st.multiselect(
                "選擇製程步驟（可多選）",
                process_list,
                default=[s for s in default_multi if s in process_list],
                key="sidebar_multi",
            )
            if not selected_steps:
                st.warning("請至少選擇一個步驟。")
                selected_steps = [process_list[0]]
            selected_process = "  +  ".join(selected_steps)

        # 偵測步驟變更 → 清除 clean_df 避免舊特徵殘留
        if set(selected_steps) != set(prev_steps):
            st.session_state["clean_df"] = None

        # 更新 session state
        st.session_state["selected_steps"]   = selected_steps
        st.session_state["selected_process"] = selected_process

        merged = _merge_process_steps(
            st.session_state["raw_df"],
            st.session_state["dfs_dict"],
            selected_steps,
        )
        st.session_state["selected_process_df"] = merged

        # 摘要資訊
        if merged is not None:
            n_param = merged.shape[1] - (1 if "BatchID" in merged.columns else 0)
            mode_icon = "🔀" if len(selected_steps) > 1 else "📋"
            st.caption(f"{mode_icon} {merged.shape[0]} 批次 × {n_param} 參數")
            if len(selected_steps) > 1:
                st.info("多步驟模式：欄位保留 `步驟:參數` 完整名稱", icon="ℹ️")

    st.markdown("---")
    st.caption("Bioprocess Analytics v2.0")


# ── Guard ─────────────────────────────────────────────────────────────────────
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
selected_steps      = st.session_state.get("selected_steps", [])

# 標題列顯示當前步驟 badges
if selected_steps:
    badges = "".join(f'<span class="step-badge">{s}</span>' for s in selected_steps)
    st.markdown(f"**當前分析步驟：** {badges}", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔭 跨製程監控",
    "📊 資料總覽",
    "📈 趨勢圖",
    "🔧 特徵工程",
    "🔍 缺失值分析",
    "🔗 相關性分析",
    "🧩 PCA 分析",
    "🌲 特徵重要性",
    "📐 統計檢定",
    "📚 文獻佐證分析",
])

with tabs[0]:
    tab_cross_process.render(raw_df)

with tabs[1]:
    tab_overview.render(raw_df, dfs_dict, selected_process_df, selected_process)

with tabs[2]:
    tab_trend.render(selected_process_df)

with tabs[3]:
    tab_feature_eng.render(selected_process_df)

with tabs[4]:
    tab_missing.render(selected_process_df)

with tabs[5]:
    tab_correlation.render(selected_process_df)

with tabs[6]:
    tab_pca.render(selected_process_df)

with tabs[7]:
    tab_feature_importance.render(selected_process_df)

with tabs[8]:
    tab_stat_test.render(selected_process_df)

with tabs[9]:
    tab_literature.render()
