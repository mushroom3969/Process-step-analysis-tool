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
# ── 多重共線性診斷（VIF + Pairwise |r|）────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個數值特徵的 VIF（Variance Inflation Factor）。
    VIF_i = 1 / (1 - R²_i)，其中 R²_i 是以其餘特徵迴歸第 i 個特徵的 R²。
    缺失值以欄位中位數填補後計算。
    """
    from sklearn.linear_model import LinearRegression
    num_cols = [c for c in df.select_dtypes(include='number').columns
                if c.lower() != 'batchid']
    if len(num_cols) < 2:
        return pd.DataFrame(columns=['Feature', 'VIF'])

    X = df[num_cols].copy()
    X = X.fillna(X.median())

    vif_vals = []
    for col in num_cols:
        y = X[col].values
        others = X.drop(columns=[col]).values
        if others.shape[1] == 0:
            vif_vals.append(np.nan)
            continue
        r2 = LinearRegression().fit(others, y).score(others, y)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_vals.append(round(float(vif), 2))

    vif_df = pd.DataFrame({'Feature': num_cols, 'VIF': vif_vals})
    return vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)


def _iterative_vif_elimination(
    df: pd.DataFrame,
    vif_threshold: float = 5.0,
    max_iter: int = 20,
    protected_cols: list = None,
) -> tuple:
    """
    迭代 VIF 剔除：每輪從可刪除特徵中移除 VIF 最高者，重新計算，
    直到所有可刪除特徵 VIF ≤ threshold，或達到 max_iter 輪次。
    回傳 (處理後 DataFrame, 剔除記錄 list[dict])
    """
    from sklearn.linear_model import LinearRegression
    if protected_cols is None:
        protected_cols = []
    protected_set = set(protected_cols)
    df_work = df.copy()
    log = []

    for iteration in range(1, max_iter + 1):
        num_cols = [c for c in df_work.select_dtypes(include='number').columns
                    if c.lower() != 'batchid']
        eligible = [c for c in num_cols if c not in protected_set]

        if len(num_cols) < 2 or not eligible:
            break

        X = df_work[num_cols].copy()
        X = X.fillna(X.median())

        # 計算保護欄外的每個可刪除特徵之 VIF（保護欄仍參與迴歸）
        vif_eligible = {}
        for col in eligible:
            y = X[col].values
            others = X.drop(columns=[col]).values
            if others.shape[1] == 0:
                continue
            r2 = LinearRegression().fit(others, y).score(others, y)
            vif_eligible[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

        if not vif_eligible:
            break

        max_vif_val = max(vif_eligible.values())
        max_vif_col = max(vif_eligible, key=vif_eligible.get)

        if max_vif_val <= vif_threshold:
            log.append({
                '輪次': f'第 {iteration} 輪',
                '動作': '✅ 收斂',
                '移除特徵': '—',
                '該輪最高 VIF': f'{max_vif_val:.2f}',
                '剩餘特徵數': len(num_cols),
            })
            break

        log.append({
            '輪次': f'第 {iteration} 輪',
            '動作': '刪除',
            '移除特徵': max_vif_col.split(':')[-1][:60],
            '該輪最高 VIF': f'{max_vif_val:.2f}',
            '剩餘特徵數': len(num_cols) - 1,
        })
        df_work = df_work.drop(columns=[max_vif_col])
    else:
        num_remaining = len([c for c in df_work.select_dtypes(include='number').columns
                             if c.lower() != 'batchid'])
        log.append({
            '輪次': f'第 {max_iter} 輪（達上限）',
            '動作': '⚠️ 未收斂',
            '移除特徵': '—',
            '該輪最高 VIF': '—',
            '剩餘特徵數': num_remaining,
        })

    return df_work, log


def _high_vif_pairs(df: pd.DataFrame, cols: list[str],
                    method: str = 'pearson') -> pd.DataFrame:
    """
    計算指定欄位間的 pairwise 相關係數絕對值，回傳上三角配對表。
    method: 'pearson'（線性）或 'spearman'（單調非線性）
    """
    sub = df[cols].fillna(df[cols].median())
    corr = sub.corr(method=method).abs()
    col_label = '|r|' if method == 'pearson' else '|ρ|'
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({
                'Feature A': cols[i],
                'Feature B': cols[j],
                col_label: round(float(corr.iloc[i, j]), 3),
            })
    df_rows = pd.DataFrame(rows)
    if df_rows.empty:
        return pd.DataFrame(columns=['Feature A', 'Feature B', col_label])
    return df_rows.sort_values(col_label, ascending=False).reset_index(drop=True)


def _compute_mi_pairs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    計算指定欄位間的 pairwise Mutual Information（sklearn kNN 估計）。
    MI 值單位為 nats，越大代表依賴性越強，0 = 完全獨立。
    可偵測任意非線性關係（非單調亦可）。
    """
    from sklearn.feature_selection import mutual_info_regression
    sub = df[cols].fillna(df[cols].median()).values
    rows = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_info_regression(
                sub[:, i].reshape(-1, 1), sub[:, j], random_state=42
            )[0]
            rows.append({'Feature A': cols[i], 'Feature B': cols[j],
                         'MI': round(float(mi), 4)})
    df_mi = pd.DataFrame(rows)
    if df_mi.empty:
        return pd.DataFrame(columns=['Feature A', 'Feature B', 'MI'])
    return df_mi.sort_values('MI', ascending=False).reset_index(drop=True)


def _render_auto_vif(clean_df: pd.DataFrame, num_cols: list):
    """自動迭代 VIF 剔除 UI（在 _render_collinearity_merge 中以模式切換呼叫）"""
    st.caption(
        "每輪自動移除 VIF 最高的特徵（可刪除範圍內），重新計算，"
        "直到所有特徵 VIF ≤ 門檻或達到最大輪次。\n\n"
        "適合需要快速篩選大量特徵的情境；保護欄位不會被刪除，但仍參與 VIF 計算。"
    )

    a1, a2 = st.columns(2)
    auto_thr = a1.slider(
        "VIF 剔除門檻", 2.0, 20.0, 5.0, 0.5, key="fe_auto_thr",
        help="高於此值的特徵（由高到低）逐輪被移除",
    )
    auto_max_iter = a2.number_input(
        "最大迭代輪次", 1, 100, 20, step=1, key="fe_auto_max",
        help="超過此輪次即使未收斂也停止",
    )

    protected = st.multiselect(
        "保護欄位（不允許被自動刪除）",
        options=num_cols,
        key="fe_auto_protected",
        help="選擇不希望被自動剔除的特徵；它們仍會參與其他欄位的 VIF 計算",
    )

    btn_c1, btn_c2 = st.columns([2, 1])
    run_auto = btn_c1.button("🤖 執行迭代自動剔除", key="fe_auto_run", type="primary")
    if btn_c2.button("🔄 清除結果", key="fe_auto_clear"):
        st.session_state["fe_vif_result"] = None  # 修改這行
        st.session_state["fe_vif_log"] = None     # 修改這行
        st.rerun()
    if run_auto:
        with st.spinner(f"正在迭代 VIF 剔除..."):
            df_result, elim_log = _iterative_vif_elimination(
                clean_df,           # 傳入目前的 DataFrame
                float(auto_thr),    # 傳入 VIF 門檻
                int(auto_max_iter), # 傳入最大迭代次數
                protected           # 傳入保護欄位清單
                )
            # 順便把最終 VIF 算好存起來，避免 rerun 時重複計算
            final_vif_df = _compute_vif(df_result) 
            
        st.session_state["fe_vif_result"] = df_result
        st.session_state["fe_vif_log"] = elim_log
        st.session_state["fe_auto_vif_final"] = final_vif_df # 新增快取
        st.rerun()

    # 讀取時
    final_vif = st.session_state.get("fe_auto_vif_final")

    auto_result = st.session_state.get("fe_vif_result")
    auto_log    = st.session_state.get("fe_vif_log")

    if auto_result is None or auto_log is None:
        st.info("設定參數後，點擊「🤖 執行迭代自動剔除」開始。")
        return

    # ── 統計摘要 ──────────────────────────────────────────────
    before_n   = len(num_cols)
    after_cols = [c for c in auto_result.select_dtypes(include='number').columns
                  if c.lower() != 'batchid']
    after_n    = len(after_cols)
    removed_n  = before_n - after_n

    rm1, rm2, rm3 = st.columns(3)
    rm1.metric("原始特徵數", before_n)
    rm2.metric("移除特徵數", removed_n, delta=f"-{removed_n}", delta_color="inverse")
    rm3.metric("保留特徵數", after_n)

    # ── 逐輪剔除記錄 ──────────────────────────────────────────
    st.markdown("#### 📋 逐輪剔除記錄")
    st.dataframe(pd.DataFrame(auto_log), use_container_width=True, hide_index=True)

    # ── 移除欄位清單 ──────────────────────────────────────────
    removed_cols = [c for c in num_cols if c not in set(after_cols)]
    if removed_cols:
        with st.expander(f"🗑️ 已移除欄位清單（共 {len(removed_cols)} 欄）"):
            st.dataframe(
                pd.DataFrame({"移除特徵": removed_cols}),
                use_container_width=True, hide_index=True,
            )

    # ── 剩餘特徵最終 VIF ──────────────────────────────────────
    st.markdown("#### 📊 剩餘特徵最終 VIF")
    with st.spinner("計算剩餘特徵 VIF..."):
        final_vif = _compute_vif(auto_result)

    def _auto_color(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ''
        if val >= 10: return 'background-color:#FFCCCC'
        if val >= 5:  return 'background-color:#FFF3CC'
        return 'background-color:#CCFFCC'

    st.dataframe(
        final_vif.style.map(_auto_color, subset=['VIF']),
        use_container_width=True, hide_index=True, height=250,
    )

    # ── 套用按鈕 ──────────────────────────────────────────────
    st.markdown("---")
    if st.button("✅ 套用結果至資料集", key="fe_auto_apply", type="primary"):
        snap         = clean_df.copy()
        all_removed  = [c for c in clean_df.columns if c not in auto_result.columns]
        _push_op("vif_reduce", all_removed, [], snap)
        st.session_state["clean_df"]       = auto_result
        st.session_state["fe_vif_df"]      = None
        st.session_state["fe_vif_result"] = None
        st.session_state["fe_vif_log"] = None
        st.success(f"✅ 已套用！共移除 {len(all_removed)} 個特徵。")
        st.rerun()


def _render_collinearity_merge(show_mean: bool = True):
    """Step 2.5 UI：VIF + Pairwise 相關 多重共線性診斷 & 處理"""
    st.markdown("---")
    st.markdown("### 🔗 Step 2.5：多重共線性診斷（VIF + Pairwise 相關）")
    st.caption(
        "Step 1 僅過濾無效欄位（關鍵字）。"
        "此步驟針對**所有剩餘特徵**計算 VIF，找出 Max/Min、Before/After、編號欄等配對共線性，"
        "再由您選擇合併（mean）、計算差值（diff）或直接刪除。\n\n"
        "⚠️ **VIF 僅偵測線性共線性**。"
        "Spearman 可偵測**單調非線性**；MI 可偵測**任意非線性（含非單調）**。"
    )

    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        return

    num_cols = [c for c in clean_df.select_dtypes(include='number').columns
                if c.lower() != 'batchid']
    if len(num_cols) < 2:
        st.info("數值特徵不足 2 欄，無法計算 VIF。")
        return

    # ── 模式切換 ──────────────────────────────────────────────────
    mode = st.radio(
        "診斷模式",
        ["🔍 手動配對診斷", "🤖 迭代自動剔除"],
        horizontal=True,
        key="fe_vif_mode",
    )

    # 確保在進入特定模式前，基本的 clean_df 判斷已經完成
    if mode == "🤖 迭代自動剔除":
        # 這裡建議加上一個 Spinner，讓使用者知道正在載入自動模式的 UI
        with st.container():
            _render_auto_vif(clean_df, num_cols)
        # 務必在這裡 return，避免跑到底下的手動模式參數設定
        return

    # ── 以下為手動模式 ─────────────────────────────────────────────
    # ── 參數設定 ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    vif_warn  = c1.slider("VIF 警示門檻", 2.0, 20.0, 5.0,  0.5, key="fe_vif_warn",
                           help="VIF ≥ 此值 → 黃色警示")
    vif_high  = c2.slider("VIF 嚴重門檻", 5.0, 50.0, 10.0, 1.0, key="fe_vif_high",
                           help="VIF ≥ 此值 → 紅色嚴重")
    corr_method_sel = c3.selectbox(
        "配對相關方法",
        ["Pearson |r|（線性）", "Spearman |ρ|（單調非線性）", "Mutual Information（任意非線性）"],
        key="fe_corr_method",
        help="Pearson: 線性｜Spearman: 單調非線性｜MI: 任意非線性（含非單調）"
    )

    if "Mutual" in corr_method_sel:
        corr_method, corr_label, corr_col = 'mi', 'MI (nats)', 'MI'
        r_thr = st.slider("MI 門檻 (nats)", 0.0, 2.0, 0.1, 0.01, key="fe_r_thr_mi",
                          help="MI ≥ 此值 → 顯示為高依賴性配對；完全獨立時 MI ≈ 0")
    elif "Spearman" in corr_method_sel:
        corr_method, corr_label, corr_col = 'spearman', '|ρ| (Spearman)', '|ρ|'
        r_thr = st.slider("配對 |ρ| 門檻", 0.5, 1.0, 0.85, 0.01, key="fe_r_thr_sp",
                          help="|ρ| ≥ 此值 → 顯示為高度相關配對")
    else:
        corr_method, corr_label, corr_col = 'pearson', '|r| (Pearson)', '|r|'
        r_thr = st.slider("配對 |r| 門檻", 0.5, 1.0, 0.85, 0.01, key="fe_r_thr_pe",
                          help="|r| ≥ 此值 → 顯示為高度相關配對")

    # ── 執行 / 清除按鈕 ───────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([2, 1])
    if btn_col1.button("🔍 執行共線性診斷", key="run_vif", type="primary"):
        with st.spinner(f"計算 {len(num_cols)} 個特徵的 VIF..."):
            vif_df = _compute_vif(clean_df)
        st.session_state["fe_vif_df"]     = vif_df
        st.session_state["fe_pair_cache"] = None   # 清除舊配對快取
        st.rerun()

    if btn_col2.button("🔄 清除診斷結果", key="clear_vif"):
        st.session_state["fe_vif_df"]     = None
        st.session_state["fe_pair_cache"] = None
        st.rerun()

    # ── 讀取已儲存的 VIF 結果 ─────────────────────────────────────
    vif_df = st.session_state.get("fe_vif_df")
    if vif_df is not None:
        # 若 clean_df 的欄位已改變（執行處理後），自動失效快取
        current_cols = set(clean_df.columns)
        cached_cols  = set(vif_df['Feature'].tolist())
        if not cached_cols.issubset(current_cols):
            st.session_state["fe_vif_df"] = None
            vif_df = None
    if vif_df is None:
        st.info("點擊「🔍 執行共線性診斷」開始。")
        return

    # ── VIF 總覽表 ──────────────────────────────────────────────────
    st.markdown("#### 📋 VIF 總覽")

    def _vif_color(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ''
        if val >= vif_high: return 'background-color:#FFCCCC'
        if val >= vif_warn: return 'background-color:#FFF3CC'
        return 'background-color:#CCFFCC'

    n_ok   = int((vif_df['VIF'] < vif_warn).sum())
    n_warn = int(((vif_df['VIF'] >= vif_warn) & (vif_df['VIF'] < vif_high)).sum())
    n_high = int((vif_df['VIF'] >= vif_high).sum())

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("✅ VIF 正常 (< {:.0f})".format(vif_warn), n_ok)
    mc2.metric("⚠️ 警示 ({:.0f}–{:.0f})".format(vif_warn, vif_high), n_warn)
    mc3.metric("🚨 嚴重 (≥ {:.0f})".format(vif_high), n_high)

    st.dataframe(
        vif_df.style.map(_vif_color, subset=['VIF']),
        use_container_width=True, hide_index=True, height=300
    )

    # ── VIF 長條圖（Top 20）──────────────────────────────────────────
    top_vif = vif_df.head(20)
    bar_colors = [
        '#FF6B6B' if v >= vif_high else '#FFD93D' if v >= vif_warn else '#6BCB77'
        for v in top_vif['VIF']
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, len(top_vif) * 0.38)))
    ax.barh(top_vif['Feature'].str[-50:], top_vif['VIF'],
            color=bar_colors, alpha=0.85)
    ax.axvline(vif_warn, color='#FFD93D', lw=1.5, ls='--',
               label=f'VIF={vif_warn:.0f} (警示)')
    ax.axvline(vif_high, color='#FF6B6B', lw=1.5, ls='--',
               label=f'VIF={vif_high:.0f} (嚴重)')
    ax.set_xlabel('VIF'); ax.invert_yaxis()
    ax.set_title('Top 20 Features by VIF', fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── 高 VIF 特徵的 Pairwise 分析 ──────────────────────────────────
    high_vif_cols = [c for c in vif_df.loc[vif_df['VIF'] >= vif_warn, 'Feature'].tolist()
                     if c in clean_df.columns]

    if not high_vif_cols:
        st.success("✅ 所有特徵 VIF 均低於警示門檻，無需處理。")
        return

    # ── 配對計算按鈕（獨立觸發，避免 spinner 打斷 render）──────
    cache_key = f"{corr_method}_{','.join(sorted(high_vif_cols))}"
    pair_cache = st.session_state.get("fe_pair_cache")
    pair_df    = pair_cache["df"] if (pair_cache and pair_cache.get("key") == cache_key) else None

    n_pairs_possible = len(high_vif_cols) * (len(high_vif_cols) - 1) // 2
    btn_label = (
        f"📊 計算配對相關（{corr_label}，共 {n_pairs_possible} 對）"
        if pair_df is None
        else f"🔄 重新計算配對相關（{corr_label}）"
    )
    if st.button(btn_label, key="run_pair", type="secondary"):
        if corr_method == 'mi':
            if len(high_vif_cols) > 30:
                st.warning(f"⚠️ 高 VIF 特徵共 {len(high_vif_cols)} 個，MI 計算可能較慢。")
            try:
                pair_df = _compute_mi_pairs(clean_df, high_vif_cols)
            except Exception as e:
                st.error(f"MI 計算失敗：{e}")
                return
        else:
            try:
                pair_df = _high_vif_pairs(clean_df, high_vif_cols, method=corr_method)
            except Exception as e:
                st.error(f"配對相關計算失敗：{e}")
                return
        st.session_state["fe_pair_cache"] = {"key": cache_key, "df": pair_df}
        # 不呼叫 st.rerun()，直接往下 render 結果

    if pair_df is None:
        st.info("點擊上方按鈕計算配對相關。")
        return

    st.markdown("#### 🔍 高 VIF 特徵配對分析")

    if pair_df.empty:
        st.info("高 VIF 特徵數不足（需 ≥ 2 個），無法計算配對相關。")
        return

    # MI：Top-20 長條圖（全部 score 顯示，門檻線標示）
    if corr_method == 'mi':
        top_all = pair_df.head(20)
        pair_labels = [
            f"{r['Feature A'].split(':')[-1][:20]}↔{r['Feature B'].split(':')[-1][:20]}"
            for _, r in top_all.iterrows()
        ]
        fig2, ax2 = plt.subplots(figsize=(10, max(4, len(top_all) * 0.4)))
        colors = ['#9B59B6' if v >= r_thr else '#CCCCCC' for v in top_all['MI']]
        ax2.barh(pair_labels, top_all['MI'], color=colors, alpha=0.85)
        ax2.axvline(r_thr, color='red', lw=1.5, ls='--', label=f'門檻={r_thr}')
        ax2.set_xlabel('Mutual Information (nats)')
        ax2.set_title('Top 20 MI Pairs（紫色 ≥ 門檻）', fontsize=10)
        ax2.invert_yaxis(); ax2.legend(fontsize=8); ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
        st.caption(f"共 {len(pair_df)} 個配對，最大 MI = {pair_df['MI'].max():.4f}，目前門檻 = {r_thr}")

    elif len(high_vif_cols) <= 25:
        sub = clean_df[high_vif_cols].fillna(clean_df[high_vif_cols].median())
        corr_mat = sub.corr(method=corr_method).abs()
        short_names = {c: c.split(':')[-1][:35] for c in high_vif_cols}
        corr_mat.index   = [short_names[c] for c in high_vif_cols]
        corr_mat.columns = [short_names[c] for c in high_vif_cols]
        sz = max(6, len(high_vif_cols) * 0.55)
        fig2, ax2 = plt.subplots(figsize=(sz, sz))
        sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    vmin=0, vmax=1, linewidths=0.4, ax=ax2, annot_kws={'size': 7})
        ax2.set_title(f'High-VIF Feature Pairwise {corr_label}', fontsize=10)
        plt.xticks(rotation=35, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    # ── 處理選項（data_editor）──────────────────────────────────
    high_pairs = pair_df[pair_df[corr_col] >= r_thr].reset_index(drop=True)
    st.markdown("#### ⚡ 處理高度相關配對")

    if high_pairs.empty:
        st.info(f"目前門檻 {r_thr} 下無配對超過門檻，請降低門檻或查看上方長條圖確認分佈。")
        return

    st.caption(f"共 {len(high_pairs)} 個配對 ≥ {r_thr}。在表格選擇動作，全部選完後點「⚡ 執行處理」。")
    edit_df = high_pairs[[c for c in ['Feature A', 'Feature B', corr_col] if c in high_pairs.columns]].copy()
    edit_df['處理方式'] = '保留兩欄'
    edit_df['新欄位名稱'] = [
        f"{str(fa).split(':')[-1][:25]}_mean" for fa in edit_df['Feature A']
    ]

    edited_df = st.data_editor(
        edit_df,
        column_config={
            'Feature A':  st.column_config.TextColumn('Feature A',  disabled=True),
            'Feature B':  st.column_config.TextColumn('Feature B',  disabled=True),
            corr_col:     st.column_config.NumberColumn(corr_col,   disabled=True, format='%.3f'),
            '處理方式':   st.column_config.SelectboxColumn(
                              '處理方式',
                              options=['保留兩欄', '合併為 mean', '刪除 A', '刪除 B'],
                              required=True,
                          ),
            '新欄位名稱': st.column_config.TextColumn('新欄位名稱（合併時填寫）'),
        },
        use_container_width=True,
        hide_index=True,
        num_rows='fixed',
        key=f"fe_pair_editor_{cache_key[:12]}",
    )

    if st.button("⚡ 執行處理", key="fe_vif_exec", type="primary"):
        df_work = clean_df.copy()
        ops_log = []

        for _, row in edited_df.iterrows():
            fa  = row['Feature A']
            fb  = row['Feature B']
            act = row['處理方式']
            if '保留' in act:
                continue
            if '合併' in act:
                nn = str(row.get('新欄位名稱', '')).strip() or f"{str(fa)[:20]}_mean"
                if fa in df_work.columns and fb in df_work.columns:
                    df_work[nn] = df_work[[fa, fb]].mean(axis=1)
                    df_work.drop(columns=[fa, fb], inplace=True, errors='ignore')
                    ops_log.append(('merge', [fa, fb], [nn]))
            elif '刪除 A' in act and fa in df_work.columns:
                df_work.drop(columns=[fa], inplace=True, errors='ignore')
                ops_log.append(('drop', [fa], []))
            elif '刪除 B' in act and fb in df_work.columns:
                df_work.drop(columns=[fb], inplace=True, errors='ignore')
                ops_log.append(('drop', [fb], []))

        if not ops_log:
            st.warning("所有配對均選擇「保留兩欄」，未做任何變更。")
        else:
            snap = clean_df.copy()
            st.session_state["clean_df"]      = df_work
            st.session_state["fe_vif_df"]     = None   # 執行後清除診斷結果
            st.session_state["fe_pair_cache"] = None
            removed = [c for op in ops_log for c in op[1]]
            added   = [c for op in ops_log for c in op[2]]
            _push_op("vif_reduce", removed, added, snap)
            st.success(f"✅ 完成！移除 {len(removed)} 欄，新增 {len(added)} 欄。")
            st.rerun()


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
        "fe_stat_result":      None,
        "fe_auto_result":      None,
        "df_before_step2":     None,
        "fe_vif_df":           None,   # VIF 診斷結果快取
        "fe_pair_cache":       None,   # 配對相關快取
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
    if auto_res is not None:
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
