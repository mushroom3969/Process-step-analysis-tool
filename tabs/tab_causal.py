"""Tab — 因果推論分析"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def _partial_corr_matrix(df: "pd.DataFrame") -> tuple:
    """
    用「迴歸殘差法」計算偏相關矩陣，不需要矩陣求逆，數值更穩定。

    原理：partial_corr(Xi, Xj | 其他) = corr(ei, ej)
      ei = Xi 對「所有其他變數」迴歸後的殘差
      ej = Xj 對「所有其他變數」迴歸後的殘差

    回傳 (pcorr_df, warning_msg)
    """
    from sklearn.linear_model import Ridge as _Ridge

    warns = []

    # ── Step 1：移除常數欄和含 NaN/Inf 的欄位 ───────────────
    clean = df.dropna()
    arr_raw = clean.values.astype(float)
    valid_mask = (
        np.isfinite(arr_raw).all(axis=0) &
        (np.std(arr_raw, axis=0) > 1e-10)
    )
    cols_valid = [c for c, v in zip(df.columns, valid_mask) if v]
    removed    = [c for c, v in zip(df.columns, valid_mask) if not v]
    if removed:
        warns.append(f"已自動移除常數或含異常值的欄位：{', '.join(str(r) for r in removed)}")

    if len(cols_valid) < 2:
        return None, "有效欄位不足 2 個（其餘皆為常數或含異常值）。"

    arr = arr_raw[:, valid_mask]
    n_samples, n_feats = arr.shape

    # ── Step 2：樣本數檢查 ───────────────────────────────────
    if n_samples < n_feats + 2:
        return None, (
            f"樣本數（{n_samples}）< 特徵數 + 2（{n_feats + 2}），"
            f"請將特徵數減少至 {max(2, n_samples - 3)} 個以下。"
        )

    # ── Step 3：迴歸殘差法 ───────────────────────────────────
    # 對每個欄位，用其他所有欄位做 Ridge 迴歸，取殘差
    # 用 Ridge 而非 OLS 是為了在共線性時仍能穩定求解
    alpha_ridge = 1e-3   # 很小的正則化，幾乎不影響結果
    residuals = np.zeros_like(arr)
    for i in range(n_feats):
        others = np.delete(arr, i, axis=1)
        yi     = arr[:, i]
        reg    = _Ridge(alpha=alpha_ridge, fit_intercept=True)
        reg.fit(others, yi)
        residuals[:, i] = yi - reg.predict(others)

    # ── Step 4：計算殘差之間的相關係數 = 偏相關 ─────────────
    pcorr_arr = np.corrcoef(residuals.T)
    # 確保值域在 [-1, 1] 且無 NaN（corrcoef 理論上應該是，但以防萬一）
    pcorr_arr = np.clip(pcorr_arr, -1.0, 1.0)
    np.fill_diagonal(pcorr_arr, 1.0)

    if not np.isfinite(pcorr_arr).all():
        # fallback：把 NaN 填 0
        pcorr_arr = np.nan_to_num(pcorr_arr, nan=0.0)
        warns.append("部分偏相關無法計算，已填補為 0。")

    result_df = pd.DataFrame(pcorr_arr, index=cols_valid, columns=cols_valid)
    warn_msg  = "　".join(warns) if warns else None
    return result_df, warn_msg



def _pc_skeleton(corr_mat: "np.ndarray", n_samples: int,
                 alpha: float = 0.05) -> "np.ndarray":
    """
    PC 演算法骨架：從完全連接圖出發，用費雪 Z 轉換做條件獨立檢定，
    移除 p > alpha 的邊。（0 階，即無條件相關）
    """
    from scipy.stats import norm as _norm
    n = corr_mat.shape[0]
    adj = np.ones((n, n)) - np.eye(n)   # 完全連接圖
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_mat[i, j]
            r = np.clip(r, -0.9999, 0.9999)
            z = 0.5 * np.log((1 + r) / (1 - r))
            se = 1.0 / np.sqrt(max(n_samples - 3, 1))
            p  = 2 * (1 - _norm.cdf(abs(z) / se))
            if p > alpha:
                adj[i, j] = adj[j, i] = 0
    return adj


def _lingam_pairwise(xi: "np.ndarray", xj: "np.ndarray") -> float:
    """
    PairwiseLiNGAM：用非高斯性（偏度）判斷因果方向。
    回傳 > 0 代表 xi → xj，< 0 代表 xj → xi。
    """
    from scipy.stats import pearsonr as _pr
    xi = (xi - xi.mean()) / (xi.std() + 1e-9)
    xj = (xj - xj.mean()) / (xj.std() + 1e-9)
    r, _ = _pr(xi, xj)
    r = np.clip(r, -0.9999, 0.9999)
    # residuals of xi on xj and xj on xi
    res_ij = xi - r * xj   # residual of xi after removing xj
    res_ji = xj - r * xi
    # score = skewness of residual * skewness of cause
    # if xi causes xj: residual(xj|xi) should be more non-Gaussian
    score = float(np.mean(res_ji**3)) * float(np.mean(xj**3))           - float(np.mean(res_ij**3)) * float(np.mean(xi**3))
    return score


def _regression_ate(X: "pd.DataFrame", treatment: str,
                    outcome: str, covariates: list,
                    n_grid: int = 20) -> tuple:
    """
    回歸調整法 ATE：
    用 OLS 估計 E[Y | do(T=t)]，掃描 T 的分位數，
    回傳 (t_vals, ate_vals, conf_low, conf_high)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import warnings

    cols = [treatment] + covariates + [outcome]
    df_clean = X[cols].dropna()
    if len(df_clean) < 10:
        return None

    T = df_clean[treatment].values
    Y = df_clean[outcome].values
    C = df_clean[covariates].values if covariates else np.zeros((len(df_clean), 1))

    # 標準化特徵
    sc = StandardScaler()
    C_sc = sc.fit_transform(C) if C.shape[1] > 0 else C

    # 回歸 Y ~ T + covariates
    Xmat = np.column_stack([T.reshape(-1, 1), C_sc])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(Xmat, Y)

    # 掃描 T 的分位數
    t_vals = np.percentile(T, np.linspace(5, 95, n_grid))
    ate_vals = []
    conf_low, conf_high = [], []

    # Bootstrap CI（200 次）
    n_boot = 200
    boot_coefs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(df_clean), len(df_clean), replace=True)
        Xb = Xmat[idx]; Yb = Y[idx]
        try:
            rb = LinearRegression().fit(Xb, Yb)
            boot_coefs.append(rb.coef_[0])
        except Exception:
            pass

    boot_coefs = np.array(boot_coefs) if boot_coefs else np.array([reg.coef_[0]])
    se = float(np.std(boot_coefs))
    coef_T = reg.coef_[0]

    # E[Y | do(T=t)] = coef_T * t + E[coef_C @ C_mean]
    C_mean = C_sc.mean(axis=0)
    base = reg.intercept_ + C_mean @ reg.coef_[1:]
    for t in t_vals:
        pred = base + coef_T * t
        ate_vals.append(float(pred))
        conf_low.append(float(pred - 1.96 * se * abs(t)))
        conf_high.append(float(pred + 1.96 * se * abs(t)))

    return t_vals, np.array(ate_vals), np.array(conf_low), np.array(conf_high), coef_T, se



def _spring_layout(nodes: list, edges: list, seed: int = 42,
                   iterations: int = 80, k: float = None) -> dict:
    """
    簡易 Fruchterman-Reingold spring layout，不依賴 networkx。
    回傳 {node: (x, y)} dict。
    """
    rng = np.random.default_rng(seed)
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.5, 0.5)}

    # 初始位置：圓形均勻分布
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        pos[node] = np.array([0.5 + 0.4 * np.cos(angle),
                               0.5 + 0.4 * np.sin(angle)], dtype=float)

    if k is None:
        k = 1.0 / np.sqrt(n)

    node_idx = {node: i for i, node in enumerate(nodes)}
    adj = set((node_idx[u], node_idx[v]) for u, v in edges)
    adj |= set((node_idx[v], node_idx[u]) for u, v in edges)  # 無向

    for _ in range(iterations):
        disp = {node: np.zeros(2) for node in nodes}
        # 斥力
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i >= j:
                    continue
                delta = pos[ni] - pos[nj]
                dist  = max(np.linalg.norm(delta), 1e-6)
                force = k**2 / dist
                disp[ni] +=  force * delta / dist
                disp[nj] -=  force * delta / dist
        # 引力（有邊的節點）
        for i, j in adj:
            if i >= j:
                continue
            ni, nj = nodes[i], nodes[j]
            delta = pos[ni] - pos[nj]
            dist  = max(np.linalg.norm(delta), 1e-6)
            force = dist**2 / k
            disp[ni] -= force * delta / dist
            disp[nj] += force * delta / dist
        # 更新位置
        temp = 0.1 * (1 - _ / iterations)
        for node in nodes:
            d = disp[node]
            dn = max(np.linalg.norm(d), 1e-6)
            pos[node] += min(dn, temp) * d / dn
            # 限制在 [0.05, 0.95]
            pos[node] = np.clip(pos[node], 0.05, 0.95)

    return {node: tuple(pos[node].tolist()) for node in nodes}


def _draw_graph(ax, nodes: list, edges: list, pos: dict,
                node_colors: list, node_sizes: list,
                directed: bool = False,
                edge_weights: list = None,
                label_fontsize: int = 7):
    """用 matplotlib patches 畫圖，不依賴 networkx。"""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 邊
    for idx, (u, v) in enumerate(edges):
        pu = np.array(pos[u]); pv = np.array(pos[v])
        lw = float(edge_weights[idx]) * 3 if edge_weights else 1.5
        lw = max(0.5, min(lw, 5.0))

        if directed:
            # 箭頭（縮短起點終點避免被節點遮住）
            delta = pv - pu
            dist  = max(np.linalg.norm(delta), 1e-6)
            r_node = 0.025
            pu_adj = pu + delta / dist * r_node
            pv_adj = pv - delta / dist * r_node
            ax.annotate("", xy=pv_adj, xytext=pu_adj,
                        arrowprops=dict(arrowstyle="-|>", color="#555555",
                                        lw=lw, mutation_scale=15))
        else:
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                    color="#888888", lw=lw, alpha=0.7, zorder=1)

    # 節點
    for i, node in enumerate(nodes):
        px, py = pos[node]
        r = np.sqrt(node_sizes[i]) / 1800
        circle = plt.Circle((px, py), r, color=node_colors[i],
                             zorder=2, alpha=0.88)
        ax.add_patch(circle)
        ax.text(px, py, str(node)[:30],
                ha="center", va="center", fontsize=label_fontsize,
                color="black", fontweight="bold", zorder=3,
                wrap=True)

def _render_causal_tab(X_fi: "pd.DataFrame", y_fi: "pd.Series", top_n_fi: int):
    """因果推論分析 Tab"""

    st.markdown("#### 🔗 因果推論分析")
    st.caption(
        "注意：觀測數據的因果推論結果僅供參考，需結合領域知識驗證。"
        "所有方法均假設「無未測量的混淆因子」（ignorability）。"
    )

    target_col = st.session_state.get("fi_target_col")
    if target_col is None:
        st.warning("請先在特徵重要性頁面設定目標欄位 Y 並準備資料。")
        return

    # 合併 X 和 Y
    all_df = X_fi.copy()
    all_df[target_col] = y_fi.values

    causal_tabs = st.tabs([
        "1️⃣ 偏相關分析",
        "2️⃣ 因果結構發現（PC）",
        "3️⃣ 因果方向（LiNGAM）",
        "4️⃣ 平均處理效果（ATE）",
        "5️⃣ 反事實分析",
    ])

    # ══════════════════════════════════════════════════════════
    # Tab 0：偏相關分析
    # ══════════════════════════════════════════════════════════
    with causal_tabs[0]:
        st.markdown("#### 偏相關矩陣（Partial Correlation）")
        st.caption(
            "偏相關 = 控制所有其他變數後，X 和 Y 之間的直接線性關聯。"
            "普通相關可能因為共同原因（confounding）而虛高，偏相關更接近直接效應。"
        )

        # 選擇納入計算的特徵
        perm_df = st.session_state.get("fi_perm_df")
        if perm_df is not None:
            default_feats = perm_df["Feature"].head(top_n_fi).tolist()
        else:
            default_feats = X_fi.columns[:min(top_n_fi, len(X_fi.columns))].tolist()

        sel_feats_pc = st.multiselect(
            "選擇納入偏相關的特徵（建議 ≤ 20 個，樣本數需 > 特徵數）",
            X_fi.columns.tolist(),
            default=default_feats[:min(15, len(default_feats))],
            key="pc_sel_feats"
        )

        if len(sel_feats_pc) < 2:
            st.warning("請選擇至少 2 個特徵。")
        else:
            sub_df = all_df[sel_feats_pc + ([target_col] if target_col not in sel_feats_pc else [])].dropna()

            if st.button("🧮 計算偏相關", key="run_partial_corr"):
                with st.spinner("計算中..."):
                    pcorr, warn_msg = _partial_corr_matrix(sub_df)
                    corr = sub_df.corr()
                if pcorr is None:
                    st.error(f"❌ 無法計算偏相關：{warn_msg}")
                    st.info(f"目前樣本數：{len(sub_df)}　特徵數：{len(sub_df.columns)}　"
                            f"需要：樣本數 > 特徵數 + 2 = {len(sub_df.columns) + 2}")
                else:
                    if warn_msg:
                        st.warning(warn_msg)
                    st.session_state["causal_pcorr"] = pcorr
                    st.session_state["causal_corr"]  = corr
                    st.success(f"✅ 完成！樣本數={len(sub_df)}，特徵數={len(sub_df.columns)}")

            if st.session_state.get("causal_pcorr") is not None:
                pcorr = st.session_state["causal_pcorr"]
                corr  = st.session_state["causal_corr"]

                # 只顯示有共同欄位的部分
                common_cols = [c for c in pcorr.columns if c in sub_df.columns]
                if not common_cols:
                    st.error("偏相關矩陣與原始欄位沒有交集，可能所有欄位都被自動移除（常數或含異常值）。請減少選擇的特徵數或檢查資料品質。")
                    st.stop()
                pcorr = pcorr.loc[common_cols, common_cols]
                corr  = corr.loc[common_cols, common_cols]

                compare_mode = st.radio(
                    "顯示模式", ["偏相關", "普通相關", "差值（偏相關 − 普通相關）"],
                    horizontal=True, key="pc_mode"
                )
                show_mat = (pcorr if compare_mode == "偏相關"
                            else corr if compare_mode == "普通相關"
                            else pcorr - corr)
                title_map = {"偏相關": "Partial Correlation",
                             "普通相關": "Pearson Correlation",
                             "差值（偏相關 − 普通相關）": "Partial − Pearson（正=直接效應更大）"}

                n_feat = len(common_cols)
                fig_sz = max(8, n_feat * 0.55)
                fig, ax = plt.subplots(figsize=(fig_sz + 1, fig_sz))
                annot_fs = max(6, min(9, int(80 / max(n_feat, 1))))
                sns.heatmap(show_mat, annot=True, fmt=".2f", cmap="RdBu_r",
                            center=0, vmin=-1, vmax=1, ax=ax,
                            annot_kws={"size": annot_fs}, linewidths=0.3,
                            cbar_kws={"shrink": 0.7})
                ax.set_title(title_map[compare_mode], fontsize=12)
                plt.xticks(rotation=45, ha="right", fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                # Y 的偏相關排名
                if target_col in pcorr.columns:
                    st.markdown(f"**{target_col} 的偏相關排名**（直接效應大小）")

                    # 取 Y 欄，並確保兩個 Series 的 index 完全對齊
                    y_pcorr = pcorr[target_col].drop(target_col, errors="ignore")
                    y_corr  = corr[target_col].drop(target_col, errors="ignore")

                    # 取交集 index，確保可以相減
                    shared_idx = y_pcorr.index.intersection(y_corr.index)
                    y_pcorr = y_pcorr.reindex(shared_idx).astype(float)
                    y_corr  = y_corr.reindex(shared_idx).astype(float)

                    # 依偏相關絕對值排序
                    order   = y_pcorr.abs().sort_values(ascending=False).index
                    y_pcorr = y_pcorr.reindex(order)
                    y_corr  = y_corr.reindex(order)
                    diff    = (y_pcorr - y_corr).round(4)

                    rank_df = pd.DataFrame({
                        "Feature":         list(order),
                        "偏相關":          y_pcorr.round(4).tolist(),
                        "普通相關":        y_corr.round(4).tolist(),
                        "差值（偏-普）":   diff.tolist(),
                    })

                    def _explain(row):
                        pc, cc = row["偏相關"], row["普通相關"]
                        if pd.isna(pc) or pd.isna(cc):
                            return "—"
                        if abs(pc) >= abs(cc) * 0.8:
                            return "🎯 直接效應為主"
                        if abs(pc) < 0.05:
                            return "⬜ 直接效應微弱"
                        return "⚠️ 可能受混淆因子影響"

                    rank_df["解讀"] = rank_df.apply(_explain, axis=1)

                    # 顯示前確保數值欄都是 float，避免 None/NaN 顯示問題
                    for col in ["偏相關", "普通相關", "差值（偏-普）"]:
                        rank_df[col] = pd.to_numeric(rank_df[col], errors="coerce")

                    st.dataframe(
                        rank_df.style
                            .background_gradient(cmap="RdBu_r", subset=["偏相關", "普通相關"],
                                                  vmin=-1, vmax=1)
                            .background_gradient(cmap="PiYG", subset=["差值（偏-普）"],
                                                  vmin=-0.5, vmax=0.5)
                            .format({
                                "偏相關":      lambda x: f"{x:.4f}" if pd.notna(x) else "—",
                                "普通相關":    lambda x: f"{x:.4f}" if pd.notna(x) else "—",
                                "差值（偏-普）": lambda x: f"{x:+.4f}" if pd.notna(x) else "—",
                            }),
                        use_container_width=True, hide_index=True
                    )

    # ══════════════════════════════════════════════════════════
    # Tab 1：PC 演算法因果結構發現
    # ══════════════════════════════════════════════════════════
    with causal_tabs[1]:
        st.markdown("#### 因果結構發現（PC 演算法骨架）")
        st.caption(
            "PC 演算法透過條件獨立性檢定，從數據中學習變數間的因果骨架（無向圖）。"
            "連線代表兩變數存在直接依賴，無連線代表條件獨立（p > α）。"
        )

        pc1, pc2 = st.columns(2)
        alpha_pc = pc1.select_slider("顯著水準 α", [0.01, 0.05, 0.10, 0.20], value=0.05, key="pc_alpha")
        n_top_pc = pc2.slider("納入特徵數（Top N）", 3, min(20, X_fi.shape[1]), min(10, X_fi.shape[1]), key="pc_topn")

        if st.button("🔍 執行 PC 骨架發現", key="run_pc", type="primary"):
            perm_df = st.session_state.get("fi_perm_df")
            if perm_df is not None:
                top_feats_pc = perm_df["Feature"].head(n_top_pc).tolist()
            else:
                top_feats_pc = X_fi.columns[:n_top_pc].tolist()

            if target_col not in top_feats_pc:
                top_feats_pc.append(target_col)

            sub_pc = all_df[top_feats_pc].dropna()
            corr_pc = sub_pc.corr().values
            with st.spinner("計算條件獨立性..."):
                adj = _pc_skeleton(corr_pc, len(sub_pc), alpha=alpha_pc)
                st.session_state["causal_pc_adj"]   = adj
                st.session_state["causal_pc_feats"] = top_feats_pc
            st.success(f"✅ 完成！發現 {int(adj.sum() / 2):.0f} 條邊")

        if st.session_state.get("causal_pc_adj") is not None:
            adj        = st.session_state["causal_pc_adj"]
            feat_names = st.session_state["causal_pc_feats"]
            n_nodes    = len(feat_names)

            # 建立邊列表
            edges_pc = [(feat_names[i], feat_names[j])
                        for i in range(n_nodes) for j in range(i+1, n_nodes)
                        if adj[i, j] > 0]
            node_colors_pc = ["#e84855" if n == target_col else "#2e86ab"
                               for n in feat_names]
            node_sizes_pc  = [1200 if n == target_col else 800
                               for n in feat_names]
            pos_pc = _spring_layout(feat_names, edges_pc, seed=42,
                                    k=2.5 / max(np.sqrt(n_nodes), 1))

            fig, ax = plt.subplots(figsize=(12, 8))
            _draw_graph(ax, feat_names, edges_pc, pos_pc,
                        node_colors_pc, node_sizes_pc, directed=False)
            ax.set_title(f"PC 因果骨架（α={alpha_pc}，紅色=目標變數）", fontsize=13)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # 連接度排名
            degree_map = {n: 0 for n in feat_names}
            for u, v in edges_pc:
                degree_map[u] += 1
                degree_map[v] += 1
            y_neighbors = {v for u, v in edges_pc if u == target_col} |                           {u for u, v in edges_pc if v == target_col}
            degree_df = pd.DataFrame({
                "Feature": feat_names,
                "連接數（Degree）": [degree_map[n] for n in feat_names],
            }).sort_values("連接數（Degree）", ascending=False).reset_index(drop=True)
            degree_df["是否連接Y"] = degree_df["Feature"].apply(
                lambda f: "✅" if f in y_neighbors or f == target_col else "—"
            )
            st.dataframe(degree_df, use_container_width=True, hide_index=True)
            st.caption(
                "連接數高的節點是「中樞變數」，可能是混淆因子或關鍵中介變數。"
                "建議結合偏相關結果一起解讀。"
            )

    # ══════════════════════════════════════════════════════════
    # Tab 2：LiNGAM 因果方向
    # ══════════════════════════════════════════════════════════
    with causal_tabs[2]:
        st.markdown("#### 因果方向判斷（Pairwise LiNGAM）")
        st.caption(
            "LiNGAM（Linear Non-Gaussian Acyclic Model）利用殘差的非高斯性判斷因果方向。"
            "原理：若 X → Y，則 Y 對 X 的殘差比 X 對 Y 的殘差更接近高斯分布。"
            "此方法在數據非高斯（偏態）時效果最佳；若數據接近高斯則方向判斷可靠性下降。"
        )

        lg1, lg2 = st.columns(2)
        n_top_lg = lg1.slider("分析 Top N 特徵", 3, min(15, X_fi.shape[1]),
                               min(8, X_fi.shape[1]), key="lg_topn")
        min_corr_lg = lg2.slider("最低相關性門檻（過濾弱相關對）", 0.0, 0.5, 0.1, 0.05, key="lg_mincorr")

        if st.button("🧭 執行 LiNGAM 方向分析", key="run_lingam", type="primary"):
            perm_df = st.session_state.get("fi_perm_df")
            top_feats_lg = (perm_df["Feature"].head(n_top_lg).tolist()
                            if perm_df is not None else X_fi.columns[:n_top_lg].tolist())
            if target_col not in top_feats_lg:
                top_feats_lg.append(target_col)

            sub_lg = all_df[top_feats_lg].dropna()
            results_lg = []
            with st.spinner("計算因果方向..."):
                for i, fi in enumerate(top_feats_lg):
                    for fj in top_feats_lg[i+1:]:
                        r = float(sub_lg[[fi, fj]].corr().iloc[0, 1])
                        if abs(r) < min_corr_lg:
                            continue
                        xi = sub_lg[fi].values.astype(float)
                        xj = sub_lg[fj].values.astype(float)
                        score = _lingam_pairwise(xi, xj)
                        if score > 0:
                            cause, effect = fi, fj
                        else:
                            cause, effect = fj, fi
                        confidence = min(abs(score) * 10, 1.0)
                        results_lg.append({
                            "因（Cause）": cause,
                            "果（Effect）": effect,
                            "相關係數": round(r, 3),
                            "方向信心": round(confidence, 3),
                            "涉及目標Y": "✅" if target_col in (cause, effect) else "—",
                        })

            if results_lg:
                st.session_state["causal_lingam"] = results_lg
                st.success(f"✅ 完成！分析了 {len(results_lg)} 對特徵")
            else:
                st.warning("沒有符合相關性門檻的特徵對，請降低門檻。")

        if st.session_state.get("causal_lingam"):
            lg_df = pd.DataFrame(st.session_state["causal_lingam"])
            lg_df = lg_df.sort_values("方向信心", ascending=False).reset_index(drop=True)

            # 畫有向圖
            # 建立有向圖資料結構
            nodes_lg = list(dict.fromkeys(
                [r["因（Cause）"] for _, r in lg_df.iterrows()] +
                [r["果（Effect）"] for _, r in lg_df.iterrows()]
            ))
            edges_lg  = [(r["因（Cause）"], r["果（Effect）"]) for _, r in lg_df.iterrows()]
            weights_lg = lg_df["方向信心"].tolist()
            node_c_lg = ["#e84855" if n == target_col else "#2e86ab" for n in nodes_lg]
            node_s_lg = [1200 if n == target_col else 800 for n in nodes_lg]
            pos_lg = _spring_layout(nodes_lg, edges_lg, seed=42,
                                    k=2.5 / max(np.sqrt(len(nodes_lg)), 1))

            fig, ax = plt.subplots(figsize=(12, 8))
            _draw_graph(ax, nodes_lg, edges_lg, pos_lg,
                        node_c_lg, node_s_lg,
                        directed=True, edge_weights=weights_lg)
            ax.set_title("LiNGAM 因果有向圖（紅色=目標變數，箭頭粗細=方向信心）", fontsize=12)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # 找目標 Y 的直接原因
            direct_causes = [row["因（Cause）"] for _, row in lg_df.iterrows()
                             if row["果（Effect）"] == target_col]
            if direct_causes:
                st.success(f"🎯 **{target_col} 的直接原因（LiNGAM 判定）**：{', '.join(direct_causes)}")

            st.dataframe(
                lg_df.style.background_gradient(cmap="Blues", subset=["方向信心"])
                           .background_gradient(cmap="RdBu_r",  subset=["相關係數"]),
                use_container_width=True, hide_index=True
            )
            st.caption(
                "方向信心 > 0.5 代表方向判斷相對可靠。"
                "若數據接近高斯分布（Shapiro-Wilk p 值高），LiNGAM 的方向判斷可靠性較低，請謹慎解讀。"
            )

    # ══════════════════════════════════════════════════════════
    # Tab 3：ATE 平均處理效果
    # ══════════════════════════════════════════════════════════
    with causal_tabs[3]:
        st.markdown("#### 平均處理效果（ATE — Average Treatment Effect）")
        st.caption(
            "ATE = E[Y | do(T=t₂)] − E[Y | do(T=t₁)]，估計「主動將參數 T 從 t₁ 調整到 t₂」對 Y 的期望影響。"
            "使用**回歸調整法**：用線性模型控制協變量後，估計 T 對 Y 的邊際效應。"
        )

        all_feats = [c for c in X_fi.columns if c != target_col]
        a1, a2 = st.columns(2)
        treatment_col = a1.selectbox("處理變數（T，想改變的參數）", all_feats, key="ate_treatment")
        covariate_opts = [c for c in all_feats if c != treatment_col]

        # 預設協變量：RF 前幾名（排除 treatment）
        perm_df = st.session_state.get("fi_perm_df")
        default_cov = []
        if perm_df is not None:
            default_cov = [f for f in perm_df["Feature"].head(8).tolist()
                           if f != treatment_col][:5]

        covariates_sel = a2.multiselect(
            "控制的協變量（排除混淆因子影響）",
            covariate_opts, default=default_cov[:min(5, len(default_cov))],
            key="ate_covariates"
        )

        t_vals_range = all_df[treatment_col].dropna()
        t_p05  = float(t_vals_range.quantile(0.05))
        t_p95  = float(t_vals_range.quantile(0.95))
        t_p25  = float(t_vals_range.quantile(0.25))
        t_p75  = float(t_vals_range.quantile(0.75))
        t_step = max((t_p95 - t_p05) / 100, 1e-6)

        st.caption(
            f"📌 數據實際範圍：**{t_p05:.4f}**（5th pct）~ **{t_p95:.4f}**（95th pct）　"
            f"建議 t₁、t₂ 設在此範圍內，避免外插。"
        )

        b1, b2 = st.columns(2)
        t_from = b1.slider(
            "T 從（t₁）", min_value=t_p05, max_value=t_p95,
            value=t_p25, step=t_step, format="%.4f", key="ate_from"
        )
        t_to = b2.slider(
            "T 到（t₂）", min_value=t_p05, max_value=t_p95,
            value=t_p75, step=t_step, format="%.4f", key="ate_to"
        )

        if t_from >= t_to:
            st.warning("⚠️ t₁ 必須小於 t₂，請重新調整。")

        if st.button("📐 估計 ATE", key="run_ate", type="primary"):
            with st.spinner("ATE 估計中..."):
                result = _regression_ate(
                    all_df, treatment_col, target_col,
                    covariates_sel, n_grid=30
                )
                if result is None:
                    st.error("有效樣本數不足，請減少協變量數量。")
                else:
                    t_grid, ate_grid, ci_low, ci_high, coef_t, se_t = result
                    st.session_state.update({
                        "ate_result":    (t_grid, ate_grid, ci_low, ci_high),
                        "ate_coef_t":    coef_t,
                        "ate_se_t":      se_t,
                        "ate_treatment_val": treatment_col,
                        "ate_from_val":      t_from,
                        "ate_to_val":        t_to,
                    })
            st.success("✅ ATE 估計完成！")

        if st.session_state.get("ate_result") is not None:
            t_grid, ate_grid, ci_low, ci_high = st.session_state["ate_result"]
            coef_t   = st.session_state["ate_coef_t"]
            se_t     = st.session_state["ate_se_t"]
            treat_nm = st.session_state.get("ate_treatment_val", treatment_col)
            t_f      = st.session_state.get("ate_from_val", t_from)
            t_t      = st.session_state.get("ate_to_val", t_to)

            ate_point = float(coef_t * (t_t - t_f))
            ate_ci    = 1.96 * se_t * abs(t_t - t_f)

            col1, col2, col3 = st.columns(3)
            col1.metric("ATE（點估計）",  f"{ate_point:+.4f}",
                        help=f"{treat_nm}: {t_f:.3f} → {t_t:.3f}")
            col2.metric("95% CI",
                        f"[{ate_point - ate_ci:.4f}, {ate_point + ate_ci:.4f}]")
            col3.metric("單位效應（每增加1單位）", f"{coef_t:+.4f}")

            if abs(ate_point) < ate_ci:
                st.warning("⚠️ ATE 估計的 95% CI 包含 0，此效應在統計上不顯著。")
            else:
                direction = "增加" if ate_point > 0 else "降低"
                st.success(
                    f"✅ 將 **{treat_nm}** 從 {t_f:.3f} 調整到 {t_t:.3f}，"
                    f"預期使 **{target_col}** {direction} **{abs(ate_point):.4f}**（95% CI: ±{ate_ci:.4f}）"
                )

            # Dose-Response 曲線
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(t_grid, ci_low, ci_high, alpha=0.2, color="#2e86ab", label="95% CI")
            ax.plot(t_grid, ate_grid, color="#2e86ab", lw=2.5, label=f"E[Y | do(T=t)]")
            ax.axvline(t_f, color="#f4a261", ls="--", lw=1.5, label=f"t₁={t_f:.3f}")
            ax.axvline(t_t, color="#e84855", ls="--", lw=1.5, label=f"t₂={t_t:.3f}")
            # 標出 ATE
            y_f = float(np.interp(t_f, t_grid, ate_grid))
            y_t = float(np.interp(t_t, t_grid, ate_grid))
            ax.annotate("", xy=(t_t, y_t), xytext=(t_f, y_f),
                        arrowprops=dict(arrowstyle="<->", color="#e84855", lw=2))
            ax.text((t_f + t_t) / 2, (y_f + y_t) / 2 + (y_t - y_f) * 0.1,
                    f"ATE={ate_point:+.4f}", color="#e84855", fontsize=10, ha="center")
            # 散點（觀測值）
            ax.scatter(all_df[treat_nm].dropna(),
                       all_df[target_col].dropna().reindex(all_df[treat_nm].dropna().index),
                       color="gray", s=20, alpha=0.4, zorder=0, label="觀測值")
            ax.set_xlabel(f"{treat_nm}（處理變數 T）")
            ax.set_ylabel(f"{target_col}（目標 Y）")
            ax.set_title(f"Dose-Response Curve：E[Y | do(T=t)]")
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ══════════════════════════════════════════════════════════
    # Tab 4：反事實分析
    # ══════════════════════════════════════════════════════════
    with causal_tabs[4]:
        st.markdown("#### 反事實分析（Counterfactual Analysis）")
        st.caption(
            "選擇一個批次，模擬「如果當時某個參數不同」，預測 Y 會如何變化。"
            "使用 RF 模型（需先在 RF tab 訓練）做預測，並與實際觀測值比較。"
        )

        rf_model = st.session_state.get("fi_rf")
        if rf_model is None:
            st.warning("請先在「🌲 RF 重要性」分頁訓練 RF 模型。")
        else:
            # 選擇批次
            batch_ids = (st.session_state["raw_df"]["BatchID"].tolist()
                         if "raw_df" in st.session_state
                            and "BatchID" in st.session_state["raw_df"].columns
                         else list(range(len(X_fi))))
            # 對應 X_fi 的 index
            cf_idx = st.selectbox(
                "選擇要分析的樣本（批次）",
                options=list(range(len(X_fi))),
                format_func=lambda i: str(batch_ids[i]) if i < len(batch_ids) else str(i),
                key="cf_idx"
            )

            x_sample = X_fi.iloc[cf_idx].copy()
            y_actual = float(y_fi.iloc[cf_idx])
            y_pred_orig = float(rf_model.predict(x_sample.values.reshape(1, -1))[0])

            st.markdown(f"**樣本 {batch_ids[cf_idx] if cf_idx < len(batch_ids) else cf_idx}**")
            ci1, ci2 = st.columns(2)
            ci1.metric("實際 Y", f"{y_actual:.4f}")
            ci2.metric("RF 預測 Y（原始）", f"{y_pred_orig:.4f}",
                       delta=f"{y_pred_orig - y_actual:+.4f}")

            st.markdown("**調整特徵值（模擬反事實場景）**")

            # 選要調整的特徵（優先顯示重要特徵）
            perm_df = st.session_state.get("fi_perm_df")
            feat_order = (perm_df["Feature"].tolist() if perm_df is not None
                          else X_fi.columns.tolist())
            cf_feats = st.multiselect(
                "選擇要改變的特徵", feat_order,
                default=feat_order[:min(3, len(feat_order))],
                key="cf_feats"
            )

            cf_new_vals = {}
            if cf_feats:
                cf_cols = st.columns(min(3, len(cf_feats)))
                for idx_f, feat in enumerate(cf_feats):
                    orig_val = float(x_sample[feat])
                    feat_min = float(X_fi[feat].min())
                    feat_max = float(X_fi[feat].max())
                    new_val  = cf_cols[idx_f % len(cf_cols)].slider(
                        f"{feat[:30]}（原={orig_val:.3f}）",
                        min_value=feat_min, max_value=feat_max,
                        value=orig_val, step=(feat_max - feat_min) / 100,
                        key=f"cf_slider_{feat}"
                    )
                    cf_new_vals[feat] = new_val

                # 建立反事實樣本
                x_cf = x_sample.copy()
                for feat, val in cf_new_vals.items():
                    x_cf[feat] = val
                y_cf = float(rf_model.predict(x_cf.values.reshape(1, -1))[0])

                delta_cf = y_cf - y_pred_orig
                st.markdown("---")
                cf1, cf2, cf3 = st.columns(3)
                cf1.metric("反事實預測 Y", f"{y_cf:.4f}")
                cf2.metric("預測變化量（Δ）", f"{delta_cf:+.4f}",
                           delta_color="normal")
                cf3.metric("相對變化", f"{delta_cf / (abs(y_pred_orig) + 1e-9) * 100:+.2f}%")

                # 特徵變化 vs Y 變化的瀑布圖
                st.markdown("**特徵變化貢獻（近似瀑布圖）**")
                contrib_data = []
                x_incremental = x_sample.copy()
                y_prev = y_pred_orig
                for feat in cf_feats:
                    x_incremental[feat] = cf_new_vals[feat]
                    y_curr = float(rf_model.predict(x_incremental.values.reshape(1, -1))[0])
                    delta  = y_curr - y_prev
                    change = cf_new_vals[feat] - float(x_sample[feat])
                    contrib_data.append({
                        "特徵": feat[:35],
                        "原始值": round(float(x_sample[feat]), 4),
                        "新值": round(cf_new_vals[feat], 4),
                        "變化量": round(change, 4),
                        "Y 貢獻（Δ）": round(delta, 4),
                    })
                    y_prev = y_curr

                contrib_df = pd.DataFrame(contrib_data)

                fig, ax = plt.subplots(figsize=(10, max(4, len(cf_feats) * 0.5 + 2)))
                colors_cf = ["#e84855" if v > 0 else "#2e86ab"
                             for v in contrib_df["Y 貢獻（Δ）"]]
                ax.barh(contrib_df["特徵"], contrib_df["Y 貢獻（Δ）"],
                        color=colors_cf, alpha=0.85)
                ax.axvline(0, color="black", lw=1.2)
                ax.set_xlabel("對 Y 的邊際貢獻（Δ）")
                ax.set_title("反事實：各特徵調整對 Y 的貢獻")
                ax.invert_yaxis(); ax.grid(axis="x", alpha=0.4)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.dataframe(
                    contrib_df.style
                        .background_gradient(cmap="RdBu_r", subset=["Y 貢獻（Δ）"])
                        .format({"原始值": "{:.4f}", "新值": "{:.4f}",
                                 "變化量": "{:+.4f}", "Y 貢獻（Δ）": "{:+.4f}"}),
                    use_container_width=True, hide_index=True
                )




def render(selected_process_df):
    """Tab 入口：從 session_state 取得 X, y，呼叫主邏輯"""
    st.header("因果推論分析")

    work_df = selected_process_df
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    target_col = st.session_state.get("fi_target_col")
    if target_col is None or target_col not in work_df.columns:
        st.warning("請先至「特徵重要性」分頁設定目標欄位 Y 並準備資料，因果推論需要 X 和 Y。")
        return

    numeric_cols = [c for c in work_df.select_dtypes(include=["number"]).columns
                    if c != "BatchID"]
    top_n = st.slider("分析用 Top N 特徵（依 RF 重要性，未執行 RF 則依欄位順序）",
                      3, min(20, len(numeric_cols) - 1), min(10, len(numeric_cols) - 1),
                      key="causal_topn")

    # 取 X（排除 target 和 BatchID）
    feat_cols = [c for c in numeric_cols if c != target_col]
    X_fi = work_df[feat_cols].copy()
    y_fi = work_df[target_col].copy()
    valid = y_fi.notna() & X_fi.notna().all(axis=1)
    X_fi = X_fi[valid].reset_index(drop=True)
    y_fi = y_fi[valid].reset_index(drop=True)

    if len(X_fi) < 10:
        st.error("有效樣本數不足 10 筆，無法執行因果分析。")
        return

    _render_causal_tab(X_fi, y_fi, top_n)
