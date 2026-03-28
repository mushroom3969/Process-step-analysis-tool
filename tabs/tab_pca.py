"""Tab — PCA 分析（Hotelling T² + SPE + 貢獻分析 + 載荷分析）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skPCA
from scipy.stats import f as f_dist, chi2
from utils import extract_number


# ══════════════════════════════════════════════════════════════════════════════
# ── 統計閾值計算 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _ht2_threshold(alpha: float, n: int, p: int) -> float:
    """Hotelling T² 控制限（Jackson & Mudholkar, 1979）"""
    f_crit = f_dist.ppf(1 - alpha, p, n - p)
    return (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_crit


def _spe_threshold(spe_vals: np.ndarray, alpha: float) -> float:
    """SPE 控制限（chi² 近似）"""
    m = np.mean(spe_vals)
    v = np.var(spe_vals, ddof=1)
    if v == 0:
        return m
    g = v / (2 * m)
    h = (2 * m ** 2) / v
    return g * chi2.ppf(1 - alpha, h)


# ══════════════════════════════════════════════════════════════════════════════
# ── 主 render ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render(selected_process_df):
    st.header("PCA 主成分分析")

    work_df = selected_process_df  # app.py 已統一傳入 active_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_options = [c for c in work_df.select_dtypes(include=["number"]).columns
                       if c != "BatchID"]
    if len(numeric_options) < 2:
        st.warning("有效數值欄位不足 2 個。")
        return

    stored_target = st.session_state.get("target_col")

    # ── 設定區 ────────────────────────────────────────────────
    with st.expander("⚙️ PCA 設定", expanded=True):
        pc1, pc2, pc3 = st.columns(3)

        default_excl = [stored_target] if stored_target and stored_target in numeric_options else []
        excl_cols = pc1.multiselect(
            "排除的欄位（目標變數 Y 等）",
            numeric_options,
            default=default_excl,
            key="pca_excl",
        )
        max_comp = max(3, min(15, len(numeric_options) - len(excl_cols) - 1))
        if max_comp <= 2:
            pc2.warning("欄位數不足，無法調整主成分數。")
            n_components = 2
        else:
            n_components = pc2.slider("最大主成分數", 2, max_comp, min(5, max_comp), key="pca_n_comp")
        alpha_pca = pc3.select_slider("顯著水準 α", [0.01, 0.05, 0.10], value=0.05, key="pca_alpha")

        feat_cols = [c for c in numeric_options if c not in excl_cols]
        pc1.caption(f"納入 PCA：{len(feat_cols)} 個欄位")

    # ── Train / Test Batch 手動選擇 ───────────────────────────
    all_batches = (
        work_df["BatchID"].astype(str).tolist()
        if "BatchID" in work_df.columns
        else [str(i) for i in range(len(work_df))]
    )
    with st.expander("🗂️ Train / Test Batch 分配", expanded=True):
        st.caption(
            "**Train**：用來建立 PCA 模型與控制限，建議選**已知正常**批次。　"
            "**Test**：投影到 Train 空間比對控制限，用來驗證新批次或可疑批次。　"
            "同一批次不能同時出現在 Train 和 Test。"
        )

        col_tr, col_te = st.columns(2)

        with col_tr:
            st.markdown("**✅ Train Batch（建模用）**")
            # 預設：全部批次都在 Train
            prev_test = st.session_state.get("pca_test_batches", [])
            default_train = [b for b in all_batches if b not in prev_test]
            train_batches = st.multiselect(
                "選擇 Train Batch",
                options=all_batches,
                default=default_train,
                key="pca_train_batches",
                help="建議選已知正常的批次來建立 PCA 控制限",
            )
            st.info(f"共 **{len(train_batches)}** 批")

        with col_te:
            st.markdown("**🔍 Test Batch（驗證用）**")
            # Test 只能從「不在 Train 裡」的批次選，防止重疊
            test_options = [b for b in all_batches if b not in train_batches]
            prev_test_valid = [b for b in st.session_state.get("pca_test_batches", [])
                               if b in test_options]
            test_batches = st.multiselect(
                "選擇 Test Batch",
                options=test_options,
                default=prev_test_valid,
                key="pca_test_batches",
                help="未被選為 Train 的批次才會出現在這裡",
            )
            if test_options:
                st.info(f"共 **{len(test_batches)}** 批（可選範圍：{len(test_options)} 批）")
            else:
                st.warning("所有批次都已在 Train，請先從 Train 移除部分批次。")

        # 未分配的批次提示
        unassigned = [b for b in all_batches
                      if b not in train_batches and b not in test_batches]
        if unassigned:
            st.warning(
                f"⚠️ 以下 **{len(unassigned)}** 批次尚未分配（不會納入分析）：　"
                + "、".join(unassigned[:10])
                + ("…" if len(unassigned) > 10 else "")
            )

    if st.button("🧩 執行 PCA", key="run_pca", type="primary"):
        if len(train_batches) < 3:
            st.error("Train Batch 數量至少需要 3 筆才能建立 PCA 模型。")
            return

        # ── 依 Batch 分割資料 ─────────────────────────────
        batch_col = work_df["BatchID"].astype(str) if "BatchID" in work_df.columns                     else pd.Series([str(i) for i in range(len(work_df))])
        train_mask = batch_col.isin(train_batches)
        test_mask  = batch_col.isin(test_batches)

        df_train = work_df[train_mask].reset_index(drop=True)
        df_test  = work_df[test_mask].reset_index(drop=True) if test_batches else None

        X_train = df_train[feat_cols].dropna(axis=1, how="all")
        X_train = X_train.fillna(X_train.median())
        feat_cols_valid = list(X_train.columns)   # 去掉全空欄後的欄位

        if X_train.shape[1] < 2:
            st.error("有效欄位不足 2 個。")
            return

        labels_train = (df_train["BatchID"].values if "BatchID" in df_train.columns
                        else np.arange(len(df_train)))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        n_comp = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)

        with st.spinner("PCA 計算中..."):
            sk_pca = skPCA(n_components=n_comp)
            scores_train = sk_pca.fit_transform(X_train_scaled)
            loadings     = sk_pca.components_
            eigenvalues  = sk_pca.explained_variance_
            variance_ratio = sk_pca.explained_variance_ratio_
            cumulative_ev  = np.cumsum(variance_ratio)

            # ── Train T² / SPE ──────────────────────────────
            X_recon_train   = scores_train @ loadings
            ht2_train       = np.sum((scores_train ** 2) / eigenvalues, axis=1)
            spe_train       = np.sum((X_train_scaled - X_recon_train) ** 2, axis=1)

            n_obs = X_train_scaled.shape[0]
            ht2_thres = {
                "68%": _ht2_threshold(0.32, n_obs, n_comp),
                "95%": _ht2_threshold(0.05, n_obs, n_comp),
                "99%": _ht2_threshold(0.01, n_obs, n_comp),
            }
            spe_thres = {
                "95%": _spe_threshold(spe_train, 0.05),
                "99%": _spe_threshold(spe_train, 0.01),
            }

            # ── Test 投影（若有）────────────────────────────
            if df_test is not None and len(df_test) > 0:
                X_test = df_test[feat_cols_valid].fillna(
                    X_train[feat_cols_valid].median())
                labels_test    = (df_test["BatchID"].values if "BatchID" in df_test.columns
                                  else np.arange(len(df_test)))
                X_test_scaled  = scaler.transform(X_test)
                scores_test    = sk_pca.transform(X_test_scaled)
                X_recon_test   = scores_test @ loadings
                ht2_test       = np.sum((scores_test ** 2) / eigenvalues, axis=1)
                spe_test        = np.sum((X_test_scaled - X_recon_test) ** 2, axis=1)
            else:
                labels_test = scores_test = ht2_test = spe_test = X_test_scaled = None

        st.session_state.update({
            # Train
            "pca_scores":        scores_train,
            "pca_X_scaled":      X_train_scaled,
            "pca_X_df":          X_train,
            "pca_labels":        labels_train,
            "pca_ht2":           ht2_train,
            "pca_spe":           spe_train,
            # Test
            "pca_scores_test":   scores_test,
            "pca_X_scaled_test": X_test_scaled,
            "pca_labels_test":   labels_test,
            "pca_ht2_test":      ht2_test,
            "pca_spe_test":      spe_test,
            # 共用
            "pca_loadings":      loadings,
            "pca_eigenvalues":   eigenvalues,
            "pca_vr":            variance_ratio,
            "pca_cumev":         cumulative_ev,
            "pca_feat":          feat_cols_valid,
            "pca_ht2_thres":     ht2_thres,
            "pca_spe_thres":     spe_thres,
            "pca_n_comp_val":    n_comp,
            "pca_alpha_val":     alpha_pca,
            "pca_scaler":        scaler,
            "pca_sk":            sk_pca,
        })
        has_test_str = f" + Test {len(labels_test)} 批" if labels_test is not None else ""
        st.success(
            f"✅ PCA 完成！Train {len(labels_train)} 批{has_test_str}　｜　"
            f"保留 {n_comp} 個主成分，累計解釋 {cumulative_ev[n_comp-1]*100:.1f}%"
        )

    if st.session_state.get("pca_scores") is None:
        return

    scores      = st.session_state["pca_scores"]
    loadings    = st.session_state["pca_loadings"]
    eigenvalues = st.session_state["pca_eigenvalues"]
    vr          = st.session_state["pca_vr"]
    cumev       = st.session_state["pca_cumev"]
    X_scaled    = st.session_state["pca_X_scaled"]
    X_df        = st.session_state["pca_X_df"]
    labels      = st.session_state["pca_labels"]
    feat_names  = st.session_state["pca_feat"]
    ht2_vals    = st.session_state["pca_ht2"]
    spe_vals    = st.session_state["pca_spe"]
    ht2_thres   = st.session_state["pca_ht2_thres"]
    spe_thres   = st.session_state["pca_spe_thres"]
    n_comp      = st.session_state.get("pca_n_comp_val", st.session_state.get("pca_n_comp", 5))
    alpha_val   = st.session_state.get("pca_alpha_val", st.session_state.get("pca_alpha", 0.05))
    n_obs       = X_scaled.shape[0]

    # Test set（可能為 None）
    scores_test   = st.session_state.get("pca_scores_test")
    X_scaled_test = st.session_state.get("pca_X_scaled_test")
    labels_test   = st.session_state.get("pca_labels_test")
    ht2_test      = st.session_state.get("pca_ht2_test")
    spe_test      = st.session_state.get("pca_spe_test")
    has_test      = scores_test is not None and len(scores_test) > 0

    # ── 摘要 metrics ──────────────────────────────────────────
    cols_m = st.columns(min(n_comp, 6))
    for i, c in enumerate(cols_m):
        c.metric(f"PC{i+1}", f"{vr[i]*100:.1f}%", delta=f"累計 {cumev[i]*100:.1f}%")
    st.markdown("---")

    subtabs = st.tabs([
        "📈 Scree & 解釋變異",
        "🔵 Score Plot & Biplot",
        "🚨 Hotelling T²",
        "📉 SPE（Q 統計量）",
        "🔬 單筆貢獻分析",
        "📋 載荷矩陣",
    ])

    # ══════════════════════════════════════════════════════════
    # Tab 0：Scree & 解釋變異
    # ══════════════════════════════════════════════════════════
    with subtabs[0]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        pc_labels = [f"PC{i+1}" for i in range(n_comp)]
        ax.bar(pc_labels, vr * 100, color="#2e86ab", alpha=0.8, label="各PC解釋比例")
        ax2 = ax.twinx()
        ax2.plot(pc_labels, cumev * 100, "o-", color="#e84855", lw=2, label="累計解釋比例")
        ax2.axhline(80, color="#f4a261", ls="--", lw=1, alpha=0.7, label="80%")
        ax2.axhline(95, color="#2ca02c", ls="--", lw=1, alpha=0.7, label="95%")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("累計解釋比例 (%)")
        ax.set_ylabel("各PC解釋比例 (%)")
        ax.set_title("Scree Plot")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="center right")

        ax = axes[1]
        ax.axis("off")
        table_data = [["PC", "Eigenvalue", "解釋比例 (%)", "累計 (%)"]]
        for i in range(n_comp):
            table_data.append([f"PC{i+1}", f"{eigenvalues[i]:.4f}",
                                f"{vr[i]*100:.2f}%", f"{cumev[i]*100:.2f}%"])
        tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax.set_title("主成分解釋變異彙整", pad=12)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        n_kaiser = int((eigenvalues > 1).sum())
        st.info(
            f"**Kaiser 準則**（eigenvalue > 1）：建議保留 **{n_kaiser}** 個主成分　｜　"
            f"**80% 累計解釋**：需 **{int(np.searchsorted(cumev, 0.80)) + 1}** 個　｜　"
            f"**95% 累計解釋**：需 **{int(np.searchsorted(cumev, 0.95)) + 1}** 個"
        )

    # ══════════════════════════════════════════════════════════
    # Tab 1：Score Plot & Biplot
    # ══════════════════════════════════════════════════════════
    with subtabs[1]:
        b1, b2, b3 = st.columns(3)
        pc_opts  = list(range(n_comp))
        pc_x_idx = b1.selectbox("X 軸", pc_opts, index=0, format_func=lambda x: f"PC{x+1}", key="sp_pcx")
        pc_y_idx = b2.selectbox("Y 軸", pc_opts, index=min(1, n_comp-1), format_func=lambda x: f"PC{x+1}", key="sp_pcy")
        n_arrow  = b3.slider("Biplot 箭頭數", 3, min(20, len(feat_names)), 8, key="sp_arrow")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ── Score Plot：Train 藍點（T²熱色），Test 紅星 ──────
        ax = axes[0]
        sc = ax.scatter(scores[:, pc_x_idx], scores[:, pc_y_idx],
                        c=ht2_vals, cmap="RdYlGn_r", s=60, edgecolors="white", lw=0.5,
                        zorder=3, label="Train")
        plt.colorbar(sc, ax=ax, label="Hotelling T²（Train）")
        for i, lbl in enumerate(labels):
            ax.annotate(str(lbl)[-6:], (scores[i, pc_x_idx], scores[i, pc_y_idx]),
                        fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        if has_test:
            ax.scatter(scores_test[:, pc_x_idx], scores_test[:, pc_y_idx],
                       marker="*", s=120, color="#e84855", edgecolors="black", lw=0.5,
                       zorder=4, label="Test")
            for i, lbl in enumerate(labels_test):
                ax.annotate(str(lbl)[-6:], (scores_test[i, pc_x_idx], scores_test[i, pc_y_idx]),
                            fontsize=6, color="#e84855", alpha=0.9, xytext=(3, 3),
                            textcoords="offset points")
            ax.legend(fontsize=8)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(f"PC{pc_x_idx+1} ({vr[pc_x_idx]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y_idx+1} ({vr[pc_y_idx]*100:.1f}%)")
        ax.set_title("Score Plot（🔵 Train T²色階 ／ ★ Test）")
        ax.grid(alpha=0.3)

        # ── Biplot：Train 灰點 + Test 紅星 + 載荷箭頭 ────────
        ax = axes[1]
        all_scores_for_scale = scores
        if has_test:
            all_scores_for_scale = np.vstack([scores, scores_test])
        ax.scatter(scores[:, pc_x_idx], scores[:, pc_y_idx],
                   color="#aaaaaa", s=40, edgecolors="white", lw=0.5, alpha=0.6,
                   zorder=2, label="Train")
        if has_test:
            ax.scatter(scores_test[:, pc_x_idx], scores_test[:, pc_y_idx],
                       marker="*", s=120, color="#e84855", edgecolors="black", lw=0.5,
                       zorder=3, label="Test")
            ax.legend(fontsize=8)
        load_x = loadings[pc_x_idx]
        load_y = loadings[pc_y_idx]
        scale  = np.max(np.abs(all_scores_for_scale[:, [pc_x_idx, pc_y_idx]])) /                  (np.max(np.sqrt(load_x**2 + load_y**2)) + 1e-9)
        top_idx = np.argsort(np.sqrt(load_x**2 + load_y**2))[-n_arrow:]
        for i in top_idx:
            ax.annotate("", xy=(load_x[i]*scale, load_y[i]*scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#2e86ab", lw=1.5))
            ax.text(load_x[i]*scale*1.05, load_y[i]*scale*1.05,
                    feat_names[i][:25], fontsize=7, color="#2e86ab")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(f"PC{pc_x_idx+1} ({vr[pc_x_idx]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y_idx+1} ({vr[pc_y_idx]*100:.1f}%)")
        ax.set_title(f"Biplot（Top {n_arrow} 特徵）")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ══════════════════════════════════════════════════════════
    # Tab 2：Hotelling T²
    # ══════════════════════════════════════════════════════════
    with subtabs[2]:
        st.markdown("#### Hotelling T² 管制圖")
        st.caption(
            f"公式：$T^2_i = \\sum_{{j=1}}^{{p}} t_{{ij}}^2 / \\lambda_j$　｜　"
            f"閾值由 **Train** 建立（n={n_obs}, p={n_comp}）；Test 批次投影比對相同閾值。"
        )
        pca_show_mean = st.checkbox("顯示 Train T² 平均線", value=True, key="pca_ht2_mean")

        def _ht2_bar(ax, vals, lbs, title, show_mean=False):
            idx_s = np.argsort([extract_number(str(b)) for b in lbs])
            sv    = vals[idx_s]
            sl    = [lbs[i] for i in idx_s]
            colors = [
                "#e84855" if v > ht2_thres["99%"] else
                "#f4a261" if v > ht2_thres["95%"] else
                "#e9c46a" if v > ht2_thres["68%"] else
                "#2e86ab" for v in sv
            ]
            ax.bar(range(len(sv)), sv, color=colors, alpha=0.85, width=0.7)
            for th, col, lbl in [
                (ht2_thres["68%"], "#e9c46a", f"68% ({ht2_thres['68%']:.2f})"),
                (ht2_thres["95%"], "#f4a261", f"95% ({ht2_thres['95%']:.2f})"),
                (ht2_thres["99%"], "#e84855", f"99% ({ht2_thres['99%']:.2f})"),
            ]:
                ax.axhline(th, color=col, ls="--", lw=1.5, label=lbl)
            if show_mean:
                mu = float(np.mean(vals))
                ax.axhline(mu, color="#7209b7", lw=1.2, ls=":", label=f"Mean={mu:.2f}")
            ax.set_xticks(range(len(sv)))
            ax.set_xticklabels([str(b)[-6:] for b in sl], rotation=90, fontsize=7)
            ax.set_ylabel("Hotelling T²")
            ax.set_title(title)
            ax.legend(title="控制限", fontsize=8)
            ax.grid(axis="y", ls="--", alpha=0.4)

        n_plots = 2 if has_test else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(14 * n_plots, 5))
        if not has_test:
            axes = [axes]
        _ht2_bar(axes[0], ht2_vals, labels,
                 f"Train — Hotelling T² per Batch（α={alpha_val}）", pca_show_mean)
        if has_test:
            _ht2_bar(axes[1], ht2_test, labels_test,
                     f"Test — Hotelling T² per Batch（控制限來自 Train）")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**📋 Train 統計**")
        m1, m2, m3 = st.columns(3)
        m1.metric("超過 68% 閾值", f"{(ht2_vals > ht2_thres['68%']).sum()} 批",
                  delta=f"{(ht2_vals > ht2_thres['68%']).sum()/n_obs*100:.1f}%")
        m2.metric("超過 95% 閾值", f"{(ht2_vals > ht2_thres['95%']).sum()} 批",
                  delta=f"{(ht2_vals > ht2_thres['95%']).sum()/n_obs*100:.1f}%")
        m3.metric("超過 99% 閾值", f"{(ht2_vals > ht2_thres['99%']).sum()} 批",
                  delta=f"{(ht2_vals > ht2_thres['99%']).sum()/n_obs*100:.1f}%")

        if has_test:
            st.markdown("**📋 Test 統計（對比 Train 控制限）**")
            t1, t2, t3 = st.columns(3)
            t1.metric("超過 68% 閾值", f"{(ht2_test > ht2_thres['68%']).sum()} 批",
                      delta=f"{(ht2_test > ht2_thres['68%']).sum()/len(ht2_test)*100:.1f}%")
            t2.metric("超過 95% 閾值", f"{(ht2_test > ht2_thres['95%']).sum()} 批",
                      delta=f"{(ht2_test > ht2_thres['95%']).sum()/len(ht2_test)*100:.1f}%")
            t3.metric("超過 99% 閾值", f"{(ht2_test > ht2_thres['99%']).sum()} 批",
                      delta=f"{(ht2_test > ht2_thres['99%']).sum()/len(ht2_test)*100:.1f}%")

        def _ht2_table(vals, lbs, tag):
            df = pd.DataFrame({"Batch": lbs, "T²": vals.round(4),
                                "T²/閾值95%": (vals / ht2_thres["95%"]).round(3),
                                "來源": tag})
            df["狀態"] = df["T²"].apply(
                lambda v: "🔴 >99%" if v > ht2_thres["99%"] else
                          "🟠 >95%" if v > ht2_thres["95%"] else
                          "🟡 >68%" if v > ht2_thres["68%"] else "🟢 Normal")
            return df

        all_rows = [_ht2_table(ht2_vals, labels, "Train")]
        if has_test:
            all_rows.append(_ht2_table(ht2_test, labels_test, "Test"))
        st.dataframe(
            pd.concat(all_rows).sort_values("T²", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # Tab 3：SPE（Q 統計量）
    # ══════════════════════════════════════════════════════════
    with subtabs[3]:
        st.markdown("#### SPE（Squared Prediction Error / Q 統計量）")
        st.caption(
            "SPE = $\\|x_i - \\hat{x}_i\\|^2$，衡量樣本偏離 PCA 模型空間的程度。"
            "　T² 偵測模型空間**內**的異常；SPE 偵測模型空間**外**的異常，兩者互補。"
            "　閾值由 **Train** 建立，Test 批次比對相同閾值。"
        )

        def _spe_bar(ax, vals, lbs, title):
            idx_s = np.argsort([extract_number(str(b)) for b in lbs])
            sv    = vals[idx_s]
            sl    = [lbs[i] for i in idx_s]
            colors = [
                "#e84855" if v > spe_thres["99%"] else
                "#f4a261" if v > spe_thres["95%"] else
                "#2e86ab" for v in sv
            ]
            ax.bar(range(len(sv)), sv, color=colors, alpha=0.85, width=0.7)
            ax.axhline(spe_thres["95%"], color="#f4a261", ls="--", lw=1.5,
                       label=f"95% ({spe_thres['95%']:.2f})")
            ax.axhline(spe_thres["99%"], color="#e84855", ls="--", lw=1.5,
                       label=f"99% ({spe_thres['99%']:.2f})")
            ax.set_xticks(range(len(sv)))
            ax.set_xticklabels([str(b)[-6:] for b in sl], rotation=90, fontsize=7)
            ax.set_ylabel("SPE")
            ax.set_title(title)
            ax.legend(title="控制限", fontsize=8)
            ax.grid(axis="y", ls="--", alpha=0.4)

        n_plots = 2 if has_test else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(14 * n_plots, 5))
        if not has_test:
            axes = [axes]
        _spe_bar(axes[0], spe_vals, labels, "Train — SPE（Q）per Batch")
        if has_test:
            _spe_bar(axes[1], spe_test, labels_test,
                     "Test — SPE（Q）per Batch（控制限來自 Train）")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**📋 Train 統計**")
        s1, s2 = st.columns(2)
        s1.metric("超過 95% 閾值", f"{(spe_vals > spe_thres['95%']).sum()} 批")
        s2.metric("超過 99% 閾值", f"{(spe_vals > spe_thres['99%']).sum()} 批")

        if has_test:
            st.markdown("**📋 Test 統計（對比 Train 控制限）**")
            ts1, ts2 = st.columns(2)
            ts1.metric("超過 95% 閾值", f"{(spe_test > spe_thres['95%']).sum()} 批")
            ts2.metric("超過 99% 閾值", f"{(spe_test > spe_thres['99%']).sum()} 批")

        # Train 同時超標警告
        both_tr = np.where((ht2_vals > ht2_thres["95%"]) & (spe_vals > spe_thres["95%"]))[0]
        if len(both_tr) > 0:
            st.warning(
                f"⚠️ **[Train] 同時超過 T² 和 SPE 95% 閾值（共 {len(both_tr)} 批）：**　"
                + "、".join([str(labels[i])[-10:] for i in both_tr])
            )
        if has_test:
            both_te = np.where((ht2_test > ht2_thres["95%"]) & (spe_test > spe_thres["95%"]))[0]
            if len(both_te) > 0:
                st.error(
                    f"🚨 **[Test] 同時超過 T² 和 SPE 95% 閾值（共 {len(both_te)} 批，最需關注）：**　"
                    + "、".join([str(labels_test[i])[-10:] for i in both_te])
                )

        def _spe_table(vals, lbs, tag):
            df = pd.DataFrame({"Batch": lbs, "SPE": vals.round(4), "來源": tag})
            df["狀態"] = df["SPE"].apply(
                lambda v: "🔴 >99%" if v > spe_thres["99%"] else
                          "🟠 >95%" if v > spe_thres["95%"] else "🟢 Normal")
            return df

        all_rows = [_spe_table(spe_vals, labels, "Train")]
        if has_test:
            all_rows.append(_spe_table(spe_test, labels_test, "Test"))
        st.dataframe(
            pd.concat(all_rows).sort_values("SPE", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # Tab 4：單筆貢獻分析
    # ══════════════════════════════════════════════════════════
    with subtabs[4]:
        st.markdown("#### 選擇要分析的 Batch")
        # 合併 Train + Test 供選擇
        _all_labels  = list(labels)
        _all_scores  = scores
        _all_Xsc     = X_scaled
        _all_src     = ["Train"] * len(labels)
        if has_test:
            _all_labels += list(labels_test)
            _all_scores  = np.vstack([scores, scores_test])
            _all_Xsc     = np.vstack([X_scaled, X_scaled_test])
            _all_src    += ["Test"] * len(labels_test)

        batch_opts = [f"{s}_{b}" for s, b in zip(_all_src, _all_labels)]
        sel_batch  = st.selectbox("選擇 Batch（Train / Test）", batch_opts, key="pca_sel_batch")
        sample_i   = batch_opts.index(sel_batch)
        # 更新局部變數供下方貢獻計算使用
        scores   = _all_scores
        X_scaled = _all_Xsc
        labels   = _all_labels
        top_n      = st.slider("顯示前 N 個特徵", 5, min(30, len(feat_names)), 15, key="pca_top_n")
        view_mode  = st.radio(
            "分析模式",
            ["總 T² 貢獻（所有 PC）", "單一 PC 貢獻", "SPE 貢獻"],
            horizontal=True, key="pca_view_mode"
        )

        if view_mode == "總 T² 貢獻（所有 PC）":
            contribs = np.zeros(len(feat_names))
            for j in range(n_comp):
                contribs += (scores[sample_i, j] / eigenvalues[j]) * loadings[j, :] * X_scaled[sample_i, :]

            df_c = pd.DataFrame({"Feature": feat_names, "Contribution": contribs})
            df_c = df_c.reindex(df_c["Contribution"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.4)))
            colors = ["#e84855" if v > 0 else "#2e86ab" for v in df_c["Contribution"]]
            axes[0].barh(df_c["Feature"], df_c["Contribution"], color=colors, alpha=0.85)
            axes[0].axvline(0, color="black", lw=1)
            axes[0].set_title(f"Total T² Contribution — {sel_batch}", fontsize=11)
            axes[0].invert_yaxis()
            axes[0].grid(axis="x", ls="--", alpha=0.5)

            t2_per_pc = (scores[sample_i, :] ** 2) / eigenvalues
            bars = axes[1].bar([f"PC{j+1}" for j in range(n_comp)], t2_per_pc, color="#2e86ab", alpha=0.8)
            for bar, val in zip(bars, t2_per_pc):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            axes[1].axhline(ht2_thres["95%"] / n_comp, color="#f4a261", ls="--", lw=1,
                            label=f"95%閾值/{n_comp}PC={ht2_thres['95%']/n_comp:.2f}")
            axes[1].set_title(f"T² per PC — {sel_batch}")
            axes[1].legend(fontsize=8)
            axes[1].grid(axis="y", ls="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(df_c, use_container_width=True, hide_index=True)

        elif view_mode == "單一 PC 貢獻":
            sel_pc = st.selectbox("選擇 PC", list(range(n_comp)),
                                  format_func=lambda x: f"PC{x+1} ({vr[x]*100:.1f}%)",
                                  key="pca_sel_pc")
            pc_c = (scores[sample_i, sel_pc] / eigenvalues[sel_pc]) * loadings[sel_pc, :] * X_scaled[sample_i, :]
            df_pc = pd.DataFrame({"Feature": feat_names, "Contribution": pc_c})
            df_pc = df_pc.reindex(df_pc["Contribution"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.4)))
            colors = ["#e84855" if v > 0 else "#2e86ab" for v in df_pc["Contribution"]]
            ax.barh(df_pc["Feature"], df_pc["Contribution"], color=colors, alpha=0.85)
            ax.axvline(0, color="black", lw=1)
            ax.set_title(f"PC{sel_pc+1} Contribution — {sel_batch}", fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis="x", ls="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(df_pc, use_container_width=True, hide_index=True)

        else:  # SPE 貢獻
            x_i     = X_scaled[sample_i, :]
            x_recon = scores[sample_i, :] @ loadings
            spe_c   = (x_i - x_recon) ** 2
            df_spe  = pd.DataFrame({"Feature": feat_names, "SPE Contribution": spe_c})
            df_spe  = df_spe.sort_values("SPE Contribution", ascending=False).head(top_n).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.4)))
            ax.barh(df_spe["Feature"], df_spe["SPE Contribution"], color="#7209b7", alpha=0.8)
            ax.set_title(f"SPE Contribution — {sel_batch}", fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis="x", ls="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(df_spe, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # Tab 5：載荷矩陣
    # ══════════════════════════════════════════════════════════
    with subtabs[5]:
        st.markdown("#### 載荷矩陣（Loadings）")
        st.caption("數值越大（絕對值），代表該特徵對該主成分的貢獻越大。紅色=正載荷，藍色=負載荷。")

        n_show_pc   = st.slider("顯示主成分數", 1, n_comp, min(3, n_comp), key="load_npc")
        n_show_feat = st.slider("顯示特徵數（依 PC1 排序）", 5, len(feat_names),
                                min(20, len(feat_names)), key="load_nfeat")

        load_df = pd.DataFrame(
            loadings[:n_show_pc, :].T,
            index=feat_names,
            columns=[f"PC{i+1}" for i in range(n_show_pc)]
        )
        load_df = load_df.reindex(load_df["PC1"].abs().sort_values(ascending=False).index).head(n_show_feat)

        fig, ax = plt.subplots(figsize=(max(6, n_show_pc * 1.5), max(6, n_show_feat * 0.35)))
        im = ax.imshow(load_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n_show_pc))
        ax.set_xticklabels(load_df.columns, fontsize=10)
        ax.set_yticks(range(len(load_df)))
        ax.set_yticklabels([f[:50] for f in load_df.index], fontsize=8)
        for i in range(len(load_df)):
            for j in range(n_show_pc):
                val = load_df.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.5 else "black")
        ax.set_title("Loadings Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            load_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.4f}"),
            use_container_width=True
        )
