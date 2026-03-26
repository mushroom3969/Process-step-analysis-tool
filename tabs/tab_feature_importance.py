"""Tab 6 — 特徵重要性（RF + Lasso + SHAP + PLS-VIP + 係數解釋）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
        
import math
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _wrap_tick_labels(ax, tick_getter, reverse_map, width=40, fontsize=8):
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), width))
         for t in tick_getter()],
        fontsize=fontsize)


# ═══════════════════════════════════════════════════════════
#  RF Tab
# ═══════════════════════════════════════════════════════════

def _render_rf_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        st.markdown("#### Random Forest — 調整 Hyperparameter")

        with st.expander("⚙️ RF Hyperparameter 設定", expanded=False):
            r1, r2, r3, r4 = st.columns(4)
            n_est     = r1.slider("n_estimators",   50, 500, 200, 50, key="rf_n_est")
            max_dep   = r2.slider("max_depth",        2,  20,   5,  1, key="rf_max_dep")
            min_leaf  = r3.slider("min_samples_leaf", 1,  20,   4,  1, key="rf_min_leaf")
            n_rep     = r4.slider("n_repeats (perm)", 5,  30,  15,  5, key="rf_n_rep")

        if st.button("🌲 訓練 Random Forest", key="run_rf"):
            with st.spinner("RF 訓練中..."):
                rf = RandomForestRegressor(
                    n_estimators=n_est, max_features="sqrt",
                    max_depth=max_dep, min_samples_leaf=min_leaf, random_state=42)
                rf.fit(X_fi, y_fi)
                perm = permutation_importance(rf, X_fi, y_fi,
                                              n_repeats=n_rep, random_state=42)
                perm_df = pd.DataFrame({
                    "Feature": X_fi.columns,
                    "Perm_Importance": perm.importances_mean,
                    "Std": perm.importances_std,
                }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)
                r2 = r2_score(y_fi, rf.predict(X_fi))
                st.session_state.update({"fi_rf": rf, "fi_perm_df": perm_df, "fi_r2": r2,
                                          "shap_vals": None})
            st.success(f"✅ 完成！訓練 R² = {r2:.3f}")

        if st.session_state.get("fi_perm_df") is None:
            return

        perm_df = st.session_state["fi_perm_df"]
        rf      = st.session_state["fi_rf"]
        r2      = st.session_state.get("fi_r2", 0)

        top_perm = perm_df.head(top_n_fi)
        fig, ax  = plt.subplots(figsize=(10, max(5, top_n_fi * 0.45)))
        colors   = ["#2e86ab" if v >= 0 else "#e84855" for v in top_perm["Perm_Importance"]]
        ax.barh(top_perm["Feature"], top_perm["Perm_Importance"],
                xerr=top_perm["Std"], color=colors, alpha=0.85,
                error_kw={"ecolor": "gray", "capsize": 3})
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"RF Permutation Importance (Top {top_n_fi})", fontsize=13)
        ax.set_xlabel("Mean Importance ± Std"); ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        c1.metric("訓練集 R²", f"{r2:.3f}")
        c2.metric("特徵數", X_fi.shape[1])
        st.dataframe(perm_df.style.background_gradient(cmap="Blues", subset=["Perm_Importance"]),
                     width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════
#  Lasso Tab (new)
# ═══════════════════════════════════════════════════════════

def _render_lasso_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        st.markdown("#### Lasso Regression — 係數重要性")
        st.caption("Lasso 透過 L1 正則化自動將不重要特徵的係數壓縮為 0，達到特徵選擇效果。")

        with st.expander("⚙️ Lasso Hyperparameter 設定", expanded=False):
            la1, la2 = st.columns(2)
            use_cv = la1.checkbox("自動選擇 α（LassoCV）", value=True, key="lasso_use_cv")
            alpha_manual = la2.number_input("手動 α（use_cv=False 時）",
                                             min_value=1e-6, max_value=10.0,
                                             value=0.01, format="%.4f", key="lasso_alpha")
            la3, la4 = st.columns(2)
            max_iter = la3.slider("max_iter", 500, 5000, 1000, 500, key="lasso_max_iter")
            cv_folds = la4.slider("CV folds（LassoCV）", 3, 10, 5, key="lasso_cv")

        if st.button("🔍 執行 Lasso", key="run_lasso"):
            with st.spinner("Lasso 計算中..."):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_fi)

                if use_cv:
                    model = LassoCV(cv=cv_folds, max_iter=max_iter, random_state=42)
                    model.fit(X_scaled, y_fi)
                    best_alpha = model.alpha_
                else:
                    model = Lasso(alpha=alpha_manual, max_iter=max_iter, random_state=42)
                    model.fit(X_scaled, y_fi)
                    best_alpha = alpha_manual

                coef_df = pd.DataFrame({
                    "Feature": X_fi.columns,
                    "Coefficient": model.coef_,
                    "Abs_Coef": np.abs(model.coef_),
                }).sort_values("Abs_Coef", ascending=False).reset_index(drop=True)

                r2_lasso = r2_score(y_fi, model.predict(X_scaled))
                n_nonzero = (coef_df["Coefficient"] != 0).sum()

                st.session_state.update({
                    "lasso_model": model, "lasso_coef_df": coef_df,
                    "lasso_alpha": best_alpha, "lasso_r2": r2_lasso,
                    "lasso_scaler": scaler, "lasso_n_nonzero": n_nonzero,
                })
            st.success(f"✅ 完成！α={best_alpha:.4f}，R²={r2_lasso:.3f}，非零係數：{n_nonzero}/{X_fi.shape[1]}")

        if st.session_state.get("lasso_coef_df") is None:
            return

        coef_df   = st.session_state["lasso_coef_df"]
        best_alpha= st.session_state["lasso_alpha"]
        r2_lasso  = st.session_state["lasso_r2"]
        n_nonzero = st.session_state["lasso_n_nonzero"]

        m1, m2, m3 = st.columns(3)
        m1.metric("最佳 α", f"{best_alpha:.4f}")
        m2.metric("R²", f"{r2_lasso:.3f}")
        m3.metric("選出特徵數", f"{n_nonzero}")

        # Plot: all vs non-zero toggle
        show_nonzero_only = st.checkbox("只顯示非零係數特徵", value=True, key="lasso_nonzero_only")
        plot_df = coef_df[coef_df["Coefficient"] != 0] if show_nonzero_only else coef_df
        top_plot = plot_df.head(top_n_fi)

        if top_plot.empty:
            st.warning("所有係數均為 0，嘗試降低 α 值。"); return

        fig, ax = plt.subplots(figsize=(10, max(4, len(top_plot)*0.4)))
        colors = ["#e84855" if v > 0 else "#2e86ab" for v in top_plot["Coefficient"]]
        ax.barh(top_plot["Feature"], top_plot["Coefficient"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"Lasso Coefficients (α={best_alpha:.4f}, Top {len(top_plot)})", fontsize=13)
        ax.set_xlabel("Standardised Coefficient（正=增加Y，負=降低Y）")
        ax.invert_yaxis(); ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.caption("係數已在標準化特徵上計算，數值可直接比較特徵相對影響力。")
        st.dataframe(coef_df.style.background_gradient(cmap="RdBu_r", subset=["Coefficient"]),
                     width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════
#  SHAP Tab
# ═══════════════════════════════════════════════════════════

def _render_shap_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        st.markdown("#### SHAP 分析")
        shap_subtabs = st.tabs(["Beeswarm","Bar (全局)","Waterfall (單筆)","Dependence Plot"])

        if st.button("🔮 計算 SHAP Values", key="run_shap"):
            rf = st.session_state.get("fi_rf")
            if rf is None:
                st.error("請先在「RF 重要性」分頁訓練模型。"); return
            try:
                import shap as shap_lib
                with st.spinner("計算 SHAP..."):
                    explainer = shap_lib.TreeExplainer(rf)
                    shap_vals = explainer.shap_values(X_fi)
                st.session_state.update({"shap_vals": shap_vals,
                                          "shap_explainer": explainer, "shap_lib": shap_lib})
                st.success("✅ SHAP 完成！")
            except Exception as e:
                st.error(f"SHAP 失敗：{e}")

        if st.session_state.get("shap_vals") is None:
            return

        shap_vals = st.session_state["shap_vals"]
        shap_lib  = st.session_state["shap_lib"]
        explainer = st.session_state["shap_explainer"]
        rf        = st.session_state["fi_rf"]

        short_map   = {c: f"F{i:02d}" for i, c in enumerate(X_fi.columns)}
        reverse_map = {v: k for k, v in short_map.items()}
        X_short     = X_fi.rename(columns=short_map)
        shap_arr    = np.array(shap_vals)

        def fix_labels(ax):
            _wrap_tick_labels(ax, ax.get_yticklabels, reverse_map)
            plt.subplots_adjust(left=0.38)

        with shap_subtabs[0]:
            plt.figure(figsize=(11, max(6, top_n_fi*0.55)))
            shap_lib.summary_plot(shap_arr, X_short, max_display=top_n_fi,
                                  plot_type="dot", show=False)
            fix_labels(plt.gca()); st.pyplot(plt.gcf()); plt.close()

        with shap_subtabs[1]:
            plt.figure(figsize=(11, max(5, top_n_fi*0.55)))
            shap_lib.summary_plot(shap_arr, X_short, max_display=top_n_fi,
                                  plot_type="bar", show=False)
            fix_labels(plt.gca()); st.pyplot(plt.gcf()); plt.close()

        with shap_subtabs[2]:
            idx = st.slider("選擇樣本", 0, len(X_fi)-1, 0, key="shap_sample")
            ev_raw = explainer.expected_value
            base_val = float(ev_raw[0]) if hasattr(ev_raw, "__len__") else float(ev_raw)
            expl_obj = shap_lib.Explanation(
                values=shap_arr[idx], base_values=base_val,
                data=X_fi.iloc[idx].values, feature_names=X_short.columns.tolist())
            plt.figure(figsize=(12, max(6, top_n_fi*0.55)))
            shap_lib.plots.waterfall(expl_obj, max_display=top_n_fi, show=False)
            fix_labels(plt.gca()); st.pyplot(plt.gcf()); plt.close()
            st.caption(f"樣本{idx}｜預測:{rf.predict(X_fi.iloc[[idx]])[0]:.3f}｜實際:{y_fi.iloc[idx]:.3f}｜基準:{base_val:.3f}")

        with shap_subtabs[3]:
            st.markdown("#### 特徵交互作用分析 (Dependence Plot)")
            st.caption("觀察主特徵數值變化時，SHAP 值如何隨交互特徵（顏色）改變。")
            
            c1, c2 = st.columns(2)
            dep_feat = c1.selectbox("主特徵 (X軸)", X_fi.columns.tolist(), key="shap_dep_feat")
            
            # 增加一個選項讓使用者決定要不要自動尋找最強交互項
            auto_interaction = c2.checkbox("自動尋找最強交互特徵", value=True)
            
            if auto_interaction:
                dep_int = "auto"
            else:
                dep_int = c2.selectbox("交互著色特徵 (Color)", X_fi.columns.tolist(), key="shap_dep_int")
        
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 使用 shap_lib 繪製
            shap_lib.dependence_plot(
                dep_feat, 
                shap_arr, 
                X_fi, # 這裡建議傳入原始 X_fi 保持 Label 可讀性
                interaction_index=dep_int,
                ax=ax, 
                show=False
            )
            
            # 美化標籤
            ax.set_title(f"Interaction: {dep_feat} vs {dep_int if dep_int != 'auto' else 'Most Correlated'}", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

def _render_interaction_heatmap(X_fi, top_n_fi):
    st.markdown("#### Top 特徵相關性矩陣")
    st.caption("高相關性的特徵對通常隱含較強的交互作用。")
    
    top_feats = st.session_state["fi_perm_df"]["Feature"].head(top_n_fi).tolist()
    corr = X_fi[top_feats].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════
#  PLS-VIP Tab (enhanced with coefficients)
# ═══════════════════════════════════════════════════════════

def _render_pls_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        st.markdown("#### PLS — VIP Score + 係數解釋")

        max_comp = min(10, X_fi.shape[1], len(y_fi)-1)
        if max_comp < 2:
            st.warning("樣本數或特徵數不足。"); return

        with st.expander("⚙️ PLS Hyperparameter 設定", expanded=False):
            p1, p2 = st.columns(2)
            cv_folds_pls = p1.slider("CV folds", 3, 10, 5, key="pls_cv_folds")
            scale_pls    = p2.checkbox("標準化 X (推薦)", value=True, key="pls_scale")

        st.markdown("**Step 1：CV 選最佳主成分數**")
        if st.button("📉 計算 PLS CV MSE", key="run_pls_cv"):
            with st.spinner("PLS CV..."):
                X_arr = StandardScaler().fit_transform(X_fi) if scale_pls else X_fi.values
                y_arr = y_fi.values.ravel()
                comp_range = list(range(1, max_comp+1))
                mse_list = []
                for n_c in comp_range:
                    pls_cv = PLSRegression(n_components=n_c)
                    y_cv = cross_val_predict(pls_cv, X_arr, y_arr,
                                             cv=min(cv_folds_pls, len(y_arr)))
                    mse_list.append(mean_squared_error(y_arr, y_cv))
                st.session_state.update({"pls_mse": mse_list, "pls_range": comp_range})

        if st.session_state.get("pls_mse"):
            mse_list   = st.session_state["pls_mse"]
            comp_range = st.session_state["pls_range"]
            best_n     = comp_range[int(np.argmin(mse_list))]
            fig, ax    = plt.subplots(figsize=(8, 4))
            ax.plot(comp_range, mse_list, marker="o", color="#2e86ab")
            ax.axvline(best_n, color="#e84855", linestyle="--", label=f"Best={best_n}")
            ax.set(xlabel="Components", ylabel="CV MSE", title="PLS Cross-Validation MSE")
            ax.legend(); ax.grid(linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.info(f"建議主成分數：**{best_n}**")

        st.markdown("**Step 2：訓練 PLS 並計算 VIP + 係數**")
        n_pls = st.slider("PLS 主成分數", 1, max_comp, min(3, max_comp), key="pls_n_comp")

        if st.button("📐 計算 PLS", key="run_pls_vip"):
            with st.spinner("PLS 訓練中..."):
                scaler_pls = StandardScaler() if scale_pls else None
                X_arr = scaler_pls.fit_transform(X_fi) if scale_pls else X_fi.values
                y_arr = y_fi.values.ravel()

                pls = PLSRegression(n_components=n_pls)
                pls.fit(X_arr, y_arr)

                # VIP scores
                t = np.array(pls.x_scores_, dtype=float)
                w = np.array(pls.x_weights_, dtype=float)
                q = np.array(pls.y_loadings_, dtype=float).ravel()
                p_feat, h = w.shape
                s = np.array([float(t[:,j]@t[:,j]) * float(q[j])**2 for j in range(h)])
                w_normed = (w / np.linalg.norm(w, axis=0))**2
                vips = np.sqrt(p_feat * (w_normed @ s) / float(s.sum()))

                vip_df = pd.DataFrame({"Feature": X_fi.columns, "VIP": vips})\
                           .sort_values("VIP", ascending=False).reset_index(drop=True)

                # Regression coefficients (in original space if scaled)
                coef_raw = pls.coef_.ravel()  # (p,) in scaled space
                reg_coef_df = pd.DataFrame({
                    "Feature": X_fi.columns,
                    "PLS_Coef": coef_raw,
                    "Abs_Coef": np.abs(coef_raw),
                }).sort_values("Abs_Coef", ascending=False).reset_index(drop=True)

                # X loadings (P) — how features relate to each PC
                x_load = pd.DataFrame(
                    pls.x_loadings_,
                    index=X_fi.columns,
                    columns=[f"PC{i+1}" for i in range(n_pls)])

                # X weights (W) — how features contribute to latent scores
                x_weights = pd.DataFrame(
                    pls.x_weights_,
                    index=X_fi.columns,
                    columns=[f"PC{i+1}" for i in range(n_pls)])

                # Y explained variance per PC
                y_pred = pls.predict(X_arr).ravel()
                r2_pls = r2_score(y_arr, y_pred)

                st.session_state.update({
                    "pls_vip_df": vip_df, "pls_reg_coef_df": reg_coef_df,
                    "pls_x_loadings": x_load, "pls_x_weights": x_weights,
                    "pls_r2": r2_pls, "pls_model": pls,
                })
            st.success(f"✅ 完成！R² = {r2_pls:.3f}")

        if st.session_state.get("pls_vip_df") is None:
            return
        vip_df       = st.session_state["pls_vip_df"]
        reg_coef_df  = st.session_state["pls_reg_coef_df"]
        x_load       = st.session_state["pls_x_loadings"]
        x_weights    = st.session_state["pls_x_weights"]
        r2_pls       = st.session_state["pls_r2"]

        st.metric("PLS R²", f"{r2_pls:.3f}")

        pls_subtabs = st.tabs(["VIP Score", "迴歸係數", "X Loadings", "X Weights"])

        with pls_subtabs[0]:
            top_vip = vip_df.head(top_n_fi)
            fig, ax = plt.subplots(figsize=(10, max(5, top_n_fi*0.45)))
            ax.barh(top_vip["Feature"], top_vip["VIP"],
                    color=["#e84855" if v>=1.0 else "#2e86ab" for v in top_vip["VIP"]], alpha=0.85)
            ax.axvline(1.0, color="#e84855", linestyle="--", lw=1.5, label="VIP=1")
            ax.set_title(f"PLS VIP Score (Top {top_n_fi})", fontsize=13)
            ax.set_xlabel("VIP Score"); ax.invert_yaxis(); ax.legend()
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.caption("VIP ≥ 1.0（紅色）= 對目標變數有顯著影響。")
            st.dataframe(vip_df.style.background_gradient(cmap="Reds", subset=["VIP"]),
                         width="stretch", hide_index=True)

        with pls_subtabs[1]:
            st.markdown("**PLS 迴歸係數**（標準化空間，反映各特徵對 Y 的線性貢獻方向和大小）")
            top_coef = reg_coef_df.head(top_n_fi)
            fig, ax  = plt.subplots(figsize=(10, max(5, top_n_fi*0.45)))
            ax.barh(top_coef["Feature"], top_coef["PLS_Coef"],
                    color=["#e84855" if v>0 else "#2e86ab" for v in top_coef["PLS_Coef"]], alpha=0.85)
            ax.axvline(0, color="black", lw=1)
            ax.set_title("PLS Regression Coefficients", fontsize=13)
            ax.set_xlabel("Coefficient（正=增加Y，負=降低Y）"); ax.invert_yaxis()
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(reg_coef_df.style.background_gradient(cmap="RdBu_r", subset=["PLS_Coef"]),
                         width="stretch", hide_index=True)

        with pls_subtabs[2]:
            st.markdown("**X Loadings（P）** — 原始特徵在各主成分上的載荷（特徵與主成分的相關性）")
            st.dataframe(x_load.style.background_gradient(cmap="RdBu_r"), width="stretch")
            # Heatmap
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(max(6, n_pls*1.5), max(6, len(X_fi.columns)*0.35)))
            sns.heatmap(x_load, annot=True, fmt=".3f", cmap="RdBu_r",
                        center=0, linewidths=0.3, ax=ax, annot_kws={"size":7})
            ax.set_title("X Loadings (P) Heatmap")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with pls_subtabs[3]:
            st.markdown("**X Weights（W）** — 特徵對 PLS 潛在變數（latent score）的貢獻權重")
            st.dataframe(x_weights.style.background_gradient(cmap="RdBu_r"), width="stretch")
            fig, ax = plt.subplots(figsize=(max(6, n_pls*1.5), max(6, len(X_fi.columns)*0.35)))
            import seaborn as sns
            sns.heatmap(x_weights, annot=True, fmt=".3f", cmap="RdBu_r",
                        center=0, linewidths=0.3, ax=ax, annot_kws={"size":7})
            ax.set_title("X Weights (W) Heatmap")
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
#  Main render
# ═══════════════════════════════════════════════════════════
def render(selected_process_df):
    st.header("特徵重要性分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。"); return

    numeric_cols  = work_df.select_dtypes(include=["number"]).columns.tolist()
    stored_target = st.session_state.get("target_col")
    default_fi    = numeric_cols.index(stored_target) if stored_target in numeric_cols else 0
    target_col_fi = st.selectbox("目標欄位（Y）", numeric_cols, index=default_fi, key="fi_target")
    top_n_fi      = st.slider("顯示前 N 個特徵", 5, 30, 15, key="fi_topn")

    # Prepare X, y once
    if st.button("📦 準備資料", key="run_fi_prepare"):
        try:
            exclude = [c for c in ["BatchID", target_col_fi] if c in work_df.columns]
            X_fi = work_df.drop(columns=exclude, errors="ignore").select_dtypes(include=["number"])
            y_fi = work_df[target_col_fi]
            valid = y_fi.notna() & X_fi.notna().all(axis=1)
            X_fi = X_fi[valid].reset_index(drop=True)
            y_fi = y_fi[valid].reset_index(drop=True)
            st.session_state.update({
                "fi_X": X_fi, "fi_y": y_fi, "fi_target_col": target_col_fi,
                "fi_rf": None, "shap_vals": None,
                "lasso_coef_df": None, "pls_vip_df": None,
            })
            st.success(f"✅ 資料準備完成：{X_fi.shape[0]} 筆 × {X_fi.shape[1]} 特徵")
        except Exception as e:
            st.error(f"失敗：{e}")

    if st.session_state.get("fi_X") is None:
        return

    X_fi = st.session_state["fi_X"]
    y_fi = st.session_state["fi_y"]

    fi_subtabs = st.tabs(["🌲 RF 重要性", "🔪 Lasso 重要性", "🔮 SHAP 分析", "📐 PLS-VIP", "🔄 交互作用探索"])
    _render_rf_tab(fi_subtabs[0], X_fi, y_fi, top_n_fi)
    _render_lasso_tab(fi_subtabs[1], X_fi, y_fi, top_n_fi)
    _render_shap_tab(fi_subtabs[2], X_fi, y_fi, top_n_fi)
    _render_pls_tab(fi_subtabs[3], X_fi, y_fi, top_n_fi)

    with fi_subtabs[4]:
        if st.session_state.get("fi_X") is not None:
            _render_interaction_heatmap(st.session_state["fi_X"], top_n_fi)
