"""Tab 6 — 特徵重要性（RF + Lasso + SHAP + PLS-VIP + 交互作用）"""
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
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    cross_val_predict, train_test_split, KFold, learning_curve
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, median_absolute_error, max_error,
)
from scipy import stats as _scipy_stats


def _adj_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)"""
    if n <= p + 1:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _wrap_tick_labels(ax, tick_getter, reverse_map, width=40, fontsize=8):
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), width))
         for t in tick_getter()],
        fontsize=fontsize)

def _regression_metrics(y_true, y_pred, n_feat: int, label: str = "") -> dict:
    """計算一組回歸評估指標"""
    n = len(y_true)
    r2  = r2_score(y_true, y_pred)
    return {
        "label":   label,
        "n":       n,
        "R²":      r2,
        "Adj.R²":  _adj_r2(r2, n, n_feat),
        "RMSE":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":     float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "MedAE":   float(median_absolute_error(y_true, y_pred)),
        "MaxErr":  float(max_error(y_true, y_pred)),
    }


def _render_eval_section(
    model_name: str,
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    n_feat: int,
    cv_results: dict | None = None,   # {"R²":[...], "RMSE":[...], ...}
    y_test: pd.Series | None = None,
    y_test_pred: np.ndarray | None = None,
    key_prefix: str = "eval",
):
    """
    通用模型評估區塊：指標表 + Actual vs Pred + Residual + Q-Q + Learning Curve。
    cv_results: KFold 各 fold 的指標 dict，key=指標名, value=list of fold values。
    """
    st.markdown(f"#### 📊 {model_name} — 模型評估")

    # ── 指標彙整表 ─────────────────────────────────────────────
    train_m = _regression_metrics(y_train, y_train_pred, n_feat, "Train")
    rows = [train_m]

    if y_test is not None and y_test_pred is not None:
        test_m = _regression_metrics(y_test, y_test_pred, n_feat, "Test")
        rows.append(test_m)

    if cv_results:
        cv_row = {"label": f"CV mean±std (k={len(next(iter(cv_results.values())))})",
                  "n": "—"}
        for k, vals in cv_results.items():
            cv_row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        rows.append(cv_row)

    metric_keys = ["R²", "Adj.R²", "RMSE", "MAE", "MAPE(%)", "MedAE", "MaxErr"]
    display_rows = []
    for r in rows:
        display_rows.append({
            "來源":      r["label"],
            "n":         r["n"],
            **{k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "—"))
               for k in metric_keys},
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    # ── 圖表 ──────────────────────────────────────────────────
    eval_tabs = st.tabs(["Actual vs Pred", "Residual Plot", "Residual 分布 & Q-Q", "CV Fold 分布"])

    residuals_tr = np.array(y_train) - y_train_pred

    # Tab 0：Actual vs Predicted
    with eval_tabs[0]:
        n_plots = 2 if (y_test is not None and y_test_pred is not None) else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, y_t, y_p, label, color in zip(
            axes,
            [y_train] + ([y_test] if n_plots == 2 else []),
            [y_train_pred] + ([y_test_pred] if n_plots == 2 else []),
            ["Train"] + (["Test"] if n_plots == 2 else []),
            ["#2e86ab", "#e84855"],
        ):
            mn = min(float(np.min(y_t)), float(np.min(y_p)))
            mx = max(float(np.max(y_t)), float(np.max(y_p)))
            ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Perfect", alpha=0.6)
            ax.scatter(y_t, y_p, color=color, s=45, alpha=0.75, edgecolors="white", lw=0.5)
            r2_val = r2_score(y_t, y_p)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"{label} — Actual vs Predicted (R²={r2_val:.3f})")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 1：Residual vs Predicted
    with eval_tabs[1]:
        n_plots = 2 if (y_test is not None and y_test_pred is not None) else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, y_p, res, label, color in zip(
            axes,
            [y_train_pred] + ([y_test_pred] if n_plots == 2 else []),
            [residuals_tr] + (
                [np.array(y_test) - y_test_pred] if n_plots == 2 else []
            ),
            ["Train"] + (["Test"] if n_plots == 2 else []),
            ["#2e86ab", "#e84855"],
        ):
            ax.scatter(y_p, res, color=color, s=45, alpha=0.75, edgecolors="white", lw=0.5)
            ax.axhline(0, color="black", lw=1.2, ls="--")
            ax.axhline( np.std(res), color="gray", lw=0.8, ls=":", alpha=0.7, label="+1σ")
            ax.axhline(-np.std(res), color="gray", lw=0.8, ls=":", alpha=0.7, label="-1σ")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual − Pred)")
            ax.set_title(f"{label} — Residual vs Predicted")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("殘差應隨機分布在 0 附近（無趨勢），若有漏斗形代表異方差（heteroscedasticity）。")

    # Tab 2：殘差分布 + Q-Q plot
    with eval_tabs[2]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # 分布直方圖
        axes[0].hist(residuals_tr, bins=20, color="#2e86ab", alpha=0.75, edgecolor="white")
        mu_r, std_r = residuals_tr.mean(), residuals_tr.std()
        x_r = np.linspace(residuals_tr.min(), residuals_tr.max(), 200)
        axes[0].plot(x_r, _scipy_stats.norm.pdf(x_r, mu_r, std_r) * len(residuals_tr) *
                     (residuals_tr.max() - residuals_tr.min()) / 20,
                     color="#e84855", lw=2, label=f"N({mu_r:.3f}, {std_r:.3f})")
        axes[0].axvline(0, color="black", lw=1.2, ls="--")
        axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Count")
        axes[0].set_title("Train Residual Distribution"); axes[0].legend(fontsize=8)

        # Q-Q plot
        _scipy_stats.probplot(residuals_tr, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot（Train Residuals）")
        axes[1].get_lines()[0].set(markersize=4, alpha=0.7)
        axes[1].get_lines()[1].set(color="#e84855", lw=1.5)

        # Shapiro-Wilk 常態性檢定
        if len(residuals_tr) <= 5000:
            sw_stat, sw_p = _scipy_stats.shapiro(residuals_tr)
            normality = "✅ 常態" if sw_p > 0.05 else "⚠️ 非常態"
            axes[0].set_xlabel(f"Residual  |  Shapiro-Wilk p={sw_p:.4f} {normality}")

        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 3：CV Fold 分布
    with eval_tabs[3]:
        if not cv_results:
            st.info("尚未執行 K-Fold CV。")
        else:
            n_metrics = len(cv_results)
            fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
            if n_metrics == 1:
                axes = [axes]
            for ax, (metric, vals) in zip(axes, cv_results.items()):
                ax.boxplot(vals, patch_artist=True,
                           boxprops=dict(facecolor="#2e86ab", alpha=0.6),
                           medianprops=dict(color="#e84855", lw=2))
                for v in vals:
                    ax.scatter(1, v, color="#2e86ab", s=30, alpha=0.8, zorder=3)
                ax.set_title(f"{metric}")
                ax.set_xticks([1]); ax.set_xticklabels([f"mean={np.mean(vals):.3f}"])
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle(f"K-Fold CV 各折分布（k={len(next(iter(cv_results.values())))}）",
                         fontsize=11)
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
#  RF Tab
# ═══════════════════════════════════════════════════════════

def _render_rf_tab(fi_subtab, X_fi, y_fi, top_n_fi, X_te=None, y_te=None):
    with fi_subtab:
        st.markdown("#### Random Forest — 調整 Hyperparameter")

        with st.expander("⚙️ RF Hyperparameter 設定", expanded=False):
            r1, r2, r3, r4, r5 = st.columns(5)
            n_est    = r1.slider("n_estimators",    50, 500, 200, 50, key="rf_n_est")
            max_dep  = r2.slider("max_depth",         2,  20,   5,  1, key="rf_max_dep")
            min_leaf = r3.slider("min_samples_leaf",  1,  20,   4,  1, key="rf_min_leaf")
            n_rep    = r4.slider("n_repeats (perm)",  5,  30,  15,  5, key="rf_n_rep")
            kf_rf    = r5.slider("K-Fold 折數",       3,  10,   5,  1, key="rf_kfold")

        if st.button("🌲 訓練 Random Forest", key="run_rf"):
            with st.spinner("RF 訓練 + K-Fold CV 中..."):
                rf = RandomForestRegressor(
                    n_estimators=n_est, max_features="sqrt",
                    max_depth=max_dep, min_samples_leaf=min_leaf,
                    oob_score=True, random_state=42)
                rf.fit(X_fi, y_fi)

                # Permutation importance
                perm = permutation_importance(rf, X_fi, y_fi, n_repeats=n_rep, random_state=42)
                perm_df = pd.DataFrame({
                    "Feature":         X_fi.columns,
                    "Perm_Importance": perm.importances_mean,
                    "Std":             perm.importances_std,
                }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)

                # MDI
                mdi_df = pd.DataFrame({
                    "Feature":        X_fi.columns,
                    "MDI_Importance": rf.feature_importances_,
                }).sort_values("MDI_Importance", ascending=False).reset_index(drop=True)

                # K-Fold CV
                kf = KFold(n_splits=kf_rf, shuffle=True, random_state=42)
                cv_metrics = {"R²": [], "RMSE": [], "MAE": [], "MAPE(%)": [], "MedAE": []}
                for tr_idx, te_idx in kf.split(X_fi):
                    rf_cv = RandomForestRegressor(
                        n_estimators=n_est, max_features="sqrt",
                        max_depth=max_dep, min_samples_leaf=min_leaf, random_state=42)
                    rf_cv.fit(X_fi.iloc[tr_idx], y_fi.iloc[tr_idx])
                    yp = rf_cv.predict(X_fi.iloc[te_idx])
                    yt = y_fi.iloc[te_idx].values
                    cv_metrics["R²"].append(r2_score(yt, yp))
                    cv_metrics["RMSE"].append(float(np.sqrt(mean_squared_error(yt, yp))))
                    cv_metrics["MAE"].append(float(mean_absolute_error(yt, yp)))
                    cv_metrics["MAPE(%)"].append(float(mean_absolute_percentage_error(yt, yp)*100))
                    cv_metrics["MedAE"].append(float(median_absolute_error(yt, yp)))

                y_tr_pred = rf.predict(X_fi)
                y_te_pred = rf.predict(X_te) if X_te is not None else None

                st.session_state.update({
                    "fi_rf": rf, "fi_perm_df": perm_df, "fi_mdi_df": mdi_df,
                    "fi_oob_r2": rf.oob_score_,
                    "fi_rf_cv_metrics": cv_metrics,
                    "fi_rf_y_tr_pred": y_tr_pred,
                    "fi_rf_y_te_pred": y_te_pred,
                    "shap_vals": None,  # Reset SHAP to force recalculation if model changes
                })
            st.success(f"✅ 完成！CV R² = {np.mean(cv_metrics['R²']):.3f} ± {np.std(cv_metrics['R²']):.3f}　OOB R² = {rf.oob_score_:.3f}")

        if st.session_state.get("fi_perm_df") is None:
            return

        perm_df    = st.session_state["fi_perm_df"]
        mdi_df     = st.session_state.get("fi_mdi_df")
        rf         = st.session_state["fi_rf"]
        cv_metrics = st.session_state.get("fi_rf_cv_metrics")
        y_tr_pred  = st.session_state.get("fi_rf_y_tr_pred")
        y_te_pred  = st.session_state.get("fi_rf_y_te_pred")

        rf_tabs = st.tabs(["🌲 特徵重要性", "📊 模型評估"])

        with rf_tabs[0]:
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

            if mdi_df is not None:
                st.markdown("**MDI vs Permutation 比較**")
                merge_df = perm_df[["Feature","Perm_Importance"]].merge(
                    mdi_df[["Feature","MDI_Importance"]], on="Feature").head(top_n_fi)
                merge_df["Perm_norm"] = merge_df["Perm_Importance"] / (merge_df["Perm_Importance"].abs().max() + 1e-9)
                merge_df["MDI_norm"]  = merge_df["MDI_Importance"]  / (merge_df["MDI_Importance"].max() + 1e-9)
                merge_df = merge_df.sort_values("Perm_norm", ascending=True)
                fig, ax = plt.subplots(figsize=(10, max(5, len(merge_df)*0.4)))
                y_pos = range(len(merge_df))
                ax.barh([p - 0.18 for p in y_pos], merge_df["Perm_norm"], height=0.35,
                        color="#2e86ab", alpha=0.85, label="Permutation (norm)")
                ax.barh([p + 0.18 for p in y_pos], merge_df["MDI_norm"],  height=0.35,
                        color="#f4a261", alpha=0.85, label="MDI (norm)")
                ax.set_yticks(list(y_pos)); ax.set_yticklabels(merge_df["Feature"].tolist(), fontsize=8)
                ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.4)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            st.dataframe(perm_df.style.background_gradient(cmap="Blues", subset=["Perm_Importance"]),
                         use_container_width=True, hide_index=True)

        with rf_tabs[1]:
            if y_tr_pred is not None:
                _render_eval_section(
                    model_name="Random Forest",
                    y_train=y_fi, y_train_pred=y_tr_pred,
                    n_feat=X_fi.shape[1], cv_results=cv_metrics,
                    y_test=y_te if X_te is not None else None,
                    y_test_pred=y_te_pred, key_prefix="rf_eval",
                )

# ═══════════════════════════════════════════════════════════
#  Lasso Tab
# ═══════════════════════════════════════════════════════════

def _render_lasso_tab(fi_subtab, X_fi, y_fi, top_n_fi, X_te=None, y_te=None):
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

                n_tr_l, p_tr_l = X_fi.shape
                r2_lasso     = r2_score(y_fi, model.predict(X_scaled))
                r2_adj_lasso = _adj_r2(r2_lasso, n_tr_l, p_tr_l)
                n_nonzero    = (coef_df["Coefficient"] != 0).sum()
                
                # Test set
                r2_lasso_test = r2_adj_lasso_test = None
                if X_te is not None and y_te is not None:
                    X_te_scaled       = scaler.transform(X_te)
                    r2_lasso_test     = r2_score(y_te, model.predict(X_te_scaled))
                    r2_adj_lasso_test = _adj_r2(r2_lasso_test, len(y_te), p_tr_l)

                st.session_state["lasso_model"]         = model
                st.session_state["lasso_coef_df"]       = coef_df
                st.session_state["lasso_alpha_val"]     = best_alpha
                st.session_state["lasso_r2"]            = r2_lasso
                st.session_state["lasso_r2_adj"]        = r2_adj_lasso
                st.session_state["lasso_r2_test"]       = r2_lasso_test
                st.session_state["lasso_r2_adj_test"]   = r2_adj_lasso_test
                st.session_state["lasso_scaler"]        = scaler
                st.session_state["lasso_n_nonzero"]     = n_nonzero
            st.success(f"✅ 完成！α={best_alpha:.4f}，Train Adj.R²={r2_adj_lasso:.3f}，非零係數：{n_nonzero}/{X_fi.shape[1]}")

        if st.session_state.get("lasso_coef_df") is None:
            return

        coef_df      = st.session_state["lasso_coef_df"]
        best_alpha   = st.session_state.get("lasso_alpha_val", st.session_state.get("lasso_alpha", 0.01))
        n_nonzero    = st.session_state["lasso_n_nonzero"]
        cv_metrics_l = st.session_state.get("lasso_cv_metrics")
        y_tr_pred_l  = st.session_state.get("lasso_y_tr_pred")
        y_te_pred_l  = st.session_state.get("lasso_y_te_pred")

        lasso_tabs = st.tabs(["🔪 係數重要性", "📊 模型評估"])

        with lasso_tabs[0]:
            show_nonzero = st.checkbox("只顯示非零係數", value=True, key="lasso_nonzero_only")
            plot_df  = coef_df[coef_df["Coefficient"] != 0] if show_nonzero else coef_df
            top_plot = plot_df.head(top_n_fi)
            if top_plot.empty:
                st.warning("所有係數均為 0，嘗試降低 α 值。")
            else:
                fig, ax = plt.subplots(figsize=(10, max(4, len(top_plot)*0.4)))
                colors = ["#e84855" if v > 0 else "#2e86ab" for v in top_plot["Coefficient"]]
                ax.barh(top_plot["Feature"], top_plot["Coefficient"], color=colors, alpha=0.85)
                ax.axvline(0, color="black", lw=1)
                ax.set_title(f"Lasso Coefficients (α={best_alpha:.4f})", fontsize=13)
                ax.set_xlabel("Standardised Coefficient（正=增加Y，負=降低Y）")
                ax.invert_yaxis(); ax.grid(axis="x", alpha=0.5)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(coef_df.style.background_gradient(cmap="RdBu_r", subset=["Coefficient"]),
                         use_container_width=True, hide_index=True)

            with st.expander("📈 Regularization Path", expanded=False):
                from sklearn.linear_model import lasso_path
                from sklearn.preprocessing import StandardScaler as _SS2
                _X_path = _SS2().fit_transform(X_fi)
                alphas_path, coefs_path, _ = lasso_path(_X_path, y_fi, eps=1e-3, n_alphas=80)
                nz_mask = np.any(coefs_path != 0, axis=1)
                top_idx = np.argsort(np.abs(coefs_path[:, -1]))[-min(top_n_fi, max(1, int(nz_mask.sum()))):]
                fig, ax = plt.subplots(figsize=(10, 5))
                for i in top_idx:
                    ax.plot(np.log10(alphas_path), coefs_path[i], label=X_fi.columns[i][:25], lw=1.2)
                ax.axvline(np.log10(best_alpha + 1e-12), color="#e84855", ls="--", lw=1.5, label=f"最佳 α={best_alpha:.4f}")
                ax.set_xlabel("log10(α)"); ax.set_ylabel("Coefficient")
                ax.set_title("Lasso Regularization Path")
                ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1)); ax.grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with lasso_tabs[1]:
            if y_tr_pred_l is not None:
                _render_eval_section(
                    model_name="Lasso",
                    y_train=y_fi, y_train_pred=np.array(y_tr_pred_l),
                    n_feat=X_fi.shape[1], cv_results=cv_metrics_l,
                    y_test=y_te if X_te is not None else None,
                    y_test_pred=np.array(y_te_pred_l) if y_te_pred_l is not None else None,
                    key_prefix="lasso_eval",
                )
            else:
                st.info("請重新執行 Lasso 以產生 K-Fold 評估數據。")


# ═══════════════════════════════════════════════════════════
#  SHAP Tab (Enhanced with interaction feature)
# ═══════════════════════════════════════════════════════════

def _render_shap_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        st.markdown("#### SHAP 分析")
        shap_subtabs = st.tabs(["Beeswarm","Bar (全局)","Waterfall (單筆)","Dependence Plot","🔄 交互作用排名"])

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

        # Dependence Plot 區塊
        with shap_subtabs[3]:
            st.markdown("#### 特徵交互作用分析 (Dependence Plot)")
            
            c1, c2 = st.columns(2)
            dep_feat = c1.selectbox("主特徵 (X軸)", X_fi.columns.tolist(), key="shap_dep_feat")
            auto_interaction = c2.checkbox("自動尋找最強交互特徵", value=True, key="shap_dep_auto")
            
            if auto_interaction:
                dep_int_index = "auto" 
            else:
                dep_int_sel = c2.selectbox("交互著色特徵 (Color)", X_fi.columns.tolist(), key="shap_dep_int_manual")
                dep_int_index = short_map[dep_int_sel]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            shap_lib.dependence_plot(
                short_map[dep_feat], 
                shap_arr, 
                X_short, 
                interaction_index=dep_int_index,
                ax=ax, 
                show=False
            )
            
            ax.set_title(f"Interaction Analysis: {dep_feat}", fontsize=12)
            ax.set_xlabel(f"{dep_feat} (Actual Value)", fontsize=10)
            ax.set_ylabel(f"SHAP Value (Impact)", fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with shap_subtabs[4]:
            st.markdown("#### 🔄 SHAP Interaction Matrix")
            st.caption(
                "以 `std(SHAP_fi | bins of fj)` 估計兩兩特徵間的非線性交互強度。"
                "欄位名稱使用 FXX 代號，對照請見上方對照表。"
            )

            # ── 設定 ────────────────────────────────────────────────
            im_c1, im_c2 = st.columns(2)
            top_n_inter = im_c1.slider(
                "納入計算的 Top 特徵數（依 RF 重要性）",
                5, min(30, X_fi.shape[1]), 15, key="inter_top_n"
            )
            n_bins_inter = im_c2.slider("分箱數（bins）", 2, 10, 4, key="inter_bins")

            cmap_inter = im_c1.selectbox(
                "色圖", ["YlOrRd", "Reds", "plasma", "hot_r", "viridis"],
                key="inter_cmap"
            )
            annot_inter = im_c2.checkbox("顯示數值標注", value=True, key="inter_annot")

            if st.button("🧮 計算 Interaction Matrix", key="run_inter_matrix", type="primary"):
                perm_df = st.session_state.get("fi_perm_df")
                if perm_df is None:
                    st.error("請先在「RF 重要性」分頁訓練 RF 模型。")
                else:
                    with st.spinner("計算 SHAP 交互矩陣中..."):
                        try:
                            shap_arr_im = np.array(st.session_state["shap_vals"])
                            top_feats = perm_df["Feature"].head(top_n_inter).tolist()
                            feat_indices = [list(X_fi.columns).index(f) for f in top_feats]
                            n = len(top_feats)

                            inter_matrix = np.zeros((n, n))
                            for ii, fi_idx in enumerate(feat_indices):
                                shap_fi = shap_arr_im[:, fi_idx]
                                for jj, fj_idx in enumerate(feat_indices):
                                    if ii == jj:
                                        inter_matrix[ii, jj] = 0.0
                                        continue
                                    fj_vals = X_fi.iloc[:, fj_idx].values.astype(float)
                                    valid = ~np.isnan(fj_vals)
                                    if valid.sum() < n_bins_inter * 2:
                                        inter_matrix[ii, jj] = 0.0
                                        continue
                                    
                                    try:
                                        quantiles = np.percentile(
                                            fj_vals[valid],
                                            np.linspace(0, 100, n_bins_inter + 1)
                                        )
                                        quantiles = np.unique(quantiles)
                                        if len(quantiles) < 2:
                                            inter_matrix[ii, jj] = 0.0
                                            continue
                                        bin_means = []
                                        for b in range(len(quantiles) - 1):
                                            lo, hi = quantiles[b], quantiles[b + 1]
                                            mask = (fj_vals >= lo) & (fj_vals <= hi) & valid
                                            if mask.sum() > 0:
                                                bin_means.append(shap_fi[mask].mean())
                                        inter_matrix[ii, jj] = (
                                            float(np.std(bin_means)) if len(bin_means) > 1 else 0.0
                                        )
                                    except Exception:
                                        inter_matrix[ii, jj] = 0.0

                            short_map_im = {c: f"F{i:02d}" for i, c in enumerate(X_fi.columns)}
                            labels_im = [short_map_im[f] for f in top_feats]

                            st.session_state["inter_matrix"]   = inter_matrix
                            st.session_state["inter_labels"]   = labels_im
                            st.session_state["inter_top_feats"] = top_feats
                        except Exception as e:
                            st.error(f"計算失敗：{e}")
                            import traceback; st.code(traceback.format_exc())

            # ── 顯示熱力矩陣 ────────────────────────────────────────────
            if st.session_state.get("inter_matrix") is not None:
                inter_matrix   = st.session_state["inter_matrix"]
                labels_im      = st.session_state["inter_labels"]
                top_feats      = st.session_state["inter_top_feats"]
                n              = len(labels_im)

                cell_size  = max(0.55, min(1.1, 14.0 / n))
                fig_side   = max(7, n * cell_size)
                annot_fs   = max(6, min(10, int(80 / n)))
                tick_fs    = max(7, min(11, int(90 / n)))

                fig, ax = plt.subplots(figsize=(fig_side, fig_side))
                
                sns.heatmap(
                    inter_matrix,
                    annot=annot_inter,
                    fmt=".3f" if inter_matrix.max() < 0.1 else ".2f",
                    cmap=st.session_state.get("inter_cmap", cmap_inter),
                    xticklabels=labels_im,
                    yticklabels=labels_im,
                    cbar_kws={"label": "Interaction Strength (std of SHAP)"},
                    ax=ax,
                    annot_kws={"size": annot_fs}
                )
                
                ax.set_title("SHAP Feature Interaction Matrix", fontsize=14, pad=15)
                ax.tick_params(axis='x', rotation=45, labelsize=tick_fs)
                ax.tick_params(axis='y', rotation=0, labelsize=tick_fs)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
