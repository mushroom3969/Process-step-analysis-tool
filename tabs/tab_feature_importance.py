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


# ═══════════════════════════════════════════════════════════
#  RF Tab
# ═══════════════════════════════════════════════════════════


def _adj_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)"""
    if n <= p + 1:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


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
                    "shap_vals": None,
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
#  Lasso Tab (new)
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

        # 在 _render_shap_tab 內的 Dependence Plot 區塊
        with shap_subtabs[3]:
            st.markdown("#### 特徵交互作用分析 (Dependence Plot)")
            
            c1, c2 = st.columns(2)
            dep_feat = c1.selectbox("主特徵 (X軸)", X_fi.columns.tolist(), key="shap_dep_feat")
            auto_interaction = c2.checkbox("自動尋找最強交互特徵", value=True, key="shap_dep_auto")
        
            # 這裡修正：明確取得 SHAP 推薦的交互特徵名稱
            if auto_interaction:
                # 讓 SHAP 幫我們找 index
                dep_int_index = "auto" 
            else:
                dep_int_sel = c2.selectbox("交互著色特徵 (Color)", X_fi.columns.tolist(), key="shap_dep_int_manual")
                dep_int_index = short_map[dep_int_sel]
        
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 關鍵修正：確保傳入的是 X_short 以匹配 shap_arr 的維度
            shap_lib.dependence_plot(
                short_map[dep_feat], 
                shap_arr, 
                X_short, # 使用簡化後的 DataFrame
                interaction_index=dep_int_index,
                ax=ax, 
                show=False
            )
            
            # 強制重繪標題，顯示到底跟誰交互了
            # SHAP 在 auto 模式下會自動在右側加上顏色條標籤，如果沒出現，代表沒找到強交互
            ax.set_title(f"Interaction Analysis: {dep_feat}", fontsize=12)
            ax.set_xlabel(f"{dep_feat} (Actual Value)", fontsize=10)
            ax.set_ylabel(f"SHAP Value (Impact on Yield)", fontsize=10)
            
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
                            # 取 Top N 特徵（依 RF permutation importance）
                            top_feats = perm_df["Feature"].head(top_n_inter).tolist()
                            feat_indices = [list(X_fi.columns).index(f) for f in top_feats]
                            n = len(top_feats)

                            # 建立 n×n 矩陣：entry(i,j) = std of SHAP_i grouped by bins of fj
                            inter_matrix = np.zeros((n, n))
                            for ii, fi_idx in enumerate(feat_indices):
                                shap_fi = shap_arr_im[:, fi_idx]  # SHAP values of feature i
                                for jj, fj_idx in enumerate(feat_indices):
                                    if ii == jj:
                                        inter_matrix[ii, jj] = 0.0
                                        continue
                                    fj_vals = X_fi.iloc[:, fj_idx].values.astype(float)
                                    valid = ~np.isnan(fj_vals)
                                    if valid.sum() < n_bins_inter * 2:
                                        inter_matrix[ii, jj] = 0.0
                                        continue
                                    # Bin fj into quantile-based groups
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

                            # 用 FXX 代號作為標籤
                            short_map_im = {c: f"F{i:02d}" for i, c in enumerate(X_fi.columns)}
                            labels_im = [short_map_im[f] for f in top_feats]

                            st.session_state["inter_matrix"]   = inter_matrix
                            st.session_state["inter_labels"]   = labels_im
                            st.session_state["inter_top_feats"] = top_feats
                        except Exception as e:
                            st.error(f"計算失敗：{e}")
                            import traceback; st.code(traceback.format_exc())

            # ── 顯示矩陣 ────────────────────────────────────────────
            if st.session_state.get("inter_matrix") is not None:
                inter_matrix   = st.session_state["inter_matrix"]
                labels_im      = st.session_state["inter_labels"]
                top_feats      = st.session_state["inter_top_feats"]
                n              = len(labels_im)

                # 動態調整圖尺寸與字體
                cell_size  = max(0.55, min(1.1, 14.0 / n))
                fig_side   = max(7, n * cell_size)
                annot_fs   = max(6, min(10, int(80 / n)))
                tick_fs    = max(7, min(11, int(90 / n)))

                fig, ax = plt.subplots(figsize=(fig_side + 1.5, fig_side))
                im_df = pd.DataFrame(inter_matrix, index=labels_im, columns=labels_im)

                sns.heatmap(
                    im_df,
                    ax=ax,
                    cmap=cmap_inter,
                    annot=annot_inter,
                    fmt=".3f" if annot_inter else "",
                    annot_kws={"size": annot_fs},
                    linewidths=0.4,
                    linecolor="white",
                    cbar_kws={"label": "Interaction Strength (std of SHAP_i | bins of fj)", "shrink": 0.75},
                    square=True,
                )
                ax.set_title(
                    f"SHAP Interaction Matrix  (Top {n} features, bins={n_bins_inter})",
                    fontsize=12, pad=14,
                )
                ax.set_xlabel("fj  (conditioning feature)", fontsize=10)
                ax.set_ylabel("fi  (target SHAP)", fontsize=10)
                ax.tick_params(axis="x", labelsize=tick_fs, rotation=45)
                ax.tick_params(axis="y", labelsize=tick_fs, rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── 排行榜（上三角展開）─────────────────────────────
                st.markdown("#### 📊 交互強度排行榜 Top 15")
                rows_rank = []
                for ii in range(n):
                    for jj in range(n):
                        if ii == jj:
                            continue
                        rows_rank.append({
                            "fi（SHAP 受影響）": labels_im[ii],
                            "fj（條件特徵）":    labels_im[jj],
                            "fi 原始名稱":       top_feats[ii],
                            "fj 原始名稱":       top_feats[jj],
                            "Interaction Strength": round(float(inter_matrix[ii, jj]), 4),
                        })
                rank_df = (
                    pd.DataFrame(rows_rank)
                    .sort_values("Interaction Strength", ascending=False)
                    .reset_index(drop=True)
                    .head(15)
                )
                st.dataframe(
                    rank_df.style.background_gradient(
                        cmap="YlOrRd", subset=["Interaction Strength"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "解讀：Interaction Strength 越大，代表「當 fj 改變時，fi 的 SHAP 值波動越大」，"
                    "即兩特徵存在較強的非線性交互效應。"
                )

            # ══════════════════════════════════════════════════════
            # ── 精確版：shap_interaction_values() ─────────────────
            # ══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("#### 🎯 精確版 SHAP Interaction Values")
            st.info(
                "**原理**：呼叫 `TreeExplainer.shap_interaction_values()`，"
                "每個格子 (i,j) = 平均 |SHAP interaction value(i,j)|，"
                "與 Dependence Plot 的 `interaction_index='auto'` 使用相同底層計算，結果最精確。\n\n"
                "⚠️ **注意**：計算量為 O(n² × samples × trees)，特徵數或樣本數多時需要幾十秒。"
                "建議先用 Top N 特徵縮小範圍。",
                icon="ℹ️",
            )

            sv_c1, sv_c2 = st.columns(2)
            top_n_sv = sv_c1.slider(
                "Top 特徵數（精確版）", 5, min(25, X_fi.shape[1]), 10, key="sv_top_n"
            )
            cmap_sv   = sv_c2.selectbox(
                "色圖", ["YlOrRd", "Reds", "plasma", "hot_r", "viridis"],
                key="sv_cmap"
            )
            sv_c3, sv_c4 = st.columns(2)
            annot_sv  = sv_c3.checkbox("顯示數值標注", value=True, key="sv_annot")
            symm_sv   = sv_c4.checkbox(
                "對稱化（取 (i,j)+(j,i) 平均）", value=True, key="sv_symm",
                help="SHAP interaction matrix 理論上對稱，對稱化後更易閱讀。"
            )

            if st.button("🎯 計算精確 SHAP Interaction Values", key="run_sv_inter", type="primary"):
                rf_model = st.session_state.get("fi_rf")
                perm_df  = st.session_state.get("fi_perm_df")

                # ── 前置檢查，給出明確原因 ────────────────────────
                _sv_ready = True
                if rf_model is None:
                    st.error("❌ 尚未訓練 RF 模型，請先至「🌲 RF 重要性」分頁點擊「訓練 Random Forest」。")
                    _sv_ready = False
                if perm_df is None:
                    st.error("❌ 找不到特徵重要性資料，請先執行 RF 訓練。")
                    _sv_ready = False

                if _sv_ready:
                    top_feats_sv = perm_df["Feature"].head(top_n_sv).tolist()
                    X_sv = X_fi[top_feats_sv]
                    n_samples_sv = X_sv.shape[0]
                    n_feats_sv   = X_sv.shape[1]

                    # ── 算力預估警告 ──────────────────────────────
                    estimated_sec = (n_samples_sv * n_feats_sv ** 2) / 5000
                    if estimated_sec > 60:
                        st.warning(
                            f"⚠️ 預估計算時間約 **{estimated_sec:.0f} 秒**（{n_samples_sv} 筆 × {n_feats_sv} 特徵）。"
                            f"建議將「Top 特徵數」降至 8～10 以內再執行。"
                        )
                    else:
                        st.info(f"ℹ️ 預估計算時間約 {max(1, estimated_sec):.0f} 秒，請稍候。")

                    with st.spinner(f"計算中：{n_samples_sv} 筆 × {n_feats_sv} 特徵 × {n_feats_sv} 特徵..."):
                        try:
                            import shap as shap_lib_sv
                            explainer_sv = shap_lib_sv.TreeExplainer(rf_model)
                            # shape: (n_samples, n_features_sv, n_features_sv)
                            sv_interact  = explainer_sv.shap_interaction_values(X_sv)
                            sv_arr       = np.array(sv_interact)  # (n, p, p)

                            if sv_arr.ndim != 3:
                                st.error(
                                    f"❌ shap_interaction_values 回傳維度異常（shape={sv_arr.shape}）。"
                                    "可能是多輸出模型，目前僅支援單一輸出的 RandomForest。"
                                )
                            else:
                                # 取平均絕對值：shape (p, p)
                                mean_abs_sv = np.mean(np.abs(sv_arr), axis=0)

                                if symm_sv:
                                    mean_abs_sv = (mean_abs_sv + mean_abs_sv.T) / 2

                                # 對角線設為 0（self-interaction 無意義）
                                np.fill_diagonal(mean_abs_sv, 0)

                                short_map_sv = {col: f"F{i:02d}" for i, col in enumerate(X_fi.columns)}
                                labels_sv    = [short_map_sv[f] for f in top_feats_sv]

                                st.session_state["sv_interact_mat"]    = mean_abs_sv
                                st.session_state["sv_interact_labels"] = labels_sv
                                st.session_state["sv_interact_feats"]  = top_feats_sv
                                st.success(f"✅ 計算完成！（{n_samples_sv} 筆 × {n_feats_sv} 特徵）")

                        except MemoryError:
                            st.error(
                                "❌ 記憶體不足（MemoryError）。請降低「Top 特徵數」或減少樣本數後重試。"
                            )
                        except Exception as e:
                            err_msg = str(e)
                            # 給出更友善的錯誤說明
                            if "timeout" in err_msg.lower() or "time" in err_msg.lower():
                                st.error("❌ 計算逾時，請降低特徵數後重試。")
                            elif "memory" in err_msg.lower() or "alloc" in err_msg.lower():
                                st.error("❌ 記憶體不足，請降低特徵數或樣本數後重試。")
                            elif "tree" in err_msg.lower() or "model" in err_msg.lower():
                                st.error(
                                    f"❌ 模型不相容：{err_msg}\n"
                                    "shap_interaction_values 僅支援 Tree 系列模型（如 RandomForest）。"
                                )
                            else:
                                st.error(f"❌ 計算失敗：{err_msg}")
                            import traceback
                            with st.expander("🔍 查看完整錯誤訊息"):
                                st.code(traceback.format_exc())

            # ── 顯示精確版矩陣 ───────────────────────────────────
            if st.session_state.get("sv_interact_mat") is not None:
                mean_abs_sv  = st.session_state["sv_interact_mat"]
                labels_sv    = st.session_state["sv_interact_labels"]
                top_feats_sv = st.session_state["sv_interact_feats"]
                n_sv = len(labels_sv)

                # 動態圖尺寸與字體
                cell_sv   = max(0.55, min(1.1, 14.0 / n_sv))
                fig_sv    = max(7, n_sv * cell_sv)
                annot_fs_sv = max(6, min(10, int(80 / n_sv)))
                tick_fs_sv  = max(7, min(11, int(90 / n_sv)))

                fig, ax = plt.subplots(figsize=(fig_sv + 1.5, fig_sv))
                sv_df = pd.DataFrame(mean_abs_sv, index=labels_sv, columns=labels_sv)

                sns.heatmap(
                    sv_df,
                    ax=ax,
                    cmap=cmap_sv,
                    annot=annot_sv,
                    fmt=".4f" if annot_sv else "",
                    annot_kws={"size": annot_fs_sv},
                    linewidths=0.4,
                    linecolor="white",
                    cbar_kws={"label": "Mean |SHAP Interaction Value|", "shrink": 0.75},
                    square=True,
                )
                ax.set_title(
                    f"Exact SHAP Interaction Matrix  (Top {n_sv} features)",
                    fontsize=12, pad=14,
                )
                ax.set_xlabel("Feature j", fontsize=10)
                ax.set_ylabel("Feature i", fontsize=10)
                ax.tick_params(axis="x", labelsize=tick_fs_sv, rotation=45)
                ax.tick_params(axis="y", labelsize=tick_fs_sv, rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── 精確版排行榜（只取上三角，因已對稱化）──────────
                st.markdown("#### 📊 精確版交互強度排行榜 Top 15")
                rows_sv = []
                for ii in range(n_sv):
                    for jj in range(ii + 1, n_sv):
                        rows_sv.append({
                            "代碼 i": labels_sv[ii],
                            "代碼 j": labels_sv[jj],
                            "原始名稱 i": top_feats_sv[ii],
                            "原始名稱 j": top_feats_sv[jj],
                            "Mean |SHAP Interaction|": round(float(mean_abs_sv[ii, jj]), 5),
                        })
                rank_sv = (
                    pd.DataFrame(rows_sv)
                    .sort_values("Mean |SHAP Interaction|", ascending=False)
                    .reset_index(drop=True)
                    .head(15)
                )
                st.dataframe(
                    rank_sv.style.background_gradient(
                        cmap="YlOrRd", subset=["Mean |SHAP Interaction|"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── 與 Dependence Plot auto 的對應說明 ─────────────
                top1 = rank_sv.iloc[0]
                st.success(
                    f"🏆 最強交互：**{top1['代碼 i']}** ↔ **{top1['代碼 j']}**  "
                    f"（{top1['原始名稱 i'][:30]} ↔ {top1['原始名稱 j'][:30]}）\n\n"
                    f"在 Dependence Plot 中選擇 **{top1['代碼 i']}** 作為主特徵、"
                    f"取消勾選「自動尋找」並手動選 **{top1['代碼 j']}** 作為交互著色，"
                    f"即可看到最精確的交互效應圖。"
                )
                st.caption(
                    "此結果與 Dependence Plot `interaction_index='auto'` 使用相同底層計算（TreeExplainer），"
                    "是三種方法中最精確的交互衡量。"
                )


# ═══════════════════════════════════════════════════════════
#  PLS-VIP Tab (enhanced with coefficients)
# ═══════════════════════════════════════════════════════════

def _render_pls_tab(fi_subtab, X_fi, y_fi, top_n_fi, X_te=None, y_te=None):
    with fi_subtab:
        st.markdown("#### PLS — VIP Score + 係數解釋")

        max_comp = min(10, X_fi.shape[1], len(y_fi)-1)
        if max_comp < 2:
            st.warning("樣本數或特徵數不足。"); return

        with st.expander("⚙️ PLS Hyperparameter 設定", expanded=False):
            p1, p2 = st.columns(2)
            cv_folds_pls = p1.slider("CV folds", 3, 10, 5, key="pls_cv_folds")
            scale_pls     = p2.checkbox("標準化 X (推薦)", value=True, key="pls_scale")

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
            fig, ax     = plt.subplots(figsize=(8, 4))
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
        q2_pls       = st.session_state.get("pls_q2")
        cv_metrics_p = st.session_state.get("pls_cv_metrics")
        y_tr_pred_p  = st.session_state.get("pls_y_tr_pred")
        y_te_pred_p  = st.session_state.get("pls_y_te_pred")

        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Train R²",  f"{r2_pls:.3f}")
        pm2.metric("Q²（CV）",  f"{q2_pls:.3f}" if q2_pls is not None else "—",
                   help="Q² > 0.5 有預測力；Q² 接近 R² 代表無過擬合")
        pm3.metric("過擬合風險", "低 ✅" if q2_pls and r2_pls - q2_pls < 0.15 else "注意 ⚠️" if q2_pls else "—")

        pls_subtabs = st.tabs(["VIP Score", "迴歸係數", "X Loadings", "X Weights", "📊 模型評估"])

        with pls_subtabs[0]:
            top_vip = vip_df.head(top_n_fi)
            fig, ax = plt.subplots(figsize=(10, max(5, top_n_fi*0.45)))
            ax.barh(top_vip["Feature"], top_vip["VIP"],
                    color=["#e84855" if v>=1.0 else "#2e86ab" for v in top_vip["VIP"]], alpha=0.85)
            ax.axvline(1.0, color="#e84855", linestyle="--", lw=1.5, label="VIP=1")
            ax.set_title(f"PLS VIP Score (Top {top_n_fi})", fontsize=13)
            ax.set_xlabel("VIP Score"); ax.invert_yaxis(); ax.legend()
            ax.grid(axis="x", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.caption("VIP ≥ 1.0（紅色）= 對目標變數有顯著影響。")
            st.dataframe(vip_df.style.background_gradient(cmap="Reds", subset=["VIP"]),
                         use_container_width=True, hide_index=True)

        with pls_subtabs[1]:
            st.markdown("**PLS 迴歸係數**（標準化空間）")
            top_coef = reg_coef_df.head(top_n_fi)
            fig, ax  = plt.subplots(figsize=(10, max(5, top_n_fi*0.45)))
            ax.barh(top_coef["Feature"], top_coef["PLS_Coef"],
                    color=["#e84855" if v>0 else "#2e86ab" for v in top_coef["PLS_Coef"]], alpha=0.85)
            ax.axvline(0, color="black", lw=1)
            ax.set_title("PLS Regression Coefficients", fontsize=13)
            ax.set_xlabel("Coefficient（正=增加Y，負=降低Y）"); ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(reg_coef_df.style.background_gradient(cmap="RdBu_r", subset=["PLS_Coef"]),
                         use_container_width=True, hide_index=True)

        with pls_subtabs[2]:
            st.markdown("**X Loadings（P）**")
            st.dataframe(x_load.style.background_gradient(cmap="RdBu_r"), use_container_width=True)
            fig, ax = plt.subplots(figsize=(max(6, n_pls*1.5), max(6, len(X_fi.columns)*0.35)))
            sns.heatmap(x_load, annot=True, fmt=".3f", cmap="RdBu_r",
                        center=0, linewidths=0.3, ax=ax, annot_kws={"size":7})
            ax.set_title("X Loadings (P) Heatmap")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with pls_subtabs[3]:
            st.markdown("**X Weights（W）**")
            st.dataframe(x_weights.style.background_gradient(cmap="RdBu_r"), use_container_width=True)
            fig, ax = plt.subplots(figsize=(max(6, n_pls*1.5), max(6, len(X_fi.columns)*0.35)))
            sns.heatmap(x_weights, annot=True, fmt=".3f", cmap="RdBu_r",
                        center=0, linewidths=0.3, ax=ax, annot_kws={"size":7})
            ax.set_title("X Weights (W) Heatmap")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with pls_subtabs[4]:
            if y_tr_pred_p is not None:
                _render_eval_section(
                    model_name="PLS",
                    y_train=y_fi, y_train_pred=np.array(y_tr_pred_p),
                    n_feat=X_fi.shape[1], cv_results=cv_metrics_p,
                    y_test=y_te if X_te is not None else None,
                    y_test_pred=np.array(y_te_pred_p) if y_te_pred_p is not None else None,
                    key_prefix="pls_eval",
                )
            else:
                st.info("請重新執行 PLS 以產生評估數據。")


# ═══════════════════════════════════════════════════════════
#  🔄 [新功能] — 相關性熱點圖函式
# ═══════════════════════════════════════════════════════════

def _render_correlation_heatmap(X_fi, top_n_fi):
    """
    繪製 Top N 個重要特徵之間的相關性熱點圖，協助初步發現潛在交互作用。
    """
    with st.spinner("生成相關性矩陣中..."):
        st.markdown("#### Top 特徵相關性矩陣")
        st.caption("觀察 Top 特徵之間的聯動關係。")
        
        fi_df = st.session_state.get("fi_perm_df")
        if fi_df is None:
            st.warning("請先執行 RF 訓練。")
            return
            
        top_feats = fi_df["Feature"].head(top_n_fi).tolist()
        corr = X_fi[top_feats].corr()
        
        # 💡 關鍵修正 1：根據特徵數量動態調整圖表大小
        fig_width = max(10, top_n_fi * 0.7)
        fig_height = max(8, top_n_fi * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 💡 關鍵修正 2：動態調整 Annot 字體大小，避免數字擠在一起
        annot_size = 8 if top_n_fi <= 10 else 6
        
        sns.heatmap(
            corr, 
            annot=True, 
            fmt=".2f", 
            cmap="RdBu_r", 
            center=0, 
            ax=ax, 
            annot_kws={"size": annot_size},
            cbar_kws={"shrink": 0.8}
        )
        
        # 💡 關鍵修正 3：旋轉標籤並進行截斷或調整
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        
        ax.set_title(f"Top {top_n_fi} Features Correlation Heatmap", fontsize=14, pad=20)
        
        # 💡 關鍵修正 4：強制使用 tight_layout 並增加邊距
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()

def _render_interaction_ranking(X_fi, top_n_fi):
    """
    計算並顯示潛在交互作用排名
    """
    st.markdown("#### 🔄 潛在交互作用排名 (Top Pairs)")
    st.caption("評分 = |相關係數| × (特徵 A 重要性 + 特徵 B 重要性)。得分越高，代表這兩個參數越值得去 Dependence Plot 觀察。")
    
    fi_df = st.session_state.get("fi_perm_df")
    if fi_df is None:
        st.warning("請先在「RF 重要性」分頁訓練模型以取得特徵權重。")
        return
    
    # 1. 取得 Top 特徵與其重要性分數
    top_feats = fi_df.head(top_n_fi)
    feat_list = top_feats["Feature"].tolist()
    importance_map = dict(zip(top_feats["Feature"], top_feats["Perm_Importance"]))
    
    # 2. 計算兩兩組合的評分
    corr_matrix = X_fi[feat_list].corr().abs()
    pairs = []
    for i in range(len(feat_list)):
        for j in range(i + 1, len(feat_list)):
            f1, f2 = feat_list[i], feat_list[j]
            # 取得相關係數（避免 NaN）
            c_val = corr_matrix.loc[f1, f2]
            if np.isnan(c_val): c_val = 0
            
            # 計算綜合評分
            score = c_val * (importance_map[f1] + importance_map[f2])
            pairs.append({
                "特徵組合": f"{f1} ↔ {f2}",
                "綜合推薦得分": score,
                "相關性(Abs)": c_val
            })
    
    if not pairs:
        st.write("數據不足以計算排名。")
        return

    # 3. 轉為 DataFrame 並顯示前 10 名
    rank_df = pd.DataFrame(pairs).sort_values("綜合推薦得分", ascending=False).head(10)
    
    st.table(rank_df.style.format({
        "綜合推薦得分": "{:.4f}",
        "相關性(Abs)": "{:.2f}"
    }))
    
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

    # ── 資料準備設定 ──────────────────────────────────────
    prep_c1, prep_c2 = st.columns(2)
    use_split  = prep_c1.checkbox("切分 Train / Test Set", value=False, key="fi_use_split")
    test_size  = prep_c2.slider("Test Set 比例", 0.1, 0.4, 0.2, 0.05,
                                 key="fi_test_size", disabled=not use_split)

    if st.button("📦 準備資料", key="run_fi_prepare"):
        try:
            exclude = [c for c in ["BatchID", target_col_fi] if c in work_df.columns]
            X_all = work_df.drop(columns=exclude, errors="ignore").select_dtypes(include=["number"])
            y_all = work_df[target_col_fi]
            valid = y_all.notna() & X_all.notna().all(axis=1)
            X_all = X_all[valid].reset_index(drop=True)
            y_all = y_all[valid].reset_index(drop=True)

            if use_split:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all, test_size=test_size, random_state=42
                )
                X_tr = X_tr.reset_index(drop=True)
                X_te = X_te.reset_index(drop=True)
                y_tr = y_tr.reset_index(drop=True)
                y_te = y_te.reset_index(drop=True)
            else:
                X_tr, y_tr = X_all, y_all
                X_te, y_te = None, None

            st.session_state.update({
                "fi_X": X_tr, "fi_y": y_tr,
                "fi_X_test": X_te, "fi_y_test": y_te,
                "fi_use_split": use_split,
                "fi_target_col": target_col_fi,
                "fi_rf": None, "shap_vals": None,
                "lasso_coef_df": None, "pls_vip_df": None,
                "mi_df": None,
            })
            msg = f"✅ 資料準備完成：{X_tr.shape[0]} 筆訓練"
            if use_split:
                msg += f" + {X_te.shape[0]} 筆測試（共 {X_all.shape[0]} 筆）× {X_tr.shape[1]} 特徵"
            else:
                msg += f" × {X_tr.shape[1]} 特徵（未切 Test Set）"
            st.success(msg)
        except Exception as e:
            st.error(f"失敗：{e}")

    if st.session_state.get("fi_X") is None:
        return

    X_fi      = st.session_state["fi_X"]
    y_fi      = st.session_state["fi_y"]
    X_te      = st.session_state.get("fi_X_test")
    y_te      = st.session_state.get("fi_y_test")
    use_split = st.session_state.get("fi_use_split", False)

    if use_split and X_te is not None:
        st.info(f"📊 Train: {len(y_fi)} 筆　｜　Test: {len(y_te)} 筆", icon="ℹ️")

    # 🔄 [新功能] — 在 Tabs 清單最後加一個「🔄 交互作用探索」
    fi_subtabs = st.tabs(["🌲 RF 重要性", "🔪 Lasso 重要性", "🔮 SHAP 分析", "📐 PLS-VIP", "🔄 交互作用探索"])
    _render_rf_tab(fi_subtabs[0], X_fi, y_fi, top_n_fi, X_te, y_te)
    _render_lasso_tab(fi_subtabs[1], X_fi, y_fi, top_n_fi, X_te, y_te)
    _render_shap_tab(fi_subtabs[2], X_fi, y_fi, top_n_fi)
    _render_pls_tab(fi_subtabs[3], X_fi, y_fi, top_n_fi, X_te, y_te)
    
    # 🔄 [新功能] — 渲染交互作用分頁的熱點圖
    with fi_subtabs[4]:
        _render_correlation_heatmap(X_fi, top_n_fi)
        # --- 加上下面這一行 ---
        st.divider() # 加一條分割線
        _render_interaction_ranking(X_fi, top_n_fi)
