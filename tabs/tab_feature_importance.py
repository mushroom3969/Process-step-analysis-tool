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
from scipy import stats as _sp_stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    cross_val_predict, cross_val_score, KFold,
)
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error, max_error,
)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _wrap_tick_labels(ax, tick_getter, reverse_map, width=40, fontsize=8):
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), width))
         for t in tick_getter()],
        fontsize=fontsize)


def _adj_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R²  =  1 - (1-R²)(n-1)/(n-p-1)"""
    if n is None or p is None or (n - p - 1) <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def _fmt(v, d=4):
    """Format a float or return '—' for nan/None."""
    if v is None:
        return "—"
    try:
        if math.isnan(float(v)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(v):.{d}f}"


# ═══════════════════════════════════════════════════════════
#  Shared Evaluation Section
# ═══════════════════════════════════════════════════════════

def _render_eval_section(
    model_name: str,
    y_train, y_train_pred,
    y_test=None, y_test_pred=None,
    cv_scores: dict = None,
    n_features: int = 1,
    oob_r2: float = None,
):
    """
    Render a full evaluation panel reused by RF, Lasso, and PLS.

    Parameters
    ----------
    model_name      display title
    y_train / y_train_pred  training set actuals + preds
    y_test  / y_test_pred   test set actuals + preds (or None)
    cv_scores               dict  metric_name -> np.ndarray  fold scores
                            建議只傳 R²、RMSE、MAE（MedAE/MaxErr 不適合 K-fold ±）
    n_features              number of input features (for Adj R²)
    oob_r2                  OOB R² (RF only, or None)

    顯示邏輯
    --------
    有 Test split  → 指標表（Train + Test 點估計）+ 各圖並列 + K-fold 作補充
    無 Test split  → K-fold ± std 升格為主角 + Train 圖（標注「僅供參考」）
    """
    y_train      = np.asarray(y_train,      dtype=float)
    y_train_pred = np.asarray(y_train_pred, dtype=float)
    has_test = y_test is not None and y_test_pred is not None
    if has_test:
        y_test      = np.asarray(y_test,      dtype=float)
        y_test_pred = np.asarray(y_test_pred, dtype=float)

    p = n_features

    # ── 內部小工具 ────────────────────────────────────────

    def _metrics(yt, yp, label):
        n    = len(yt)
        r2   = float(r2_score(yt, yp))
        adj  = _adj_r2(r2, n, p)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae  = float(mean_absolute_error(yt, yp))
        nz   = yt != 0
        mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.any() else float("nan")
        mede = float(median_absolute_error(yt, yp))
        maxe = float(max_error(yt, yp))
        return {
            "集合":    label,
            "R²":      round(r2,   4),
            "Adj.R²":  round(adj,  4) if not math.isnan(adj) else "—",
            "RMSE":    round(rmse, 4),
            "MAE":     round(mae,  4),
            "MAPE(%)": round(mape, 2) if not math.isnan(mape) else "—",
            "MedAE":   round(mede, 4),
            "MaxErr":  round(maxe, 4),
        }

    def _avp_ax(ax, yt, yp, label, color):
        mn = min(yt.min(), yp.min()); mx = max(yt.max(), yp.max())
        ax.scatter(yt, yp, alpha=0.6, color=color, s=30, edgecolors="none")
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.5)
        r2v  = r2_score(yt, yp)
        adjv = _adj_r2(r2v, len(yt), p)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{label}\nR²={r2v:.3f}  Adj.R²={_fmt(adjv)}", fontsize=10)
        ax.grid(alpha=0.3)

    def _resid_ax(ax, yt, yp, label, color):
        resid = yt - yp
        sigma = float(np.std(resid))
        ax.scatter(yp, resid, alpha=0.6, color=color, s=30, edgecolors="none")
        ax.axhline(0, color="black", lw=1.5)
        ax.axhline( sigma, color="gray", lw=1, ls="--", label=f"+1σ={sigma:.3f}")
        ax.axhline(-sigma, color="gray", lw=1, ls="--", label=f"−1σ={sigma:.3f}")
        ax.fill_between([yp.min(), yp.max()], -sigma, sigma, alpha=0.1, color="gray")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
        ax.set_title(f"Residual vs Predicted ({label})", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    def _hist_sw_ax(ax, yt, yp, label, color):
        resid  = yt - yp
        n_bins = min(30, max(10, len(resid) // 5))
        ax.hist(resid, bins=n_bins, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", lw=1.5)
        if len(resid) >= 3:
            try:
                sw_stat, sw_p = _sp_stats.shapiro(resid[:5000])
                tag      = "✅ 常態 (p>0.05)" if sw_p > 0.05 else "⚠️ 非常態 (p≤0.05)"
                norm_str = f"Shapiro W={sw_stat:.3f}, p={sw_p:.4f}  {tag}"
            except Exception:
                norm_str = "Shapiro: 無法計算"
        else:
            norm_str = "樣本數不足"
        ax.set_xlabel("Residual"); ax.set_ylabel("Count")
        ax.set_title(f"殘差直方圖 ({label})\n{norm_str}", fontsize=9)
        ax.grid(alpha=0.3)

    def _qq_ax(ax, yt, yp, label, color):
        resid = yt - yp
        (osm, osr), (slope, intercept, r) = _sp_stats.probplot(resid)
        ax.scatter(osm, osr, alpha=0.6, color=color, s=20, edgecolors="none")
        xl = np.array([osm[0], osm[-1]])
        ax.plot(xl, slope * xl + intercept, "k-", lw=1.5)
        ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"Q-Q Plot ({label})  Corr={r:.3f}", fontsize=10)
        ax.grid(alpha=0.3)

    # K-fold 只顯示 R²、RMSE、MAE（MedAE/MaxErr fold 間變異太大，不適合 ±）
    CV_SHOW_METRICS = {"R²", "RMSE", "MAE", "R² (Q²-fold)"}

    def _render_cv(cv_scores: dict, title_prefix: str = ""):
        """K-fold ± std 表 + Boxplot，只顯示適合的指標。"""
        filtered = {k: v for k, v in cv_scores.items() if k in CV_SHOW_METRICS}
        if not filtered:
            return
        st.markdown(f"**📦 {title_prefix}K-Fold CV mean ± std**")
        cv_rows = []
        for metric, vals in filtered.items():
            arr = np.asarray(vals, dtype=float)
            cv_rows.append({
                "指標":          metric,
                "Mean":          round(float(arr.mean()), 4),
                "Std":           round(float(arr.std()),  4),
                "Min":           round(float(arr.min()),  4),
                "Max":           round(float(arr.max()),  4),
                "CV mean ± std": f"{arr.mean():.4f} ± {arr.std():.4f}",
            })
        st.dataframe(pd.DataFrame(cv_rows).set_index("指標"), use_container_width=True)

        st.markdown(f"**📦 {title_prefix}CV Fold 分布 Boxplot**")
        mk  = list(filtered.keys())
        pal = ["#2e86ab", "#e84855", "#f4a261", "#43aa8b", "#9b5de5"]
        fig, ax = plt.subplots(figsize=(max(5, len(mk) * 1.8), 4))
        bp = ax.boxplot([np.asarray(filtered[m], dtype=float) for m in mk], patch_artist=True)
        for patch, c in zip(bp["boxes"], pal * 10):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(mk) + 1))
        ax.set_xticklabels(mk, fontsize=11)
        ax.set_title("K-Fold CV Fold Distribution（R²、RMSE、MAE）")
        ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ════════════════════════════════════════════════════════
    st.markdown(f"##### 📊 {model_name} — 模型評估")

    # ════════════════════════════════════════════════════════
    #  分支 A：有 Train / Test Split
    # ════════════════════════════════════════════════════════
    if has_test:
        # ── 指標表（點估計） ──────────────────────────────
        st.markdown("**📋 評估指標（點估計）**")
        rows = [_metrics(y_train, y_train_pred, "Train"),
                _metrics(y_test,  y_test_pred,  "Test")]
        if oob_r2 is not None:
            rows.append({
                "集合": "OOB", "R²": round(float(oob_r2), 4), "Adj.R²": "—",
                "RMSE": "—", "MAE": "—", "MAPE(%)": "—", "MedAE": "—", "MaxErr": "—",
            })
        st.dataframe(pd.DataFrame(rows).set_index("集合"), use_container_width=True)

        # ── 圖表（Train + Test 並列） ─────────────────────
        st.markdown("**📈 Actual vs Predicted**")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 5))
            _avp_ax(ax, y_train, y_train_pred, "Train", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 5))
            _avp_ax(ax, y_test, y_test_pred, "Test", "#e84855")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**📉 Residual vs Predicted（含 ±1σ 帶）**")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 5))
            _resid_ax(ax, y_train, y_train_pred, "Train", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 5))
            _resid_ax(ax, y_test, y_test_pred, "Test", "#e84855")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**🔔 殘差分布直方圖 + Shapiro-Wilk 常態性檢定**")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 5))
            _hist_sw_ax(ax, y_train, y_train_pred, "Train", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 5))
            _hist_sw_ax(ax, y_test, y_test_pred, "Test", "#e84855")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**📊 Q-Q Plot（常態性視覺化）**")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 5))
            _qq_ax(ax, y_train, y_train_pred, "Train", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 5))
            _qq_ax(ax, y_test, y_test_pred, "Test", "#e84855")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── K-fold 作補充 ─────────────────────────────────
        if cv_scores:
            st.divider()
            st.caption("📌 以下 K-Fold CV 為補充驗證（R²、RMSE、MAE），MedAE / MaxErr 不納入 CV ±。")
            _render_cv(cv_scores)

    # ════════════════════════════════════════════════════════
    #  分支 B：無 Train / Test Split → K-fold 是主角
    # ════════════════════════════════════════════════════════
    else:
        st.info(
            "⚠️ 目前使用**全資料訓練**，無獨立 Test Set。"
            "K-Fold CV 為主要泛化能力評估；Train 圖表僅供診斷參考，**請勿以 Train 指標判斷泛化能力**。",
            icon="ℹ️",
        )

        # ── K-fold 升格為主角 ─────────────────────────────
        if cv_scores:
            _render_cv(cv_scores, title_prefix="【主要評估】")
        elif oob_r2 is not None:
            st.markdown("**📦 OOB R²（袋外估計，不偏）**")
            st.metric("OOB R²", f"{oob_r2:.4f}",
                      help="OOB = Out-Of-Bag，相當於不偏的 Test R²，可作為泛化能力參考。")
        else:
            st.warning("未提供 CV scores，無法顯示泛化評估。")

        st.divider()

        # ── Train 診斷圖（僅供參考） ──────────────────────
        with st.expander("🔍 Train 診斷圖（全資料，僅供參考）", expanded=False):
            # 指標表
            st.markdown("**📋 Train 指標（全資料點估計，非泛化指標）**")
            rows = [_metrics(y_train, y_train_pred, "Train（全資料）")]
            if oob_r2 is not None:
                rows.append({
                    "集合": "OOB", "R²": round(float(oob_r2), 4), "Adj.R²": "—",
                    "RMSE": "—", "MAE": "—", "MAPE(%)": "—", "MedAE": "—", "MaxErr": "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("集合"), use_container_width=True)

            st.markdown("**📈 Actual vs Predicted（Train，僅供參考）**")
            fig, ax = plt.subplots(figsize=(6, 5))
            _avp_ax(ax, y_train, y_train_pred, "Train（全資料）", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("**📉 Residual vs Predicted（含 ±1σ 帶）**")
            fig, ax = plt.subplots(figsize=(6, 5))
            _resid_ax(ax, y_train, y_train_pred, "Train（全資料）", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("**🔔 殘差分布直方圖 + Shapiro-Wilk 常態性檢定**")
            fig, ax = plt.subplots(figsize=(6, 5))
            _hist_sw_ax(ax, y_train, y_train_pred, "Train（全資料）", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("**📊 Q-Q Plot（常態性視覺化）**")
            fig, ax = plt.subplots(figsize=(6, 5))
            _qq_ax(ax, y_train, y_train_pred, "Train（全資料）", "#2e86ab")
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
#  RF Tab
# ═══════════════════════════════════════════════════════════

def _render_rf_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        rf_subtabs = st.tabs(["🌲 重要性分析", "📊 模型評估"])

        # ── Sub-tab 0: importance ────────────────────────
        with rf_subtabs[0]:
            st.markdown("#### Random Forest — 調整 Hyperparameter")

            with st.expander("⚙️ RF Hyperparameter 設定", expanded=False):
                r1, r2, r3, r4, r5 = st.columns(5)
                n_est      = r1.slider("n_estimators",    50, 500, 200, 50, key="rf_n_est")
                max_dep    = r2.slider("max_depth",         2,  20,   5,  1, key="rf_max_dep")
                min_leaf   = r3.slider("min_samples_leaf",  1,  20,   4,  1, key="rf_min_leaf")
                n_rep      = r4.slider("n_repeats (perm)",  5,  30,  15,  5, key="rf_n_rep")
                cv_folds_r = r5.slider("K-Fold 折數",       3,  10,   5,  1, key="rf_cv_folds")

            if st.button("🌲 訓練 Random Forest", key="run_rf"):
                with st.spinner("RF 訓練中..."):
                    rf = RandomForestRegressor(
                        n_estimators=n_est, max_features="sqrt",
                        max_depth=max_dep, min_samples_leaf=min_leaf,
                        oob_score=True, random_state=42)
                    rf.fit(X_fi, y_fi)

                    perm = permutation_importance(rf, X_fi, y_fi,
                                                  n_repeats=n_rep, random_state=42)
                    perm_df = pd.DataFrame({
                        "Feature":         X_fi.columns,
                        "Perm_Importance": perm.importances_mean,
                        "Std":             perm.importances_std,
                    }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)

                    mdi_df = pd.DataFrame({
                        "Feature":        X_fi.columns,
                        "MDI_Importance": rf.feature_importances_,
                    }).sort_values("MDI_Importance", ascending=False).reset_index(drop=True)

                    y_pred_rf = rf.predict(X_fi)
                    r2_rf     = float(r2_score(y_fi, y_pred_rf))
                    oob_r2    = float(rf.oob_score_) if hasattr(rf, "oob_score_") else None

                    # K-Fold CV (integrated)
                    kf       = KFold(n_splits=cv_folds_r, shuffle=True, random_state=42)
                    cv_r2    = cross_val_score(rf, X_fi, y_fi, cv=kf, scoring="r2")
                    cv_rmse  = np.sqrt(-cross_val_score(rf, X_fi, y_fi, cv=kf,
                                                         scoring="neg_mean_squared_error"))
                    cv_mae   = -cross_val_score(rf, X_fi, y_fi, cv=kf,
                                                scoring="neg_mean_absolute_error")

                    st.session_state.update({
                        "fi_rf": rf, "fi_perm_df": perm_df, "fi_mdi_df": mdi_df,
                        "fi_r2": r2_rf, "fi_oob_r2": oob_r2, "shap_vals": None,
                        # eval cache
                        "_rf_y":          np.asarray(y_fi,     dtype=float),
                        "_rf_y_pred":     y_pred_rf,
                        "_rf_cv_scores":  {"R²": cv_r2, "RMSE": cv_rmse, "MAE": cv_mae},
                    })

                n, p = len(y_fi), X_fi.shape[1]
                adj  = _adj_r2(r2_rf, n, p)
                oob_str = f"  OOB R²={oob_r2:.3f}" if oob_r2 else ""
                st.success(
                    f"✅ 完成！訓練 R²={r2_rf:.3f}  Adj.R²={_fmt(adj)}{oob_str}"
                )

            if st.session_state.get("fi_perm_df") is None:
                st.info("點擊「🌲 訓練 Random Forest」開始。")
                return

            perm_df = st.session_state["fi_perm_df"]
            oob_r2  = st.session_state.get("fi_oob_r2")
            mdi_df  = st.session_state.get("fi_mdi_df")
            r2_rf   = st.session_state.get("fi_r2", 0)

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

            n, p = len(y_fi), X_fi.shape[1]
            adj  = _adj_r2(r2_rf, n, p)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("訓練集 R²",        f"{r2_rf:.3f}")
            c2.metric("訓練集 Adj.R²",    _fmt(adj))
            c3.metric("OOB R²（不偏）",   f"{oob_r2:.3f}" if oob_r2 is not None else "—")
            c4.metric("特徵數",            X_fi.shape[1])

            st.dataframe(perm_df.style.background_gradient(cmap="Blues", subset=["Perm_Importance"]),
                         width="stretch", hide_index=True)

            if mdi_df is not None:
                st.markdown("#### MDI vs Permutation Importance 比較")
                st.caption(
                    "**MDI**（Mean Decrease Impurity）：訓練集內計算，易受高基數特徵影響，速度快。"
                    "　**Permutation**：亂序後測量預測力下降，更可靠但較慢。兩者方向一致代表結果穩健。"
                )
                merge_df = perm_df[["Feature", "Perm_Importance"]].merge(
                    mdi_df[["Feature", "MDI_Importance"]], on="Feature"
                ).head(top_n_fi)
                merge_df["Perm_norm"] = merge_df["Perm_Importance"] / (merge_df["Perm_Importance"].abs().max() + 1e-9)
                merge_df["MDI_norm"]  = merge_df["MDI_Importance"]  / (merge_df["MDI_Importance"].max() + 1e-9)
                merge_df = merge_df.sort_values("Perm_norm", ascending=True)
                fig, ax = plt.subplots(figsize=(10, max(5, len(merge_df)*0.4)))
                y_pos = range(len(merge_df))
                ax.barh([p2 - 0.18 for p2 in y_pos], merge_df["Perm_norm"], height=0.35,
                        color="#2e86ab", alpha=0.85, label="Permutation (norm)")
                ax.barh([p2 + 0.18 for p2 in y_pos], merge_df["MDI_norm"],  height=0.35,
                        color="#f4a261", alpha=0.85, label="MDI (norm)")
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(merge_df["Feature"].tolist(), fontsize=8)
                ax.set_xlabel("正規化重要性（0-1）")
                ax.set_title(f"MDI vs Permutation Importance（Top {len(merge_df)}）")
                ax.legend(fontsize=9)
                ax.grid(axis="x", linestyle="--", alpha=0.4)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Sub-tab 1: eval ──────────────────────────────
        with rf_subtabs[1]:
            if st.session_state.get("fi_rf") is None:
                st.info("請先在「🌲 重要性分析」分頁訓練 RF 模型。")
            else:
                _render_eval_section(
                    "Random Forest",
                    st.session_state["_rf_y"],
                    st.session_state["_rf_y_pred"],
                    cv_scores=st.session_state.get("_rf_cv_scores"),
                    n_features=X_fi.shape[1],
                    oob_r2=st.session_state.get("fi_oob_r2"),
                )


# ═══════════════════════════════════════════════════════════
#  Lasso Tab
# ═══════════════════════════════════════════════════════════

def _render_lasso_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        la_subtabs = st.tabs(["🔪 係數分析", "📊 模型評估"])

        with la_subtabs[0]:
            st.markdown("#### Lasso Regression — 係數重要性")
            st.caption("Lasso 透過 L1 正則化自動將不重要特徵的係數壓縮為 0，達到特徵選擇效果。")

            with st.expander("⚙️ Lasso Hyperparameter 設定", expanded=False):
                la1, la2 = st.columns(2)
                use_cv       = la1.checkbox("自動選擇 α（LassoCV）", value=True, key="lasso_use_cv")
                alpha_manual = la2.number_input("手動 α（use_cv=False 時）",
                                                 min_value=1e-6, max_value=10.0,
                                                 value=0.01, format="%.4f", key="lasso_alpha")
                la3, la4 = st.columns(2)
                max_iter   = la3.slider("max_iter", 500, 5000, 1000, 500, key="lasso_max_iter")
                cv_folds_l = la4.slider("CV folds（LassoCV）", 3, 10, 5, key="lasso_cv")

            if st.button("🔍 執行 Lasso", key="run_lasso"):
                with st.spinner("Lasso 計算中..."):
                    scaler   = StandardScaler()
                    X_scaled = scaler.fit_transform(X_fi)

                    if use_cv:
                        model = LassoCV(cv=cv_folds_l, max_iter=max_iter, random_state=42)
                        model.fit(X_scaled, y_fi)
                        best_alpha = model.alpha_
                    else:
                        model = Lasso(alpha=alpha_manual, max_iter=max_iter, random_state=42)
                        model.fit(X_scaled, y_fi)
                        best_alpha = alpha_manual

                    coef_df = pd.DataFrame({
                        "Feature":     X_fi.columns,
                        "Coefficient": model.coef_,
                        "Abs_Coef":    np.abs(model.coef_),
                    }).sort_values("Abs_Coef", ascending=False).reset_index(drop=True)

                    y_pred_la = model.predict(X_scaled)
                    r2_la     = float(r2_score(y_fi, y_pred_la))
                    n_nonzero = int((coef_df["Coefficient"] != 0).sum())

                    # K-Fold CV (integrated)
                    lasso_fixed = Lasso(alpha=best_alpha, max_iter=max_iter, random_state=42)
                    kf_l       = KFold(n_splits=cv_folds_l, shuffle=True, random_state=42)
                    cv_r2_l    = cross_val_score(lasso_fixed, X_scaled, y_fi, cv=kf_l, scoring="r2")
                    cv_rmse_l  = np.sqrt(-cross_val_score(lasso_fixed, X_scaled, y_fi, cv=kf_l,
                                                           scoring="neg_mean_squared_error"))
                    cv_mae_l   = -cross_val_score(lasso_fixed, X_scaled, y_fi, cv=kf_l,
                                                   scoring="neg_mean_absolute_error")

                    st.session_state.update({
                        "lasso_model":     model,
                        "lasso_coef_df":   coef_df,
                        "lasso_alpha_val": best_alpha,
                        "lasso_r2":        r2_la,
                        "lasso_scaler":    scaler,
                        "lasso_n_nonzero": n_nonzero,
                        # eval cache
                        "_la_y":         np.asarray(y_fi, dtype=float),
                        "_la_y_pred":    y_pred_la,
                        "_la_cv_scores": {"R²": cv_r2_l, "RMSE": cv_rmse_l, "MAE": cv_mae_l},
                    })

                n, p = len(y_fi), X_fi.shape[1]
                adj  = _adj_r2(r2_la, n, p)
                st.success(
                    f"✅ 完成！α={best_alpha:.4f}  R²={r2_la:.3f}  Adj.R²={_fmt(adj)}"
                    f"  非零係數：{n_nonzero}/{X_fi.shape[1]}"
                )

            if st.session_state.get("lasso_coef_df") is None:
                st.info("點擊「🔍 執行 Lasso」開始。")
            else:
                coef_df    = st.session_state["lasso_coef_df"]
                best_alpha = st.session_state.get("lasso_alpha_val",
                                                   st.session_state.get("lasso_alpha", 0.01))
                r2_la      = st.session_state["lasso_r2"]
                n_nonzero  = st.session_state["lasso_n_nonzero"]

                n, p = len(y_fi), X_fi.shape[1]
                adj  = _adj_r2(r2_la, n, p)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("最佳 α",    f"{best_alpha:.4f}")
                m2.metric("R²",        f"{r2_la:.3f}")
                m3.metric("Adj.R²",    _fmt(adj))
                m4.metric("選出特徵數", f"{n_nonzero}")

                show_nonzero_only = st.checkbox("只顯示非零係數特徵", value=True, key="lasso_nonzero_only")
                plot_df  = coef_df[coef_df["Coefficient"] != 0] if show_nonzero_only else coef_df
                top_plot = plot_df.head(top_n_fi)

                if top_plot.empty:
                    st.warning("所有係數均為 0，嘗試降低 α 值。")
                else:
                    fig, ax = plt.subplots(figsize=(10, max(4, len(top_plot)*0.4)))
                    colors  = ["#e84855" if v > 0 else "#2e86ab" for v in top_plot["Coefficient"]]
                    ax.barh(top_plot["Feature"], top_plot["Coefficient"], color=colors, alpha=0.85)
                    ax.axvline(0, color="black", lw=1)
                    ax.set_title(f"Lasso Coefficients (α={best_alpha:.4f}, Top {len(top_plot)})", fontsize=13)
                    ax.set_xlabel("Standardised Coefficient（正=增加Y，負=降低Y）")
                    ax.invert_yaxis(); ax.grid(axis="x", linestyle="--", alpha=0.5)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                st.caption("係數已在標準化特徵上計算，數值可直接比較特徵相對影響力。")
                st.dataframe(coef_df.style.background_gradient(cmap="RdBu_r", subset=["Coefficient"]),
                             width="stretch", hide_index=True)

                with st.expander("📈 Regularization Path（Alpha vs 係數）", expanded=False):
                    st.caption("觀察不同 α 值下各特徵的係數變化，了解特徵進入模型的順序。")
                    from sklearn.linear_model import lasso_path
                    scaler_path = StandardScaler()
                    X_path = scaler_path.fit_transform(X_fi)
                    alphas_path, coefs_path, _ = lasso_path(X_path, y_fi, eps=1e-3, n_alphas=80)
                    nonzero_mask = np.any(coefs_path != 0, axis=1)
                    top_feat_idx = np.argsort(np.abs(coefs_path[:, -1]))[-min(top_n_fi, nonzero_mask.sum()):]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for i in top_feat_idx:
                        ax.plot(np.log10(alphas_path), coefs_path[i], label=X_fi.columns[i][:25], lw=1.2)
                    ax.axvline(np.log10(best_alpha), color="#e84855", ls="--", lw=1.5,
                               label=f"最佳 α={best_alpha:.4f}")
                    ax.set_xlabel("log10(α)"); ax.set_ylabel("Coefficient")
                    ax.set_title("Lasso Regularization Path")
                    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
                    ax.grid(alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

        with la_subtabs[1]:
            if st.session_state.get("lasso_model") is None:
                st.info("請先在「🔪 係數分析」分頁執行 Lasso。")
            else:
                _render_eval_section(
                    "Lasso Regression",
                    st.session_state["_la_y"],
                    st.session_state["_la_y_pred"],
                    cv_scores=st.session_state.get("_la_cv_scores"),
                    n_features=X_fi.shape[1],
                )


# ═══════════════════════════════════════════════════════════
#  SHAP Tab (Enhanced with interaction feature) — unchanged
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
            ev_raw   = explainer.expected_value
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
            c1, c2 = st.columns(2)
            dep_feat = c1.selectbox("主特徵 (X軸)", X_fi.columns.tolist(), key="shap_dep_feat")
            auto_interaction = c2.checkbox("自動尋找最強交互特徵", value=True, key="shap_dep_auto")
            if auto_interaction:
                dep_int_index = "auto"
            else:
                dep_int_sel   = c2.selectbox("交互著色特徵 (Color)", X_fi.columns.tolist(), key="shap_dep_int_manual")
                dep_int_index = short_map[dep_int_sel]
            fig, ax = plt.subplots(figsize=(10, 6))
            shap_lib.dependence_plot(
                short_map[dep_feat], shap_arr, X_short,
                interaction_index=dep_int_index, ax=ax, show=False)
            ax.set_title(f"Interaction Analysis: {dep_feat}", fontsize=12)
            ax.set_xlabel(f"{dep_feat} (Actual Value)", fontsize=10)
            ax.set_ylabel("SHAP Value (Impact on Yield)", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with shap_subtabs[4]:
            st.markdown("#### 🔄 SHAP Interaction Matrix")
            st.caption(
                "以 `std(SHAP_fi | bins of fj)` 估計兩兩特徵間的非線性交互強度。"
                "欄位名稱使用 FXX 代號，對照請見上方對照表。"
            )
            im_c1, im_c2 = st.columns(2)
            top_n_inter  = im_c1.slider("納入計算的 Top 特徵數（依 RF 重要性）",
                                         5, min(30, X_fi.shape[1]), 15, key="inter_top_n")
            n_bins_inter = im_c2.slider("分箱數（bins）", 2, 10, 4, key="inter_bins")
            cmap_inter   = im_c1.selectbox("色圖", ["YlOrRd","Reds","plasma","hot_r","viridis"],
                                            key="inter_cmap")
            annot_inter  = im_c2.checkbox("顯示數值標注", value=True, key="inter_annot")

            if st.button("🧮 計算 Interaction Matrix", key="run_inter_matrix", type="primary"):
                perm_df = st.session_state.get("fi_perm_df")
                if perm_df is None:
                    st.error("請先在「RF 重要性」分頁訓練 RF 模型。")
                else:
                    with st.spinner("計算 SHAP 交互矩陣中..."):
                        try:
                            shap_arr_im  = np.array(st.session_state["shap_vals"])
                            top_feats    = perm_df["Feature"].head(top_n_inter).tolist()
                            feat_indices = [list(X_fi.columns).index(f) for f in top_feats]
                            n = len(top_feats)
                            inter_matrix = np.zeros((n, n))
                            for ii, fi_idx in enumerate(feat_indices):
                                shap_fi = shap_arr_im[:, fi_idx]
                                for jj, fj_idx in enumerate(feat_indices):
                                    if ii == jj: inter_matrix[ii,jj] = 0.0; continue
                                    fj_vals = X_fi.iloc[:, fj_idx].values.astype(float)
                                    valid   = ~np.isnan(fj_vals)
                                    if valid.sum() < n_bins_inter * 2:
                                        inter_matrix[ii,jj] = 0.0; continue
                                    try:
                                        qs = np.unique(np.percentile(fj_vals[valid],
                                                                      np.linspace(0,100,n_bins_inter+1)))
                                        if len(qs) < 2: inter_matrix[ii,jj] = 0.0; continue
                                        bm = []
                                        for b in range(len(qs)-1):
                                            m = (fj_vals >= qs[b]) & (fj_vals <= qs[b+1]) & valid
                                            if m.sum() > 0: bm.append(shap_fi[m].mean())
                                        inter_matrix[ii,jj] = float(np.std(bm)) if len(bm)>1 else 0.0
                                    except Exception: inter_matrix[ii,jj] = 0.0
                            short_map_im = {c: f"F{i:02d}" for i,c in enumerate(X_fi.columns)}
                            labels_im    = [short_map_im[f] for f in top_feats]
                            st.session_state.update({"inter_matrix": inter_matrix,
                                                      "inter_labels": labels_im,
                                                      "inter_top_feats": top_feats})
                        except Exception as e:
                            st.error(f"計算失敗：{e}")
                            import traceback; st.code(traceback.format_exc())

            if st.session_state.get("inter_matrix") is not None:
                inter_matrix = st.session_state["inter_matrix"]
                labels_im    = st.session_state["inter_labels"]
                top_feats    = st.session_state["inter_top_feats"]
                n = len(labels_im)
                cell_size = max(0.55, min(1.1, 14.0/n))
                fig_side  = max(7, n*cell_size)
                annot_fs  = max(6, min(10, int(80/n)))
                tick_fs   = max(7, min(11, int(90/n)))
                fig, ax = plt.subplots(figsize=(fig_side+1.5, fig_side))
                im_df   = pd.DataFrame(inter_matrix, index=labels_im, columns=labels_im)
                sns.heatmap(im_df, ax=ax, cmap=cmap_inter, annot=annot_inter,
                            fmt=".3f" if annot_inter else "",
                            annot_kws={"size": annot_fs}, linewidths=0.4, linecolor="white",
                            cbar_kws={"label":"Interaction Strength (std of SHAP_i | bins of fj)","shrink":0.75},
                            square=True)
                ax.set_title(f"SHAP Interaction Matrix  (Top {n} features, bins={n_bins_inter})",
                             fontsize=12, pad=14)
                ax.set_xlabel("fj  (conditioning feature)", fontsize=10)
                ax.set_ylabel("fi  (target SHAP)", fontsize=10)
                ax.tick_params(axis="x", labelsize=tick_fs, rotation=45)
                ax.tick_params(axis="y", labelsize=tick_fs, rotation=0)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.markdown("#### 📊 交互強度排行榜 Top 15")
                rows_rank = []
                for ii in range(n):
                    for jj in range(n):
                        if ii == jj: continue
                        rows_rank.append({
                            "fi（SHAP 受影響）": labels_im[ii], "fj（條件特徵）": labels_im[jj],
                            "fi 原始名稱": top_feats[ii], "fj 原始名稱": top_feats[jj],
                            "Interaction Strength": round(float(inter_matrix[ii,jj]),4),
                        })
                rank_df = (pd.DataFrame(rows_rank)
                           .sort_values("Interaction Strength", ascending=False)
                           .reset_index(drop=True).head(15))
                st.dataframe(rank_df.style.background_gradient(cmap="YlOrRd",
                             subset=["Interaction Strength"]),
                             use_container_width=True, hide_index=True)
                st.caption("解讀：Interaction Strength 越大，代表「當 fj 改變時，fi 的 SHAP 值波動越大」，即兩特徵存在較強的非線性交互效應。")

            st.markdown("---")
            st.markdown("#### 🎯 精確版 SHAP Interaction Values")
            st.info(
                "**原理**：呼叫 `TreeExplainer.shap_interaction_values()`，"
                "每個格子 (i,j) = 平均 |SHAP interaction value(i,j)|，"
                "與 Dependence Plot 的 `interaction_index='auto'` 使用相同底層計算，結果最精確。\n\n"
                "⚠️ **注意**：計算量為 O(n² × samples × trees)，特徵數或樣本數多時需要幾十秒。"
                "建議先用 Top N 特徵縮小範圍。", icon="ℹ️",
            )
            sv_c1, sv_c2 = st.columns(2)
            top_n_sv = sv_c1.slider("Top 特徵數（精確版）", 5, min(25,X_fi.shape[1]), 10, key="sv_top_n")
            cmap_sv  = sv_c2.selectbox("色圖", ["YlOrRd","Reds","plasma","hot_r","viridis"], key="sv_cmap")
            sv_c3, sv_c4 = st.columns(2)
            annot_sv = sv_c3.checkbox("顯示數值標注", value=True, key="sv_annot")
            symm_sv  = sv_c4.checkbox("對稱化（取 (i,j)+(j,i) 平均）", value=True, key="sv_symm",
                                       help="SHAP interaction matrix 理論上對稱，對稱化後更易閱讀。")

            if st.button("🎯 計算精確 SHAP Interaction Values", key="run_sv_inter", type="primary"):
                rf_model = st.session_state.get("fi_rf")
                perm_df  = st.session_state.get("fi_perm_df")
                _sv_ready = True
                if rf_model is None:
                    st.error("❌ 尚未訓練 RF 模型，請先至「🌲 RF 重要性」分頁點擊「訓練 Random Forest」。")
                    _sv_ready = False
                if perm_df is None:
                    st.error("❌ 找不到特徵重要性資料，請先執行 RF 訓練。"); _sv_ready = False
                if _sv_ready:
                    top_feats_sv = perm_df["Feature"].head(top_n_sv).tolist()
                    X_sv = X_fi[top_feats_sv]
                    ns, nf = X_sv.shape
                    est   = (ns * nf**2) / 5000
                    if est > 60:
                        st.warning(f"⚠️ 預估計算時間約 **{est:.0f} 秒**（{ns} 筆 × {nf} 特徵）。建議將「Top 特徵數」降至 8～10 以內再執行。")
                    else:
                        st.info(f"ℹ️ 預估計算時間約 {max(1,est):.0f} 秒，請稍候。")
                    with st.spinner(f"計算中：{ns} 筆 × {nf} 特徵 × {nf} 特徵..."):
                        try:
                            import shap as shap_lib_sv
                            expl_sv     = shap_lib_sv.TreeExplainer(rf_model)
                            sv_interact = expl_sv.shap_interaction_values(X_sv)
                            sv_arr      = np.array(sv_interact)
                            if sv_arr.ndim != 3:
                                st.error(f"❌ shap_interaction_values 回傳維度異常（shape={sv_arr.shape}）。可能是多輸出模型，目前僅支援單一輸出的 RandomForest。")
                            else:
                                mean_abs_sv = np.mean(np.abs(sv_arr), axis=0)
                                if symm_sv: mean_abs_sv = (mean_abs_sv + mean_abs_sv.T) / 2
                                np.fill_diagonal(mean_abs_sv, 0)
                                short_map_sv = {col: f"F{i:02d}" for i,col in enumerate(X_fi.columns)}
                                labels_sv    = [short_map_sv[f] for f in top_feats_sv]
                                st.session_state.update({"sv_interact_mat":    mean_abs_sv,
                                                          "sv_interact_labels": labels_sv,
                                                          "sv_interact_feats":  top_feats_sv})
                                st.success(f"✅ 計算完成！（{ns} 筆 × {nf} 特徵）")
                        except MemoryError:
                            st.error("❌ 記憶體不足（MemoryError）。請降低「Top 特徵數」或減少樣本數後重試。")
                        except Exception as e:
                            em = str(e)
                            if "timeout" in em.lower() or "time" in em.lower():
                                st.error("❌ 計算逾時，請降低特徵數後重試。")
                            elif "memory" in em.lower() or "alloc" in em.lower():
                                st.error("❌ 記憶體不足，請降低特徵數或樣本數後重試。")
                            elif "tree" in em.lower() or "model" in em.lower():
                                st.error(f"❌ 模型不相容：{em}\nshap_interaction_values 僅支援 Tree 系列模型（如 RandomForest）。")
                            else:
                                st.error(f"❌ 計算失敗：{em}")
                            import traceback
                            with st.expander("🔍 查看完整錯誤訊息"): st.code(traceback.format_exc())

            if st.session_state.get("sv_interact_mat") is not None:
                mean_abs_sv  = st.session_state["sv_interact_mat"]
                labels_sv    = st.session_state["sv_interact_labels"]
                top_feats_sv = st.session_state["sv_interact_feats"]
                n_sv = len(labels_sv)
                cell_sv    = max(0.55, min(1.1, 14.0/n_sv))
                fig_sv     = max(7, n_sv*cell_sv)
                annot_fs_sv = max(6, min(10, int(80/n_sv)))
                tick_fs_sv  = max(7, min(11, int(90/n_sv)))
                fig, ax = plt.subplots(figsize=(fig_sv+1.5, fig_sv))
                sv_df = pd.DataFrame(mean_abs_sv, index=labels_sv, columns=labels_sv)
                sns.heatmap(sv_df, ax=ax, cmap=cmap_sv, annot=annot_sv,
                            fmt=".4f" if annot_sv else "",
                            annot_kws={"size": annot_fs_sv}, linewidths=0.4, linecolor="white",
                            cbar_kws={"label":"Mean |SHAP Interaction Value|","shrink":0.75},
                            square=True)
                ax.set_title(f"Exact SHAP Interaction Matrix  (Top {n_sv} features)", fontsize=12, pad=14)
                ax.set_xlabel("Feature j", fontsize=10); ax.set_ylabel("Feature i", fontsize=10)
                ax.tick_params(axis="x", labelsize=tick_fs_sv, rotation=45)
                ax.tick_params(axis="y", labelsize=tick_fs_sv, rotation=0)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.markdown("#### 📊 精確版交互強度排行榜 Top 15")
                rows_sv = []
                for ii in range(n_sv):
                    for jj in range(ii+1, n_sv):
                        rows_sv.append({"代碼 i": labels_sv[ii], "代碼 j": labels_sv[jj],
                                        "原始名稱 i": top_feats_sv[ii], "原始名稱 j": top_feats_sv[jj],
                                        "Mean |SHAP Interaction|": round(float(mean_abs_sv[ii,jj]),5)})
                rank_sv = (pd.DataFrame(rows_sv)
                           .sort_values("Mean |SHAP Interaction|", ascending=False)
                           .reset_index(drop=True).head(15))
                st.dataframe(rank_sv.style.background_gradient(cmap="YlOrRd",
                             subset=["Mean |SHAP Interaction|"]),
                             use_container_width=True, hide_index=True)
                top1 = rank_sv.iloc[0]
                st.success(
                    f"🏆 最強交互：**{top1['代碼 i']}** ↔ **{top1['代碼 j']}**  "
                    f"（{top1['原始名稱 i'][:30]} ↔ {top1['原始名稱 j'][:30]}）\n\n"
                    f"在 Dependence Plot 中選擇 **{top1['代碼 i']}** 作為主特徵、"
                    f"取消勾選「自動尋找」並手動選 **{top1['代碼 j']}** 作為交互著色，即可看到最精確的交互效應圖。"
                )
                st.caption("此結果與 Dependence Plot `interaction_index='auto'` 使用相同底層計算（TreeExplainer），是三種方法中最精確的交互衡量。")


# ═══════════════════════════════════════════════════════════
#  PLS-VIP Tab
# ═══════════════════════════════════════════════════════════

def _render_pls_tab(fi_subtab, X_fi, y_fi, top_n_fi):
    with fi_subtab:
        pls_outer = st.tabs(["📐 VIP 分析", "📊 模型評估"])

        with pls_outer[0]:
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
                    X_arr  = StandardScaler().fit_transform(X_fi) if scale_pls else X_fi.values
                    y_arr  = y_fi.values.ravel()
                    comp_r = list(range(1, max_comp+1))
                    mse_l  = []
                    for nc in comp_r:
                        pls_cv = PLSRegression(n_components=nc)
                        y_cv   = cross_val_predict(pls_cv, X_arr, y_arr,
                                                    cv=min(cv_folds_pls, len(y_arr)))
                        mse_l.append(mean_squared_error(y_arr, y_cv))
                    st.session_state.update({"pls_mse": mse_l, "pls_range": comp_r})

            if st.session_state.get("pls_mse"):
                mse_l  = st.session_state["pls_mse"]
                comp_r = st.session_state["pls_range"]
                best_n = comp_r[int(np.argmin(mse_l))]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(comp_r, mse_l, marker="o", color="#2e86ab")
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

                    t = np.array(pls.x_scores_,   dtype=float)
                    w = np.array(pls.x_weights_,  dtype=float)
                    q = np.array(pls.y_loadings_, dtype=float).ravel()
                    p_feat, h = w.shape
                    s = np.array([float(t[:,j]@t[:,j]) * float(q[j])**2 for j in range(h)])
                    w_normed = (w / np.linalg.norm(w, axis=0))**2
                    vips = np.sqrt(p_feat * (w_normed @ s) / float(s.sum()))

                    vip_df = (pd.DataFrame({"Feature": X_fi.columns, "VIP": vips})
                              .sort_values("VIP", ascending=False).reset_index(drop=True))

                    coef_raw   = pls.coef_.ravel()
                    reg_coef_df = (pd.DataFrame({
                        "Feature":  X_fi.columns,
                        "PLS_Coef": coef_raw,
                        "Abs_Coef": np.abs(coef_raw),
                    }).sort_values("Abs_Coef", ascending=False).reset_index(drop=True))

                    x_load    = pd.DataFrame(pls.x_loadings_, index=X_fi.columns,
                                             columns=[f"PC{i+1}" for i in range(n_pls)])
                    x_weights = pd.DataFrame(pls.x_weights_,  index=X_fi.columns,
                                             columns=[f"PC{i+1}" for i in range(n_pls)])

                    y_pred_pls = pls.predict(X_arr).ravel()
                    r2_pls     = float(r2_score(y_arr, y_pred_pls))

                    # K-Fold Q² (integrated)
                    kf_pls    = KFold(n_splits=min(cv_folds_pls, len(y_arr)), shuffle=True, random_state=42)
                    q2_scores = []
                    cv_rmse_p = []
                    cv_mae_p  = []
                    for tr, te in kf_pls.split(X_arr):
                        pls2 = PLSRegression(n_components=n_pls)
                        pls2.fit(X_arr[tr], y_arr[tr])
                        yp   = pls2.predict(X_arr[te]).ravel()
                        q2_scores.append(float(r2_score(y_arr[te], yp)))
                        cv_rmse_p.append(float(np.sqrt(mean_squared_error(y_arr[te], yp))))
                        cv_mae_p.append(float(mean_absolute_error(y_arr[te], yp)))
                    q2_pls = float(np.mean(q2_scores))

                    y_var_per_comp = []
                    for nc in range(1, n_pls+1):
                        pt = PLSRegression(n_components=nc); pt.fit(X_arr, y_arr)
                        y_var_per_comp.append(r2_score(y_arr, pt.predict(X_arr).ravel()))
                    y_var_each = ([y_var_per_comp[0]] +
                                  [y_var_per_comp[i]-y_var_per_comp[i-1] for i in range(1, n_pls)])

                    st.session_state.update({
                        "pls_vip_df": vip_df, "pls_reg_coef_df": reg_coef_df,
                        "pls_x_loadings": x_load, "pls_x_weights": x_weights,
                        "pls_r2": r2_pls, "pls_q2": q2_pls,
                        "pls_y_var_each": y_var_each, "pls_model": pls,
                        "pls_n_comp": n_pls,
                        # eval cache
                        "_pls_y":         np.asarray(y_fi, dtype=float),
                        "_pls_y_pred":    y_pred_pls,
                        "_pls_cv_scores": {
                            "R² (Q²-fold)": np.array(q2_scores),
                            "RMSE": np.array(cv_rmse_p),
                            "MAE":  np.array(cv_mae_p),
                        },
                    })

                n, p_ = len(y_fi), X_fi.shape[1]
                adj   = _adj_r2(r2_pls, n, p_)
                st.success(
                    f"✅ 完成！R²={r2_pls:.3f}  Adj.R²={_fmt(adj)}  Q²={q2_pls:.3f}"
                )

            if st.session_state.get("pls_vip_df") is None:
                st.info("點擊「📐 計算 PLS」開始。")
                return

            vip_df      = st.session_state["pls_vip_df"]
            reg_coef_df = st.session_state["pls_reg_coef_df"]
            x_load      = st.session_state["pls_x_loadings"]
            x_weights   = st.session_state["pls_x_weights"]
            r2_pls      = st.session_state["pls_r2"]
            q2_pls      = st.session_state.get("pls_q2")
            y_var_each  = st.session_state.get("pls_y_var_each", [])
            n_pls       = st.session_state.get("pls_n_comp", 3)

            n, p_ = len(y_fi), X_fi.shape[1]
            adj   = _adj_r2(r2_pls, n, p_)
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("PLS R²（訓練集）", f"{r2_pls:.3f}")
            pm2.metric("Adj.R²",          _fmt(adj))
            pm3.metric("Q²（CV，不偏）",   f"{q2_pls:.3f}" if q2_pls is not None else "—",
                       help="Q² > 0.5 代表模型有預測力；Q² 接近 R² 代表無過擬合")
            pm4.metric("過擬合風險",
                       "低 ✅" if q2_pls and r2_pls-q2_pls<0.15 else "注意 ⚠️" if q2_pls else "—")

            if y_var_each:
                st.markdown("**各 Component 新增的 Y 解釋變異（%）**")
                fig_pls, ax_pls = plt.subplots(figsize=(7, 3))
                ax_pls.bar([f"PC{i+1}" for i in range(len(y_var_each))],
                           [v*100 for v in y_var_each], color="#2e86ab", alpha=0.8)
                for i, v in enumerate(y_var_each):
                    ax_pls.text(i, v*100+0.5, f"{v*100:.1f}%", ha="center", fontsize=9)
                ax_pls.set_ylabel("新增解釋 Y 變異 (%)"); ax_pls.set_title("PLS 各主成分對 Y 的邊際解釋力")
                ax_pls.grid(axis="y", alpha=0.3)
                plt.tight_layout(); st.pyplot(fig_pls); plt.close()

            pls_subtabs = st.tabs(["VIP Score","迴歸係數","X Loadings","X Weights"])
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
                fig, ax = plt.subplots(figsize=(max(6,n_pls*1.5), max(6,len(X_fi.columns)*0.35)))
                sns.heatmap(x_load, annot=True, fmt=".3f", cmap="RdBu_r",
                            center=0, linewidths=0.3, ax=ax, annot_kws={"size":7})
                ax.set_title("X Loadings (P) Heatmap")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with pls_subtabs[3]:
                st.markdown("**X Weights（W）** — 特徵對 PLS 潛在變數（latent score）的貢獻權重")
                st.dataframe(x_weights.style.background_gradient(cmap="RdBu_r"), width="stretch")
                fig, ax = plt.subplots(figsize=(max(6,n_pls*1.5), max(6,len(X_fi.columns)*0.35)))
                sns.heatmap(x_weights, annot=True, fmt=".3f", cmap="RdBu_r",
                            center=0, linewidths=0.3, ax=ax, annot_kws={"size":7})
                ax.set_title("X Weights (W) Heatmap")
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with pls_outer[1]:
            if st.session_state.get("pls_model") is None:
                st.info("請先在「📐 VIP 分析」分頁訓練 PLS 模型。")
            else:
                _render_eval_section(
                    "PLS Regression",
                    st.session_state["_pls_y"],
                    st.session_state["_pls_y_pred"],
                    cv_scores=st.session_state.get("_pls_cv_scores"),
                    n_features=X_fi.shape[1],
                )


# ═══════════════════════════════════════════════════════════
#  Correlation Heatmap + Interaction Ranking (unchanged)
# ═══════════════════════════════════════════════════════════

def _render_correlation_heatmap(X_fi, top_n_fi):
    with st.spinner("生成相關性矩陣中..."):
        st.markdown("#### Top 特徵相關性矩陣")
        st.caption("觀察 Top 特徵之間的聯動關係。")
        fi_df = st.session_state.get("fi_perm_df")
        if fi_df is None:
            st.warning("請先執行 RF 訓練。"); return
        top_feats  = [f for f in fi_df["Feature"].head(top_n_fi).tolist()
                      if f in X_fi.columns]
        if not top_feats:
            st.warning("目前特徵與上次 RF 訓練的特徵不符，請重新執行 RF 訓練。"); return
        corr       = X_fi[top_feats].corr()
        fig_width  = max(10, top_n_fi * 0.7)
        fig_height = max(8,  top_n_fi * 0.6)
        fig, ax    = plt.subplots(figsize=(fig_width, fig_height))
        annot_size = 8 if top_n_fi <= 10 else 6
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                    annot_kws={"size": annot_size}, cbar_kws={"shrink": 0.8})
        plt.xticks(rotation=45, ha="right", fontsize=9); plt.yticks(fontsize=9)
        ax.set_title(f"Top {top_n_fi} Features Correlation Heatmap", fontsize=14, pad=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()


def _render_interaction_ranking(X_fi, top_n_fi):
    st.markdown("#### 🔄 潛在交互作用排名 (Top Pairs)")
    st.caption("評分 = |相關係數| × (特徵 A 重要性 + 特徵 B 重要性)。得分越高，代表這兩個參數越值得去 Dependence Plot 觀察。")
    fi_df = st.session_state.get("fi_perm_df")
    if fi_df is None:
        st.warning("請先在「RF 重要性」分頁訓練模型以取得特徵權重。"); return
    top_feats      = fi_df.head(top_n_fi)
    feat_list      = [f for f in top_feats["Feature"].tolist() if f in X_fi.columns]
    if not feat_list:
        st.warning("目前特徵與上次 RF 訓練的特徵不符，請重新執行 RF 訓練。"); return
    importance_map = dict(zip(top_feats["Feature"], top_feats["Perm_Importance"]))
    corr_matrix    = X_fi[feat_list].corr().abs()
    pairs = []
    for i in range(len(feat_list)):
        for j in range(i+1, len(feat_list)):
            f1, f2  = feat_list[i], feat_list[j]
            c_val   = corr_matrix.loc[f1, f2]
            if np.isnan(c_val): c_val = 0
            score   = c_val * (importance_map[f1] + importance_map[f2])
            pairs.append({"特徵組合": f"{f1} ↔ {f2}", "綜合推薦得分": score, "相關性(Abs)": c_val})
    if not pairs:
        st.write("數據不足以計算排名。"); return
    rank_df = pd.DataFrame(pairs).sort_values("綜合推薦得分", ascending=False).head(10)
    st.table(rank_df.style.format({"綜合推薦得分": "{:.4f}", "相關性(Abs)": "{:.2f}"}))


# ═══════════════════════════════════════════════════════════
#  Mutual Information Tab (unchanged)
# ═══════════════════════════════════════════════════════════

def _render_mi_tab(X_fi, y_fi, top_n_fi):
    st.markdown("#### 🤝 Mutual Information（互資訊）")
    st.caption(
        "MI 衡量特徵與目標之間的**非線性**相依程度，不限於線性關係。"
        "單位為 bit，越大代表共享的資訊量越多。"
        "與 Pearson 相關係數互補：相關係數接近 0 但 MI 高，代表存在非線性關係。"
    )
    with st.expander("⚙️ MI 設定", expanded=False):
        mi1, mi2 = st.columns(2)
        n_neighbors_mi = mi1.slider("n_neighbors", 3, 15, 5, key="mi_neighbors",
                                     help="KNN 估計 MI 的鄰近數，越大越平滑")
        random_mi = mi2.number_input("random_state", value=42, step=1, key="mi_random")

    if st.button("🤝 計算 Mutual Information", key="run_mi", type="primary"):
        from sklearn.feature_selection import mutual_info_regression
        with st.spinner("計算 MI..."):
            mi_vals = mutual_info_regression(X_fi, y_fi,
                                              n_neighbors=n_neighbors_mi,
                                              random_state=int(random_mi))
            mi_df = pd.DataFrame({"Feature": X_fi.columns, "MI": mi_vals})\
                      .sort_values("MI", ascending=False).reset_index(drop=True)
            pearson_vals          = X_fi.corrwith(y_fi).reindex(X_fi.columns).values
            mi_df["Pearson_r"]    = pearson_vals
            mi_df["Pearson_abs"]  = np.abs(pearson_vals)
            mi_df["MI_norm"]      = mi_df["MI"] / (mi_df["MI"].max() + 1e-9)
            mi_df["Pearson_norm"] = mi_df["Pearson_abs"] / (mi_df["Pearson_abs"].max() + 1e-9)
            mi_df["NonLinear_signal"] = mi_df["MI_norm"] - mi_df["Pearson_norm"]
            st.session_state["mi_df"] = mi_df
        st.success(f"✅ 完成！Top 特徵：{mi_df.iloc[0]['Feature']}")

    if st.session_state.get("mi_df") is None: return
    mi_df  = st.session_state["mi_df"]
    top_mi = mi_df.head(top_n_fi)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_n_fi*0.42)))
    axes[0].barh(top_mi["Feature"], top_mi["MI"], color="#7209b7", alpha=0.85)
    axes[0].set_xlabel("Mutual Information (bits)")
    axes[0].set_title(f"Mutual Information（Top {top_n_fi}）")
    axes[0].invert_yaxis(); axes[0].grid(axis="x", linestyle="--", alpha=0.4)
    merge = top_mi.sort_values("MI_norm", ascending=True)
    y_pos = range(len(merge))
    axes[1].barh([p3-0.18 for p3 in y_pos], merge["MI_norm"],     height=0.35,
                 color="#7209b7", alpha=0.85, label="MI (norm)")
    axes[1].barh([p3+0.18 for p3 in y_pos], merge["Pearson_norm"], height=0.35,
                 color="#2e86ab", alpha=0.85, label="|Pearson r| (norm)")
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(merge["Feature"].tolist(), fontsize=8)
    axes[1].set_xlabel("正規化分數（0-1）"); axes[1].set_title("MI vs |Pearson r| 比較")
    axes[1].legend(fontsize=9); axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### 🔍 非線性信號偵測")
    st.caption("**MI_norm - |Pearson|_norm > 0** 的特徵，代表線性相關低但資訊量高，可能存在非線性關係，值得在 SHAP Dependence Plot 進一步觀察。")
    nonlinear_df = mi_df[mi_df["NonLinear_signal"] > 0.05]\
                     .sort_values("NonLinear_signal", ascending=False).head(top_n_fi)
    if nonlinear_df.empty:
        st.info("未偵測到顯著非線性信號（所有特徵的 Pearson 已足夠解釋 MI）。")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, max(4, len(nonlinear_df)*0.4)))
        ax2.barh(nonlinear_df["Feature"], nonlinear_df["NonLinear_signal"],
                 color="#e84855", alpha=0.85)
        ax2.axvline(0, color="black", lw=1)
        ax2.set_xlabel("MI_norm − |Pearson|_norm（正值=非線性信號）")
        ax2.set_title("非線性信號強度排名"); ax2.invert_yaxis(); ax2.grid(axis="x", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.dataframe(
        mi_df.head(top_n_fi).style
            .background_gradient(cmap="Purples", subset=["MI"])
            .background_gradient(cmap="RdBu_r",  subset=["Pearson_r"])
            .format({"MI":"{:.4f}","Pearson_r":"{:.3f}",
                     "MI_norm":"{:.3f}","Pearson_norm":"{:.3f}","NonLinear_signal":"{:.3f}"}),
        use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
#  綜合排名 Tab  ← 新增 RF_MDI + 方法勾選
# ═══════════════════════════════════════════════════════════

def _render_consensus_tab(X_fi, top_n_fi):
    st.markdown("#### 🏆 綜合重要性排名")
    st.caption(
        "整合所有已執行的方法（RF_Perm、RF_MDI、Lasso、SHAP、PLS_VIP、MI），"
        "以**名次平均**計算綜合排名。方法越多、分析越全面。"
    )

    # ── collect available methods ─────────────────────────
    available: dict = {}

    perm_df = st.session_state.get("fi_perm_df")
    if perm_df is not None and "Perm_Importance" in perm_df.columns:
        available["RF_Perm"] = (perm_df.set_index("Feature")["Perm_Importance"]
                                .reindex(X_fi.columns).fillna(0))

    mdi_df = st.session_state.get("fi_mdi_df")
    if mdi_df is not None and "MDI_Importance" in mdi_df.columns:
        available["RF_MDI"] = (mdi_df.set_index("Feature")["MDI_Importance"]
                               .reindex(X_fi.columns).fillna(0))

    lasso_df = st.session_state.get("lasso_coef_df")
    if lasso_df is not None and "Abs_Coef" in lasso_df.columns:
        available["Lasso"] = (lasso_df.set_index("Feature")["Abs_Coef"]
                              .reindex(X_fi.columns).fillna(0))

    shap_vals = st.session_state.get("shap_vals")
    if shap_vals is not None:
        available["SHAP"] = pd.Series(np.abs(np.array(shap_vals)).mean(axis=0),
                                       index=X_fi.columns)

    vip_df = st.session_state.get("pls_vip_df")
    if vip_df is not None and "VIP" in vip_df.columns:
        available["PLS_VIP"] = (vip_df.set_index("Feature")["VIP"]
                                .reindex(X_fi.columns).fillna(0))

    mi_df = st.session_state.get("mi_df")
    if mi_df is not None and "MI" in mi_df.columns:
        available["MI"] = (mi_df.set_index("Feature")["MI"]
                           .reindex(X_fi.columns).fillna(0))

    if not available:
        st.warning("尚未執行任何方法，請至少完成一種分析後再來看綜合排名。")
        return

    # ── method checkboxes ─────────────────────────────────
    st.markdown("**✅ 選擇納入綜合排名的方法**（預設全選）")
    chk_cols = st.columns(min(len(available), 6))
    methods: dict = {}
    for col, (name, series) in zip(chk_cols, available.items()):
        if col.checkbox(name, value=True, key=f"consensus_chk_{name}"):
            methods[name] = series

    if not methods:
        st.warning("請至少勾選一種方法。"); return

    st.info(f"目前已納入 **{len(methods)}** 種方法：{', '.join(methods.keys())}")

    # ── rank-mean aggregation ──────────────────────────────
    rank_df  = pd.DataFrame(index=X_fi.columns)
    score_df = pd.DataFrame(index=X_fi.columns)
    for method, series in methods.items():
        mn, mx = series.min(), series.max()
        score_df[method] = ((series - mn) / (mx - mn + 1e-9)).clip(0, 1)
        rank_df[method]  = series.rank(ascending=False, method="average")

    rank_df["平均名次"]  = rank_df.mean(axis=1)
    score_df["綜合分數"] = score_df.mean(axis=1)
    consensus = (score_df[["綜合分數"]]
                 .join(rank_df[["平均名次"]])
                 .join(score_df[list(methods.keys())]))
    consensus = consensus.sort_values("綜合分數", ascending=False).reset_index()
    consensus.rename(columns={"index": "Feature"}, inplace=True)

    top_cons = consensus.head(top_n_fi)

    # ── bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(5, len(top_cons)*0.45)))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_cons)))[::-1]
    ax.barh(top_cons["Feature"], top_cons["綜合分數"], color=colors, alpha=0.9)
    ax.set_xlabel("綜合分數（各方法正規化後平均，0-1）")
    ax.set_title(f"🏆 綜合重要性 Top {len(top_cons)}（{', '.join(methods.keys())}）")
    ax.invert_yaxis(); ax.grid(axis="x", linestyle="--", alpha=0.4)
    for i, (_, row) in enumerate(top_cons.iterrows()):
        ax.text(row["綜合分數"]+0.005, i, f"{row['綜合分數']:.3f}", va="center", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── heatmap ────────────────────────────────────────────
    st.markdown("#### 各方法正規化分數 Heatmap")
    heat_data = top_cons.set_index("Feature")[list(methods.keys())]
    fig2, ax2 = plt.subplots(figsize=(max(8, len(methods)*1.5), max(5, len(top_cons)*0.4)))
    im = ax2.imshow(heat_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, shrink=0.6, label="正規化分數")
    ax2.set_xticks(range(len(methods))); ax2.set_yticks(range(len(heat_data)))
    ax2.set_xticklabels(list(methods.keys()), fontsize=10)
    ax2.set_yticklabels([f[:45] for f in heat_data.index], fontsize=8)
    for i in range(len(heat_data)):
        for j in range(len(methods)):
            val = heat_data.values[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if val < 0.4 else "black")
    ax2.set_title("各方法正規化分數（綠=高重要性，紅=低重要性）")
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.dataframe(
        consensus.head(top_n_fi).style
            .background_gradient(cmap="RdYlGn",   subset=["綜合分數"])
            .background_gradient(cmap="RdYlGn_r", subset=["平均名次"])
            .format({"綜合分數": "{:.4f}", "平均名次": "{:.1f}",
                     **{m: "{:.3f}" for m in methods.keys()}}),
        use_container_width=True, hide_index=True)

    # ── inter-method agreement ─────────────────────────────
    if len(methods) >= 2:
        st.markdown("#### 📊 方法間一致性")
        st.caption("各方法名次之間的 Spearman 相關，相關越高代表方法結論越一致。")
        from scipy.stats import spearmanr
        mn_list = list(methods.keys())
        n_m     = len(mn_list)
        corr_mat = np.eye(n_m)
        for i in range(n_m):
            for j in range(i+1, n_m):
                r, _ = spearmanr(rank_df[mn_list[i]], rank_df[mn_list[j]])
                corr_mat[i,j] = corr_mat[j,i] = r
        fig3, ax3 = plt.subplots(figsize=(max(5, n_m*1.2), max(4, n_m)))
        im3 = ax3.imshow(corr_mat, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax3.set_xticks(range(n_m)); ax3.set_yticks(range(n_m))
        ax3.set_xticklabels(mn_list, fontsize=10); ax3.set_yticklabels(mn_list, fontsize=10)
        for i in range(n_m):
            for j in range(n_m):
                ax3.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                         fontsize=10, color="white" if abs(corr_mat[i,j])<0.4 else "black")
        ax3.set_title("方法間 Spearman 名次相關（1=完全一致，-1=完全相反）")
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ═══════════════════════════════════════════════════════════
#  Main render
# ═══════════════════════════════════════════════════════════

def render(selected_process_df):
    st.header("特徵重要性分析")
    _cd     = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。"); return

    numeric_cols  = work_df.select_dtypes(include=["number"]).columns.tolist()
    stored_target = st.session_state.get("target_col")
    default_fi    = numeric_cols.index(stored_target) if stored_target in numeric_cols else 0
    target_col_fi = st.selectbox("目標欄位（Y）", numeric_cols, index=default_fi, key="fi_target")
    top_n_fi      = st.slider("顯示前 N 個特徵", 5, 30, 15, key="fi_topn")

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

    fi_subtabs = st.tabs([
        "🌲 RF 重要性", "🔪 Lasso 重要性", "🔮 SHAP 分析",
        "📐 PLS-VIP", "🔄 交互作用探索",
        "🤝 Mutual Information", "🏆 綜合排名",
    ])
    _render_rf_tab(fi_subtabs[0], X_fi, y_fi, top_n_fi)
    _render_lasso_tab(fi_subtabs[1], X_fi, y_fi, top_n_fi)
    _render_shap_tab(fi_subtabs[2], X_fi, y_fi, top_n_fi)
    _render_pls_tab(fi_subtabs[3], X_fi, y_fi, top_n_fi)

    with fi_subtabs[4]:
        _render_correlation_heatmap(X_fi, top_n_fi)
        st.divider()
        _render_interaction_ranking(X_fi, top_n_fi)

    with fi_subtabs[5]:
        _render_mi_tab(X_fi, y_fi, top_n_fi)

    with fi_subtabs[6]:
        _render_consensus_tab(X_fi, top_n_fi)
