"""
機器學習模型工具：Random Forest、PLS-VIP、相關性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error


def analyze_correlation(
    df: pd.DataFrame,
    target_col: str,
    method: str = "pearson",
    top_n: int = 10,
) -> tuple | None:
    """
    計算各特徵與目標欄位的相關係數，回傳 (圖、排行 DataFrame) 或 None。
    """
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if target_col not in numeric_df.columns:
        return None

    corr_matrix = numeric_df.corr(method=method)
    corr_series = corr_matrix[target_col].drop(target_col)
    corr_rank = pd.DataFrame({
        "Feature": corr_series.index,
        "Correlation": corr_series.values,
        "Abs_Correlation": corr_series.abs().values,
    }).sort_values(by="Abs_Correlation", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_corr = corr_rank.head(top_n)
    sns.barplot(data=top_corr, x="Correlation", y="Feature", palette="vlag", ax=ax)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_title(f"Top {top_n} Features Correlated with\n{target_col}", fontsize=12)
    ax.set_xlabel(f"{method.capitalize()} Correlation Coefficient")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return fig, corr_rank[["Feature", "Correlation"]].reset_index(drop=True)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 5,
    min_samples_leaf: int = 4,
    n_repeats: int = 15,
) -> tuple[pd.DataFrame, RandomForestRegressor]:
    """訓練 RF 並計算 Permutation Importance。"""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    perm = permutation_importance(rf, X_train, y_train, n_repeats=n_repeats, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Perm_Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)
    return perm_df, rf


def compute_pls_vip(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 3,
) -> pd.DataFrame:
    """
    訓練 PLS 並以 VIP 公式計算各特徵重要性。
    使用純 numpy 實作，相容各版本 sklearn。
    """
    X_arr = X.values
    y_arr = np.array(y).ravel()

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_arr, y_arr)

    t = np.array(pls.x_scores_, dtype=float)    # (n, h)
    w = np.array(pls.x_weights_, dtype=float)    # (p, h)
    q = np.array(pls.y_loadings_, dtype=float).ravel()  # (h,)
    p_feat, h = w.shape

    s = np.array([
        float(t[:, j] @ t[:, j]) * float(q[j]) ** 2
        for j in range(h)
    ], dtype=float)
    total_s = float(s.sum())
    w_col_norms = np.linalg.norm(w, axis=0)
    w_normed = (w / w_col_norms) ** 2
    vips = np.sqrt(p_feat * (w_normed @ s) / total_s)

    return pd.DataFrame({
        "Feature": X.columns,
        "VIP": vips,
    }).sort_values("VIP", ascending=False).reset_index(drop=True)


def compute_pls_cv_mse(
    X: pd.DataFrame,
    y: pd.Series,
    max_comp: int,
    cv_folds: int = 5,
) -> tuple[list[int], list[float]]:
    """計算各主成分數的交叉驗證 MSE，回傳 (component_range, mse_list)。"""
    X_arr = X.values
    y_arr = np.array(y).ravel()
    comp_range = list(range(1, max_comp + 1))
    mse_list = []
    for n_c in comp_range:
        pls_cv = PLSRegression(n_components=n_c)
        y_cv = cross_val_predict(pls_cv, X_arr, y_arr, cv=min(cv_folds, len(y_arr)))
        mse_list.append(mean_squared_error(y_arr, y_cv))
    return comp_range, mse_list
