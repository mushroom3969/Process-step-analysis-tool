"""
特徵工程與統計篩選工具函式
"""

import re
import numpy as np
import pandas as pd


def clean_process_features_with_log(
    df: pd.DataFrame,
    id_col: str = "BatchID",
    protected_cols: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    自動清理製程特徵欄位：
      A. 關鍵字過濾（Verification Result / No (na)）
    注意：Max/Min、Before/After、編號欄等配對合併已移至 Step 2.5（VIF 共線性診斷後決定）。
    回傳 (清理後 DataFrame, 刪除記錄 DataFrame)
    """
    if protected_cols is None:
        protected_cols = []
    whitelist = set([id_col] + protected_cols)

    df = df.loc[:, ~df.columns.duplicated()]
    new_df = df.copy().reset_index(drop=True)
    drop_log: list[dict] = []

    # ── Rule A: 關鍵字過濾 ────────────────────────────────────
    target_keywords = ["Verification Result", "No (na)"]
    to_drop_kw = [
        c for c in new_df.columns
        if (
            any(kw in c for kw in target_keywords)
            and not c.strip().lower().endswith("(times)")
        )
        and c not in whitelist
    ]
    for c in to_drop_kw:
        drop_log.append({"Column": c, "Reason": "Keyword Filter"})
    new_df = new_df.drop(columns=to_drop_kw)

    return new_df, pd.DataFrame(drop_log)


def filter_columns_by_stats(
    df: pd.DataFrame,
    batch_col: str = "BatchID",
    cv_threshold: float = 0.01,
    jump_ratio_threshold: float = 0.3,
    acf_threshold: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """
    統計篩選：移除低資訊量欄位（低 CV、高 Jump Ratio、低 ACF）。
    回傳 (篩選後 DataFrame, 被剔除欄位與原因字典)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols: list[str] = []
    dropped_info: dict[str, str] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        std_val = series.std()
        if std_val == 0 or series.nunique() <= 1:
            dropped_info[col] = "Constant/Zero Variance"
            continue

        mean_val = series.mean()
        cv = (std_val / abs(mean_val)) if mean_val != 0 else float("inf")
        data_range = series.max() - series.min()
        avg_jump = series.diff().abs().mean()
        jump_ratio = (avg_jump / data_range) if data_range != 0 else 0
        acf_1 = series.autocorr(lag=1)

        reasons = []
        if cv < cv_threshold:
            reasons.append(f"Low CV({cv:.4f})")
        if jump_ratio > jump_ratio_threshold:
            reasons.append(f"High Jump({jump_ratio:.2f})")
        if not np.isnan(acf_1) and acf_1 < acf_threshold:
            reasons.append(f"Low ACF({acf_1:.2f})")

        if reasons:
            dropped_info[col] = " & ".join(reasons)
        else:
            keep_cols.append(col)

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return df[keep_cols + non_numeric], dropped_info
