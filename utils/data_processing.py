"""
資料處理工具函式
"""

import re
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def process_step_count(df: pd.DataFrame) -> pd.DataFrame:
    """統計各製程步驟的參數欄位數量。"""
    process_cols = [col for col in df.columns if ":" in col]
    steps = [col.split(":")[0] for col in process_cols]
    step_counts = pd.Series(steps).value_counts().reset_index()
    step_counts.columns = ["Process Step", "Parameter Count"]
    return step_counts


def split_process_df(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """將寬表依製程步驟前綴拆分成多個子 DataFrame。"""
    common_cols = list(dict.fromkeys(col for col in df.columns if ":" not in col))
    process_names = {col.split(":")[0] for col in df.columns if ":" in col}
    split_dfs = {}
    for process in process_names:
        process_specific_cols = [col for col in df.columns if col.startswith(f"{process}:")]
        all_cols = common_cols + process_specific_cols
        process_df = df.loc[:, ~df.columns.duplicated()][all_cols].copy()
        process_df.columns = [
            col.split(":")[-1] if ":" in col else col
            for col in process_df.columns
        ]
        process_df = process_df.dropna(axis=1, how="all")
        split_dfs[process] = process_df
    return split_dfs


def filt_specific_name(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """依關鍵字（不區分大小寫）篩選欄位。"""
    selected_cols = df.columns[df.columns.str.contains(query, case=False)]
    return df[selected_cols]


def extract_batch_logic(s) -> int:
    """從批次 ID 字串末尾提取 4 位數字。"""
    match = re.search(r"(\d{4})$", str(s))
    return int(match.group(1)) if match else 0


def extract_number(s) -> int:
    """從字串提取第一個數字。"""
    match = re.search(r"\d+", str(s))
    return int(match.group()) if match else 0


def smooth_process_data(
    df: pd.DataFrame,
    target_cols: list[str],
    id_cols: list[str] = None,
    method: str = "loess",
    frac: float = 0.3,
    span: int = 10,
) -> pd.DataFrame:
    """對指定數值欄位套用 LOESS 或 EWMA 平滑。"""
    if id_cols is None:
        id_cols = ["BatchID"]
    existing_ids = [c for c in id_cols if c in df.columns]
    smoothed_df = df[existing_ids].copy()
    x = np.arange(len(df))

    for col in target_cols:
        if col not in df.columns:
            continue
        y = df[col].values
        mask = ~np.isnan(y)

        if method.lower() == "loess":
            if np.sum(mask) > 10:
                res = lowess(y[mask], x[mask], frac=frac)
                res_y = np.full(len(y), np.nan)
                res_y[mask] = res[:, 1]
                smoothed_df[col] = (
                    pd.Series(res_y, index=df.index).interpolate(limit_direction="both")
                )
            else:
                smoothed_df[col] = y
        elif method.lower() == "ewma":
            smoothed_df[col] = df[col].ewm(span=span, adjust=False).mean()

    return smoothed_df


def missing_col(df: pd.DataFrame) -> pd.DataFrame:
    """統計各欄位的缺失數量與比率。"""
    na_counts = df.isnull().sum()
    total = len(df)
    na_ratio = (na_counts / total) * 100
    mask = na_counts > 0
    missing_summary = pd.concat(
        [na_counts[mask].sort_values(ascending=False),
         na_ratio[mask].sort_values(ascending=False)],
        axis=1,
    )
    missing_summary.columns = ["Missing Count", "Missing Ratio (%)"]
    return missing_summary
