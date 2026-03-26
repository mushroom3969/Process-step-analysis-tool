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
      B. 配對差值（Max-Min、After-Before、End-Start）
      C. 數字編號欄位取平均後合併
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

    def _clean_col_name(name: str) -> str:
        return re.sub(r"\s?\(.*\)$", "", name).strip()

    # ── Rule B: 配對差值 ──────────────────────────────────────
    pairs = [
        (["Maximum", "Maximun"], ["Minimum", "Minimun"], "Diff_MaxMin"),
        (["After"], ["Before"], "Diff_AfterBefore"),
        (["End"], ["Start"], "Diff_EndStart"),
    ]
    current_cols = new_df.columns.tolist()
    for high_keys, low_keys, suffix in pairs:
        for k_high in high_keys:
            for k_low in low_keys:
                high_cols = [c for c in current_cols if k_high in c]
                for c_h in high_cols:
                    base_h = _clean_col_name(c_h).replace(k_high, "")
                    for c_l in current_cols:
                        if k_low in c_l:
                            base_l = _clean_col_name(c_l).replace(k_low, "")
                            if base_h == base_l and c_h != c_l:
                                new_col = f"{base_h.strip('_')}_{suffix}"
                                if new_col not in new_df.columns:
                                    new_df[new_col] = new_df[c_h] - new_df[c_l]

    # ── Rule C: 數字編號欄位 → 取平均 ─────────────────────────
    pattern = r"^(.*)_(\d+)\s?(\(.*\))$"
    group_dict: dict[str, list] = {}
    for c in new_df.columns:
        if c in whitelist:
            continue
        m = re.match(pattern, c)
        if m:
            base_name, _, unit = m.groups()
            key = f"{base_name.strip('_')} {unit}"
            group_dict.setdefault(key, []).append(c)

    for key, grouped_cols in group_dict.items():
        if len(grouped_cols) > 1:
            for c in grouped_cols:
                drop_log.append({"Column": c, "Reason": f"Averaged into: {key}"})
            new_df[key] = new_df[grouped_cols].mean(axis=1)
            new_df = new_df.drop(columns=grouped_cols)

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
        # 資料點太少 → 直接保留，不做統計篩選
        if len(series) < 5:
            keep_cols.append(col)
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

        # ACF 計算可能因資料不足或常數而回傳 NaN → 視為「無法判斷」，不剔除
        try:
            acf_1 = series.autocorr(lag=1)
        except Exception:
            acf_1 = float("nan")

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
    # 保持原始欄位順序
    original_order = [c for c in df.columns if c in keep_cols + non_numeric]
    return df[original_order], dropped_info
