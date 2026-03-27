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

    # ── Rule B: Min/Max → mean + range；Before/After、Start/End → diff ────
    MIN_ALIASES = {"min", "minimum", "minimun", "low", "lower", "lo"}
    MAX_ALIASES = {"max", "maximum", "maximun", "high", "upper", "hi"}
    BEF_ALIASES = {"before", "bef", "bf", "pre", "initial", "init", "start"}
    AFT_ALIASES = {"after", "aft", "af", "post", "final", "fin", "end"}

    def _find_kw(name: str, aliases: set):
        n = name.lower()
        for kw in sorted(aliases, key=len, reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
            if re.search(pattern, n):
                return kw
        return None

    def _base(col: str, kw: str) -> str:
        n = col.lower()
        pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
        return re.sub(pattern, "", n).strip("_- ")

    current_cols = list(new_df.columns)
    min_map, max_map, bef_map, aft_map = {}, {}, {}, {}

    for col in current_cols:
        if col in whitelist:
            continue
        kw = _find_kw(col, MIN_ALIASES)
        if kw:
            b = _base(col, kw)
            min_map.setdefault(b, col)
            continue
        kw = _find_kw(col, MAX_ALIASES)
        if kw:
            b = _base(col, kw)
            max_map.setdefault(b, col)
            continue
        kw = _find_kw(col, BEF_ALIASES)
        if kw:
            b = _base(col, kw)
            bef_map.setdefault(b, col)
            continue
        kw = _find_kw(col, AFT_ALIASES)
        if kw:
            b = _base(col, kw)
            aft_map.setdefault(b, col)

    # Min/Max → mean + range
    for b in sorted(set(min_map) & set(max_map)):
        c_min, c_max = min_map[b], max_map[b]
        label = b.strip("_- ") or f"{c_min}_{c_max}"
        mean_col  = f"{label}_mean"
        range_col = f"{label}_range"
        new_df[mean_col]  = (new_df[c_min] + new_df[c_max]) / 2
        new_df[range_col] = new_df[c_max] - new_df[c_min]
        drop_log.append({"Column": c_min, "Reason": f"Merged into {mean_col}, {range_col}"})
        drop_log.append({"Column": c_max, "Reason": f"Merged into {mean_col}, {range_col}"})
        new_df = new_df.drop(columns=[c_min, c_max])

    # Before/After、Start/End → diff (after/end − before/start)
    for b in sorted(set(bef_map) & set(aft_map)):
        c_bef, c_aft = bef_map[b], aft_map[b]
        label = b.strip("_- ") or f"{c_bef}_{c_aft}"
        diff_col = f"{label}_diff"
        new_df[diff_col] = new_df[c_aft] - new_df[c_bef]
        drop_log.append({"Column": c_bef, "Reason": f"Merged into {diff_col}"})
        drop_log.append({"Column": c_aft, "Reason": f"Merged into {diff_col}"})
        new_df = new_df.drop(columns=[c_bef, c_aft])

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
