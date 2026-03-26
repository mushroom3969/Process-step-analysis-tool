import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .data_processing import (
    process_step_count, split_process_df, filt_specific_name,
    extract_batch_logic, extract_number, smooth_process_data, missing_col,
)
from .feature_engineering import (
    clean_process_features_with_log, filter_columns_by_stats,
)
from .models import (
    analyze_correlation, train_random_forest, compute_pls_vip, compute_pls_cv_mse,
)
from .plotting import (
    plot_indexed_lineplots, plot_clean_lineplots,
    plot_correlation_bar, plot_missing_heatmap, plot_yield_tracking,
)
from .pubmed_gemini import (
    pubmed_search, pubmed_fetch_abstracts, build_search_queries_with_gemini,
    call_gemini, search_pubmed_for_features, build_literature_prompt,
)

__all__ = [
    "process_step_count","split_process_df","filt_specific_name",
    "extract_batch_logic","extract_number","smooth_process_data","missing_col",
    "clean_process_features_with_log","filter_columns_by_stats",
    "analyze_correlation","train_random_forest","compute_pls_vip","compute_pls_cv_mse",
    "plot_indexed_lineplots","plot_clean_lineplots",
    "plot_correlation_bar","plot_missing_heatmap","plot_yield_tracking",
    "pubmed_search","pubmed_fetch_abstracts","build_search_queries_with_gemini",
    "call_gemini","search_pubmed_for_features","build_literature_prompt",
]
