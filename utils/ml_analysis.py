"""
Machine learning analysis utilities: correlation, PCA, RF, SHAP, PLS-VIP.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import f as f_dist


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_correlation(df, target_col, method="pearson"):
    """
    Compute correlation between all numeric columns and target_col.
    Returns sorted DataFrame with Feature, Correlation, Abs_Correlation.
    IMPROVEMENT: returns None cleanly when target not found.
    """
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if target_col not in numeric_df.columns:
        return None
    corr_series = numeric_df.corr(method=method)[target_col].drop(target_col)
    return pd.DataFrame({
        "Feature": corr_series.index,
        "Correlation": corr_series.values,
        "Abs_Correlation": corr_series.abs().values,
    }).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)


# ── PCA helpers ───────────────────────────────────────────────────────────────

def compute_ht2_thresholds(n_obs, n_pc):
    """
    Compute Hotelling T² control limits for 68%, 95%, 99% confidence.
    Uses exact F-distribution critical values.
    """
    def threshold(alpha):
        f_crit = f_dist.ppf(1 - alpha, n_pc, n_obs - n_pc)
        return (n_pc * (n_obs - 1) * (n_obs + 1)) / (n_obs * (n_obs - n_pc)) * f_crit
    return threshold(0.32), threshold(0.05), threshold(0.01)


def compute_ht2_per_sample(scores, ev):
    """Compute Hotelling T² value for each sample."""
    return np.sum((scores ** 2) / ev, axis=1)


def compute_total_contribution(scores, loadings, ev, x_scaled, sample_i):
    """
    Compute feature contribution to Hotelling T² for a single sample
    summed across all PCs.
    Formula: contribution[feat] = sum_pc( (score/eigenval) * loading * x_scaled )
    """
    n_pc = scores.shape[1]
    contributions = np.zeros(loadings.shape[1])
    for a in range(n_pc):
        contributions += (scores[sample_i, a] / ev[a]) * loadings[a, :] * x_scaled[sample_i, :]
    return contributions


def compute_pc_contribution(scores, loadings, ev, x_scaled, sample_i, pc_idx):
    """Compute feature contribution to a specific PC for a single sample."""
    return (scores[sample_i, pc_idx] / ev[pc_idx]) * loadings[pc_idx, :] * x_scaled[sample_i, :]


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_rf_and_importance(X_train, y_train,
                             n_estimators=200, max_depth=5,
                             min_samples_leaf=4, n_repeats=15):
    """
    Train RandomForestRegressor and compute permutation importance.
    IMPROVEMENT: max_features changed from hardcoded 5 to 'sqrt' (scales with features).
    Returns (perm_importance_df, rf_model, r2_train).
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    perm = permutation_importance(rf, X_train, y_train,
                                  n_repeats=n_repeats, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Perm_Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)

    r2 = r2_score(y_train, rf.predict(X_train))
    return perm_df, rf, r2


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def make_short_feature_map(X):
    """
    Map original (long) feature names to short codes F00, F01, ...
    Returns (X_short, short_map, reverse_map).
    """
    short_map = {c: f"F{i:02d}" for i, c in enumerate(X.columns)}
    reverse_map = {v: k for k, v in short_map.items()}
    return X.rename(columns=short_map), short_map, reverse_map


def get_shap_base_value(explainer):
    """
    Safely extract scalar base value from a SHAP TreeExplainer.
    BUGFIX: expected_value may be an array for multi-output models.
    """
    ev = explainer.expected_value
    if hasattr(ev, "__len__"):
        return float(ev[0])
    return float(ev)


def restore_shap_yticklabels(ax, reverse_map, wrap_width=40):
    """Replace short F-codes on y-axis with wrapped original feature names."""
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), wrap_width))
         for t in ax.get_yticklabels()],
        fontsize=8,
    )


# ── PLS-VIP ───────────────────────────────────────────────────────────────────

def compute_pls_vip(X_train, y_train, n_components=3):
    """
    Train PLSRegression and compute VIP scores.

    BUGFIX (original): used s.T @ weight which returned an array not scalar,
    causing 'setting array element with sequence' ValueError.
    FIX: fully vectorised with (w / w_col_norms)^2 @ s pattern.

    Returns (vip_df, pls_model).
    """
    X_arr = np.array(X_train, dtype=float)
    y_arr = np.array(y_train, dtype=float).ravel()

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_arr, y_arr)

    t = np.array(pls.x_scores_,   dtype=float)   # (n, h)
    w = np.array(pls.x_weights_,  dtype=float)   # (p, h)
    q = np.array(pls.y_loadings_, dtype=float).ravel()  # (h,)
    p_feat, h = w.shape

    # s[j]: explained variance of component j weighted by y-loading²
    s = np.array([float(t[:, j] @ t[:, j]) * float(q[j]) ** 2 for j in range(h)], dtype=float)
    total_s = float(s.sum())

    w_col_norms = np.linalg.norm(w, axis=0)   # (h,)
    w_normed = (w / w_col_norms) ** 2          # (p, h)
    vips = np.sqrt(p_feat * (w_normed @ s) / total_s)  # (p,)

    vip_df = pd.DataFrame({
        "Feature": X_train.columns,
        "VIP": vips,
    }).sort_values("VIP", ascending=False).reset_index(drop=True)

    return vip_df, pls


def compute_pls_cv_mse(X_train, y_train, max_components=10, cv_folds=5):
    """
    Compute cross-validation MSE for PLS with 1..max_components.
    Returns list of MSE values indexed by n_components.
    """
    X_arr = np.array(X_train, dtype=float)
    y_arr = np.array(y_train, dtype=float).ravel()
    mse_list = []
    for n_c in range(1, max_components + 1):
        pls_cv = PLSRegression(n_components=n_c)
        y_cv = cross_val_predict(pls_cv, X_arr, y_arr, cv=min(cv_folds, len(y_arr)))
        mse_list.append(mean_squared_error(y_arr, y_cv))
    return mse_list


# ── PubMed + Gemini ───────────────────────────────────────────────────────────

def pubmed_search(query, max_results=5):
    """Search PubMed via E-utilities and return list of PMIDs."""
    import urllib.parse as up
    import urllib.request as ur
    import json as js
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
           f"?db=pubmed&retmode=json&retmax={max_results}&term={up.quote(query)}")
    try:
        with ur.urlopen(url, timeout=10) as r:
            return js.loads(r.read())["esearchresult"].get("idlist", [])
    except Exception:
        return []


def pubmed_fetch_abstracts(pmids):
    """
    Fetch title, abstract, year, journal for a list of PMIDs via E-utilities XML.
    Returns list of dicts with keys: pmid, title, abstract, year, journal, url.
    """
    import re as _re
    import urllib.request as ur
    if not pmids:
        return []
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           f"?db=pubmed&id={','.join(pmids)}&rettype=abstract&retmode=xml")
    try:
        with ur.urlopen(url, timeout=15) as r:
            xml = r.read().decode("utf-8")
    except Exception:
        return []

    def clean_xml(s):
        return _re.sub(r"<[^>]+>", "", s).strip() if s else ""

    articles = []
    for block in _re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, _re.DOTALL):
        pmid_m    = _re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        title_m   = _re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, _re.DOTALL)
        abs_m     = _re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, _re.DOTALL)
        year_m    = _re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, _re.DOTALL)
        journal_m = _re.search(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>", block)
        pmid = pmid_m.group(1) if pmid_m else "?"
        articles.append({
            "pmid":     pmid,
            "title":    clean_xml(title_m.group(1))    if title_m    else "No title",
            "abstract": clean_xml(abs_m.group(1))[:800] if abs_m      else "No abstract",
            "year":     year_m.group(1)                if year_m     else "?",
            "journal":  clean_xml(journal_m.group(1))  if journal_m  else "?",
            "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


def build_pubmed_queries_with_gemini(features, target, context, api_key):
    """
    Use Gemini to translate raw process parameter names into PubMed search queries.
    Falls back to simple rule-based extraction on API failure.

    IMPROVEMENT: prompt is explicit about JSON-only output to reduce parsing failures.
    """
    import re as _re
    import json as _js
    import urllib.request as _ur

    feat_list = "\n".join(["- " + f for f in features])
    prompt = (
        "You are a bioprocess scientist helping to search PubMed for literature. "
        "Convert each process parameter name into a concise PubMed search query "
        "(3-6 words max) using standard scientific terminology. "
        "Process context: " + context + " "
        "Target variable: " + target + " "
        "Parameters:\n" + feat_list + "\n"
        "Also add 2 broad topic queries. "
        'Reply ONLY with a JSON array like: [{"param":"X","query":"HIC protein yield"}] '
        "No markdown, no explanation."
    )
    try:
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.5-flash:generateContent?key=" + api_key)
        payload = _js.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.1},
        }).encode("utf-8")
        req = _ur.Request(url, data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=30) as resp:
            result = _js.loads(resp.read())
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = _re.sub(r"^```[\w]*\n?|```$", "", raw).strip()
        parsed = _js.loads(raw)
        qs = [(item["param"], item["query"]) for item in parsed
              if "param" in item and "query" in item]
        if qs:
            return qs
    except Exception:
        pass

    # Fallback: rule-based
    proc_kw = context.split(",")[0].strip() if context else "bioprocess"
    qs = []
    for feat in features:
        import re as _re2
        clean = _re2.sub(r"\(.*?\)", "", feat).replace("_", " ").replace("-", " ")
        words = [w for w in clean.split() if len(w) > 3]
        kw = " ".join(words[:4])
        if kw:
            qs.append((feat, kw + " " + proc_kw + " chromatography yield"))
    qs.append(("Overall", proc_kw + " yield optimization process parameters"))
    return qs


# ── Model Validation: K-fold / LOOCV / Bootstrap ──────────────────────────────

def _cv_metrics(y_true, y_pred):
    """Compute R², RMSE, MAE for one fold."""
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def compute_kfold_cv(model, X, y, n_splits=5, shuffle=True, random_state=42):
    """
    K-Fold Cross-Validation.

    Parameters
    ----------
    model       : sklearn estimator (unfitted clone used per fold)
    X, y        : features and target (array-like)
    n_splits    : number of folds (k)

    Returns
    -------
    dict with keys 'R²', 'RMSE', 'MAE' → np.ndarray of per-fold scores
    and 'y_true', 'y_pred' → concatenated out-of-fold predictions
    """
    from sklearn.base import clone
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    r2_list, rmse_list, mae_list = [], [], []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X_arr):
        m = clone(model)
        m.fit(X_arr[train_idx], y_arr[train_idx])
        y_hat = m.predict(X_arr[test_idx])
        r2, rmse, mae = _cv_metrics(y_arr[test_idx], y_hat)
        r2_list.append(r2); rmse_list.append(rmse); mae_list.append(mae)
        y_true_all.extend(y_arr[test_idx]); y_pred_all.extend(y_hat)

    return {
        "method":  f"{n_splits}-Fold CV",
        "R²":      np.array(r2_list),
        "RMSE":    np.array(rmse_list),
        "MAE":     np.array(mae_list),
        "y_true":  np.array(y_true_all),
        "y_pred":  np.array(y_pred_all),
    }


def compute_loocv(model, X, y):
    """
    Leave-One-Out Cross-Validation (LOOCV).
    Each sample is held out once; n folds total.

    Returns
    -------
    Same structure as compute_kfold_cv.
    Note: per-fold R² is not meaningful for n=1 test sets;
    overall R² is computed from the full OOF prediction vector.
    """
    from sklearn.base import clone
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    loo = LeaveOneOut()
    y_pred_all = np.empty(n)
    rmse_list, mae_list = [], []

    for train_idx, test_idx in loo.split(X_arr):
        m = clone(model)
        m.fit(X_arr[train_idx], y_arr[train_idx])
        y_hat = m.predict(X_arr[test_idx])
        y_pred_all[test_idx[0]] = y_hat[0]
        rmse_list.append(abs(y_arr[test_idx[0]] - y_hat[0]))   # |e| per fold
        mae_list.append(abs(y_arr[test_idx[0]] - y_hat[0]))

    # Overall R² from the full OOF vector
    overall_r2 = float(r2_score(y_arr, y_pred_all))
    overall_rmse = float(np.sqrt(mean_squared_error(y_arr, y_pred_all)))
    overall_mae  = float(mean_absolute_error(y_arr, y_pred_all))

    return {
        "method":       "LOOCV",
        "R²":           np.array([overall_r2]),         # single overall value
        "RMSE":         np.array([overall_rmse]),
        "MAE":          np.array([overall_mae]),
        "R²_per_fold":  None,                           # undefined for n=1 folds
        "abs_err":      np.array(mae_list),             # per-sample absolute error
        "y_true":       y_arr,
        "y_pred":       y_pred_all,
    }


def compute_bootstrap_cv(model, X, y, n_boot=200, random_state=42):
    """
    Bootstrap .632 Cross-Validation.

    Each iteration:
      1. Draw n samples with replacement → bootstrap training set
      2. Out-of-bag (OOB) samples → test set
      3. Compute metrics on OOB set

    The .632 estimator (Efron & Tibshirani 1997) combines bootstrap OOB
    and train error to reduce optimism bias:
        err_.632 = 0.368 * train_err + 0.632 * oob_err

    Returns
    -------
    dict with keys 'R²', 'RMSE', 'MAE' → np.ndarray of per-bootstrap scores
    and scalar '0.632_R²', '0.632_RMSE', '0.632_MAE'
    and 'y_true_concat', 'y_pred_concat' for all OOB predictions
    """
    from sklearn.base import clone
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)
    rng = np.random.default_rng(random_state)

    r2_list, rmse_list, mae_list = [], [], []
    y_true_concat, y_pred_concat = [], []

    for _ in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        if oob_mask.sum() < 2:
            continue
        m = clone(model)
        m.fit(X_arr[boot_idx], y_arr[boot_idx])
        y_hat = m.predict(X_arr[oob_mask])
        r2, rmse, mae = _cv_metrics(y_arr[oob_mask], y_hat)
        r2_list.append(r2); rmse_list.append(rmse); mae_list.append(mae)
        y_true_concat.extend(y_arr[oob_mask]); y_pred_concat.extend(y_hat)

    r2_arr   = np.array(r2_list)
    rmse_arr = np.array(rmse_list)
    mae_arr  = np.array(mae_list)

    # .632 correction: train error on full dataset
    m_full = clone(model)
    m_full.fit(X_arr, y_arr)
    y_train_pred = m_full.predict(X_arr)
    _, train_rmse, train_mae = _cv_metrics(y_arr, y_train_pred)
    train_r2 = float(r2_score(y_arr, y_train_pred))

    # .632 blended estimates
    r2_632   = 0.368 * train_r2   + 0.632 * float(r2_arr.mean())
    rmse_632 = 0.368 * train_rmse + 0.632 * float(rmse_arr.mean())
    mae_632  = 0.368 * train_mae  + 0.632 * float(mae_arr.mean())

    return {
        "method":       f"Bootstrap ({n_boot} iter)",
        "R²":           r2_arr,
        "RMSE":         rmse_arr,
        "MAE":          mae_arr,
        "0.632_R²":     r2_632,
        "0.632_RMSE":   rmse_632,
        "0.632_MAE":    mae_632,
        "y_true":       np.array(y_true_concat),
        "y_pred":       np.array(y_pred_concat),
    }


def compare_cv_methods(model, X, y,
                        kfold_k=5, boot_n=200, random_state=42):
    """
    Run all three CV methods and return results dict keyed by method name.

    Returns
    -------
    {'K-Fold': {...}, 'LOOCV': {...}, 'Bootstrap': {...}}
    """
    results = {}
    results["K-Fold"]    = compute_kfold_cv(model, X, y,
                                             n_splits=kfold_k,
                                             random_state=random_state)
    results["LOOCV"]     = compute_loocv(model, X, y)
    results["Bootstrap"] = compute_bootstrap_cv(model, X, y,
                                                 n_boot=boot_n,
                                                 random_state=random_state)
    return results


def call_gemini(api_key, prompt, max_tokens=6000):
    """
    Call Gemini REST API and return response text.
    IMPROVEMENT: timeout increased to 60s for long analyses.
    """
    import json as _js
    import urllib.request as _ur
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash:generateContent?key=" + api_key)
    payload = _js.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
    }).encode("utf-8")
    req = _ur.Request(url, data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
    with _ur.urlopen(req, timeout=60) as resp:
        result = _js.loads(resp.read())
    return result["candidates"][0]["content"]["parts"][0]["text"]
