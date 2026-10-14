import numpy as np
from scipy import stats


def residualize(values, design):
    """values minus their least-squares fit on the columns of design."""
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def partial_rank_corr(x, y, Z=None, min_n=4):
    """Spearman partial correlation of x and y controlling for columns of Z:
    rank everything, OLS-residualize ranked x and y on ranked Z, Pearson the residuals."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if Z is None:
        Z = np.empty((len(x), 0))
    else:
        Z = np.asarray(Z, float).reshape(len(x), -1)
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    n_ok = ok.sum()
    if n_ok < max(min_n, Z.shape[1] + 3):
        return np.nan

    rank_x = stats.rankdata(x[ok])
    rank_y = stats.rankdata(y[ok])
    columns = [np.ones(n_ok)]
    for j in range(Z.shape[1]):
        columns.append(stats.rankdata(Z[ok, j]))
    design = np.column_stack(columns)
    resid_x = residualize(rank_x, design)
    resid_y = residualize(rank_y, design)
    if resid_x.std() < 1e-12 or resid_y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def one_sample(v) -> dict:
    """Summary and tests of a distribution of per-unit effects against zero."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    out = dict(n=len(v), mean=np.nan, median=np.nan, frac_negative=np.nan,
               p_t=np.nan, p_wilcoxon=np.nan, p_sign=np.nan)
    if len(v) == 0:
        return out
    out["mean"] = v.mean()
    out["median"] = np.median(v)
    out["frac_negative"] = np.mean(v < 0)
    if len(v) > 1:
        out["p_t"] = stats.ttest_1samp(v, 0).pvalue
    nonzero = v[v != 0]
    if len(nonzero) >= 3:
        out["p_wilcoxon"] = stats.wilcoxon(nonzero).pvalue
        n_negative = int((nonzero < 0).sum())
        out["p_sign"] = stats.binomtest(n_negative, len(nonzero), 0.5).pvalue
    return out


def paired(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    out = dict(n=len(d), mean_diff=np.nan, frac_positive=np.nan, p_t=np.nan, p_wilcoxon=np.nan)
    if len(d):
        out["mean_diff"] = d.mean()
        out["frac_positive"] = np.mean(d > 0)
    if len(d) > 1:
        out["p_t"] = stats.ttest_1samp(d, 0).pvalue
    if len(d) >= 3 and np.any(d != 0):
        out["p_wilcoxon"] = stats.wilcoxon(d).pvalue
    return out


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    from_largest = ranked[::-1]
    running_min = np.minimum.accumulate(from_largest)
    adjusted_sorted = running_min[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted_sorted, 0, 1)
    return out


def circular_null(x, y, Z=None):
    """Within-run temporal null: correlations of y with every circular shift of x."""
    x = np.asarray(x, float)
    rhos = []
    for shift in range(1, len(x)):
        rhos.append(partial_rank_corr(np.roll(x, shift), y, Z))
    return np.array(rhos)
