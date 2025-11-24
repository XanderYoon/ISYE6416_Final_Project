from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def paired_statistical_comparison(df: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    """
    Paired comparison between single and mixed features.

    Assumes columns of the form:
        '{label_prefix}_single_<feature>',
        '{label_prefix}_mixed_<feature>'.

    Returns
    -------
    stats_df : pd.DataFrame with columns
        feature, mean_single, mean_mixed, diff_mean,
        p_ttest, p_wilcoxon, significant_0.05
    """
    single_cols = [c for c in df.columns if f"{label_prefix}_single_" in c]
    mixed_cols = [c for c in df.columns if f"{label_prefix}_mixed_" in c]

    single_cols = sorted(single_cols)
    mixed_cols = sorted(mixed_cols)

    results = []

    for single_col, mixed_col in zip(single_cols, mixed_cols):
        feature_name = single_col.replace(f"{label_prefix}_single_", "")

        single_vals = df[single_col].astype(float)
        mixed_vals = df[mixed_col].astype(float)
        diff = mixed_vals - single_vals

        ttest_p = ttest_rel(mixed_vals, single_vals, nan_policy="omit").pvalue
        try:
            wilcoxon_p = wilcoxon(mixed_vals, single_vals).pvalue
        except ValueError:
            wilcoxon_p = np.nan

        results.append(
            {
                "feature": feature_name,
                "mean_single": float(np.mean(single_vals)),
                "mean_mixed": float(np.mean(mixed_vals)),
                "diff_mean": float(np.mean(diff)),
                "p_ttest": float(ttest_p),
                "p_wilcoxon": float(wilcoxon_p),
                "significant_0.05": bool(ttest_p < 0.05),
            }
        )

    return pd.DataFrame(results)
