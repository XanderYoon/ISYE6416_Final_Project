from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


def one_way_anova_by_factor(
    df: pd.DataFrame,
    factor_col: str,
    min_group_size: int = 3,
    excluded_features: List[str] | None = None,
) -> pd.DataFrame:
    """
    Run one-way ANOVA for each numeric feature against a categorical factor.

    Parameters
    ----------
    df : pd.DataFrame
    factor_col : str
        Factor column name (e.g., 'gender', 'location', 'sound_type').
    min_group_size : int
        Minimum group size for inclusion.
    excluded_features : list[str] or None
        List of feature names to skip.

    Returns
    -------
    anova_df : pd.DataFrame
        Columns: factor, feature, F, p_value
    """
    if excluded_features is None:
        excluded_features = ["duration"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []

    for feature in numeric_cols:
        if feature in excluded_features:
            continue

        groups = []
        for _, sub_df in df.groupby(factor_col):
            values = sub_df[feature].dropna().values
            if len(values) >= min_group_size:
                groups.append(values)

        if len(groups) < 2:
            continue

        F_stat, p_val = f_oneway(*groups)
        results.append(
            {
                "factor": factor_col,
                "feature": feature,
                "F": float(F_stat),
                "p_value": float(p_val),
            }
        )

    result = mark_significance(pd.DataFrame(results))
    return pd.DataFrame(result).sort_values("p_value")


def mark_significance(anova_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Add a boolean 'significant' column to ANOVA results based on alpha.
    """
    result = anova_df.copy()
    result["significant"] = result["p_value"] < alpha
    return result
