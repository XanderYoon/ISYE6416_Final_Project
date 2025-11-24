from typing import List

import pandas as pd


def inner_join(
    single_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    reference_col: str,
    join_keys: List[str],
) -> pd.DataFrame:
    """
    Perform a standardized inner join between a single-source dataset
    and a mixed dataset that contains a corresponding reference column.

    Parameters
    ----------
    single_df : pd.DataFrame
        Single-source dataset (e.g., HS or LS) with column 'filename'.
    mixed_df : pd.DataFrame
        Mixed dataset (e.g., Mix_HS or Mix_LS).
    reference_col : str
        Column in mixed_df representing the reference file
        (e.g., 'heart_ref_file' or 'lung_ref_file').
    join_keys : list of str
        Columns used as join keys (e.g., ['gender', 'heart_sound_type', 'location']).

    Returns
    -------
    merged_df : pd.DataFrame
        Inner-joined dataframe with unified column names:
        - 'single_source_filename'
        - 'mixed_source_filename'
    """
    single_renamed = single_df.rename(columns={"filename": "single_source_filename"})
    mixed_reduced = mixed_df[join_keys + [reference_col]].copy()
    mixed_reduced = mixed_reduced.rename(columns={reference_col: "mixed_source_filename"})

    merged_df = pd.merge(single_renamed, mixed_reduced, on=join_keys, how="inner")

    print(f"Matched rows: {len(merged_df)}")
    return merged_df
