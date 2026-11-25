"""
HER360 Fertility - Ovulation Predictor: data preparation (v1)

Turns the raw source-labelled candidate file into a model-ready table
WITHOUT changing any label and WITHOUT inventing any value.

Corrections made (each is logged):
  1. Sort rows by participant and cycle_number.
  2. Remove repeats of (participant, cycle_number): keep first occurrence in file order.
  3. Drop columns not known at prediction time (leakage audit).
  4. Build features from the participant's PREVIOUS cycles only.
  5. Label how much history each row has.
Not done on purpose: no label edits, no imputation, no changing the splits.
"""
import json
import numpy as np
import pandas as pd

TARGET = "target_estimated_ovulation_day"
GROUP = "participant_group_id"
SPLIT = "split"

LEAKAGE_COLUMNS = {
    "cycle_length_days": "Length of the CURRENT cycle - unknown until the cycle ends (kept only to build lags of later cycles)",
    "first_day_high_fertility_sign": "Observed during the cycle; not guaranteed at prediction time",
    "total_high_days": "Accumulated over time; post-start information",
    "high_days_post_peak": "Post-peak information",
    "total_peak_days": "Accumulated / after-event information",
}

N_LAGS = 3


def load_raw(path):
    return pd.read_csv(path)


def prepare(df_raw):
    """Return (model_table, log_dict)."""
    log = {"rows_in": int(len(df_raw))}

    # repeated (participant, cycle_number): keep first occurrence in ORIGINAL file order
    df = df_raw.reset_index().rename(columns={"index": "_orig_row"})
    df = df.sort_values([GROUP, "cycle_number", "_orig_row"], kind="stable")
    repeated = df[df.duplicated([GROUP, "cycle_number"], keep="first")]
    log["repeated_rows_removed"] = int(len(repeated))
    log["repeated_rows_removed_original_row_numbers"] = repeated["_orig_row"].tolist()
    df = df.drop_duplicates([GROUP, "cycle_number"], keep="first").reset_index(drop=True)

    # lag features from PREVIOUS cycles of the same participant only
    length_lookup = df.set_index([GROUP, "cycle_number"])["cycle_length_days"]
    for k in range(1, N_LAGS + 1):
        keys = pd.MultiIndex.from_arrays([df[GROUP], df["cycle_number"] - k])
        df[f"prev_cycle_length_{k}"] = length_lookup.reindex(keys).to_numpy()

    lag_cols = [f"prev_cycle_length_{k}" for k in range(1, N_LAGS + 1)]
    df["n_prev_cycles"] = df[lag_cols].notna().sum(axis=1)
    df["prev_cycle_mean"] = df[lag_cols].mean(axis=1)
    df["prev_cycle_std"] = df[lag_cols].std(axis=1)  # NaN when fewer than 2 previous cycles

    # history status - a PROJECT convention (not a clinical rule)
    df["history_status"] = np.select(
        [df["n_prev_cycles"] == 0, df["n_prev_cycles"] < N_LAGS],
        ["insufficient", "limited"],
        default="sufficient",
    )

    keep = [GROUP, SPLIT, "cycle_number", "reproductive_category_code",
            *lag_cols, "n_prev_cycles", "prev_cycle_mean", "prev_cycle_std",
            "history_status", TARGET]
    out = df[keep].copy()

    log["rows_out"] = int(len(out))
    log["history_status_counts"] = out["history_status"].value_counts().to_dict()
    log["history_status_by_split"] = (
        out.groupby([SPLIT, "history_status"]).size().unstack(fill_value=0).to_dict("index")
    )
    log["labels_modified"] = 0
    log["values_imputed"] = 0
    return out, log


FEATURES_V1 = [
    "prev_cycle_length_1", "prev_cycle_length_2", "prev_cycle_length_3",
    "prev_cycle_mean", "prev_cycle_std", "n_prev_cycles",
]


def validate(out):
    """Hard checks - raise if anything is wrong."""
    assert out[TARGET].notna().all(), "missing target"
    assert not out.duplicated([GROUP, "cycle_number"]).any(), "repeated participant+cycle remain"
    assert (out.groupby(GROUP)[SPLIT].nunique() == 1).all(), "participant in >1 split"
    for c in FEATURES_V1:
        assert c not in LEAKAGE_COLUMNS and c not in (GROUP, SPLIT, TARGET)
    lagc = [c for c in out.columns if c.startswith("prev_cycle_length_")]
    assert out[lagc].stack().dropna().between(15, 60).all(), "implausible previous cycle length"
    return True
