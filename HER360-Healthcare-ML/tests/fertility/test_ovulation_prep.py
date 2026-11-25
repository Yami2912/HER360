import sys, pathlib
import pandas as pd
import pytest

# find HER360-Healthcare-ML root by walking up until preprocessing/fertility exists
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "preprocessing" / "fertility").exists())
sys.path.insert(0, str(ROOT / "preprocessing" / "fertility"))
import ovulation_prep as op


def make(rows):
    cols = ["cycle_number", "reproductive_category_code", "cycle_length_days",
            "first_day_high_fertility_sign", "total_high_days", "high_days_post_peak",
            "total_peak_days", op.TARGET, op.GROUP, op.SPLIT]
    return pd.DataFrame(rows, columns=cols)


def row(cn, length, target, g="p1", split="train"):
    return (cn, 0, length, 10.0, 2.0, 0.0, 2.0, target, g, split)


def test_lags_use_only_previous_cycles():
    t, _ = op.prepare(make([row(1, 28, 14), row(2, 30, 15), row(3, 26, 13)]))
    r3 = t[t.cycle_number == 3].iloc[0]
    assert r3.prev_cycle_length_1 == 30 and r3.prev_cycle_length_2 == 28
    r1 = t[t.cycle_number == 1].iloc[0]
    assert r1.n_prev_cycles == 0 and r1.history_status == "insufficient"


def test_gap_in_cycle_numbers_is_not_bridged():
    t, _ = op.prepare(make([row(1, 28, 14), row(3, 30, 15)]))  # cycle 2 missing
    r = t[t.cycle_number == 3].iloc[0]
    assert pd.isna(r.prev_cycle_length_1)
    assert r.prev_cycle_length_2 == 28


def test_repeated_cycle_removed_first_kept_labels_untouched():
    t, log = op.prepare(make([row(1, 28, 14), row(2, 27, 13), row(2, 26, 13)]))
    assert log["repeated_rows_removed"] == 1
    assert len(t) == 2
    assert t[t.cycle_number == 2].iloc[0][op.TARGET] == 13


def test_lags_do_not_cross_participants():
    t, _ = op.prepare(make([row(1, 28, 14, "a"), row(2, 30, 15, "b")]))
    assert pd.isna(t[t[op.GROUP] == "b"].iloc[0].prev_cycle_length_1)


def test_leakage_columns_absent_from_output_and_features():
    t, _ = op.prepare(make([row(1, 28, 14)]))
    for c in op.LEAKAGE_COLUMNS:
        assert c not in t.columns
        assert c not in op.FEATURES_V1


def test_validate_catches_split_leakage():
    t, _ = op.prepare(make([row(1, 28, 14, "a", "train"), row(2, 28, 14, "a", "test")]))
    with pytest.raises(AssertionError):
        op.validate(t)
