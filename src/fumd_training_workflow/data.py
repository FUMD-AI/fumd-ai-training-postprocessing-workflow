"""
=============================================================================
FUMD-AI Training Workflow -- shared library: dataset loading & sequence windowing
=============================================================================
Used by: notebooks/step_1_load_and_window_dataset.ipynb (and reused by
step_2/step_3 for consistent feature/encoder/scaler handling).

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

Consumes `dataset_labeled_w<W>.csv`, the AI-ready output of the FUMD-AI
preprocessing workflow's Step 6. Columns present there:

    t, veh_id, x, y, angle, speed, pos, lane, slope, signals,
    Time, Object, averageCqiDl, distance, measuredSinrDl, measuredSinrUl,
    rcvdSinrDl, rlcDelayDl, rlcPduDelayDl, rlcThroughputDl,
    servingCell, servingCell-1..-7, servingCell1..7,
    migration, destination

Three column groups are deliberately kept OUT of the model's input features:

  - `servingCell1..7` (no dash) are *lead* (future) values -- see the
    preprocessing workflow's step_3 notebook, section "Add lagged/lead
    servingCell columns". Including them would hand the model part of its
    own answer. Only the lagged `servingCell-1..-7` columns are used.
  - `migration` / `destination` are themselves derived from each vehicle's
    *future* stable serving-cell trajectory (see the preprocessing
    workflow's labeling.py) -- also future information. They are carried
    through per-window (via `build_sequences`' `label_columns`) purely so
    evaluation can report accuracy separately for migration-window rows,
    never as inputs.
  - `x-1..x-7` / `y-1..y-7` (past position lags) don't exist in this
    dataset at all -- the preprocessing workflow's Step 6 drops them
    intentionally (only needed internally by its own Step 5). Only the
    current `x`, `y` survive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TIME_COLUMN = "t"
GROUP_COLUMN = "veh_id"
TARGET_COLUMN = "servingCell"

# High-cardinality per-map SUMO edge id -- the only genuinely categorical
# (string) column in this dataset. `signals` looks categorical but is a
# numeric turn-signal bitmask (0/1/2/8/9/10/...) -- left as a plain feature.
CATEGORICAL_COLUMNS = ["lane"]

# Future values -- never valid model inputs (see module docstring).
LEAD_LEAKAGE_COLUMNS = [f"servingCell{n}" for n in range(1, 8)]

# Derived-from-the-future labels -- carried alongside windows for
# evaluation, never fed to the model as a feature.
LABEL_ONLY_COLUMNS = ["migration", "destination"]

LAG_SERVINGCELL_COLUMNS = [f"servingCell-{n}" for n in range(1, 8)]

DEFAULT_FEATURE_COLUMNS = [
    "angle", "speed", "pos", "lane", "signals",
    "averageCqiDl", "distance", "measuredSinrDl", "measuredSinrUl", "rcvdSinrDl",
    "rlcDelayDl", "rlcPduDelayDl", "rlcThroughputDl",
    TARGET_COLUMN, *LAG_SERVINGCELL_COLUMNS,
    "x", "y",
]

REQUIRED_COLUMNS = {
    TIME_COLUMN, GROUP_COLUMN, TARGET_COLUMN, *LABEL_ONLY_COLUMNS,
    *DEFAULT_FEATURE_COLUMNS,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_labeled_dataset(path: str) -> pd.DataFrame:
    """Load one `dataset_labeled_w<W>.csv` and sanity-check its schema."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing expected columns {sorted(missing)} -- is this a "
            "dataset_labeled_w<W>.csv from the FUMD-AI preprocessing workflow's "
            "Step 6?"
        )
    leaked = [c for c in LEAD_LEAKAGE_COLUMNS if c in DEFAULT_FEATURE_COLUMNS]
    assert not leaked, f"lead/future columns must never be features: {leaked}"
    return df.sort_values([GROUP_COLUMN, TIME_COLUMN]).reset_index(drop=True)


def ensure_one_row_per_vehicle_second(
    df: pd.DataFrame,
    group_col: str = GROUP_COLUMN,
    time_col: str = TIME_COLUMN,
) -> pd.DataFrame:
    """
    Defensively collapse to (at most) one row per vehicle per whole second.

    The preprocessing workflow's output is already ~1 row/vehicle/second
    (SUMO fcd-output granularity), so this is normally a no-op -- unlike the
    old ad hoc training notebook, which unconditionally resampled. Detects
    duplicate (vehicle, floor(time)) pairs and only resamples (keeping the
    last reading in each second, matching the old notebook's behaviour)
    when they're actually present.
    """
    seconds = np.floor(df[time_col]).astype("int64")
    dup_mask = df.assign(_sec=seconds).duplicated(subset=[group_col, "_sec"], keep=False)
    if not dup_mask.any():
        return df.reset_index(drop=True)

    d = df.copy()
    d["_sec"] = seconds
    agg = {c: "last" for c in d.columns if c not in (group_col, "_sec")}
    out = d.groupby([group_col, "_sec"], as_index=False).agg(agg)
    out = out.rename(columns={"_sec": time_col + "_bucket"})
    out[time_col] = out[time_col + "_bucket"].astype(float)
    out = out.drop(columns=[time_col + "_bucket"])
    return out.sort_values([group_col, time_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

class SafeLabelEncoder:
    """
    Like sklearn's LabelEncoder, but `transform` maps any category not seen
    during `fit` to one extra reserved "unknown" code instead of raising.

    Needed because `lane` values are per-map SUMO edge ids: a road segment
    that happened not to be visited in the training run's traffic instance
    can legitimately appear in another evaluation run over the same map.
    The old notebook side-stepped this by re-fitting a fresh LabelEncoder on
    every single file it loaded -- which silently makes the codes
    inconsistent across files (encoder fit on file A assigns different
    integers than one fit on file B). Fitting once (on the training split)
    and reusing it everywhere else is the fix; this class is what makes
    that safe to do.
    """

    def __init__(self):
        self.classes_: list[str] = []
        self._index: dict[str, int] = {}

    def fit(self, values) -> "SafeLabelEncoder":
        self.classes_ = sorted(pd.unique(pd.Series(values).astype(str)))
        self._index = {c: i for i, c in enumerate(self.classes_)}
        return self

    @property
    def unknown_code(self) -> int:
        return len(self.classes_)  # one past the last real class

    @property
    def num_classes(self) -> int:
        return len(self.classes_) + 1  # + the reserved "unknown" bucket

    def transform(self, values) -> np.ndarray:
        s = pd.Series(values).astype(str)
        return s.map(self._index).fillna(self.unknown_code).astype(int).to_numpy()

    def fit_transform(self, values) -> np.ndarray:
        return self.fit(values).transform(values)


def fit_label_encoders(
    df: pd.DataFrame, columns: list[str] = CATEGORICAL_COLUMNS
) -> dict[str, SafeLabelEncoder]:
    return {c: SafeLabelEncoder().fit(df[c]) for c in columns}


def apply_label_encoders(
    df: pd.DataFrame, encoders: dict[str, SafeLabelEncoder]
) -> pd.DataFrame:
    out = df.copy()
    for col, enc in encoders.items():
        out[col] = enc.transform(out[col])
    return out


# ---------------------------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------------------------

def split_vehicles(
    df: pd.DataFrame,
    group_col: str = GROUP_COLUMN,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Split by *vehicle* id, not by window. Consecutive sliding windows from
    the same vehicle overlap heavily (they share all but one row), so
    splitting the already-built window arrays (what the old notebook did,
    via plain `train_test_split(X_total, y_total, ...)`) leaks near-
    duplicate samples across train/validation. Splitting the dataframe by
    whole vehicle first -- before `build_sequences` ever runs -- means no
    window in the validation set shares a single source row with a
    training window.
    """
    vehicle_ids = df[group_col].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(vehicle_ids)
    n_val = int(round(len(vehicle_ids) * val_fraction))
    n_val = min(max(n_val, 1), len(vehicle_ids) - 1) if len(vehicle_ids) > 1 else 0
    val_ids = set(vehicle_ids[:n_val])
    train_ids = set(vehicle_ids[n_val:])
    train_df = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_ids)].reset_index(drop=True)
    print(f"split_vehicles: {len(train_ids)} train vehicles, {len(val_ids)} val vehicles")
    return train_df, val_df


# ---------------------------------------------------------------------------
# Sliding-window sequence construction
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    sequence_length: int = 6,
    future_steps: list[int] = (1, 2, 3, 4, 5, 6, 7),
    group_col: str = GROUP_COLUMN,
    label_columns: list[str] = LABEL_ONLY_COLUMNS,
):
    """
    Build (X, y, extra) sliding windows per vehicle.

    X: (n_windows, sequence_length, n_features) -- `sequence_length` seconds
       of history immediately before the forecast point.
    y: (n_windows, len(future_steps)) -- `target_column` at each of
       `future_steps` seconds after the history window (same semantics as
       the old notebook's `create_sequences`: `y[:, k]` is the value
       `future_steps[k]` seconds ahead).
    extra: {label_col: (n_windows,) array} -- `label_columns` sampled at the
       *first* forecast step (`future_steps[0]`), for evaluation only (e.g.
       breaking out accuracy on `migration`-window rows). Never part of X.

    A vehicle with fewer than `sequence_length + max(future_steps)` rows is
    skipped entirely (not enough history+horizon to build even one window).
    """
    future_steps = list(future_steps)
    max_future = max(future_steps)
    needed = sequence_length + max_future

    X_list, y_list = [], []
    extra_lists = {c: [] for c in label_columns}
    n_skipped = 0

    for _, group in df.groupby(group_col, sort=False):
        group = group.sort_values(TIME_COLUMN).reset_index(drop=True)
        n = len(group)
        if n < needed:
            n_skipped += 1
            continue

        feats = group[feature_columns].to_numpy(dtype=float)
        target = group[target_column].to_numpy()
        label_arrays = {c: group[c].to_numpy() for c in label_columns if c in group.columns}

        for i in range(n - needed + 0):
            # i ranges 0 .. n-sequence_length-max_future-1 inclusive, so the
            # furthest row touched is i+sequence_length+max_future <= n-1.
            X_list.append(feats[i : i + sequence_length])
            y_list.append([target[i + sequence_length + step] for step in future_steps])
            for c, arr in label_arrays.items():
                extra_lists[c].append(arr[i + sequence_length + future_steps[0]])

    if n_skipped:
        print(f"build_sequences: skipped {n_skipped} vehicle(s) with < {needed} rows")

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    extra = {c: np.asarray(v) for c, v in extra_lists.items()}
    print(f"build_sequences: {X.shape[0]} windows, X {X.shape}, y {y.shape}")
    return X, y, extra


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the training split only (n, seq, feat) -> flat."""
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """
    Transform-only (never fit) -- the old notebook called `fit_transform`
    on every evaluation dataset too, which silently makes loss/accuracy
    numbers across datasets non-comparable (each was scaled by its own
    statistics instead of the training set's). This function is
    transform-only on purpose; there is no fit_transform variant here.
    """
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)
