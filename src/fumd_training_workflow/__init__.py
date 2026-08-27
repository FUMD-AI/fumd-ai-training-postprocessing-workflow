"""
FUMD-AI Training & Postprocessing Workflow -- shared library.

Consumes the AI-ready dataset produced by the FUMD-AI preprocessing
workflow (`dataset_labeled_w<W>.csv`: per-row `migration`/`destination`
labels on top of the merged SUMO+OMNeT++ per-vehicle-per-second features)
and provides:

  - data.py           dataset loading, sequence windowing, scaling/encoding
  - model.py           the BiLSTM + Bahdanau-attention forecasting model
  - evaluate.py         metrics computation (loss/accuracy/precision/recall/F1/top-2/confusion matrix)
  - metrics_io.py       structured (CSV/JSON) metrics read/write for postprocessing
  - bs_location.py      real-measurement base-station location estimation
  - migration_map.py    spatial plotting of migration/handover events

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
"""

from .data import (
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    GROUP_COLUMN,
    TIME_COLUMN,
    load_labeled_dataset,
    ensure_one_row_per_vehicle_second,
    split_vehicles,
    SafeLabelEncoder,
    fit_label_encoders,
    apply_label_encoders,
    build_sequences,
    fit_scaler,
    apply_scaler,
)
from .model import BahdanauAttention, ZeroInitialState, build_model

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "GROUP_COLUMN",
    "TIME_COLUMN",
    "load_labeled_dataset",
    "ensure_one_row_per_vehicle_second",
    "split_vehicles",
    "SafeLabelEncoder",
    "fit_label_encoders",
    "apply_label_encoders",
    "build_sequences",
    "fit_scaler",
    "apply_scaler",
    "BahdanauAttention",
    "ZeroInitialState",
    "build_model",
]
