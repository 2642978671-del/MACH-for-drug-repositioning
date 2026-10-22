import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


PAIR_COLUMNS = ["drug_id", "disease_id", "values"]


def parse_positive_int_tuple(value):
    """Parse comma-separated positive integers used by multi-scale modules."""
    if isinstance(value, tuple):
        parsed = value
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        parsed = tuple(int(part) for part in parts)
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("value must contain one or more positive integers")
    return parsed


def fixed_epoch_test_schedule(max_epochs):
    """Return the only epoch at which the outer test fold may be evaluated."""
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    return (int(max_epochs),)


def load_completed_fixed_fold(path, expected_fold, expected_epoch):
    """Load a completed fixed-epoch fold only when its audit fields match."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if payload.get("fold") != int(expected_fold):
        return None
    if payload.get("fixed_epoch") != int(expected_epoch):
        return None
    if payload.get("test_read_count") != 1:
        return None
    try:
        return (
            float(payload["test_auroc"]),
            float(payload["test_aupr_trapz"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_pair_frame(frame):
    missing = set(PAIR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("pair frame is missing columns: {}".format(sorted(missing)))
    labels = set(frame["values"].unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("values must contain only binary labels 0 and 1")
    if frame.duplicated(["drug_id", "disease_id"]).any():
        raise ValueError("pair frame contains duplicate drug-disease pairs")


def _frame_summary(frame):
    counts = frame["values"].value_counts().to_dict()
    return {
        "count": int(len(frame)),
        "positive_count": int(counts.get(1, 0)),
        "unlabeled_count": int(counts.get(0, 0)),
    }


def _split_hash(inner_train, validation, seed, validation_ratio):
    payload = {
        "seed": int(seed),
        "validation_ratio": float(validation_ratio),
        "inner_train": inner_train[PAIR_COLUMNS].sort_values(PAIR_COLUMNS).values.tolist(),
        "validation": validation[PAIR_COLUMNS].sort_values(PAIR_COLUMNS).values.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stratified_inner_split(outer_train, validation_ratio=0.1, seed=2026):
    """Split an outer training fold by label without touching the outer test fold."""
    _validate_pair_frame(outer_train)
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []

    for label in (1, 0):
        label_frame = outer_train.loc[outer_train["values"] == label, PAIR_COLUMNS]
        if len(label_frame) < 2:
            raise ValueError("each label needs at least two outer-training pairs")

        permutation = rng.permutation(len(label_frame))
        validation_count = int(round(len(label_frame) * validation_ratio))
        validation_count = min(max(validation_count, 1), len(label_frame) - 1)
        validation_positions = permutation[:validation_count]
        train_positions = permutation[validation_count:]

        validation_parts.append(label_frame.iloc[validation_positions])
        train_parts.append(label_frame.iloc[train_positions])

    inner_train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    inner_train = inner_train.iloc[rng.permutation(len(inner_train))].reset_index(drop=True)
    validation = validation.iloc[rng.permutation(len(validation))].reset_index(drop=True)

    train_pairs = set(map(tuple, inner_train[["drug_id", "disease_id"]].to_numpy()))
    validation_pairs = set(map(tuple, validation[["drug_id", "disease_id"]].to_numpy()))
    if train_pairs & validation_pairs:
        raise RuntimeError("inner training and validation pairs overlap")
    if len(train_pairs | validation_pairs) != len(outer_train):
        raise RuntimeError("inner split does not cover the outer training fold")

    manifest = {
        "seed": int(seed),
        "validation_ratio": float(validation_ratio),
        "inner_train": _frame_summary(inner_train),
        "validation": _frame_summary(validation),
    }
    manifest["split_hash"] = _split_hash(
        inner_train,
        validation,
        seed=seed,
        validation_ratio=validation_ratio,
    )
    return inner_train, validation, manifest


@dataclass
class EarlyStoppingTracker:
    patience: int
    min_delta: float = 1e-4
    best_epoch: int = 0
    best_aupr: float = -float("inf")
    best_auroc: float = -float("inf")
    bad_checks: int = 0

    def __post_init__(self):
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")

    def update(self, epoch, aupr, auroc):
        primary_improved = aupr > self.best_aupr + self.min_delta
        primary_tied = abs(aupr - self.best_aupr) <= self.min_delta
        secondary_improved = primary_tied and auroc > self.best_auroc
        improved = primary_improved or secondary_improved

        if improved:
            self.best_epoch = int(epoch)
            self.best_aupr = float(aupr)
            self.best_auroc = float(auroc)
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_checks >= self.patience
