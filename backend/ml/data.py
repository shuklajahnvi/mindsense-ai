"""Dataset download, inspection and splitting for GoEmotions and Dreaddit.

GoEmotions (Demszky et al., ACL 2020) - ``google-research-datasets/go_emotions``,
"simplified" configuration: 27 emotions + neutral, multi-label, with the
authors' official train/validation/test split (43,410 / 5,426 / 5,427).

Dreaddit (Turcan & McKeown, LOUHI 2019) - the labelled subset of the corpus
(~3.5k segments) mirrored at ``andreagasparini/dreaddit`` with the authors'
original train/test split. A validation set is carved out of *train* using
StratifiedGroupKFold grouped by ``post_id`` so that segments from the same
Reddit post never appear in both train and validation. Any post that occurs
in both train and test is removed from train to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

from ml.common.text import clean_text
from ml.paths import DATA_CACHE_DIR

GOEMOTIONS_ID = "google-research-datasets/go_emotions"
DREADDIT_ID = "andreagasparini/dreaddit"
SEED = 42


@dataclass
class MultiLabelSplits:
    label_names: list[str]
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        y = np.zeros((len(df), len(self.label_names)), dtype=np.float32)
        for i, labels in enumerate(df["labels"]):
            y[i, list(labels)] = 1.0
        return y


@dataclass
class BinarySplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    leakage_report: dict


def _subsample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)


def load_goemotions(train_limit: int | None = None, eval_limit: int | None = None) -> MultiLabelSplits:
    ds = load_dataset(GOEMOTIONS_ID, "simplified", cache_dir=str(DATA_CACHE_DIR))
    label_names = ds["train"].features["labels"].feature.names
    frames = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()[["id", "text", "labels"]]
        df["labels"] = df["labels"].apply(lambda x: [int(v) for v in x])
        df["clean"] = df["text"].map(clean_text)
        df = df[df["clean"].str.len() > 0]
        frames[split] = df
    return MultiLabelSplits(
        label_names=list(label_names),
        train=_subsample(frames["train"], train_limit),
        val=_subsample(frames["validation"], eval_limit),
        test=frames["test"].reset_index(drop=True),  # test set is never subsampled
    )


def load_dreaddit(val_fraction_folds: int = 7) -> BinarySplits:
    ds = load_dataset(DREADDIT_ID, cache_dir=str(DATA_CACHE_DIR))
    cols = ["id", "post_id", "subreddit", "text", "label", "confidence"]
    train = ds["train"].to_pandas()[cols].copy()
    test = ds["test"].to_pandas()[cols].copy()
    for df in (train, test):
        df["clean"] = df["text"].map(clean_text)
        df["label"] = df["label"].astype(int)

    # --- Leakage checks ---------------------------------------------------
    overlap_posts = set(train["post_id"]) & set(test["post_id"])
    dup_text = set(train["clean"]) & set(test["clean"])
    n_before = len(train)
    train = train[~train["post_id"].isin(overlap_posts) & ~train["clean"].isin(dup_text)]
    train = train.drop_duplicates(subset=["clean"]).reset_index(drop=True)
    leakage = {
        "train_test_shared_post_ids": len(overlap_posts),
        "train_test_duplicate_texts": len(dup_text),
        "train_rows_removed": int(n_before - len(train)),
        "posts_with_multiple_segments_in_train": int((train.groupby("post_id").size() > 1).sum()),
    }

    # --- Grouped, stratified validation split ------------------------------
    sgkf = StratifiedGroupKFold(n_splits=val_fraction_folds, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(sgkf.split(train, train["label"], groups=train["post_id"]))
    tr, va = train.iloc[tr_idx].reset_index(drop=True), train.iloc[va_idx].reset_index(drop=True)
    assert not (set(tr["post_id"]) & set(va["post_id"])), "post leakage between train/val"
    leakage["train_val_shared_post_ids"] = 0
    return BinarySplits(train=tr, val=va, test=test.reset_index(drop=True), leakage_report=leakage)
