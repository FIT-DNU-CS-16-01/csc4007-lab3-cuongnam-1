from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[.,!?;]")


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def normalize_labels(series: pd.Series) -> pd.Series:
    def _norm(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v in LABEL_TO_ID:
                return v
        if int(v) == 0:
            return "negative"
        return "positive"

    return series.map(_norm)


def _finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = normalize_labels(df["label"])
    return df[["text", "label"]].reset_index(drop=True)


def _safe_split(
    df: pd.DataFrame,
    test_size: float | int,
    seed: int,
    *,
    stratify_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[stratify_col] if stratify_col in df.columns and df[stratify_col].nunique() > 1 else None
    try:
        left, right = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        left, right = train_test_split(df, test_size=test_size, random_state=seed, stratify=None)
    return left.reset_index(drop=True), right.reset_index(drop=True)


def load_local_csv_dataset(
    data_path: str | None,
    text_col: str,
    label_col: str,
) -> pd.DataFrame:
    if not data_path:
        raise ValueError("data_path is required when dataset=local_csv")
    df = pd.read_csv(data_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {text_col}, {label_col}")
    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
    return _finalize_df(df)


def load_imdb_frames(max_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    if max_rows is None:
        train_df = load_dataset("imdb", split="train").to_pandas()
        test_df = load_dataset("imdb", split="test").to_pandas()
    else:
        total = max(int(max_rows), 50)
        test_n = max(20, round(total * 0.2))
        train_source_n = max(30, total - test_n)
        train_df = load_dataset("imdb", split=f"train[:{train_source_n}]").to_pandas()
        test_df = load_dataset("imdb", split=f"test[:{test_n}]").to_pandas()

    train_df = train_df.rename(columns={"text": "text", "label": "label"})[["text", "label"]]
    test_df = test_df.rename(columns={"text": "text", "label": "label"})[["text", "label"]]
    return _finalize_df(train_df), _finalize_df(test_df)


def prepare_splits(
    name: str,
    data_path: str | None,
    text_col: str,
    label_col: str,
    max_rows: int | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if name == "imdb":
        train_source, test_df = load_imdb_frames(max_rows=max_rows)
        train_df, val_df = _safe_split(train_source, test_size=0.2, seed=seed)
        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    if name == "local_csv":
        df = load_local_csv_dataset(data_path=data_path, text_col=text_col, label_col=label_col)
        if max_rows is not None:
            df = df.sample(n=min(int(max_rows), len(df)), random_state=seed).reset_index(drop=True)
        train_val, test_df = _safe_split(df, test_size=0.2, seed=seed)
        val_share = max(0.1, len(test_df) / max(len(train_val), 1))
        train_df, val_df = _safe_split(train_val, test_size=val_share, seed=seed)
        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    raise ValueError(f"Unsupported dataset: {name}")


def build_vocab(texts: Iterable[str], max_vocab_size: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common(max_vocab_size - 2):
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], int, int, int, int]:
    tokens = simple_tokenize(text)
    orig_len = len(tokens)
    ids = [vocab.get(tok, 1) for tok in tokens]
    unk_count = sum(1 for i in ids if i == 1)
    truncated = 1 if len(ids) > max_len else 0
    ids = ids[:max_len]
    length = len(ids)
    if length < max_len:
        ids = ids + [0] * (max_len - length)
    return ids, max(length, 1), orig_len, unk_count, truncated


def encode_dataframe(df: pd.DataFrame, vocab: dict[str, int], max_len: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        input_ids, seq_len, orig_len, unk_count, truncated = encode_text(row["text"], vocab, max_len)
        rows.append(
            {
                "text": row["text"],
                "label": row["label"],
                "label_id": LABEL_TO_ID[row["label"]],
                "input_ids": input_ids,
                "seq_len": seq_len,
                "orig_len": orig_len,
                "unk_count": unk_count,
                "truncated": truncated,
            }
        )
    return pd.DataFrame(rows)


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "seq_len": torch.tensor(row["seq_len"], dtype=torch.long),
            "label_id": torch.tensor(row["label_id"], dtype=torch.long),
            "text": row["text"],
            "label": row["label"],
        }



def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "seq_len": torch.stack([x["seq_len"] for x in batch]),
        "label_id": torch.stack([x["label_id"] for x in batch]),
        "text": [x["text"] for x in batch],
        "label": [x["label"] for x in batch],
    }


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(SequenceDataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=collate_batch, generator=generator)
    val_loader = DataLoader(SequenceDataset(val_df), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(SequenceDataset(test_df), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader
