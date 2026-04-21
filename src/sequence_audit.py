from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_sequence_audit(train_texts: list[str], vocab: dict[str, int], max_len: int, encoded_train: pd.DataFrame) -> dict:
    orig_lens = encoded_train['orig_len'].to_numpy()
    seq_lens = encoded_train['seq_len'].to_numpy()
    trunc_rate = float(encoded_train['truncated'].mean()) if len(encoded_train) else 0.0
    total_tokens = int(orig_lens.sum()) if len(orig_lens) else 0
    total_unk = int(encoded_train['unk_count'].sum()) if len(encoded_train) else 0
    unk_rate = float(total_unk / max(total_tokens, 1))
    avg_pad_ratio = float(np.mean((max_len - seq_lens) / max_len)) if len(seq_lens) else 0.0
    return {
        'n_train': int(len(encoded_train)),
        'vocab_size': int(len(vocab)),
        'max_len': int(max_len),
        'orig_len_median': float(np.median(orig_lens)) if len(orig_lens) else 0.0,
        'orig_len_p95': float(np.percentile(orig_lens, 95)) if len(orig_lens) else 0.0,
        'truncation_rate': trunc_rate,
        'unk_rate': unk_rate,
        'avg_pad_ratio': avg_pad_ratio,
    }


def render_sequence_audit_md(path: str | Path, audit: dict) -> None:
    text = f"""# Sequence Audit

- n_train: {audit['n_train']}
- vocab_size: {audit['vocab_size']}
- max_len: {audit['max_len']}
- orig_len_median: {audit['orig_len_median']:.2f}
- orig_len_p95: {audit['orig_len_p95']:.2f}
- truncation_rate: {audit['truncation_rate']:.4f}
- unk_rate: {audit['unk_rate']:.4f}
- avg_pad_ratio: {audit['avg_pad_ratio']:.4f}

## Gợi ý đọc kết quả
- Nếu truncation_rate cao, `max_len` có thể đang quá nhỏ.
- Nếu avg_pad_ratio quá cao, `max_len` có thể đang quá lớn.
- Nếu unk_rate cao, vocab_size có thể đang quá nhỏ hoặc tokenization chưa phù hợp.
"""
    Path(path).write_text(text, encoding='utf-8')
