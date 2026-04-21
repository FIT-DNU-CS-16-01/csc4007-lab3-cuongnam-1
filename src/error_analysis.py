from __future__ import annotations

from pathlib import Path

import pandas as pd


def assign_error_bucket(text: str, prob_negative: float, prob_positive: float) -> str:
    text_l = (text or '').lower()
    confidence = max(prob_negative, prob_positive)
    if ' not ' in f' {text_l} ' or "n't" in text_l or ' never ' in f' {text_l} ':
        return 'negation'
    if ('but' in text_l or 'however' in text_l or 'although' in text_l) and len(text_l.split()) > 20:
        return 'mixed_sentiment'
    if len(text_l.split()) > 120:
        return 'long_review'
    if confidence >= 0.9:
        return 'confident_but_wrong'
    return 'other'


def build_error_analysis(pred_df: pd.DataFrame) -> pd.DataFrame:
    errors = pred_df[pred_df['label'] != pred_df['pred_label']].copy()
    if errors.empty:
        return errors
    errors['error_bucket'] = errors.apply(
        lambda r: assign_error_bucket(r['text'], r['prob_negative'], r['prob_positive']),
        axis=1,
    )
    errors['confidence'] = errors[['prob_negative', 'prob_positive']].max(axis=1)
    cols = ['text', 'label', 'pred_label', 'prob_negative', 'prob_positive', 'confidence', 'error_bucket']
    return errors[cols].sort_values(by='confidence', ascending=False).reset_index(drop=True)


def save_error_analysis(errors: pd.DataFrame, out_dir: str | Path, min_expected: int = 10) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_dir / 'error_analysis.csv', index=False)
    if errors.empty:
        summary = '# Error Analysis\n\nKhông có mẫu sai để phân tích.\n'
    else:
        bucket_counts = errors['error_bucket'].value_counts().to_dict()
        lines = [
            '# Error Analysis',
            '',
            f'- n_errors: {len(errors)}',
            f'- meets_min_expected_{min_expected}: {len(errors) >= min_expected}',
            '',
            '## Bucket counts',
        ]
        for k, v in bucket_counts.items():
            lines.append(f'- {k}: {v}')
        summary = '\n'.join(lines) + '\n'
    (out_dir / 'error_analysis_summary.md').write_text(summary, encoding='utf-8')
