from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def save_epoch_history(history: list[dict], path: str | Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def plot_training_curves(history: list[dict], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    df = pd.DataFrame(history)
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curve.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_macro_f1'], label='train_macro_f1')
    plt.plot(df['epoch'], df['val_macro_f1'], label='val_macro_f1')
    plt.plot(df['epoch'], df['train_accuracy'], label='train_accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Metric Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'metric_curve.png', dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=['negative', 'positive'])
    ax.set_yticks([0, 1], labels=['negative', 'positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_summary(metrics_summary: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / 'metrics_summary.json').write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md = f"""# Metrics Summary

## Validation
- loss: {metrics_summary['val']['loss']:.4f}
- accuracy: {metrics_summary['val']['accuracy']:.4f}
- macro_f1: {metrics_summary['val']['macro_f1']:.4f}

## Test
- loss: {metrics_summary['test']['loss']:.4f}
- accuracy: {metrics_summary['test']['accuracy']:.4f}
- macro_f1: {metrics_summary['test']['macro_f1']:.4f}
"""
    (out_dir / 'metrics_summary.md').write_text(md, encoding='utf-8')


def create_baseline_vs_rnn(rnn_metrics: dict, output_path: str | Path, baseline_metrics_path: str | None = None) -> None:
    rows = []
    if baseline_metrics_path and Path(baseline_metrics_path).exists():
        baseline = json.loads(Path(baseline_metrics_path).read_text(encoding='utf-8'))
        rows.append({
            'model': baseline.get('model', 'baseline_ml'),
            'input_representation': baseline.get('vectorizer', 'tfidf/bow'),
            'test_accuracy': baseline.get('test', {}).get('accuracy'),
            'test_macro_f1': baseline.get('test', {}).get('macro_f1'),
        })
    else:
        rows.append({
            'model': 'baseline_ml',
            'input_representation': 'tfidf/bow',
            'test_accuracy': None,
            'test_macro_f1': None,
        })
    rows.append({
        'model': 'embedding_rnn',
        'input_representation': 'token_sequence',
        'test_accuracy': rnn_metrics['test']['accuracy'],
        'test_macro_f1': rnn_metrics['test']['macro_f1'],
    })
    pd.DataFrame(rows).to_csv(output_path, index=False)
