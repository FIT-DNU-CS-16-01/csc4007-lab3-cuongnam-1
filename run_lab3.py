from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.data import (
    build_vocab,
    create_dataloaders,
    encode_dataframe,
    prepare_splits,
)
from src.error_analysis import build_error_analysis, save_error_analysis
from src.evaluate import (
    compute_classification_metrics,
    create_baseline_vs_rnn,
    plot_confusion_matrix,
    plot_training_curves,
    save_epoch_history,
    save_metrics_summary,
)
from src.model import RNNClassifier
from src.sequence_audit import build_sequence_audit, render_sequence_audit_md
from src.train import evaluate_epoch, predict_with_probs, train_model
from src.utils import ensure_dir, save_json, set_seed
from src.wandb_utils import init_wandb, log_epoch, safe_finish


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='imdb', choices=['imdb', 'local_csv'])
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--text_col', default='text')
    ap.add_argument('--label_col', default='label')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_rows', type=int, default=None)
    ap.add_argument('--vocab_size', type=int, default=20000)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--use_wandb', action='store_true')
    ap.add_argument('--wandb_project', default='csc4007-lab3-rnn')
    ap.add_argument('--wandb_entity', default=None)
    ap.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'])
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--baseline_metrics_path', default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path('outputs')
    for sub in ['logs', 'splits', 'metrics', 'figures', 'models', 'predictions', 'error_analysis']:
        ensure_dir(out_dir / sub)

    splits = prepare_splits(
        name=args.dataset,
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    for split_name, split_df in splits.items():
        split_df.to_csv(out_dir / 'splits' / f'{split_name}.csv', index=False)

    vocab = build_vocab(splits['train']['text'].tolist(), max_vocab_size=args.vocab_size)
    encoded = {
        split_name: encode_dataframe(split_df, vocab=vocab, max_len=args.max_len)
        for split_name, split_df in splits.items()
    }
    train_loader, val_loader, test_loader = create_dataloaders(
        encoded['train'], encoded['val'], encoded['test'], batch_size=args.batch_size, seed=args.seed
    )

    audit = build_sequence_audit(
        train_texts=splits['train']['text'].tolist(),
        vocab=vocab,
        max_len=args.max_len,
        encoded_train=encoded['train'],
    )
    render_sequence_audit_md(out_dir / 'logs' / 'sequence_audit.md', audit)

    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        pad_idx=0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    run = init_wandb(args=args, audit=audit, vocab_size=len(vocab))

    history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        epoch_logger=lambda row: log_epoch(run, row),
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / 'models' / 'best_model.pt')

    val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
    test_loss, test_metrics = evaluate_epoch(model, test_loader, criterion, device)
    pred_df, y_true, y_pred, y_prob = predict_with_probs(model, test_loader, device)

    pred_export = splits['test'].reset_index(drop=True).copy()
    pred_export['pred_label'] = pred_df['pred_label']
    pred_export['prob_negative'] = pred_df['prob_negative']
    pred_export['prob_positive'] = pred_df['prob_positive']
    pred_export.to_csv(out_dir / 'predictions' / 'test_predictions.csv', index=False)

    save_epoch_history(history, out_dir / 'metrics' / 'epoch_history.csv')
    plot_training_curves(history, out_dir / 'figures')
    plot_confusion_matrix(y_true, y_pred, out_dir / 'figures' / 'confusion_matrix.png')

    metrics_summary = {
        'dataset': args.dataset,
        'dataset_path': args.data_path if args.dataset == 'local_csv' else None,
        'seed': args.seed,
        'device': str(device),
        'model': 'embedding_rnn',
        'vocab_size_requested': args.vocab_size,
        'vocab_size_actual': len(vocab),
        'max_len': args.max_len,
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'epochs_requested': args.epochs,
        'epochs_trained': len(history),
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'splits': {k: int(len(v)) for k, v in splits.items()},
        'sequence_audit': audit,
        'val': {'loss': val_loss, **val_metrics},
        'test': {'loss': test_loss, **test_metrics},
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'notes': 'Token sequence -> Embedding -> RNN -> classifier; early stopping on validation macro-F1.',
        'split_protocol': 'IMDB uses original train/test with validation carved from train only; local_csv uses seeded stratified splits when possible.',
    }
    save_metrics_summary(metrics_summary, out_dir / 'metrics')

    errors = build_error_analysis(pred_export)
    save_error_analysis(errors, out_dir / 'error_analysis', min_expected=10)

    create_baseline_vs_rnn(
        rnn_metrics=metrics_summary,
        output_path=out_dir / 'metrics' / 'baseline_vs_rnn.csv',
        baseline_metrics_path=args.baseline_metrics_path,
    )

    run_summary = {
        'dataset': args.dataset,
        'seed': args.seed,
        'model': 'embedding_rnn',
        'test_macro_f1': test_metrics['macro_f1'],
        'test_accuracy': test_metrics['accuracy'],
        'best_val_macro_f1': max(row['val_macro_f1'] for row in history) if history else None,
        'wandb_enabled': bool(run),
        'wandb_mode': args.wandb_mode if args.use_wandb else 'disabled',
    }
    save_json(run_summary, out_dir / 'logs' / 'run_summary.json')

    if run is not None:
        run.summary.update({
            'best_val_macro_f1': run_summary['best_val_macro_f1'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_loss,
        })
    safe_finish(run)
    print('DONE.')


if __name__ == '__main__':
    main()
