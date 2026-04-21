from __future__ import annotations

from typing import Any


def init_wandb(args, audit: dict, vocab_size: int):
    if not args.use_wandb or args.wandb_mode == 'disabled':
        return None
    try:
        import wandb
    except Exception:
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        config={
            'dataset': args.dataset,
            'seed': args.seed,
            'vocab_size_requested': args.vocab_size,
            'vocab_size_actual': vocab_size,
            'max_len': args.max_len,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'sequence_audit': audit,
        },
    )
    return run


def log_epoch(run, row: dict[str, Any]) -> None:
    if run is None:
        return
    run.log(row)


def safe_finish(run) -> None:
    if run is None:
        return
    run.finish()
