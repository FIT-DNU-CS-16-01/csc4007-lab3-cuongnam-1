from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data import ID_TO_LABEL


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        lengths = batch['seq_len'].to(device)
        labels = batch['label_id'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * input_ids.size(0)
        preds = logits.argmax(dim=1)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    n = max(len(loader.dataset), 1)
    return total_loss / n, {
        'accuracy': float(accuracy_score(all_true, all_pred)),
        'macro_f1': float(f1_score(all_true, all_pred, average='macro', zero_division=0)),
    }


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['seq_len'].to(device)
            labels = batch['label_id'].to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * input_ids.size(0)
            preds = logits.argmax(dim=1)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    n = max(len(loader.dataset), 1)
    return total_loss / n, {
        'accuracy': float(accuracy_score(all_true, all_pred)),
        'macro_f1': float(f1_score(all_true, all_pred, average='macro', zero_division=0)),
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs: int, patience: int, min_delta: float, epoch_logger: Callable[[dict], None] | None = None):
    history = []
    best_metric = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_macro_f1': train_metrics['macro_f1'],
            'val_macro_f1': val_metrics['macro_f1'],
        }
        history.append(row)
        if epoch_logger:
            epoch_logger(row)

        current = val_metrics['macro_f1']
        if current > best_metric + min_delta:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    return history, best_state


def predict_with_probs(model, loader, device):
    model.eval()
    all_rows = []
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['seq_len'].to(device)
            labels = batch['label_id'].to(device)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for i in range(input_ids.size(0)):
                true_id = int(labels[i].cpu().item())
                pred_id = int(preds[i].cpu().item())
                prob_row = probs[i].cpu().tolist()
                y_true.append(true_id)
                y_pred.append(pred_id)
                y_prob.append(prob_row)
                all_rows.append({
                    'true_label': ID_TO_LABEL[true_id],
                    'pred_label': ID_TO_LABEL[pred_id],
                    'prob_negative': float(prob_row[0]),
                    'prob_positive': float(prob_row[1]),
                })
    return pd.DataFrame(all_rows), y_true, y_pred, y_prob
