#!/usr/bin/env python3
import argparse
import os
import json

import torch
import numpy as np
import pandas as pd
import plotly.express as px

from torch.utils.data import Dataset, DataLoader
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

from data import load_split, preprocess, extract_labels
from model import BasicDeepHitMLP
from utils import set_seed, DiscreteTimeLabelTransform, compute_rank_matrix, compute_brier_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate DeepHit competing-risks model'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--durations', type=int, default=16,
                        help='Number of discrete time intervals')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate for optimizer')
    return parser.parse_args()


def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, t, e in loader:
            X, t, e = X.to(device), t.to(device), e.to(device)
            preds = model(X)
            rank = compute_rank_matrix(t, e, device)
            loss_val = loss_fn(preds, t, e, rank)
            total_loss += loss_val.item() * X.size(0)
            total_samples += X.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0


def main():
    args = parse_args()
    set_seed(1234)
    os.makedirs('outputs/figures', exist_ok=True)

    # Data
    train_df, val_df, test_df = load_split(args.data)
    X_train, X_val, X_test, scaler = preprocess(train_df, val_df, test_df)
    y_tr_t, y_tr_e = extract_labels(train_df)
    y_v_t, y_v_e   = extract_labels(val_df)
    y_te_t, y_te_e = extract_labels(test_df)

    labtrans = DiscreteTimeLabelTransform(args.durations, scheme='equidistant')
    y_tr_t_disc, y_tr_e_disc = labtrans.fit_transform(y_tr_t, y_tr_e)
    y_v_t_disc, y_v_e_disc   = labtrans.transform(y_v_t, y_v_e)
    y_te_t_disc, y_te_e_disc = labtrans.transform(y_te_t, y_te_e)
    time_grid = labtrans.cuts

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class SurvivalDataset(Dataset):
        def __init__(self, X, times, events):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.times = torch.tensor(times, dtype=torch.long)
            self.events = torch.tensor(events, dtype=torch.long)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.times[idx], self.events[idx]

    train_loader = DataLoader(SurvivalDataset(X_train, y_tr_t_disc, y_tr_e_disc),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(SurvivalDataset(X_val,   y_v_t_disc, y_v_e_disc),
                              batch_size=args.batch_size)
    test_loader  = DataLoader(SurvivalDataset(X_test,  y_te_t_disc, y_te_e_disc),
                              batch_size=args.batch_size)

    # Model setup
    num_features = X_train.shape[1]
    num_events = int(y_tr_e_disc.max())
    num_time_steps = len(time_grid)
    model = BasicDeepHitMLP(num_features, num_events, num_time_steps).to(device)
    deephit = DeepHit(model, alpha=0.9, sigma=0.1,
                      device=device, duration_index=time_grid)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = deephit.loss

    # Training
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_batch, t_batch, e_batch in train_loader:
            X_batch, t_batch, e_batch = X_batch.to(device), t_batch.to(device), e_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, t_batch, e_batch, compute_rank_matrix(t_batch, e_batch, device))
            loss.backward(); optimizer.step()
        tr_loss = evaluate_loss(model, train_loader, loss_fn, device)
        vl_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history['train'].append(tr_loss); history['val'].append(vl_loss)
        print(f"Epoch {epoch:03d} - train: {tr_loss:.4f}, val: {vl_loss:.4f}")
        if vl_loss < best_val_loss:
            best_val_loss, best_epoch = vl_loss, epoch
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), 'outputs/model.pth')
    print(f"Saved best model from epoch {best_epoch}")

    # Plot loss with Plotly
    df_loss = pd.DataFrame({
        'Epoch': list(range(1, args.epochs+1)),
        'Training': history['train'],
        'Validation': history['val']
    })
    fig_loss = px.line(
        df_loss.melt(id_vars='Epoch', var_name='Set', value_name='Loss'),
        x='Epoch', y='Loss', color='Set',
        title='Training vs Validation Loss'
    )
    fig_loss.update_layout(template='plotly_dark', font=dict(size=14))
    loss_html = 'outputs/figures/loss_curve.html'
    loss_png = 'outputs/figures/loss_curve.png'
    fig_loss.write_html(loss_html)
    fig_loss.write_image(loss_png)

    # Predict CIF
    cif = deephit.predict_cif(X_test, batch_size=args.batch_size, to_cpu=True, numpy=True)
    np.random.seed(1234)
    sample_ids = np.random.choice(cif.shape[2], size=min(3, cif.shape[2]), replace=False)

    for sid in sample_ids:
        df_cif = pd.DataFrame({
            'Time': time_grid,
            **{f'Event {evt+1}': cif[evt,:,sid] for evt in range(num_events)}
        })
        fig = px.line(
            df_cif.melt(id_vars='Time', var_name='Event', value_name='CIF'),
            x='Time', y='CIF', color='Event',
            title=f'CIF Curves for Sample {sid}'
        )
        fig.update_layout(template='plotly_dark', font=dict(size=14))
        html_path = f'outputs/figures/cif_sample{sid}.html'
        png_path = f'outputs/figures/cif_sample{sid}.png'
        fig.write_html(html_path)
        fig.write_image(png_path)

    # Evaluation metrics
    metrics = {'time_dependent_cindex': {}, 'ipcw_cindex': {}, 'brier_score': {}}
    for evt in range(num_events):
        surv_df = pd.DataFrame(1 - cif[evt], index=time_grid)
        evaluator = EvalSurv(surv_df, y_te_t_disc, y_te_e_disc == (evt+1))
        metrics['time_dependent_cindex'][f'event_{evt+1}'] = float(evaluator.concordance_td('antolini'))

    train_struct = Surv.from_arrays(y_tr_e_disc>0, y_tr_t_disc)
    eval_idxs = [int(q * num_time_steps) for q in (0.25, 0.5, 0.75)]
    for evt in range(num_events):
        ipcw = {}
        test_struct = Surv.from_arrays(y_te_e_disc==(evt+1), y_te_t_disc)
        for idx in eval_idxs:
            tau = time_grid[idx]
            c_val = concordance_index_ipcw(train_struct, test_struct, cif[evt, idx, :], tau=tau)[0]
            ipcw[f'time_{tau}'] = float(c_val)
        metrics['ipcw_cindex'][f'event_{evt+1}'] = ipcw

    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(y_tr_t_disc, event_observed=(y_tr_e_disc==0))
    for evt in range(num_events):
        brier = {}
        for idx in eval_idxs:
            tau = time_grid[idx]
            brier[f'time_{tau}'] = float(compute_brier_score(cif[evt, idx, :], kmf, y_te_t_disc, y_te_e_disc, event_id=evt+1, horizon=tau))
        metrics['brier_score'][f'event_{evt+1}'] = brier

    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to outputs/metrics.json")

if __name__ == '__main__':
    main()
