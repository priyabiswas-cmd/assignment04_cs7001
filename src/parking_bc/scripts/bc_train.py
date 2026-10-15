#!/usr/bin/env python3
"""
Train a behavioural-cloning policy on the recorded demonstrations.

    python3 scripts/bc_dataset.py --data-dir data --out dataset.npz
    python3 scripts/bc_train.py  --dataset dataset.npz --out policy.pt

The model is a small MLP.  With 4 inputs and a few thousand samples there
is nothing for a larger architecture to buy; get this baseline working and
measure it before reaching for anything else.

The checkpoint stores the normalisation statistics alongside the weights,
so the policy node needs nothing but this one file.

A NOTE ON VALIDATION LOSS.  Low validation MSE does not mean the policy
parks.  Offline loss measures single-step agreement with the demonstrator
on states the demonstrator visited; at run time the policy visits its own
states, and small errors compound.  Treat the loss as a training-health
check, and judge the policy by closed-loop rollouts.
"""

import argparse
import json
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit(
        "PyTorch not found. Install the CPU build:\n"
        "    pip3 install torch --index-url "
        "https://download.pytorch.org/whl/cpu")


class MLPPolicy(nn.Module):
    """obs -> action, or obs -> (chunk, 2) when predicting a sequence."""

    def __init__(self, obs_dim, chunk=1, hidden=(256, 256), dropout=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.chunk = chunk

        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2 * chunk))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        if self.chunk > 1:
            return y.view(-1, self.chunk, 2)
        return y


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return d, int(d['chunk']), bool(d['history'])


def evaluate(model, X, Y, loss_fn, batch=4096):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb, yb = X[i:i + batch], Y[i:i + batch]
            total += loss_fn(model(xb), yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def per_dim_mae(model, X, Y, act_std, chunk):
    """Mean absolute error in real units (m/s and rad/s)."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
    err = (pred - Y).abs().cpu().numpy()
    if chunk > 1:
        err = err.reshape(-1, chunk, 2).mean(axis=1)
    return err.mean(axis=0) * act_std


def train(args):
    d, chunk, history = load_dataset(args.dataset)
    dev = torch.device('cuda' if (args.cuda and torch.cuda.is_available())
                       else 'cpu')

    Xtr = torch.tensor(d['X_train'], dtype=torch.float32, device=dev)
    Ytr = torch.tensor(d['Y_train'], dtype=torch.float32, device=dev)
    Xva = torch.tensor(d['X_val'], dtype=torch.float32, device=dev)
    Yva = torch.tensor(d['Y_val'], dtype=torch.float32, device=dev)

    obs_dim = Xtr.shape[1]
    model = MLPPolicy(obs_dim, chunk,
                      hidden=tuple(args.hidden),
                      dropout=args.dropout).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(3, args.patience // 3))
    # L1 is less distracted by the occasional jerky correction in human
    # demonstrations than MSE, which squares those outliers.
    loss_fn = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()

    n = len(Xtr)
    best, best_state, bad = float('inf'), None, 0
    history_log = []

    print(f"\ndevice {dev} | obs {obs_dim} | chunk {chunk} | "
          f"hidden {tuple(args.hidden)} | loss {args.loss}")
    print(f"train {n} samples, val {len(Xva)}\n")
    print(f"{'epoch':>6} {'train':>10} {'val':>10} {'lr':>9}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(idx)

        tr = running / n
        va = evaluate(model, Xva, Yva, loss_fn)
        sched.step(va)
        history_log.append((epoch, tr, va))

        if va < best - 1e-6:
            best = va
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % args.log_every == 0 or epoch == 1:
            lr = opt.param_groups[0]['lr']
            mark = " *" if bad == 0 else ""
            print(f"{epoch:6d} {tr:10.5f} {va:10.5f} {lr:9.2e}{mark}")

        if bad >= args.patience:
            print(f"\nstopping early at epoch {epoch} "
                  f"({args.patience} epochs without improvement)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    act_std = d['act_std']
    mae = per_dim_mae(model, Xva, Yva, act_std, chunk)

    torch.save(dict(
        state_dict=model.state_dict(),
        obs_dim=obs_dim,
        chunk=chunk,
        history=history,
        hidden=list(args.hidden),
        obs_mean=d['obs_mean'], obs_std=d['obs_std'],
        act_mean=d['act_mean'], act_std=d['act_std'],
        val_loss=best,
        val_mae_v=float(mae[0]), val_mae_w=float(mae[1]),
    ), args.out)

    print(f"\nbest val loss   {best:.5f}")
    print(f"val MAE         v {mae[0]:.4f} m/s   w {mae[1]:.4f} rad/s")
    print(f"                (demonstrated v spread {act_std[0]:.3f}, "
          f"w spread {act_std[1]:.3f})")

    ratio_v, ratio_w = mae[0] / act_std[0], mae[1] / act_std[1]
    if max(ratio_v, ratio_w) > 0.6:
        print("\nNOTE: error is a large fraction of the spread in the data - "
              "the policy\n      has not learned much. Usually inconsistent "
              "demonstrations, or too\n      few of them.")

    if args.curve:
        write_curve(history_log, args.curve)
        print(f"\nloss curve -> {args.curve}")

    print(f"\nsaved {args.out}")
    print("\nOffline loss is only a health check. Run the policy in the "
          "simulator\nto find out whether it actually parks.\n")


def write_curve(log, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        with open(os.path.splitext(path)[0] + '.json', 'w') as f:
            json.dump(log, f)
        return
    ep = [r[0] for r in log]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [r[1] for r in log], label='train')
    plt.plot(ep, [r[2] for r in log], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss (normalised units)')
    plt.title('Behavioural cloning training curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='dataset.npz')
    p.add_argument('--out', default='policy.pt')
    p.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--loss', choices=['l1', 'mse'], default='l1')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--curve', default='figures/training_curve.png')
    p.add_argument('--cuda', action='store_true')
    return p


if __name__ == '__main__':
    a = build_parser().parse_args()
    if a.curve:
        os.makedirs(os.path.dirname(a.curve) or '.', exist_ok=True)
    train(a)
