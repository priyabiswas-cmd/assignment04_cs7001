#!/usr/bin/env python3
"""
Train a camera-conditioned behavioural-cloning policy.

    python3 scripts/bc_dataset_camera.py --data-dir data_camera --out dataset_camera.npz
    python3 scripts/bc_train_camera.py  --dataset dataset_camera.npz --out policy_cam.pt

A small CNN (3 conv layers) encodes the image, an MLP head maps that to
(v, w) or a (chunk, 2) action sequence. This is deliberately small: with a
few thousand frames there is nothing for a large network to gain, and a
small one trains in minutes on CPU.

Images are normalised INSIDE the model (pixels/255, then the per-channel
mean/std saved by bc_dataset_camera.py) so the checkpoint is
self-contained - the policy node only needs this one file plus the raw
uint8 image from the camera.

Same caution as the pose-based trainer: validation loss measures
single-step agreement with the expert on states the expert visited. Judge
the policy by closed-loop rollouts, not by this number.
"""

import argparse
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


class CNNPolicy(nn.Module):
    """(B, H, W, 3) uint8-range image in -> (v, w) or (chunk, 2) out."""

    def __init__(self, img_size, img_mean, img_std, chunk=1,
                 channels=(16, 32, 32), fc_hidden=128, dropout=0.0):
        super().__init__()
        self.chunk = chunk
        self.register_buffer('img_mean',
                             torch.tensor(img_mean, dtype=torch.float32)
                             .view(1, 3, 1, 1))
        self.register_buffer('img_std',
                             torch.tensor(img_std, dtype=torch.float32)
                             .view(1, 3, 1, 1))

        conv_layers, c_in = [], 3
        for c_out in channels:
            conv_layers += [nn.Conv2d(c_in, c_out, kernel_size=5 if c_in == 3 else 3,
                                      stride=2),
                            nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*conv_layers)

        # Compute the flattened size with a dummy forward pass rather than
        # hardcoding it - keeps this correct for any --img-size.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        head = [nn.Flatten(), nn.Linear(flat_dim, fc_hidden), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(fc_hidden, 2 * chunk))
        self.head = nn.Sequential(*head)

    def forward(self, img_uint8_like):
        """
        img_uint8_like: (B, H, W, 3), values in [0, 255] as float32.
        Converts to (B, 3, H, W), normalises, runs the network.
        """
        x = img_uint8_like.permute(0, 3, 1, 2) / 255.0
        x = (x - self.img_mean) / self.img_std
        x = self.conv(x)
        y = self.head(x)
        return y.view(-1, self.chunk, 2) if self.chunk > 1 else y


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return d, int(d['chunk']), int(d['img_size'])


def batches(n, batch_size, shuffle, device):
    idx = torch.randperm(n, device=device) if shuffle \
        else torch.arange(n, device=device)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def evaluate(model, imgs, acts, loss_fn, device, batch_size=256):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for idx in batches(len(imgs), batch_size, False, device):
            xb = torch.tensor(imgs[idx.cpu().numpy()], dtype=torch.float32,
                              device=device)
            yb = acts[idx]
            total += loss_fn(model(xb), yb).item() * len(idx)
            n += len(idx)
    return total / max(n, 1)


def per_dim_mae(model, imgs, acts, act_std, chunk, device, batch_size=256):
    model.eval()
    errs = []
    with torch.no_grad():
        for idx in batches(len(imgs), batch_size, False, device):
            xb = torch.tensor(imgs[idx.cpu().numpy()], dtype=torch.float32,
                              device=device)
            pred = model(xb)
            errs.append((pred - acts[idx]).abs().cpu().numpy())
    err = np.concatenate(errs)
    if chunk > 1:
        err = err.reshape(len(err), chunk, 2).mean(axis=1)
    return err.mean(axis=0) * act_std


def train(args):
    d, chunk, img_size = load_dataset(args.dataset)
    dev = torch.device('cuda' if (args.cuda and torch.cuda.is_available())
                       else 'cpu')

    # Images stay as numpy uint8 on the host; only the current batch is
    # moved to a tensor, since the full set as float32 would be 4x the
    # memory for no benefit.
    Ximg_tr, Ytr_np = d['X_train_img'], d['Y_train']
    Ximg_va, Yva_np = d['X_val_img'], d['Y_val']
    Ytr = torch.tensor(Ytr_np, dtype=torch.float32, device=dev)
    Yva = torch.tensor(Yva_np, dtype=torch.float32, device=dev)

    model = CNNPolicy(img_size, d['img_mean'], d['img_std'], chunk=chunk,
                      dropout=args.dropout).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(3, args.patience // 3))
    loss_fn = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()

    n = len(Ximg_tr)
    best, best_state, bad = float('inf'), None, 0

    print(f"\ndevice {dev} | img {img_size}x{img_size} | chunk {chunk} | "
          f"loss {args.loss}")
    print(f"train {n} frames, val {len(Ximg_va)}\n")
    print(f"{'epoch':>6} {'train':>10} {'val':>10} {'lr':>9}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for idx in batches(n, args.batch, True, dev):
            idx_np = idx.cpu().numpy()
            xb = torch.tensor(Ximg_tr[idx_np], dtype=torch.float32,
                              device=dev)
            yb = Ytr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(idx)

        tr = running / n
        va = evaluate(model, Ximg_va, Yva, loss_fn, dev)
        sched.step(va)

        if va < best - 1e-6:
            best = va
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % args.log_every == 0 or epoch == 1:
            lr = opt.param_groups[0]['lr']
            print(f"{epoch:6d} {tr:10.5f} {va:10.5f} {lr:9.2e}"
                  f"{' *' if bad == 0 else ''}")

        if bad >= args.patience:
            print(f"\nstopping early at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    act_std = d['act_std']
    mae = per_dim_mae(model, Ximg_va, Yva, act_std, chunk, dev)

    torch.save(dict(
        state_dict=model.state_dict(),
        img_size=img_size, chunk=chunk,
        img_mean=d['img_mean'], img_std=d['img_std'],
        act_mean=d['act_mean'], act_std=d['act_std'],
        val_loss=best,
        val_mae_v=float(mae[0]), val_mae_w=float(mae[1]),
    ), args.out)

    print(f"\nbest val loss   {best:.5f}")
    print(f"val MAE         v {mae[0]:.4f} m/s   w {mae[1]:.4f} rad/s")
    print(f"                (demonstrated v spread {act_std[0]:.3f}, "
          f"w spread {act_std[1]:.3f})")
    print(f"\nsaved {args.out}")
    print("\nThis is a health check, not a park/no-park verdict. "
          "Run a rollout to find out.\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='dataset_camera.npz')
    p.add_argument('--out', default='policy_cam.pt')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--loss', choices=['l1', 'mse'], default='l1')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--cuda', action='store_true')
    return p


if __name__ == '__main__':
    train(build_parser().parse_args())
