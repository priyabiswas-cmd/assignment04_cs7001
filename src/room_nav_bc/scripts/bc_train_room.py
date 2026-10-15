#!/usr/bin/env python3
"""
Train a behavioural-cloning policy on camera + odometry room-navigation
demonstrations (bc_dataset_room.py's output).

    python3 bc_dataset_room.py --data-dir data_room_expert --out dataset_room.npz
    python3 bc_train_room.py   --dataset dataset_room.npz  --out policy_room.pt

THE MODEL.  A small CNN encodes the camera frame to an embedding, which is
concatenated with the odometry features and fed through a small MLP head -
unlike every other policy in this project, this one genuinely has to learn
SOMETHING LIKE PERCEPTION from pixels, not just a control law over a
handed-to-it position. Expect it to need more data and more epochs to get
anywhere than the 4-6 input MLPs elsewhere here did.

A NOTE ON WHAT "LOW LOSS" MEANS HERE.  The dataset is a clean 50/50 split
of left-of-island and right-of-island demonstrations, with NOTHING in the
observation that says which way a given episode went (see bc_dataset_room
.py's docstring). A plain MSE regression loss is expected to converge to
predicting something close to the AVERAGE of "turn left" and "turn right"
actions for states near the choice point - i.e. a LOW training loss here
is not evidence of a good policy, it may be exactly the averaging failure
mode this whole exercise exists to demonstrate. Judge it by closed-loop
rollouts (bc_policy_node_room.py), not this number, same as every other
policy in this project.
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


class CNNPolicy(nn.Module):
    """camera frame + odometry -> action, or (chunk, 2) for a sequence."""

    def __init__(self, odom_dim, img_size=64, chunk=1, img_embed=128,
                hidden=(128, 64), dropout=0.0):
        super().__init__()
        self.chunk = chunk
        self.img_size = img_size

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        conv_hw = img_size // 8
        self.img_fc = nn.Sequential(
            nn.Linear(32 * conv_hw * conv_hw, img_embed), nn.ReLU())

        layers, d = [], img_embed + odom_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2 * chunk))
        self.head = nn.Sequential(*layers)

    def forward(self, img, odom):
        """img: (B,3,H,W) float in [0,1]. odom: (B,odom_dim) normalised."""
        z = self.conv(img).flatten(1)
        z = self.img_fc(z)
        z = torch.cat([z, odom], dim=1)
        y = self.head(z)
        return y.view(-1, self.chunk, 2) if self.chunk > 1 else y


def to_img_tensor(images_uint8, device):
    """(N,H,W,3) uint8 -> (N,3,H,W) float32 in [0,1], on `device`."""
    t = torch.as_tensor(images_uint8, dtype=torch.float32, device=device)
    return t.permute(0, 3, 1, 2).contiguous() / 255.0


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return d, int(d['chunk']), bool(d['history']), int(d['img_size'])


def evaluate(model, img_u8, odom, Y, loss_fn, device, batch=256):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(odom), batch):
            xb = to_img_tensor(img_u8[i:i + batch], device)
            ob = odom[i:i + batch]
            yb = Y[i:i + batch]
            total += loss_fn(model(xb, ob), yb).item() * len(ob)
            n += len(ob)
    return total / max(n, 1)


def per_dim_mae(model, img_u8, odom, Y, act_std, chunk, device, batch=256):
    model.eval()
    errs, n = np.zeros(2, np.float64), 0
    with torch.no_grad():
        for i in range(0, len(odom), batch):
            xb = to_img_tensor(img_u8[i:i + batch], device)
            ob = odom[i:i + batch]
            pred = model(xb, ob)
            err = (pred - Y[i:i + batch]).abs().cpu().numpy()
            if chunk > 1:
                err = err.reshape(len(ob), chunk, 2).mean(axis=1)
            errs += err.sum(axis=0)
            n += len(ob)
    return (errs / max(n, 1)) * act_std


def train(args):
    d, chunk, history, img_size = load_dataset(args.dataset)
    dev = torch.device('cuda' if (args.cuda and torch.cuda.is_available())
                       else 'cpu')

    Xtr_img = d['X_train_img']                 # keep as uint8 on host RAM
    Xva_img = d['X_val_img']
    Xtr_odom = torch.tensor(d['X_train_odom'], dtype=torch.float32, device=dev)
    Ytr = torch.tensor(d['Y_train'], dtype=torch.float32, device=dev)
    Xva_odom = torch.tensor(d['X_val_odom'], dtype=torch.float32, device=dev)
    Yva = torch.tensor(d['Y_val'], dtype=torch.float32, device=dev)

    odom_dim = Xtr_odom.shape[1]
    model = CNNPolicy(odom_dim, img_size=img_size, chunk=chunk,
                      img_embed=args.img_embed, hidden=tuple(args.hidden),
                      dropout=args.dropout).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(3, args.patience // 3))
    loss_fn = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()

    n = len(Xtr_odom)
    best, best_state, bad = float('inf'), None, 0
    history_log = []

    print(f"\ndevice {dev} | odom {odom_dim} | img {img_size}x{img_size} | "
          f"chunk {chunk} | hidden {tuple(args.hidden)} | loss {args.loss}")
    print(f"train {n} samples, val {len(Xva_odom)}\n")
    print(f"{'epoch':>6} {'train':>10} {'val':>10} {'lr':>9}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = to_img_tensor(Xtr_img[idx], dev)
            ob = Xtr_odom[idx]
            yb = Ytr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb, ob), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(idx)

        tr = running / n
        va = evaluate(model, Xva_img, Xva_odom, Yva, loss_fn, dev)
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
    mae = per_dim_mae(model, Xva_img, Xva_odom, Yva, act_std, chunk, dev)

    torch.save(dict(
        state_dict=model.state_dict(),
        odom_dim=odom_dim,
        img_size=img_size,
        chunk=chunk,
        history=history,
        img_embed=args.img_embed,
        hidden=list(args.hidden),
        odom_mean=d['odom_mean'], odom_std=d['odom_std'],
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
        print("\nNOTE: error is a large fraction of the spread in the data. "
              "For THIS dataset\n      that is expected, not necessarily a "
              "bug - the left/right choice is not in\n      the "
              "observation, so some irreducible error (the averaging "
              "failure mode)\n      is exactly what this exercise is "
              "testing. See val MAE on w especially.")

    if args.curve:
        write_curve(history_log, args.curve)
        print(f"\nloss curve -> {args.curve}")

    print(f"\nsaved {args.out}")
    print("\nOffline loss is only a health check, more so here than "
          "anywhere else in\nthis project. Run the policy in the "
          "simulator to find out what it actually does\n"
          "(bc_policy_node_room.py) - including watching whether it "
          "drives into the\nisland, which would be the averaging failure "
          "mode working as expected.\n")


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
    plt.title('Room-navigation BC training curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='dataset_room.npz')
    p.add_argument('--out', default='policy_room.pt')
    p.add_argument('--img-embed', type=int, default=128)
    p.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--loss', choices=['l1', 'mse'], default='l1')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--curve', default='figures/training_curve_room.png')
    p.add_argument('--cuda', action='store_true')
    return p


if __name__ == '__main__':
    a = build_parser().parse_args()
    if a.curve:
        os.makedirs(os.path.dirname(a.curve) or '.', exist_ok=True)
    train(a)
