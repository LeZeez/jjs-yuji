import torch
import torch.nn as nn
from pathlib import Path
from natsort import natsorted
import internal.model_manager as mm
import internal.mu as mu

CKPT_PATH = "output_model.pt"
VAL_PATH = "outputs_gen_int8_val/" # made by int8_quantize.py
PROPSPTH = "outputs_gen_props.pt"  # made by train.py, for the same loss weights
SHIFTAMOUNT = 2  # should match train.py
BATCH_SPLITS = 4 # ^^^

model, sizes = mm.load_model(CKPT_PATH, mode="train")
model = model.cuda().eval()
nx, ny = sizes["num_bins_x"], sizes["num_bins_y"]
feat_h, feat_w = sizes["vsn_height"] // 16, sizes["vsn_width"] // 16

# == loss weights, same as train.py ==
props = torch.load(PROPSPTH)
k_count = props["k_count"]
key_pos_weight = (k_count[:, 0] / k_count[:, 1]).masked_fill(k_count[:, 1] == 0, 1e-4)

def bin_weight(values, bins, num_bins):
    count = mu.val_to_ix(torch.stack(values).reshape(-1), bins).bincount(minlength=num_bins).clamp(min=1)
    return count.sum() / (count * num_bins)

x_weight = bin_weight(props["m_x_all"], model.x_bins.cpu(), nx)
y_weight = bin_weight(props["m_y_all"], model.y_bins.cpu(), ny)
key_loss_fn = nn.BCEWithLogitsLoss(pos_weight=key_pos_weight.cuda())
dx_loss_fn = nn.CrossEntropyLoss(weight=x_weight.cuda())
dy_loss_fn = nn.CrossEntropyLoss(weight=y_weight.cuda())

# == model vs baselines ==
# copy current: the action 2 frames from now = the action now (pure inertia)
# idle: no keys, no movement
stats = {name: {"tp": 0, "fp": 0, "fn": 0, "x_hit": 0, "y_hit": 0, "x_err": 0.0, "y_err": 0.0, "n": 0}
         for name in ("model", "copy current", "idle")}

def add(name, keys, x_ix, y_ix, true_keys, true_x, true_y):
    s = stats[name]
    s["tp"] += (keys & true_keys).sum().item()   # pressed, and should be
    s["fp"] += (keys & ~true_keys).sum().item()  # pressed, shouldn't be
    s["fn"] += (~keys & true_keys).sum().item()  # missed a press
    s["x_hit"] += (x_ix == true_x).sum().item()
    s["y_hit"] += (y_ix == true_y).sum().item()
    # error in actual mouse movement, not in bins
    s["x_err"] += (model.x_bins[x_ix] - model.x_bins[true_x]).abs().sum().item()
    s["y_err"] += (model.y_bins[y_ix] - model.y_bins[true_y]).abs().sum().item()
    s["n"] += len(true_x)


files = natsorted(Path(VAL_PATH).iterdir())
losses = torch.zeros(3) # keys, x, y
for file in files:
    f = torch.load(file, map_location="cpu")
    features = f["features_q"].cuda().float() * f["features_scale"].cuda().view(1, -1, 1, 1) # dequantize
    features = features.reshape(BATCH_SPLITS, -1, features.shape[1], feat_h, feat_w)
    keys = f["keys"].cuda().bool()
    mouse = f["mouse"].cuda().float()
    x_ix = mu.val_to_ix(mouse[:, 0], model.x_bins)
    y_ix = mu.val_to_ix(mouse[:, 1], model.y_bins)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        (key_preds, mouse_preds), _ = model(features)
    key_preds = key_preds.reshape(len(keys), -1).float()
    mouse_preds = mouse_preds.reshape(len(keys), -1).float()

    # frame t predicts the action at t + SHIFTAMOUNT (the last SHIFTAMOUNT frames have no target, like the mask in train.py)
    now, future = slice(None, -SHIFTAMOUNT), slice(SHIFTAMOUNT, None)
    true_keys, true_x, true_y = keys[future], x_ix[future], y_ix[future]

    losses += torch.tensor([key_loss_fn(key_preds[now], true_keys.float()).item(),
                            dx_loss_fn(mouse_preds[now, :nx], true_x).item(),
                            dy_loss_fn(mouse_preds[now, nx:], true_y).item()])

    add("model", torch.sigmoid(key_preds[now]) > model.key_thresh,
        mouse_preds[now, :nx].argmax(-1), mouse_preds[now, nx:].argmax(-1),
        true_keys, true_x, true_y)
    add("copy current", keys[now], x_ix[now], y_ix[now], true_keys, true_x, true_y)
    n = len(true_x)
    still_x = mu.val_to_ix(torch.zeros(1, device="cuda"), model.x_bins).expand(n)
    still_y = mu.val_to_ix(torch.zeros(1, device="cuda"), model.y_bins).expand(n)
    add("idle", torch.zeros_like(true_keys), still_x, still_y, true_keys, true_x, true_y)


k, x, y = (losses / len(files)).tolist()
print(f"{len(files)} files | {k + x + y:.4f} | ({k:.4f}, {x:.4f}, {y:.4f})") # same format as train.py

print(f"\n{'':14s}{'key F1':>8s}{'x bin acc':>11s}{'y bin acc':>11s}{'x err':>8s}{'y err':>8s}")
for name, s in stats.items():
    f1 = 2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1)
    print(f"{name:14s}{f1:8.3f}{s['x_hit'] / s['n']:11.3f}{s['y_hit'] / s['n']:11.3f}"
          f"{s['x_err'] / s['n']:8.2f}{s['y_err'] / s['n']:8.2f}")
