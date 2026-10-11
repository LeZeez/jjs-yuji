import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from pathlib import Path
import os
import subprocess
import internal.model_manager as mm
import internal.mu as mu


# change as you like - these were my settings

SAVE_PATH = "output_model.pt"

SHIFTAMOUNT = 2
NUM_BINS_X = 25
NUM_BINS_Y = 15
MU = 3
NUM_KEYS = 15
BATCH  = 7168
HALF_BATCH = BATCH // 2
LR = 1e-4
T_MAX = 4000
PROPSPTH = "outputs_gen_props.pt" # calculates dataset props once so you don't have to redo it again
DATASETPTH = "outputs_gen_int8/" # dataset folder path containing .pt files
USEMEM = True # can cause OOM if dataset is big

# DATASETPTH_VAL = "outputs_gen_int8_val/" 
# validating, unused

class JJSDataset(IterableDataset):
    def __init__(self, folder_path: str | Path, in_memory: bool = True):
        self.fldr_pth = Path(folder_path)
        self.files = [
            self.fldr_pth / file
            for file in os.listdir(self.fldr_pth)
        ]
        self.in_memory = in_memory
        self.current_index = 0
        self.loaded_files = []

        if self.in_memory:
            self.load_files()
    def open_file(self):
        for file in self.files:
            print(file)
            try:
                if self.in_memory == True:
                    yield torch.load(file, map_location="cpu")
                    subprocess.run("free -h",shell=True)
                    print("\r")
                elif self.in_memory == False:
                    yield torch.load(file, map_location="cpu", mmap=True)
            except RuntimeError:
                print(f"Skipping {file}...")
                continue

    def load_files(self):
        self.loaded_files = list(self.open_file())

    def __iter__(self):
        while True:
            source = (
                self.loaded_files
                if self.in_memory
                else self.open_file()
            )

            for i, loaded in enumerate(source):
                self.current_index = i
                yield (
                    # # uncomment if you use raw features (not recommended)
                    # loaded["features"],

                    # == for quantized features ==
                    loaded["features_q"],
                    loaded["features_scale"],
                    # ============================
                    loaded["keys"],
                    loaded["mouse"],
                )


dataset = JJSDataset(DATASETPTH, USEMEM)
dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=0)

# dataset_val = JJSDataset(DATASETPTH_VAL, USEMEM)
# dataloader_val = DataLoader(dataset=dataset_val, batch_size=None, num_workers=0)
# validating, unused

n = 0
k_counts_per_key = torch.zeros(NUM_KEYS, 2)
m_x_all = []
m_y_all = []

if not Path(PROPSPTH).exists():
    for f in dataset.files:
        f = torch.load(f, map_location="cpu", mmap=True)
        k = f["keys"][:BATCH][:-SHIFTAMOUNT].clone()
        is_0 = (k == 0).float().sum(0)
        is_1 = (k == 1).float().sum(0)
        k_counts_per_key += torch.stack([is_0, is_1], dim=1)

        m = f["mouse"][:BATCH][:-SHIFTAMOUNT].clone()
        m_x = m[..., 0].reshape(-1)
        m_y = m[..., 1].reshape(-1)

        m_x_all.append(m_x)
        m_y_all.append(m_y)
        n += len(k)
    torch.save({
        "n":n,
        "k_count":k_counts_per_key,
        "m_x_all":m_x_all,
        "m_y_all":m_y_all
    }, PROPSPTH)
elif Path(PROPSPTH).exists():
    props = torch.load(PROPSPTH)
    n = props["n"]
    k_counts_per_key = props["k_count"]
    m_x_all = props["m_x_all"]
    m_y_all = props["m_y_all"]

m_x_all = torch.stack(m_x_all).reshape(-1)
m_y_all = torch.stack(m_y_all).reshape(-1)
m_x_all_abs = torch.abs(m_x_all)
m_y_all_abs = torch.abs(m_y_all)
scale_x = torch.quantile(m_x_all_abs.float(), 0.99)
scale_y = torch.quantile(m_y_all_abs.float(), 0.99)
camera_maxval_x = scale_x.item()
camera_maxval_y = scale_y.item()
camera_binsize_x = (2 * camera_maxval_x) / (NUM_BINS_X - 1)
camera_binsize_y = (2 * camera_maxval_y) / (NUM_BINS_Y - 1)

key_pos_weight = k_counts_per_key[:, 0] / k_counts_per_key[:, 1]
key_pos_weight = key_pos_weight.masked_fill(k_counts_per_key[:, 1] == 0, 1e-4)



qx = mu.CameraQuantizer(camera_maxval=camera_maxval_x, camera_binsize=camera_binsize_x,
                     quantization_scheme="mu_law", mu=MU)
qy = mu.CameraQuantizer(camera_maxval=camera_maxval_y, camera_binsize=camera_binsize_y,
                     quantization_scheme="mu_law", mu=MU)

x_bins = torch.tensor(qx.undiscretize(np.arange(NUM_BINS_X)))
x_bins = torch.round(x_bins)
y_bins = torch.tensor(qy.undiscretize(np.arange(NUM_BINS_Y)))
y_bins = torch.round(y_bins)

x_weight = mu.val_to_ix(m_x_all, x_bins).bincount(minlength=NUM_BINS_X).clamp(min=1)
x_weight = (x_weight.sum() / (x_weight * NUM_BINS_X))

y_weight = mu.val_to_ix(m_y_all, y_bins).bincount(minlength=NUM_BINS_Y).clamp(min=1)
y_weight = (y_weight.sum() / (y_weight * NUM_BINS_Y))


key_loss_fn = nn.BCEWithLogitsLoss(pos_weight=key_pos_weight.cuda())
dx_loss_fn = nn.CrossEntropyLoss(weight=x_weight.cuda())
dy_loss_fn = nn.CrossEntropyLoss(weight=y_weight.cuda())


model = mm.JJSYuji(mode="train", num_keys=NUM_KEYS,
                   num_bins_x=NUM_BINS_X, num_bins_y=NUM_BINS_Y)

model.register_buffer("x_bins", x_bins)
model.register_buffer("y_bins", y_bins)
model.register_buffer("key_thresh", key_pos_weight / (1 + key_pos_weight))
model = model.to("cuda")

torch.set_printoptions(sci_mode=False)
print("x_s:", scale_x)
print("y_s:",scale_y)
print("x_w:", x_weight)
print("y_w:",y_weight)
print("pos:", key_pos_weight)
print("x_bins:", model.x_bins)
print("y_bins:", model.y_bins)

optimizer = optim.AdamW(model.parameters(), fused=True, lr=LR)
updates = 0

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=T_MAX, 
    eta_min=(LR / 10)
)


bs = 4 # split individual videos into separate batches to introduce noise
seq_len = (HALF_BATCH) // bs

mouse_size = [bs * seq_len, NUM_BINS_X + NUM_BINS_Y]
key_size = [bs, seq_len, NUM_KEYS]
features_size = [bs, seq_len, 384, 15, 27]

assert 0 <= SHIFTAMOUNT < HALF_BATCH

loss_mask = torch.ones(HALF_BATCH, dtype=torch.bool, device="cuda")
if SHIFTAMOUNT > 0:
    loss_mask[-SHIFTAMOUNT:] = False

loss_mask_key = loss_mask.reshape(bs, seq_len)
loss_mask_mouse = loss_mask.clone()


for i, batch in enumerate(dataloader):
    features, scale, target_keys, target_mouse = batch
    # discard extra feats if exist
    if min(len(features), len(target_keys), len(target_mouse)) < HALF_BATCH:
        print(
            f"Too short: features={len(features)}, "
            f"keys={len(target_keys)}, mouse={len(target_mouse)}"
        )
        continue

    features = features[:HALF_BATCH].cuda()
    scale = scale.cuda().view(1, 384, 1, 1)
    target_keys = target_keys[:HALF_BATCH]
    target_mouse = target_mouse[:HALF_BATCH]

    # apply shift
    if SHIFTAMOUNT > 0:
        target_keys = torch.roll(target_keys, -SHIFTAMOUNT, dims=0)
        target_mouse = torch.roll(target_mouse, -SHIFTAMOUNT, dims=0)

        target_keys[-SHIFTAMOUNT:] = 0
        target_mouse[-SHIFTAMOUNT:] = 0

    features = (features.float() * scale).reshape(features_size) # dequantize

    target_keys = target_keys.reshape(key_size).float().cuda()
    target_mouse = target_mouse.float().cuda()
    target_x = target_mouse[..., 0]
    target_y = target_mouse[..., 1]

    target_x = mu.val_to_ix(target_x, model.x_bins)
    target_y = mu.val_to_ix(target_y, model.y_bins)

    # calc loss
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        key_preds, mouse_preds, _ = model(features)
        key_preds = key_preds.view(key_size)

        loss_keys = key_loss_fn(key_preds[loss_mask_key], target_keys[loss_mask_key])

        mouse_preds = mouse_preds.reshape(mouse_size)
        preds_x = mouse_preds[:, :NUM_BINS_X]
        preds_y = mouse_preds[:, NUM_BINS_X:]
        loss_x = dx_loss_fn(preds_x[loss_mask_mouse], target_x[loss_mask_mouse])
        loss_y = dy_loss_fn(preds_y[loss_mask_mouse], target_y[loss_mask_mouse])
    total_loss = loss_keys + loss_x + loss_y
    if (updates % 10 == 0) or (updates == 0):
        loss = total_loss.item()
        print(f"{updates} | {loss} | ({loss_keys.item():.4f}, {loss_x.item():.4f}, {loss_y.item():.4f}) | {optimizer.param_groups[0]['lr']:.6f}")
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    if updates == T_MAX:
        break
    updates += 1
torch.save(model.state_dict(), SAVE_PATH)
print("Training done!")