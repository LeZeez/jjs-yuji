import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from pathlib import Path
import os
import subprocess
import internal.model_manager as mm


# change as you like - these were my settings

SAVE_PATH = "output_model.pt"

SHIFTAMOUNT = 2
NUM_KEYS = 14 # must match KEYS in device_manager.py (E removed)
NUM_BINS_X = 25 # upper limit, repeats in the quantiles can drop a few
NUM_BINS_Y = 15 # ^^^
VSN_WIDTH = 432 # must match the resize in prepare_data.py
VSN_HEIGHT = 240 # ^^^
HEAD_CHANNELS = 4 # channels the 384 DINO channels are squeezed to before flattening
LYR_DIM = 256 # mLSTM width
BATCH_SEQS = 16 # random windows per step, each from a random file
SEQ_LEN = 896 # frames per window, must be a multiple of 128 (mLSTM chunk size)
LR = 3e-4
T_MAX = 2000
PROPSPTH = "outputs_gen_props.pt" # calculates dataset props once so you don't have to redo it again
DATASETPTH = "outputs_gen_int8/" # quantized dataset folder path containing .pt files
USEMEM = True # can cause OOM if dataset is big (for reference, 1 hour of INT8 at 30fps is around 16gb)


assert 0 < SHIFTAMOUNT < SEQ_LEN, "SHIFTAMOUNT must be greater than 0 and less than SEQ_LEN"
assert SEQ_LEN % 128 == 0, "SEQ_LEN must be a multiple of the mLSTM chunk size (128)"


class JJSDataset(IterableDataset):
    def __init__(self, folder_path: str | Path, num_seqs: int, seq_len: int, in_memory: bool = True):
        self.fldr_pth = Path(folder_path)
        self.files = [
            self.fldr_pth / file
            for file in os.listdir(self.fldr_pth)
        ]
        self.num_seqs = num_seqs
        self.seq_len = seq_len
        self.in_memory = in_memory
        self.loaded_files = []
        self.load_files() # mmap'd when not in memory, so random windows only read what they need

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
        # files too short for one window + shift are skipped
        self.loaded_files = [
            f for f in self.open_file()
            if min(len(f["features_q"]), len(f["keys"]), len(f["mouse"])) >= self.seq_len + SHIFTAMOUNT
        ]

    def random_window(self):
        f = self.loaded_files[np.random.randint(len(self.loaded_files))]
        n = min(len(f["features_q"]), len(f["keys"]), len(f["mouse"]))
        a = np.random.randint(0, n - self.seq_len - SHIFTAMOUNT + 1)
        t = slice(a + SHIFTAMOUNT, a + SHIFTAMOUNT + self.seq_len) # targets are SHIFTAMOUNT frames ahead
        return f["features_q"][a:a + self.seq_len], f["features_scale"][a:a + self.seq_len], f["keys"][t], f["mouse"][t]

    def __iter__(self):
        while True:
            windows = [self.random_window() for _ in range(self.num_seqs)]
            # features (B, T, 384, H, W) int8, scale (B, T, 384), keys (B, T, keys), mouse (B, T, 2)
            yield tuple(torch.stack(x) for x in zip(*windows))

# quantile bins instead of mu-law
def get_bins(num_bins, values):
    split = (num_bins - 1) // 2                      # bins per side
    values = values.float()
    q = torch.linspace(0, 1, split + 1)[1:-1]        # split - 1 cuts per side

    # borders sit at .5 so a whole-number move never lands on one
    pos = torch.quantile(values[values > 0], q).round() + 0.5
    neg = torch.quantile(-values[values < 0], q).round() + 0.5
    pos = pos[pos > 0.5].unique()                    # drop repeats and copies of the zero border
    neg = neg[neg > 0.5].unique()

    edges = torch.cat([-neg.flip(0), torch.tensor([-0.5, 0.5]), pos])
    ix = torch.bucketize(values, edges)
    bins = torch.stack([values[ix == i].median() for i in range(len(edges) + 1)])  # value played per bin
    return edges, bins


dataset = JJSDataset(DATASETPTH, BATCH_SEQS, SEQ_LEN, USEMEM)
dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=0)

n = 0
k_counts_per_key = torch.zeros(NUM_KEYS, 2)
m_x_all = []
m_y_all = []

if not Path(PROPSPTH).exists():
    for f in dataset.files:
        f = torch.load(f, map_location="cpu", mmap=True)
        k = f["keys"][:-SHIFTAMOUNT].clone()
        is_0 = (k == 0).float().sum(0)
        is_1 = (k == 1).float().sum(0)
        k_counts_per_key += torch.stack([is_0, is_1], dim=1)

        m = f["mouse"][:-SHIFTAMOUNT].clone()
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

m_x_all = torch.cat(m_x_all)
m_y_all = torch.cat(m_y_all)
x_edges, x_bins = get_bins(NUM_BINS_X, m_x_all)
y_edges, y_bins = get_bins(NUM_BINS_Y, m_y_all)
NUM_BINS_X, NUM_BINS_Y = len(x_bins), len(y_bins) # repeats can drop a bin, the model must match

# sqrt so rare keys are pushed up, but not as hard as the full ratio
key_pos_weight = k_counts_per_key[:, 0] / k_counts_per_key[:, 1]
key_pos_weight = key_pos_weight.masked_fill(k_counts_per_key[:, 1] == 0, 1e-4).sqrt()

key_loss_fn = nn.BCEWithLogitsLoss(pos_weight=key_pos_weight.cuda())
dx_loss_fn = nn.CrossEntropyLoss() # plain, the quantile bins are already balanced
dy_loss_fn = nn.CrossEntropyLoss()


sizes = {"num_keys": NUM_KEYS, "num_bins_x": NUM_BINS_X, "num_bins_y": NUM_BINS_Y,
         "vsn_width": VSN_WIDTH, "vsn_height": VSN_HEIGHT,
         "head_channels": HEAD_CHANNELS, "lyr_dim": LYR_DIM}
model = mm.JJSYuji(**sizes, mode="train")

model.register_buffer("x_edges", x_edges)
model.register_buffer("y_edges", y_edges)
model.register_buffer("x_bins", x_bins)
model.register_buffer("y_bins", y_bins)
model.register_buffer("key_thresh", key_pos_weight / (1 + key_pos_weight))
model = model.to("cuda")

torch.set_printoptions(sci_mode=False)
print("params:", sum(p.numel() for p in model.parameters()))
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

for i, batch in enumerate(dataloader):
    features, scale, target_keys, target_mouse = batch

    # dequantize, each frame with its own scale
    features = features.cuda().float() * scale.cuda().float().view(BATCH_SEQS, SEQ_LEN, 384, 1, 1)

    target_keys = target_keys.float().cuda()
    target_mouse = target_mouse.float().cuda()
    target_x = torch.bucketize(target_mouse[..., 0].reshape(-1).contiguous(), model.x_edges)
    target_y = torch.bucketize(target_mouse[..., 1].reshape(-1).contiguous(), model.y_edges)

    # calc loss
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        (key_preds, mouse_preds), _ = model(features)

        loss_keys = key_loss_fn(key_preds, target_keys)

        mouse_preds = mouse_preds.reshape(-1, NUM_BINS_X + NUM_BINS_Y)
        preds_x = mouse_preds[:, :NUM_BINS_X]
        preds_y = mouse_preds[:, NUM_BINS_X:]
        loss_x = dx_loss_fn(preds_x, target_x)
        loss_y = dy_loss_fn(preds_y, target_y)
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
torch.save({"state_dict": model.state_dict(), "config": sizes}, SAVE_PATH)
print("Training done!")
