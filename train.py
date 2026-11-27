import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import os
import subprocess
import internal.model_manager as mm
from internal.device_manager import KEYS


# change as you like - these were my settings (the released model)

SAVE_PATH = "output_model.pt"

SHIFTAMOUNT = 2 # frame t predicts the action at t + 2
NUM_KEYS = len(KEYS) # 14, editable from KEYS in internal.device_manager
NUM_BINS_X = 49 # upper limit, repeats in the quantiles can drop a few (my data: 45)
NUM_BINS_Y = 15 # ^^^
VSN_WIDTH = 432 # must match the resize in prepare_data.py
VSN_HEIGHT = 240 # ^^^
FILTERS = 256 # ConvLSTM memory channels per screen position
SQUEEZE_CHANNELS = 4 # channels the ConvLSTM map is squeezed to before flattening
LYR_DIM = 512 # down_proj, SwiGLU and heads width
BATCH_SEQS = 80 # random windows per step, each from a random file
SEQ_LEN = 180 # frames per window, 6 s at 30 fps
LR = 3e-4
T_MAX = 2000
MOUSE_WEIGHTS = "sqrt" # mouse bin weights:
                # "full" = inverse frequency (jittery), "sqrt" = its square root, "none" = plain loss (not recommended)
KEY_POS_WEIGHT_MAX = 50 # cap on the keys' pos_weight (rare keys like G and R are 100-200 uncapped)
PROPSPTH = "data_props.pt" # calculates dataset props
DATASETPTH = "data_int8/" # quantized dataset folder path containing .pt files
USEMEM = True # can cause OOM if dataset is big (for reference, 1 hour of INT8 at 30fps is around 16gb)

assert 0 < SHIFTAMOUNT < SEQ_LEN, "SHIFTAMOUNT must be greater than 0 and less than SEQ_LEN"


class JJSDataset(IterableDataset):
    def __init__(self, folder: str | Path, windows_per_batch: int, window_len: int, in_memory: bool = True):
        self.folder = Path(folder)
        self.file_paths = [
            self.folder / file_name
            for file_name in os.listdir(self.folder)
        ]
        self.windows_per_batch = windows_per_batch
        self.window_len = window_len
        self.in_memory = in_memory
        self.recordings = []
        self.load_recordings() # mmap'd when not in memory, so random windows only read what they need

    def read_recordings(self):
        for file_path in self.file_paths:
            print(file_path)
            try:
                if self.in_memory == True:
                    recording = torch.load(file_path, map_location="cpu")
                    subprocess.run("free -h", shell=True)
                else:
                    recording = torch.load(file_path, map_location="cpu", mmap=True)
            except RuntimeError:
                print(f"Skipping {file_path}...")
                continue
            yield recording

    def load_recordings(self):
        # recordings too short for one window + shift are skipped
        self.recordings = [
            recording for recording in self.read_recordings()
            if frame_count(recording) >= self.window_len + SHIFTAMOUNT
        ]

    def random_window(self):
        recording = self.recordings[torch.randint(len(self.recordings), (1,)).item()]
        max_start = frame_count(recording) - self.window_len - SHIFTAMOUNT
        start = torch.randint(max_start + 1, (1,)).item() # +1 so the last window can be picked too
        inputs = slice(start, start + self.window_len)                                  # frames the model sees
        targets = slice(start + SHIFTAMOUNT, start + SHIFTAMOUNT + self.window_len)   # actions SHIFTAMOUNT frames later
        features = recording["features_q"][inputs]
        scales = recording["features_scale"][inputs]
        keys = recording["keys"][targets]
        mouse = recording["mouse"][targets]
        return features, scales, keys, mouse

    def __iter__(self):
        while True:
            windows = [self.random_window() for _ in range(self.windows_per_batch)]
            # features (B, T, 384, H, W) int8, scales (B, T, 384), keys (B, T, keys), mouse (B, T, 2)
            yield tuple(torch.stack(part) for part in zip(*windows))


def frame_count(recording):
    # features, keys and mouse can differ by a frame at the end, the shortest decides
    return min(len(recording["features_q"]), len(recording["keys"]), len(recording["mouse"]))


# quantile bins instead of mu-law
def get_bins(num_bins, values):
    split = (num_bins - 1) // 2                      # bins per side
    values = values.float()
    linspace = torch.linspace(0, 1, split + 1)[1:-1]        # split - 1 cuts per side

    # borders sit at .5 so a whole-number move never lands on one
    pos = torch.quantile(values[values > 0], linspace).round() + 0.5
    neg = torch.quantile(-values[values < 0], linspace).round() + 0.5
    pos = pos[pos > 0.5].unique()                    # drop repeats and copies of the zero border
    neg = neg[neg > 0.5].unique()

    edges = torch.cat([-neg.flip(0), torch.tensor([-0.5, 0.5]), pos])
    indices = torch.bucketize(values, edges)
    bins = torch.stack([values[indices == i].median() for i in range(len(edges) + 1)])  # value played per bin
    return edges, bins


def class_weight(values, edges, num_bins):
    # inverse frequency: total / (count * bins), so a bin with an average share of the moves gets 1
    counts = torch.bucketize(values.contiguous(), edges).bincount(minlength=num_bins).clamp(min=1).float()
    inverse = counts.sum() / (counts * num_bins)
    if MOUSE_WEIGHTS == "full":
        return inverse
    if MOUSE_WEIGHTS == "sqrt":
        return inverse.sqrt()
    return torch.ones(num_bins)


dataset = JJSDataset(DATASETPTH, BATCH_SEQS, SEQ_LEN, USEMEM)
dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=0)

k_counts_per_key = torch.zeros(NUM_KEYS, 2)
m_x_all = []
m_y_all = []


for file in dataset.file_paths:
    loaded = torch.load(file, map_location="cpu", mmap=True)
    loaded_keys = loaded["keys"][:-SHIFTAMOUNT].clone()
    is_0 = (loaded_keys == 0).float().sum(0)
    is_1 = (loaded_keys == 1).float().sum(0)
    k_counts_per_key += torch.stack([is_0, is_1], dim=1)

    mouse = loaded["mouse"][:-SHIFTAMOUNT].clone()
    mouse_x = mouse[..., 0].reshape(-1)
    mouse_y = mouse[..., 1].reshape(-1)

    m_x_all.append(mouse_x)
    m_y_all.append(mouse_y)
torch.save({ # for validate.py
    "m_x_all":m_x_all, 
    "m_y_all":m_y_all
}, PROPSPTH)

m_x_all = torch.cat(m_x_all)
m_y_all = torch.cat(m_y_all)
x_edges, x_bins = get_bins(NUM_BINS_X, m_x_all)
y_edges, y_bins = get_bins(NUM_BINS_Y, m_y_all)
real_num_bins_x, real_num_bins_y = len(x_bins), len(y_bins) # repeats in the quantiles can drop some bins
x_weight = class_weight(m_x_all, x_edges, real_num_bins_x)
y_weight = class_weight(m_y_all, y_edges, real_num_bins_y)

# full ratio, capped so rare keys don't get pushed too hard
key_pos_weight = k_counts_per_key[:, 0] / k_counts_per_key[:, 1]
key_pos_weight = key_pos_weight.masked_fill(k_counts_per_key[:, 1] == 0, 1e-4).clamp(max=KEY_POS_WEIGHT_MAX)

key_loss_fn = nn.BCEWithLogitsLoss(pos_weight=key_pos_weight.cuda())
dx_loss_fn = nn.CrossEntropyLoss(weight=x_weight.cuda())
dy_loss_fn = nn.CrossEntropyLoss(weight=y_weight.cuda())


sizes = {"num_keys": NUM_KEYS, "num_bins_x": real_num_bins_x, "num_bins_y": real_num_bins_y,
         "vsn_width": VSN_WIDTH, "vsn_height": VSN_HEIGHT,
         "filters": FILTERS, "squeeze_channels": SQUEEZE_CHANNELS, "lyr_dim": LYR_DIM}
model = mm.ConvLSTMYuji(**sizes)

model.register_buffer("x_edges", x_edges)
model.register_buffer("y_edges", y_edges)
model.register_buffer("x_bins", x_bins)
model.register_buffer("y_bins", y_bins)
model.register_buffer("key_thresh", key_pos_weight / (1 + key_pos_weight))
model = model.to("cuda")

torch.set_printoptions(sci_mode=False)
print("params:", sum(p.numel() for p in model.parameters()))
print("x_w:", x_weight)
print("y_w:", y_weight)
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

for batch in dataloader:
    features, scale, target_keys, target_mouse = batch

    # dequantize, each frame with its own scale
    features = features.cuda().to(torch.bfloat16) * scale.cuda().to(torch.bfloat16).view(BATCH_SEQS, SEQ_LEN, 384, 1, 1)

    target_keys = target_keys.float().cuda()
    target_mouse = target_mouse.float().cuda()
    target_x = torch.bucketize(target_mouse[..., 0].reshape(-1).contiguous(), model.x_edges)
    target_y = torch.bucketize(target_mouse[..., 1].reshape(-1).contiguous(), model.y_edges)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        (key_preds, mouse_preds), _ = model(features)

    loss_keys = key_loss_fn(key_preds.float(), target_keys)
    mouse_preds = mouse_preds.float().reshape(-1, real_num_bins_x + real_num_bins_y)
    preds_x = mouse_preds[:, :real_num_bins_x]
    preds_y = mouse_preds[:, real_num_bins_x:]
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
