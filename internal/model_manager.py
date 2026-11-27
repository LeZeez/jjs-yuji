
import copy
import random
import timm
import queue
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from numpy.typing import NDArray
from internal.config import ModelConfig, PlayConfig
from torchvision.transforms import v2
from evdev import ecodes
from internal.device_manager import KEYS, REL
from internal.frame_editor import draw_crosshair, mask_candy

StateType = tuple[torch.Tensor, torch.Tensor] | None # ConvLSTMYuji's (hidden, cell), None = start fresh

# how play picks keys: the held keys (W A S D) are pressed when their chance is over the checkpoint's
# threshold (key_thresh = real chance 0.5) minus kbrd_thresh_soften, the rest are rolled with their real chance every frame
HELD_KEYS = [ecodes.KEY_W, ecodes.KEY_A, ecodes.KEY_S, ecodes.KEY_D]
ROLLED_KEYS = torch.tensor([key not in HELD_KEYS for key in KEYS]) # (keys,) True = rolled, in the order of KEYS
RANDOM_RESET_CHANCE = 0.001        # per frame (at 30 fps: once every ~33 s on average)

# from https://dev.to/mshojaei77/swiglu-the-ffn-upgrade-i-use-to-get-free-performance-33jc
class SwiGLU(nn.Module):
    """
    A standard SwiGLU FFN implementation.
    Reference: Noam Shazeer's "GLU Variants Improve Transformer"
    (https://arxiv.org/abs/2002.05202)
    """
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        # The SwiGLU paper recommends the hidden dimension be 2/3 of the FFN dimension
        hidden_dim = int(2 * d_ffn / 3)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # First linear projection for the gate, activated by SiLU (Swish)
        gate = F.silu(self.w1(x))
        # Second linear projection for the data
        data = self.w2(x)
        # Element-wise multiplication, followed by the final projection
        return self.w3(gate * data)


class ConvLSTM2D(nn.Module):
    """
    Keras ConvLSTM2D (keras/src/layers/rnn/conv_lstm.py, ConvLSTMCell), channels first, stride 1, no dilation.
    Same math as Keras's ConvLSTMCell.call:
        i = sigmoid(x_i + h_i)    f = sigmoid(x_f + h_f)    o = sigmoid(x_o + h_o)
        c = f * c + i * tanh(x_c + h_c)
        h = o * tanh(c)
    x_* = input conv of the frame ("valid" padding, with bias), h_* = recurrent conv of the last h ("same" padding, no bias).
    Like Keras, each conv holds the 4 gates' kernels stacked (4 * filters outputs, order i, f, c, o).
    Init as Keras: kernel glorot_uniform, recurrent kernel orthogonal, bias zeros with the forget part at 1 (unit_forget_bias).
    Not Keras: one dropout mask shared by the 4 gates (Keras draws one per gate, which needs 4x the memory in training).
    Like Keras, the masks are drawn once per window and kept for every frame.
    """
    def __init__(self, in_channels: int, filters: int, kernel_size: int = 3,
                 dropout: float = 0.0, recurrent_dropout: float = 0.0):
        super().__init__()
        self.filters = filters
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.input_conv = nn.Conv2d(in_channels, 4 * filters, kernel_size, padding="valid")
        self.recurrent_conv = nn.Conv2d(filters, 4 * filters, kernel_size, padding="same", bias=False)
        # PyTorch's fan_in/fan_out for a conv (channels x kernel area) are the same as Keras's
        nn.init.xavier_uniform_(self.input_conv.weight) # = Keras glorot_uniform
        nn.init.orthogonal_(self.recurrent_conv.weight)
        with torch.no_grad():
            self.input_conv.bias.zero_()
            self.input_conv.bias[filters:2 * filters].fill_(1.0) # the forget gate's part

    def dropout_mask(self, rate: float, reference: torch.Tensor):
        # a mask shaped like `reference`: a `rate` fraction zeroed, the rest scaled by 1 / (1 - rate)
        ones = torch.ones_like(reference)
        return F.dropout(ones, rate)

    def forward(self, x: torch.Tensor, state=None):
        # x (B, T, C, H, W) -> out (B, T, filters, H', W'), state (h, c) each (B, filters, H', W'). H' = H - kernel + 1
        # state: (h, c) from the last call, None = start from zeros
        B, T, C, H, W = x.shape
        use_dropout = self.training and 0 < self.dropout < 1
        use_recurrent_dropout = self.training and 0 < self.recurrent_dropout < 1

        if use_dropout:
            mask = self.dropout_mask(self.dropout, x[:, :1]) # (B, 1, C, H, W), the same on every frame
            x = x * mask
        # the input part of the gates doesn't need the memory, so it runs for every frame at once
        frames = x.reshape(B * T, C, H, W)
        x_gates = self.input_conv(frames)                       # (B * T, 4 * filters, H', W')
        x_gates = x_gates.reshape(B, T, *x_gates.shape[1:])
        # split once: indexing [:, t] inside the loop makes the backward pass quadratic in T
        x_gates = x_gates.unbind(dim=1)                         # T x (B, 4 * filters, H', W')

        if state is None:
            zeros = torch.zeros(B, self.filters, *x_gates[0].shape[2:], device=x.device)
            state = (zeros, zeros)
        h, c = state
        if use_recurrent_dropout:
            recurrent_mask = self.dropout_mask(self.recurrent_dropout, h) # (B, filters, H', W'), the same on every frame

        outputs = []
        for t in range(T):
            h_in = h * recurrent_mask if use_recurrent_dropout else h
            gates = x_gates[t] + self.recurrent_conv(h_in)
            x_i, x_f, x_c, x_o = gates.chunk(4, dim=1) # x_* + h_* of each gate
            i = torch.sigmoid(x_i)
            f = torch.sigmoid(x_f)
            c = f * c + i * torch.tanh(x_c)
            o = torch.sigmoid(x_o)
            h = o * torch.tanh(c)
            outputs.append(h)

        out = torch.stack(outputs, dim=1) # (B, T, filters, H', W')
        return out, (h, c)


class ConvLSTMYuji(nn.Module):
    # DINO map -> ConvLSTM2D -> 1x1 conv to squeeze_channels -> flatten -> down_proj -> dropout
    #   -> x + SwiGLU(norm x) -> norm -> key and mouse heads
    # outputs: (key logits (B, T, keys), mouse logits (B, T, bins x + bins y)), state (hidden, cell)
    def __init__(self, num_keys: int, num_bins_x: int, num_bins_y: int,
                 vsn_width: int = 432, vsn_height: int = 240,
                 filters: int = 256, squeeze_channels: int = 4, lyr_dim: int = 512,
                 dropout: float = 0.5, conv_dropout: float = 0.5):
        # filters: ConvLSTM memory channels per position. lyr_dim: down_proj, SwiGLU and heads width
        # dropout: after down_proj. conv_dropout: the ConvLSTM's dropout and recurrent_dropout
        super().__init__()
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.lyr_dim = lyr_dim
        kernel_size = 3
        # DINO's feature map is vsn size / 16 (patch size), e.g. 240x432 -> 15x27, the input conv takes kernel - 1 off
        map_height = vsn_height // 16 - kernel_size + 1
        map_width = vsn_width // 16 - kernel_size + 1
        self.conv_lstm = ConvLSTM2D(384, filters, kernel_size, dropout=conv_dropout, recurrent_dropout=conv_dropout)
        self.squeeze = nn.Conv2d(filters, squeeze_channels, 1)
        self.down_proj = nn.Linear(squeeze_channels * map_height * map_width, lyr_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.swiglu_norm = nn.LayerNorm(lyr_dim)
        self.swiglu = SwiGLU(lyr_dim, 4 * lyr_dim)
        self.last_norm = nn.LayerNorm(lyr_dim)
        self.key_layer = nn.Linear(lyr_dim, num_keys)
        self.mouse_layer = nn.Linear(lyr_dim, num_bins_x + num_bins_y)

    def forward(self, x: torch.Tensor, states: StateType = None):
        # x: DINO map (B, T, 384, 15, 27). states: (hidden, cell) from the last call, None = start fresh
        batch, frames = x.shape[:2]
        x, states = self.conv_lstm(x, states)                  # (B, T, filters, 13, 25)
        frame_maps = x.reshape(batch * frames, *x.shape[2:])
        squeezed = self.squeeze(frame_maps)                     # (B * T, squeeze_channels, 13, 25)
        x = squeezed.reshape(batch, frames, -1)                 # (B, T, squeeze_channels * 13 * 25)
        x = self.down_proj(x)                                   # (B, T, lyr_dim)
        x = self.dropout(x)
        x = x + self.swiglu(self.swiglu_norm(x))
        normed = self.last_norm(x)
        key_preds = self.key_layer(normed)                      # (B, T, keys)
        mouse_preds = self.mouse_layer(normed)                  # (B, T, bins x + bins y)
        return (key_preds, mouse_preds), states


class Vision(nn.Module):
    def __init__(self, vsn_width: int, vsn_height: int):
        super().__init__()
        self.vision = timm.create_model(
            'vit_small_patch16_dinov3.lvd1689m',
            pretrained=True,
        )

        data_config = timm.data.resolve_model_data_config(self.vision)
        self.transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((vsn_height, vsn_width), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_config["mean"], std=data_config["std"])
        ])

    def forward(self, bgr_frames):
        device = next(self.vision.parameters()).device  # cuda but kept like that for flexibility for later (e.g. cpu mode)

        # bgr_frames: (N, H, W, 3)
        bgr_frames = np.array(bgr_frames) # copy, so the marker isn't drawn on the caller's frames
        bgr_frames = draw_crosshair(bgr_frames)
        bgr_frames = mask_candy(bgr_frames)
        rgb_frames = bgr_frames[..., ::-1]
        rgb_frames = np.ascontiguousarray(rgb_frames)

        frames = torch.from_numpy(rgb_frames)
        frames = frames.permute(0, 3, 1, 2) # (N, 3, H, W)
        frames = frames.to(device)
        frames = self.transform(frames)

        features = self.vision.forward_intermediates(frames, indices=[-1], norm=True,
                                                     output_fmt="NCHW", intermediates_only=True)[0]
        return features  # (N, 384, 15, 27)

@torch.no_grad()
def get_controls(model_out, play_cfg: PlayConfig, model: "ConvLSTMYuji"):

    key_preds_raw, mouse_preds = model_out
    key_chance = F.sigmoid(key_preds_raw)
    held = key_chance > (model.key_thresh - play_cfg.kbrd_thresh_soften)
    # training's pos_weight inflates the outputs: odds(output) = w * odds(real), and logit(key_thresh) = log w
    real_chance = F.sigmoid(key_preds_raw - torch.logit(model.key_thresh))
    rolled = torch.bernoulli(real_chance).bool()

    out_keys = torch.where(ROLLED_KEYS.to(held.device), rolled, held)
    out_keys = out_keys.long().tolist()[0][0]

    mouse_preds = mouse_preds.reshape([model.num_bins_x + model.num_bins_y])
    preds_x = mouse_preds[..., :model.num_bins_x] / play_cfg.mouse_temp
    preds_y = mouse_preds[..., model.num_bins_x:] / play_cfg.mouse_temp

    # weighted average of the top bin and its two neighbours
    preds_x = soft_decode(preds_x, model.x_bins, 1)
    preds_y = soft_decode(preds_y, model.y_bins, 1)


    key_dict = {k: v for k, v in zip(KEYS, out_keys)}

    out_mouse = [float(preds_x), float(preds_y)]
    mouse_dict = {k: v for k, v in zip(REL, out_mouse)}
    return {"keys": key_dict, "mouse": mouse_dict}

def get_window(x: torch.Tensor, index: int, radius: int) -> torch.Tensor:
    n = x.size(0)
    window_len = 2 * radius + 1
    
    if window_len > n:
        raise ValueError(f"Window length ({window_len}) cannot exceed tensor size ({n}).")
    
    
    start = index - radius
    
    # clamp bounds so window stays size (2 * radius + 1)
    if start < 0:
        start = 0
    elif start + window_len > n:
        start = n - window_len

    return x[start : start + window_len], torch.arange(start, start + window_len)

def soft_decode(logits, bins, radius):
    probs = F.softmax(logits, -1)
    argidx = probs.argmax(-1)

    if bins[argidx] == 0:
        return 0
    
    window, indices = get_window(probs, argidx, radius)
    values = (bins[indices] * window).sum() / window.sum()
    return values
    
def is_conv_lstm(state_dict):
    # ConvLSTMYuji checkpoints are the only ones with a ConvLSTM2D in them
    return "conv_lstm.input_conv.weight" in state_dict


def read_checkpoint(path):
    # returns the weights and the sizes the model was trained with
    ckpt = torch.load(path)
    if "state_dict" not in ckpt or not is_conv_lstm(ckpt["state_dict"]):
        raise ValueError(f"{path} is not a ConvLSTM checkpoint (the old mLSTM models aren't supported anymore)")
    return ckpt["state_dict"], dict(ckpt["config"])


def load_model(path):
    state_dict, sizes = read_checkpoint(path)
    model = ConvLSTMYuji(**sizes)
    model.register_buffer("x_bins", torch.zeros(sizes["num_bins_x"]))
    model.register_buffer("y_bins", torch.zeros(sizes["num_bins_y"]))
    model.register_buffer("x_edges", torch.zeros(sizes["num_bins_x"] - 1))
    model.register_buffer("y_edges", torch.zeros(sizes["num_bins_y"] - 1))
    model.register_buffer("key_thresh", torch.zeros(sizes["num_keys"]))
    model.load_state_dict(state_dict)
    return model, sizes


# == memory reset in play ==
# The model's memory (states) is cleared now and then at random

def random_reset():
    # -> True on the frames where play should clear the memory
    return random.random() < RANDOM_RESET_CHANCE


class ThreadModel:
    def __init__(self, cfg: ModelConfig,  device: Device):
        self.request_queue = queue.Queue()
        self.device = device
        self.cfg = copy.copy(cfg) # what it was loaded with, the GUI compares it to know if a reload is needed
        self.ready = threading.Event() # set when loading ends (even if it failed)
        self.load_error = None
        self.sizes = None # read from the checkpoint once loaded

        self.thread = threading.Thread(
            target=self._load,
            args=(self.cfg, device),
            daemon=True
        )

        self.thread.start() # forgot to add this and wondered why it didn't work (cost: 5 minutes)


    def _load(self, cfg: ModelConfig, device: Device):

        try:
            # hardcoding eval cuz not ideal for training anyway
            model, self.sizes = load_model(cfg.ckpt_path)
            vision = Vision(self.sizes["vsn_width"], self.sizes["vsn_height"])
            model = model.to(device).eval()
            vision = vision.to(device).eval()
        except Exception as e:
            self.load_error = e # e.g. wrong ckpt_path, sent back on every request instead of hanging
        finally:
            self.ready.set()

        while True:
            item = self.request_queue.get()
            if item is None:  # shutdown cleanly
                self.request_queue.task_done()
                break

            frame, states, play_cfg, result_queue = item
            try:
                if self.load_error is not None:
                    raise self.load_error

                # forward pass
                with torch.no_grad():
                    vision_out = vision(frame[None])     # (1, 384, 15, 27)
                    vision_out = vision_out.unsqueeze(0) # (1, 1, 384, 15, 27), batch and time
                    model_out, states = model(vision_out, states)

                controls = get_controls(model_out, play_cfg, model)
                result_queue.put((controls, states, True, None))
            except Exception as e:
                result_queue.put((None, None, False, e))
            finally:
                self.request_queue.task_done()

    # play_cfg is sent with every frame so mouse temp / key eagerness apply without a reload
    def generate(self, frame: NDArray, states: StateType, play_cfg: PlayConfig):
        result_queue = queue.Queue()
        self.request_queue.put((frame, states, play_cfg, result_queue))
        controls, states, success, excep = result_queue.get()

        if success == False:
            raise excep

        result_queue.task_done()
        return controls, states

    def stop(self):
        self.request_queue.put(None)
        self.thread.join()
        
