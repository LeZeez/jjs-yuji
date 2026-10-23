from xlstm.xlstm_large.model import xLSTMLargeConfig, mLSTMBlock, mLSTMLayerStateType
import torch
import torch.nn as nn
from internal.config import Config
from math import floor
from torchvision.transforms import v2
import timm
import cv2
from typing import Literal, List

ModelMode = Literal["inference", "train"]
StateType = List[mLSTMLayerStateType]


class JJSYuji(nn.Module):
    def __init__(self, num_keys: int, num_bins_x: int,
                 num_bins_y: int,  mode: ModelMode = "train"):
        super().__init__()
        self.conv = nn.Conv2d(384, 64, 1)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.convstr = nn.Conv2d(64, 64, 3, 2, 1)
        h = floor((15 + 2*1 - 3) / 2) + 1
        w = floor((27 + 2*1 - 3) / 2) + 1
        self.img_dim = self.convstr.out_channels * h * w
        self.down_proj = nn.Linear(self.img_dim, 1024)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)

        kernel = "native_autograd" if mode == "inference" else "triton_xl_chunk"
        chunk_size = 1 if mode == "inference" else 128
        self.cfg = xLSTMLargeConfig(
            1024, 4, 1, 0, 
            chunk_size=chunk_size,
            chunkwise_kernel=f"chunkwise--{kernel}",
            return_last_states=True
        )
        self.main_layers = nn.ModuleList([mLSTMBlock(self.cfg) for _ in range(2)])
        self.last_norm = nn.LayerNorm(1024)
        self.key_layer = nn.Linear(1024, num_keys)
        self.mouse_layer = nn.Linear(1024, num_bins_x + num_bins_y)

    def forward(self, x: torch.Tensor, states: StateType = None):
        B, T, C, H, W = x.shape

        x = x.reshape([B * T, C, H, W])
        x = self.conv(x)
        x = self.dropout2d(x)
        x = self.convstr(x)
        x = x.reshape([B, T, self.img_dim])
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.down_proj(x)

        if states is None:
            states = [None] * len(self.main_layers)
        new_states = []

        for i, layer in enumerate(self.main_layers):
            x, next_state = layer(x, state=states[i])
            new_states.append(next_state)

        normed = self.last_norm(x)
        key_preds = self.key_layer(normed)
        mouse_preds = self.mouse_layer(normed)

        return key_preds, mouse_preds, new_states


class Vision(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.vision = timm.create_model(
            'vit_small_patch16_dinov3.lvd1689m',
            pretrained=True,
            features_only=True,
            out_indices=(-1,)
        )

        data_config = timm.data.resolve_model_data_config(self.vision)
        self.transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((cfg.vsn_height, cfg.vsn_width), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_config["mean"], std=data_config["std"])
        ])

    def forward(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform(frame).unsqueeze(0).cuda()
        frame = self.vision(frame)[0].unsqueeze(0)

        return frame

    
def load_model(cfg: Config, *args, **kwargs):
    ckpt = torch.load(cfg.ckpt_path)
    vision = Vision(cfg)
    model = JJSYuji(*args, **kwargs,
                    num_keys=cfg.num_keys,
                    num_bins_x=cfg.num_bins_x,
                    num_bins_y=cfg.num_bins_y)
    model.register_buffer("x_bins", torch.zeros(cfg.num_bins_x))
    model.register_buffer("y_bins", torch.zeros(cfg.num_bins_y))
    model.register_buffer("key_thresh", torch.zeros(cfg.num_keys))
    model.load_state_dict(ckpt)
    return model, vision