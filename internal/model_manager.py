
import cv2
import copy
import timm
import queue
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from math import floor
from numpy.typing import NDArray
from typing import Literal, List
from internal.config import ModelConfig, PlayConfig
from torchvision.transforms import v2
from internal.device_manager import KEYS, REL
from xlstm.xlstm_large.model import xLSTMLargeConfig, mLSTMBlock, mLSTMLayerStateType

ModelMode = Literal["inference", "train"]
StateType = List[mLSTMLayerStateType] | None


class JJSYuji(nn.Module):
    def __init__(self, num_keys: int, num_bins_x: int,
                 num_bins_y: int, vsn_width: int = 432, vsn_height: int = 240,
                 mode: ModelMode = "train"):
        super().__init__()
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.conv = nn.Conv2d(384, 64, 1)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.convstr = nn.Conv2d(64, 64, 3, 2, 1)
        # DINO's feature map is vsn size / 16 (patch size), e.g. 240x432 -> 15x27
        h = floor((vsn_height // 16 + 2*1 - 3) / 2) + 1
        w = floor((vsn_width // 16 + 2*1 - 3) / 2) + 1
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

        return (key_preds, mouse_preds), new_states


class Vision(nn.Module):
    def __init__(self, vsn_width: int, vsn_height: int):
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
        v2.Resize((vsn_height, vsn_width), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_config["mean"], std=data_config["std"])
        ])

    def forward(self, frame):
        device = next(self.vision.parameters()).device  # cuda but kept like that for flexibility for later (e.g. cpu mode)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform(frame).unsqueeze(0).to(device)
        frame = self.vision(frame)[0].unsqueeze(0)

        return frame

@torch.no_grad()
def get_controls(model_out, play_cfg: PlayConfig, model: JJSYuji):

    key_preds_raw, mouse_preds = model_out
    mem_keys = F.sigmoid(key_preds_raw)
    mem_keys = mem_keys > (model.key_thresh - play_cfg.kbrd_thresh_soften)
    mem_keys = mem_keys.long().tolist()[0][0]

    mouse_preds = mouse_preds.reshape([model.num_bins_x + model.num_bins_y])
    preds_x = mouse_preds[..., :model.num_bins_x] / play_cfg.mouse_temp
    preds_y = mouse_preds[..., model.num_bins_x:] / play_cfg.mouse_temp

    preds_x = preds_x.argmax(-1)
    preds_y = preds_y.argmax(-1)

    preds_x = model.x_bins[preds_x]
    preds_y = model.y_bins[preds_y]


    key_dct = {k: v for k, v in zip(KEYS, mem_keys)}

    exec_mouse = [preds_x.item(), preds_y.item()]
    mouse_dct = {k: v for k, v in zip(REL, exec_mouse)}
    return {"keys": key_dct, "mouse": mouse_dct}

    
def read_checkpoint(path):
    # returns the weights and the sizes the model was trained with
    ckpt = torch.load(path)
    if "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt["config"]

    # old format (plain state_dict): sizes come from the weight shapes.
    # the vision size can't be recovered from them, so it's the one the original model used
    return ckpt, {
        "num_keys": ckpt["key_layer.weight"].shape[0],
        "num_bins_x": ckpt["x_bins"].shape[0],
        "num_bins_y": ckpt["y_bins"].shape[0],
        "vsn_width": 432,
        "vsn_height": 240,
    }


def load_model(path, mode: ModelMode = "train"):
    state_dict, sizes = read_checkpoint(path)
    model = JJSYuji(**sizes, mode=mode)
    model.register_buffer("x_bins", torch.zeros(sizes["num_bins_x"]))
    model.register_buffer("y_bins", torch.zeros(sizes["num_bins_y"]))
    model.register_buffer("key_thresh", torch.zeros(sizes["num_keys"]))
    model.load_state_dict(state_dict)
    return model, sizes


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
            # hardcoding eval/inference cuz not ideal for training anyway
            model, self.sizes = load_model(cfg.ckpt_path, mode="inference")
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
                    vision_out = vision(frame)
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
        
