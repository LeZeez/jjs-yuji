import torch
import os
import time
import cv2
import torch.nn.functional as F
from plyer import notification
from pynput import keyboard as kb
from internal.capture_manager import DataRecorder
from internal.device_manager import *
import internal.model_manager as mm
import internal.config as c

cfg = c.Config().load()

@torch.no_grad()
def get_controls(frame, model, state=None):
    
    key_preds_raw, mouse_preds, state = model(frame, state)
    mem_keys = F.sigmoid(key_preds_raw)
    mem_keys = mem_keys > (model.key_thresh - cfg.kbrd_thresh_soften)
    mem_keys = mem_keys.long().tolist()[0][0]

    mouse_preds = mouse_preds.reshape([cfg.num_bins_x + cfg.num_bins_y])
    preds_x = mouse_preds[..., :cfg.num_bins_x] / cfg.mouse_temp
    preds_y = mouse_preds[..., cfg.num_bins_x:] / cfg.mouse_temp

    preds_x = preds_x.argmax(-1)
    preds_y = preds_y.argmax(-1)

    preds_x = model.x_bins[preds_x]
    preds_y = model.y_bins[preds_y]


    key_dct = {k: v for k, v in zip(KEYS, mem_keys)}

    exec_mouse = [preds_x.item(), preds_y.item()]
    mouse_dct = {k: v for k, v in zip(REL, exec_mouse)}
    return {"keys": key_dct, "mouse": mouse_dct}, state


# use ctrl + alt + x to exit
def on_combination_pressed():
    os._exit(0)
listener =  kb.GlobalHotKeys({'<ctrl>+<alt>+x': on_combination_pressed})
listener.start()

# == LOAD CKPT ==
model, vision = mm.load_model(cfg, mode="inference")
model = model.to("cuda").eval()
vision = vision.to("cuda").eval()
# ================


cam = DataRecorder(cfg.cam_idx, cfg.width, cfg.height, cfg.fps, "_")
for _ in range(5):
    model(torch.randn([1, 1, 384, 15, 27]).cuda()) #warmup

notification.notify(title="Get ready. Playing in 3 seconds.", timeout=1)
time.sleep(3)
notification.notify(title="Playing...", timeout=1)


# == INFERENCE ==
states = None
with init_input_device("TEST") as ui, torch.no_grad():
    while True:
        frame = cam.read()
        frame = vision(frame)
        frame_data, states = get_controls(frame, model, states)
        execute_frames(ui, frame_data, 1, 1)
        ui.syn()
