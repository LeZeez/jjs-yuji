import torch
import timm
import os
import time
import cv2
import torch.nn.functional as F
from plyer import notification
from torchvision.transforms import v2
from pynput import keyboard as kb
from internal.capture_manager import DataRecorder
from internal.device_manager import *
import internal.model_manager as mm



NUM_BINS_X = 25
NUM_BINS_Y = 15
NUM_KEYS = 15
CAM_IDX = 2 # OBS camera device with roblox on
CKPT = "model.pt"
MOUSE_TEMP = 1.0
KBRD_THRESH_SOFTEN = 0.2 # the more it is the more the model is
                         # eager to press all buttons (1.0 makes it press everything all at once)
@torch.no_grad()

def get_controls(frame, model, state=None):
    
    key_preds_raw, mouse_preds, state = model(frame, state)
    mem_keys = F.sigmoid(key_preds_raw)
    mem_keys = mem_keys > (model.key_thresh - KBRD_THRESH_SOFTEN)
    mem_keys = mem_keys.long().tolist()[0][0]

    mouse_preds = mouse_preds.reshape([NUM_BINS_X + NUM_BINS_Y])
    preds_x = mouse_preds[..., :NUM_BINS_X] / MOUSE_TEMP
    preds_y = mouse_preds[..., NUM_BINS_X:] / MOUSE_TEMP

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

vision = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',
    pretrained=True,
    features_only=True,
    out_indices=(-1,)
)
vision = vision.eval().to("cuda", dtype=torch.float32)
data_config = timm.data.resolve_model_data_config(vision)

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((240, 432), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=data_config["mean"], std=data_config["std"])
])


# == LOAD CKPT ==
ckpt = torch.load(CKPT)
model = mm.JJSYuji(mode="inference", num_keys=NUM_KEYS,
                   num_bins_x=NUM_BINS_X, num_bins_y=NUM_BINS_Y)
model.register_buffer("x_bins", torch.zeros(NUM_BINS_X))
model.register_buffer("y_bins", torch.zeros(NUM_BINS_Y))
model.register_buffer("key_thresh", torch.zeros(NUM_KEYS))
model.load_state_dict(ckpt)
model = model.to("cuda").eval()
# ================


cam = DataRecorder(CAM_IDX, 1280, 720, 60, "_")
for _ in range(5):
    model(torch.randn([1, 1, 384, 15, 27]).cuda()) #warmup

notification.notify(title="Get ready. Playing in 3 seconds.", timeout=1)
time.sleep(3)
notification.notify(title="Playing...", timeout=1)


# == INFERENCE ==
states = None
froze = cam.read()
framec = 0
with init_input_device("TEST") as ui, torch.no_grad():
    while True:
        frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).unsqueeze(0).cuda()
        frame = vision(frame)[0].unsqueeze(0)
        frame_data, states = get_controls(frame, model, states)
        execute_frames(ui, frame_data, 1, 1)
        ui.syn()
        framec += 1