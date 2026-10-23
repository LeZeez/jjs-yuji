import torch
import cv2
import internal.capture_manager as cm
import internal.device_manager as dm
from plyer import notification
from pynput import keyboard as kb
from pathlib import Path
import time
import json
from dataclasses import dataclass
import internal.config as c

cfg = c.Config.load()


# devices are picked in the GUI (same list since it uses the same strict setting)
mouse = dm.find_mouse(cfg.strict_devices)[cfg.mouse_idx or 0]
keyboard = dm.find_keyboard(cfg.strict_devices)[cfg.keyboard_idx or 0]
print(f"Mouse: {mouse.name} | Keyboard: {keyboard.name}")


@dataclass
class Checker:
    is_done: bool = False

checker = Checker()

def done_true():
    checker.is_done = True
    notification.notify(title="Got signal. Wait!", timeout=1)

# ctrl + alt + x to stop check and then stop recording when frames recorded % cfg.nearest_stop == 0
listener =  kb.GlobalHotKeys({'<ctrl>+<alt>+x': done_true})
listener.start()


out_pth = Path("/".join(cfg.out_prefix.split("/")[:-1]))
out_pth.mkdir(parents=True, exist_ok=True)
frames_recorded = 0

# for files
count = max([int(file.stem.split("_")[-1]) for file in out_pth.iterdir()] + [0]) + 1 or 1
print("Starting with count", count)

# I lagged so hard without these
torch.set_num_threads(1)
cv2.setNumThreads(1)
##########################

# RECORDING

##########################
fc = dm.FrameRecorder()
recorder = cm.DataRecorder(cfg.cam_idx, cfg.width, cfg.height, cfg.fps, f"{cfg.out_prefix}_{count}.mkv")
notification.notify(title="Get ready. Recording in 5 seconds.", timeout=1)
time.sleep(5)
notification.notify(title="Recording...", timeout=1)
with recorder.open_file(), open(f"{cfg.out_prefix}_{count}.jsonl", 'w') as f:
    while True:
        recorder.write()
        frame_data = fc.get_frame_controls(mouse=mouse, keyboard=keyboard)
        f.write(json.dumps(frame_data) + "\n")
        frames_recorded += 1
        if checker.is_done and frames_recorded % cfg.nearest_stop == 0:
            break
notification.notify(title="Done!", timeout=1)
recorder.release()
count += 1