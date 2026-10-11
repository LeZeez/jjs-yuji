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

NEAREST_STOP = 3584 # stop when frames are divisible by 
                    # this number (1min and divisible by 128 in my case)

FPFX = "recs/rec" # any folder then the prefix of each recorded
                  # video is added automatically (e.g., recs/rec_1, recs/recs_2)
FPS = 60
CAM_IDX = 2 # OBS for me
WIDTH = 1280
HEIGHT = 720


# you can override index for each using `ovrd_idx` instead of asking everytime
mouse = dm.find_mouse()
keyboard = dm.find_keyboard()


@dataclass
class Checker:
    is_done: bool = False

checker = Checker()

def done_true():
    checker.is_done = True

# ctrl + alt + x to stop check and then stop recording when frames recorded % NEAREST_STOP == 0 
listener =  kb.GlobalHotKeys({'<ctrl>+<alt>+x': done_true})
listener.start()


out_pth = Path("/".join(FPFX.split("/")[:-1]))
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
recorder = cm.DataRecorder(CAM_IDX, WIDTH, HEIGHT, FPS, f"{FPFX}_{count}.mkv")
notification.notify(title="Get ready. Recording in 5 seconds.", timeout=1)
time.sleep(5)
notification.notify(title="Recording...", timeout=1)
with recorder.open_file(), open(f"{FPFX}_{count}.jsonl", 'w') as f:
    while True:
        recorder.write()
        frame_data = fc.get_frame_controls(mouse=mouse, keyboard=keyboard)
        f.write(json.dumps(frame_data) + "\n")
        frames_recorded += 1
        if checker.is_done and frames_recorded % NEAREST_STOP == 0:
            break
notification.notify(title="Done!", timeout=1)
recorder.release()
count += 1