import torch
import cv2
import threading
import json
import internal.capture_manager as cm
import internal.device_manager as dm
from pynput import keyboard as kb
from pathlib import Path
import internal.config as c


def record(cfg: c.Config, event: threading.Event, on_status = dm.notfy):
    # I lagged so hard without these
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    # devices are picked in the GUI (same list since it uses the same strict setting)
    mouse = dm.find_mouse(cfg.strict_devices)[cfg.mouse_idx or 0]
    keyboard = dm.find_keyboard(cfg.strict_devices)[cfg.keyboard_idx or 0]
    print(f"Mouse: {mouse.name} | Keyboard: {keyboard.name}")

    out_pth = Path("/".join(cfg.out_prefix.split("/")[:-1]))
    out_pth.mkdir(parents=True, exist_ok=True)
    frames_recorded = 0

    # for files
    count = max([int(file.stem.split("_")[-1]) for file in out_pth.iterdir()] + [0]) + 1 or 1
    print("Starting with count", count)

    fc = dm.FrameRecorder()
    recorder = cm.DataRecorder(cfg.cam_idx, cfg.width, cfg.height, cfg.fps, f"{cfg.out_prefix}_{count}.mkv")
    on_status("Get ready. Recording in 5 seconds.")
    if event.wait(5):
        recorder.release()
        return
    on_status("Recording")

    try:
        with recorder.open_file(), open(f"{cfg.out_prefix}_{count}.jsonl", 'w') as f:
            while True:
                recorder.write()
                frame_data = fc.get_frame_controls(mouse=mouse, keyboard=keyboard)
                f.write(json.dumps(frame_data) + "\n")
                frames_recorded += 1
                if event.is_set() and frames_recorded % cfg.nearest_stop == 0:
                    break
    finally:
        recorder.release()
    on_status("Done!")

if __name__ == "__main__":
    event = threading.Event()
    # ctrl + alt + x to stop check and then stop recording when frames recorded % cfg.nearest_stop == 0
    listener =  kb.GlobalHotKeys(
        {'<ctrl>+<alt>+x': lambda: dm.stop_notfy(event, dm.notfy)})
    listener.start()
    record(c.Config.load(), event, dm.notfy)