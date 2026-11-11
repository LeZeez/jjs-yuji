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
    rec = cfg.record

    # devices are picked in the GUI (same list since it uses the same strict setting)
    mouse = dm.find_mouse(rec.strict_devices)[rec.mouse_idx or 0]
    keyboard = dm.find_keyboard(rec.strict_devices)[rec.keyboard_idx or 0]
    print(f"Mouse: {mouse.name} | Keyboard: {keyboard.name}")

    out_pth = Path(rec.out_prefix).parent
    out_pth.mkdir(parents=True, exist_ok=True)
    frames_recorded = 0

    # for files, next number after the highest rec_N (other files are ignored)
    nums = [file.stem.split("_")[-1] for file in out_pth.iterdir()]
    count = max([int(n) for n in nums if n.isdigit()] + [0]) + 1
    print("Starting with count", count)

    fc = dm.FrameRecorder(mouse, keyboard)
    recorder = cm.DataRecorder(rec.cam_idx, rec.width, rec.height, rec.fps, f"{rec.out_prefix}_{count}.mkv")
    on_status("Get ready. Recording in 5 seconds.")

    if event.wait(5):
        # if it stops mid-wait
        recorder.release()
        fc.stop()
        on_status("Stopped")
        return

    on_status("Recording")

    try:
        with recorder.open_file(), open(f"{rec.out_prefix}_{count}.jsonl", 'w') as f:
            fc.get_frame_controls() # throw away the inputs from the countdown
            while True:
                recorder.write()
                frame_data = fc.get_frame_controls()
                f.write(json.dumps(frame_data) + "\n")
                frames_recorded += 1
                if event.is_set() and frames_recorded % rec.nearest_stop == 0:
                    break
    finally:
        recorder.release()
        fc.stop()
    on_status("Done!")

if __name__ == "__main__":
    event = threading.Event()
    # ctrl + alt + x to stop check and then stop recording when frames recorded % cfg.nearest_stop == 0
    listener =  kb.GlobalHotKeys(
        {'<ctrl>+<alt>+x': lambda: dm.stop_notfy(event, dm.notfy)})
    listener.start()
    record(c.Config.load(), event, dm.notfy)
