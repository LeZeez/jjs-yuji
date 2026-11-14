import torch
import cv2
import threading
import json
import time
import internal.capture_manager as cm
import internal.device_manager as dm
import internal.sound_manager as sm
from pynput import keyboard as kb
from pathlib import Path
import internal.config as c


def record(cfg: c.Config, event: threading.Event, on_status = sm.terminal_status):
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

    # for files, next number after the highest <prefix>_N (other files are ignored)
    prefix = Path(rec.out_prefix).name + "_"  # "recorded/rec" -> "rec_"
    nums = [file.stem.removeprefix(prefix) for file in out_pth.iterdir() if file.stem.startswith(prefix)]
    count = max([int(n) for n in nums if n.isdigit()] + [0]) + 1
    print("Starting with count", count)

    fc = dm.FrameRecorder(mouse, keyboard)
    recorder = cm.DataRecorder(rec.cam_idx, rec.width, rec.height, rec.fps, f"{rec.out_prefix}_{count}.mkv")

    # the camera can ignore the requested fps, check it before recording
    on_status("Checking camera FPS...")  # no sound, it only takes a second
    seconds = recorder.measure_fps()
    if abs(seconds - 1) > 0.05:
        recorder.release()
        fc.stop()
        raise RuntimeError(f"Camera runs at {rec.fps / seconds:.1f} fps, not {rec.fps}")

    # a tick every second of the countdown
    for seconds_left in range(5, 0, -1):
        on_status(f"Get ready. Recording in {seconds_left}...", sound="countdown")
        if event.wait(1):
            # if it stops mid-wait
            recorder.release()
            fc.stop()
            on_status("Stopped")
            return

    on_status("Recording", sound="start")

    try:
        with recorder.open_file(), open(f"{rec.out_prefix}_{count}.jsonl", 'w') as f:
            fc.get_frame_controls() # throw away the inputs from the countdown
            failed = 0
            start = time.monotonic()
            while True:
                if not recorder.write():
                    # no frame, no line, the inputs stay queued for the next frame
                    failed += 1
                    if failed >= 30:
                        raise RuntimeError("Camera stopped returning frames")
                    continue
                failed = 0
                elapsed = time.monotonic() - start
                frame_data = fc.get_frame_controls()
                frame_data["timestamp"] = round(elapsed, 4) # log timestamp
                f.write(json.dumps(frame_data) + "\n")
                frames_recorded += 1
                if event.is_set() and frames_recorded % rec.nearest_stop == 0:
                    break
    finally:
        recorder.release()
        fc.stop()
    on_status("Done!", sound="done")

if __name__ == "__main__":
    event = threading.Event()
    # ctrl + alt + x to stop check and then stop recording when frames recorded % cfg.nearest_stop == 0
    listener =  kb.GlobalHotKeys(
        {'<ctrl>+<alt>+x': lambda: dm.request_stop(event, sm.terminal_status)})
    listener.start()
    record(c.Config.load(), event, sm.terminal_status)
