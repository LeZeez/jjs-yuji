import threading
import numpy as np
from pynput import keyboard as kb
from internal.capture_manager import DataRecorder
import internal.device_manager as dm
import internal.model_manager as mm
import internal.sound_manager as sm
import internal.config as c


def play(cfg: c.Config, model: mm.ThreadModel, event: threading.Event, on_status = sm.terminal_status):
    cap = cfg.play
    cam = DataRecorder(cap.cam_idx, cap.width, cap.height, cap.fps, "_")
    try:
        for _ in range(5):
            warmup = np.random.randint(0, 255, [cap.height, cap.width, 3], dtype=np.uint8)
            model.generate(warmup, None, cfg.play)

        # a tick every second of the countdown
        countdown = cap.countdown if cap.use_countdown else 0
        for seconds_left in range(countdown, 0, -1):
            on_status(f"Get ready. Playing in {seconds_left}...", sound="countdown")
            if event.wait(1):
                return
        on_status("Playing...", sound="start")

        # == INFERENCE ==
        states = None
        with dm.init_input_device("TEST") as ui:
            writer = dm.DeviceWriter(ui, cap.fps, cap.mouse_precision)
            try:
                failed = 0
                while not event.is_set():
                    frame = cam.read()
                    if frame is None:
                        # same as the recorder: a few misses are fine, a dead camera is an error
                        failed += 1
                        if failed >= 30:
                            raise RuntimeError("Camera stopped returning frames")
                        continue
                    failed = 0
                    frame_data, states = model.generate(frame, states, cfg.play)
                    writer.write(frame_data)
            finally:
                writer.stop()
    finally:
        cam.release()
        on_status("Stopped")

if __name__ == "__main__":
    event = threading.Event()
    # ctrl + alt + x to stop inference
    listener =  kb.GlobalHotKeys(
        {'<ctrl>+<alt>+x': lambda: dm.request_stop(event, sm.terminal_status)})
    listener.start()
    config = c.Config.load()
    model = mm.ThreadModel(config.model, "cuda")
    play(config, model, event, sm.terminal_status)
