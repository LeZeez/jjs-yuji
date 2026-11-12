import threading
import numpy as np
from pynput import keyboard as kb
from internal.capture_manager import DataRecorder
import internal.device_manager as dm
import internal.model_manager as mm
import internal.config as c


def play(cfg: c.Config, model: mm.ThreadModel, event: threading.Event, on_status = dm.notfy):
    cap = cfg.play
    cam = DataRecorder(cap.cam_idx, cap.width, cap.height, cap.fps, "_")
    try:
        for _ in range(5):
            warmup = np.random.randint(0, 255, [cap.height, cap.width, 3], dtype=np.uint8)
            model.generate(warmup, None, cfg.play)

        on_status("Get ready. Playing in 3 seconds.")
        if event.wait(3):
            return
        on_status("Playing...")

        # == INFERENCE ==
        states = None
        with dm.init_input_device("TEST") as ui:
            writer = dm.DeviceWriter(ui, cap.fps)
            try:
                while not event.is_set():
                    frame = cam.read()
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
        {'<ctrl>+<alt>+x': lambda: dm.stop_notfy(event, dm.notfy)})
    listener.start()
    config = c.Config.load()
    model = mm.ThreadModel(config.model, "cuda")
    play(config, model, event, dm.notfy)
