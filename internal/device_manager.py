import numpy as np
import evdev
from evdev import InputDevice, UInput, ecodes as e
from plyer import notification
import threading
import queue
import select
import time

def notfy(message):
    notification.notify(title=message, timeout=1)

def stop_notfy(event, on_status):
    event.set()
    on_status("Stopping...")

# FINDING DEVICES

ALPHABET_KEYS = {getattr(e, f"KEY_{c}") for c in "ABCDEFGHIJKLMNOPQRSTUVWXY" + "Z"}
CORE_KEYS = ALPHABET_KEYS | {e.KEY_SPACE, e.KEY_ENTER}

def find_keyboard(strict=True):
    devices = [InputDevice(path) for path in evdev.list_devices()]
    kybrds = []
    for dev in devices:
        # check key capabilities
        caps = dev.capabilities()

        keys = set(caps.get(e.EV_KEY, []))
        has_core = CORE_KEYS.issubset(keys)
        if strict == True:
            has_rel = e.EV_REL in caps and e.REL_X in caps[e.EV_REL] # check if it has any REL caps
            has_abs_axes = e.EV_ABS in caps
            if has_rel or has_abs_axes: 
                continue # skip the device
            if any(term in dev.name.lower() for term in ("mouse",)):
                continue

        if has_core: # check if it has alphabet
            kybrds.append(dev)

    return kybrds


def find_mouse(strict=True):
    devices = [InputDevice(path) for path in evdev.list_devices()]
    mice = []
    for dev in devices:
        # check if the device has relative displacement
        caps = dev.capabilities()

        if strict == True:
            keys = set(caps.get(e.EV_KEY, []))
            has_core = CORE_KEYS.issubset(keys) # check if it has any keyboard keys
            if has_core: 
                continue # skip the device
            if any(term in dev.name.lower() for term in ("kbd", "keyboard")):
                continue


        has_rel = e.EV_REL in caps and e.REL_X in caps[e.EV_REL] # check if it has any REL caps
        if has_rel:
            mice.append(dev)

    return mice

# RECORDING / READING

class DeviceReader:
    def __init__(self, device: InputDevice):
        self.device = device
        self.q = queue.Queue()
        self.thread = threading.Thread(
            target=self.run,
            args=(self.device,),
            daemon=True
        )

        self.thread.start()
    def run(self, device: InputDevice):
        while True:
            try:
                # read loop with timeout instead of device.read_loop(), which doesn't have timeout
                ready, _, _ = select.select([device.fd], [], [], 0.1) 
                if ready:
                    for evts in device.read():
                        self.q.put(evts)
            except (OSError, ValueError):
                break

    def read(self):
        items = []
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        return items

    def stop(self):
        self.device.close()
        self.thread.join()

def _get_frame_controls(mouse: DeviceReader, keyboard: DeviceReader):
    frame_keys = {}
    frame_rel = {}

    keyboard_evts = keyboard.read()
    mouse_evts = mouse.read()
    
    for evt in keyboard_evts:
        # keyboard buttons
        if evt.type == e.EV_KEY:
            frame_keys[evt.code] = evt.value

    for evt in mouse_evts:
        # mouse buttons
        if evt.type == e.EV_KEY:
            frame_keys[evt.code] = evt.value
        # relative displacement
        elif evt.type == e.EV_REL:
            frame_rel[evt.code] = frame_rel.get(evt.code, 0) + evt.value

    frame_keys = {k:v for k,v in frame_keys.items() if v != 2}

    return {"keys": frame_keys, "mouse": frame_rel}


class FrameRecorder:
    def __init__(self, mouse: InputDevice, keyboard: InputDevice):
        self.mouse = DeviceReader(mouse)
        self.keyboard = DeviceReader(keyboard)

    def get_frame_controls(self):
        return _get_frame_controls(self.mouse, self.keyboard)

    def stop(self):
        self.mouse.stop()
        self.keyboard.stop()
        



# EXECUTION

# all mouse and keyboard keys + mouse deltas
CAPABILITIES = {
    e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
    e.EV_KEY: list(e.keys.keys()) + list(e.BTN.keys())
}

# all keyboard keys, in order
KEYS = [
    e.KEY_W,
    e.KEY_A,
    e.KEY_S,
    e.KEY_D,
    e.KEY_F,
    e.KEY_G,
    e.KEY_Q,
    e.KEY_E,
    e.KEY_R,
    e.KEY_1,
    e.KEY_2,
    e.KEY_3,
    e.KEY_4,
    e.KEY_SPACE,
    e.BTN_LEFT,
]

# mouse deltas
REL = [
    e.REL_X,
    e.REL_Y,
]

def init_input_device(name: str):
    return UInput(CAPABILITIES, name=name)

def split_move(value, precision):
    weights = np.linspace(0, 1, precision)
    totals = np.cumsum(weights) / weights.sum() * value   # running total after each part
    totals = np.round(totals)
    return np.diff(totals, prepend=0)

def write_keys(ui, keys):
    for code, value in keys.items():
        ui.write(e.EV_KEY, int(code), int(value))
    ui.syn()

def write_mouse_smooth(ui, mouse, fps, precision=5):
    xs = split_move(mouse.get(str(e.REL_X), 0), precision)
    ys = split_move(mouse.get(str(e.REL_Y), 0), precision)
    for i in range(precision):
        ui.write(e.EV_REL, e.REL_X, int(xs[i]))
        ui.write(e.EV_REL, e.REL_Y, int(ys[i]))
        ui.syn()
        time.sleep(0.9 / fps / precision) # 90% of a frame, so it's done before the next frame arrives

# writes keys + mouse in its own thread, because the mouse smoothing sleeps and would slow the play loop
class DeviceWriter:
    def __init__(self, ui, fps, precision=5):
        self.ui = ui
        self.fps = fps
        self.precision = precision
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            frame_data = self.q.get()
            if frame_data is None:     # stop signal
                break
            write_keys(self.ui, frame_data["keys"])
            mouse = {str(code): value for code, value in frame_data["mouse"].items()} # the model gives int keys, jsonl gives str
            write_mouse_smooth(self.ui, mouse, self.fps, self.precision)

    def write(self, frame_data):
        self.q.put(frame_data)

    def stop(self):
        self.q.put(None)
        self.thread.join()