import numpy as np
import evdev
from evdev import InputDevice, UInput, ecodes as e

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

# unusued for now
def soft_decode(logits, quantizer, num_bins):
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    soft_index = (probs * np.arange(num_bins)).sum()
    return quantizer.undiscretize(soft_index)

def _get_frame_controls(mouse: InputDevice, keyboard: InputDevice):
    frame_keys = {}
    frame_rel = {}

    try:
        for evt in keyboard.read():
            # keyboard buttons
            if evt.type == e.EV_KEY:
                frame_keys[evt.code] = evt.value
    except BlockingIOError:
        pass
    
    try:
        for evt in mouse.read():
            # mouse buttons
            if evt.type == e.EV_KEY:
                frame_keys[evt.code] = evt.value
            # relative displacement
            elif evt.type == e.EV_REL:
                frame_rel[evt.code] = frame_rel.get(evt.code, 0) + evt.value
    except BlockingIOError:
        pass

    frame_keys = {k:v for k,v in frame_keys.items() if v != 2}

    return {"keys": frame_keys, "mouse": frame_rel}


class FrameRecorder:
    def __init__(self):
        self.done = 0

    def get_frame_controls(self, mouse: InputDevice, keyboard: InputDevice):
        if self.done == 0:
            self.done += 1

            # clear the buffer
            for _ in range(5):
                _get_frame_controls(mouse, keyboard)
            
            return {"keys": {}, "mouse": {}}

        self.done += 1
        return _get_frame_controls(mouse, keyboard)
        



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

def smooth_mouse_exec(ui: UInput, code: int, value: int, precision: int = 5):
    space = np.linspace(10, 20, precision)
    space = space / space.sum()
    values = space * value
    values = np.round(values)
    for i in range(precision):
        ui.write(e.EV_REL, int(code), int(values[i]))
        ui.syn()

def execute_frames(ui: UInput, frame_data: dict, mouse_constant: int | float = 1.0, precision: int = 5):
    # apply displacement
    for code, value in frame_data["mouse"].items():
        smooth_mouse_exec(ui, code, (value * mouse_constant), precision=precision)
    # apply pressed keys
    for code, value in frame_data["keys"].items():
        ui.write(e.EV_KEY, int(code), int(value))