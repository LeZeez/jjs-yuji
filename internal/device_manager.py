import attr
import numpy as np
import evdev
from evdev import InputDevice, UInput, ecodes as e

# FINDING DEVICES

def find_keyboard(ovrd_idx: int = None):
    devices = [InputDevice(path) for path in evdev.list_devices()]
    kybrds = []
    for dev in devices:
        # check if the device has Z key hehe
        capabilities = dev.capabilities()
        if e.EV_KEY in capabilities and e.KEY_Z in capabilities[e.EV_KEY]:
            kybrds.append(dev)
    
    if not kybrds:
        return None
    if len(kybrds) == 1:
        return kybrds[0]
    if ovrd_idx is not None:
        print(f"Using index {ovrd_idx}: {kybrds[ovrd_idx].name}")
        return kybrds[ovrd_idx]

    # nice message
    names = "\n".join(  [f"{kybrd.name}: {ix}" for ix, kybrd in enumerate(kybrds)]  )
    message = (f"We found {len(kybrds)} keyboards:\n"
               f"{names}\n"
               "Choose one: ")
    
    kybrds_idx = int(input(message))
    return kybrds[kybrds_idx]

def find_mouse(ovrd_idx: int = None):
    devices = [InputDevice(path) for path in evdev.list_devices()]
    mice = []
    for dev in devices:
        # check if the device has relative displacement
        capabilities = dev.capabilities()
        if e.EV_REL in capabilities and e.REL_X in capabilities[e.EV_REL]:
            mice.append(dev)
    
    if not mice:
        return None
    if len(mice) == 1:
        return mice[0]
    if ovrd_idx is not None:
        print(f"Using index {ovrd_idx}: {mice[ovrd_idx].name}")
        return mice[ovrd_idx]

    # mice message
    names = "\n".join(  [f"{mouse.name}: {idx}" for idx, mouse in enumerate(mice)]  )
    message = (f"We found {len(mice)} mouse devices:\n"
               f"{names}\n"
               "Choose one: ")
    
    mouse_idx = int(input(message))
    return mice[mouse_idx]

# RECORDING / READING

class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"

@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy

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

CAPABILITIES = {
    e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
    e.EV_KEY: list(e.keys.keys()) + list(e.BTN.keys())
}

KEYS = [e.KEY_W,
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
e.BTN_LEFT,]

REL = [e.REL_X,
e.REL_Y,]

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