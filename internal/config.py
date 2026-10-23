from dataclasses import dataclass, asdict
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

@dataclass
class Config:
    # record / inference
    cam_idx: int = 2
    width: int = 1280
    height: int = 720
    fps: int = 30
    mouse_idx: int | None = None
    keyboard_idx: int | None = None
    strict_devices: bool = True       # hide extra interfaces (e.g. a mouse's macro keyboard)
                                      # turn off for laptops / combo devices
    out_prefix: str = "recorded/rec"  # folder + file prefix, the number is added automatically (e.g., recorded/rec_1)
    nearest_stop: int = 3584          # recording stops when frames are divisible by this (divisible by 128)
    ckpt_path: str = "model.pt"
    num_keys: int = 15
    num_bins_x: int = 25
    num_bins_y: int = 15
    mouse_temp: float = 1.0
    kbrd_thresh_soften: float = 0.2   # the more it is the more the model is eager to press all buttons
                                      # (1.0 makes it press everything all at once)

    # vision
    vsn_width: int = 432
    vsn_height: int = 240

    def save(self, path=CONFIG_PATH):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=CONFIG_PATH):
        if not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})  # ignore removed keys