from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

# record and play each get their own copy
@dataclass
class CaptureConfig:
    cam_idx: int = 2
    width: int = 1280
    height: int = 720
    fps: int = 30
    notify: bool = True

@dataclass
class RecordConfig(CaptureConfig):
    mouse_idx: int | None = None
    keyboard_idx: int | None = None
    strict_devices: bool = True       # hide extra interfaces (e.g. a mouse's macro keyboard)
                                      # turn off for laptops / combo devices
    out_prefix: str = "recorded/rec"  # folder + file prefix, the number is added automatically (e.g., recorded/rec_1)
    nearest_stop: int = 3584          # recording stops when frames are divisible by this (divisible by 128 for mLSTM kernel)

# read every frame, no reload needed
@dataclass
class PlayConfig(CaptureConfig):
    mouse_temp: float = 1.0
    kbrd_thresh_soften: float = 0.2   # the more it is the more the model is eager to press all buttons
                                      # (1.0 makes it press everything all at once)

# sizes (keys, mouse bins, vision) are read from the checkpoint
@dataclass
class ModelConfig:
    ckpt_path: str = "model.pt"

@dataclass
class Config:
    record: RecordConfig = field(default_factory=RecordConfig)
    play: PlayConfig = field(default_factory=PlayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def save(self, path=CONFIG_PATH):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=CONFIG_PATH):
        if not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        sections = {}
        for section in fields(cls):
            known = {f.name for f in fields(section.type)}
            values = data.get(section.name, {})
            sections[section.name] = section.type(**{k: v for k, v in values.items() if k in known})  # ignore removed keys
        return cls(**sections)
