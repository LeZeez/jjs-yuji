from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

# record and play each get their own copy
@dataclass
class CaptureConfig:
    cam_idx: int = 2
    width: int = 432
    height: int = 240
    fps: int = 30
    sounds: bool = True   # play a sound on countdown, start, stop, done and errors

@dataclass
class RecordConfig(CaptureConfig):
    mouse_idx: int | None = None
    keyboard_idx: int | None = None
    strict_devices: bool = True       # hide extra interfaces (e.g. a mouse's macro keyboard)
                                      # turn off for laptops / combo devices
    out_prefix: str = "recorded/rec"  # folder + file prefix, the number is added automatically (e.g., recorded/rec_1)
    nearest_stop: int = 1          # recording stops when frames are divisible by this
    use_countdown: bool = True
    countdown: int = 3                # seconds before recording starts

# read every frame, no reload needed
@dataclass
class PlayConfig(CaptureConfig):
    mouse_temp: float = 1.0
    kbrd_thresh_soften: float = 0.1   # lowers the threshold of W A S D, the more it is the more eagerly they're pressed
    mouse_precision: int = 5          # each frame's mouse move is split into this many steps over the frame
                                      # (read when play starts, not every frame)
    use_countdown: bool = True
    countdown: int = 3                # seconds before playing starts

# sizes (keys, mouse bins, vision) are read from the checkpoint
@dataclass
class ModelConfig:
    ckpt_path: str = "jjs-yuji-2.pt"

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
