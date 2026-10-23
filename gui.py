import cv2
import customtkinter as ctk
from cv2_enumerate_cameras import enumerate_cameras
import internal.device_manager as dm
from internal.config import Config, CONFIG_PATH

if not CONFIG_PATH.exists():
    Config().save()

# settings shown as text boxes: (config field, label, type to convert to)
FIELDS = [
    ("width", "Width", int),
    ("height", "Height", int),
    ("fps", "FPS", int),
    ("out_prefix", "Output prefix", str),
    ("nearest_stop", "Stop multiple", int),
    ("ckpt_path", "Checkpoint", str),
    ("mouse_temp", "Mouse temp", float),
    ("kbrd_thresh_soften", "Key eagerness", float),
]


class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master, cfg: Config):
        super().__init__(master)
        self.cfg = cfg
        self.entries = {}

        # one row per field: label on the left, text box on the right
        for row, (name, label, _) in enumerate(FIELDS):
            ctk.CTkLabel(self, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="w")
            entry = ctk.CTkEntry(self, width=220)
            entry.insert(0, str(getattr(cfg, name)))
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.entries[name] = entry

        # devices are picked by name but saved as an index
        row = len(FIELDS)
        self.cam_menu = self.empty_menu(row, "Camera")
        self.mouse_menu = self.empty_menu(row + 1, "Mouse")
        self.kybrd_menu = self.empty_menu(row + 2, "Keyboard")

        # strict hides extra interfaces, turn off for laptops / combo devices
        self.strict_box = ctk.CTkCheckBox(self, text="Strict device filter", command=self.refresh)
        if cfg.strict_devices:
            self.strict_box.select()
        self.strict_box.grid(row=row + 3, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkButton(self, text="Refresh", command=self.refresh).grid(row=row + 3, column=1, padx=10, pady=5)

        self.status = ctk.CTkLabel(self, text="")
        self.status.grid(row=row + 4, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkButton(self, text="Save", command=self.save).grid(row=row + 4, column=1, padx=10, pady=10)

        self.refresh()

    def empty_menu(self, row, label):
        ctk.CTkLabel(self, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        menu = ctk.CTkOptionMenu(self, values=[""], width=220)
        menu.grid(row=row, column=1, padx=10, pady=5)
        return menu

    def refresh(self):
        # cameras: the number is the real /dev/videoN index that cv2.VideoCapture takes, not the list position
        names = [f"{cam.index}: {cam.name}" for cam in enumerate_cameras(cv2.CAP_V4L2)]
        current = next((n for n in names if n.split(":")[0] == str(self.cfg.cam_idx)), None)
        if current is None:
            # e.g. OBS virtual camera is off right now, keep it instead of silently switching
            current = f"{self.cfg.cam_idx}: not connected"
            names.append(current)
        self.cam_menu.configure(values=names)
        self.cam_menu.set(current)

        strict = bool(self.strict_box.get())
        self.fill_menu(self.mouse_menu, dm.find_mouse(strict), self.cfg.mouse_idx)
        self.fill_menu(self.kybrd_menu, dm.find_keyboard(strict), self.cfg.keyboard_idx)

    def fill_menu(self, menu, devices, idx):
        # "0: name" so devices with the same name can still be told apart
        names = [f"{i}: {dev.name}" for i, dev in enumerate(devices)] or ["None found"]
        menu.configure(values=names)
        menu.set(names[idx] if idx is not None and idx < len(names) else names[0])

    def save(self):
        for name, label, convert in FIELDS:
            try:
                setattr(self.cfg, name, convert(self.entries[name].get()))
            except ValueError:
                self.status.configure(text=f"{label} is not valid")
                return

        for menu, field in ((self.cam_menu, "cam_idx"), (self.mouse_menu, "mouse_idx"), (self.kybrd_menu, "keyboard_idx")):
            if menu.get() != "None found":
                setattr(self.cfg, field, int(menu.get().split(":")[0]))
        self.cfg.strict_devices = bool(self.strict_box.get())

        self.cfg.save()
        self.status.configure(text="Saved")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("JJS Yuji")
        SettingsFrame(self, Config.load()).pack(padx=20, pady=20, fill="both", expand=True)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    App().mainloop()
