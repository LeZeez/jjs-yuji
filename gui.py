import cv2
import queue
import threading
import torch
import customtkinter as ctk
from customtkinter.windows.widgets.scaling import CTkScalingBaseClass
from cv2_enumerate_cameras import enumerate_cameras
from pynput import keyboard as kb
import internal.device_manager as dm
import internal.model_manager as mm
from internal.config import Config, CONFIG_PATH
from recorder import record
from inference import play

RED, DARK_RED = "#b33a3a", "#8f2e2e"
GREEN, ORANGE = "#2fa572", "#d6a02a"

if not CONFIG_PATH.exists():
    Config().save()


def status_color(message):
    if message.startswith("Error"):
        return RED
    if message.startswith(("Get ready", "Loading", "Stopping")):
        return ORANGE
    if message.startswith(("Recording", "Playing", "Loaded")):
        return GREEN
    return "gray"


class FormFrame(ctk.CTkFrame):
    def __init__(self, master, cfg: Config):
        super().__init__(master)
        self.cfg = cfg
        self.entries = []  # (section, field, label, type, text box)
        self.row = 0

    def header(self, text):
        font = ctk.CTkFont(size=16, weight="bold")
        ctk.CTkLabel(self, text=text, font=font).grid(row=self.row, column=0, padx=10, pady=(15, 5), sticky="w")
        self.row += 1

    def note(self, text):
        ctk.CTkLabel(self, text=text, text_color="gray", font=ctk.CTkFont(size=11)).grid(
            row=self.row, column=1, padx=10, pady=(0, 5))
        self.row += 1

    def entry(self, section, name, label, convert):
        ctk.CTkLabel(self, text=label).grid(row=self.row, column=0, padx=10, pady=5, sticky="w")
        entry = ctk.CTkEntry(self, width=260)
        entry.insert(0, str(getattr(getattr(self.cfg, section), name)))
        entry.grid(row=self.row, column=1, padx=10, pady=5)
        self.entries.append((section, name, label, convert, entry))
        self.row += 1

    def menu(self, label):
        ctk.CTkLabel(self, text=label).grid(row=self.row, column=0, padx=10, pady=5, sticky="w")
        menu = ctk.CTkOptionMenu(self, values=[""], width=260, dynamic_resizing=False)
        menu.grid(row=self.row, column=1, padx=10, pady=5)
        self.row += 1
        return menu

    def save_entries(self):
        for section, name, label, convert, entry in self.entries:
            try:
                setattr(getattr(self.cfg, section), name, convert(entry.get()))
            except ValueError:
                return label
        return None


class ModelFrame(FormFrame):
    def __init__(self, master, cfg: Config, app):
        super().__init__(master, cfg)
        self.app = app
        self.shown = None  # last status, to only redraw on change

        self.header("Model")
        self.entry("model", "ckpt_path", "Checkpoint", str)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=self.row, column=0, columnspan=2, pady=10)
        self.row += 1
        self.load_btn = ctk.CTkButton(buttons, text="Load", command=self.load)
        self.reload_btn = ctk.CTkButton(buttons, text="Reload", command=self.reload,
                                        fg_color="transparent", border_width=2)
        self.unload_btn = ctk.CTkButton(buttons, text="Unload", command=self.unload,
                                        fg_color=RED, hover_color=DARK_RED)
        for btn in (self.load_btn, self.reload_btn, self.unload_btn):
            btn.pack(side="left", padx=5)

        self.status = ctk.CTkLabel(self, text="", wraplength=520, font=ctk.CTkFont(size=24, weight="bold"))
        self.status.grid(row=self.row, column=0, columnspan=2, padx=10, pady=20)

        self.poll()

    def load(self):
        bad = self.save_entries()
        if bad is not None:
            self.status.configure(text=f"{bad} is not valid", text_color=RED)
            return
        self.cfg.save()
        self.app.model = mm.ThreadModel(self.cfg.model, "cuda")

    def unload(self):
        self.app.model.stop()
        self.app.model = None
        torch.cuda.empty_cache()

    def reload(self):
        self.unload()
        self.load()

    def changed(self):
        # compared to what the model was loaded with, so editing then reverting shows nothing
        return [label for _, name, label, _, entry in self.entries
                if entry.get() != str(getattr(self.app.model.cfg, name))]

    def poll(self):
        model = self.app.model
        loading = model is not None and not model.ready.is_set()
        if model is None:
            text = "Not loaded"
        elif loading:
            text = "Loading..."
        elif model.load_error is not None:
            text = f"Error: {model.load_error}"
        else:
            s = model.sizes
            text = (f"Loaded: {model.cfg.ckpt_path}\n{s['num_keys']} keys, "
                    f"{s['num_bins_x']}x{s['num_bins_y']} mouse bins, {s['vsn_width']}x{s['vsn_height']} vision")
            if self.changed():
                text += f"\n{', '.join(self.changed())} changed, reload to apply"

        color = ORANGE if "reload to apply" in text else status_color(text)
        playing = isinstance(self.app.running, PlayFrame)
        if (text, playing) != self.shown:
            self.shown = (text, playing)
            self.status.configure(text=text, text_color=color)
            self.load_btn.configure(state="normal" if model is None else "disabled")
            # unload waits for loading to finish, and can't pull the model from under Play
            other = "disabled" if model is None or loading or playing else "normal"
            self.reload_btn.configure(state=other)
            self.unload_btn.configure(state=other)

        self.after(200, self.poll)


class RunFrame(FormFrame):
    # settings on the left, Start/Stop for record() / play() on the right (they run in their own thread)
    SECTION = None  # config section of the tab
    NOTE = None     # shown under Start/Stop

    def __init__(self, master, cfg: Config, app):
        super().__init__(master, cfg)
        self.app = app
        self.section = getattr(cfg, self.SECTION)
        self.thread = None
        self.event = None
        self.messages = queue.Queue()
        self.shown = None

        self.header("Capture")
        self.cam_menu = self.menu("Camera")
        self.entry(self.SECTION, "width", "Width", int)
        self.entry(self.SECTION, "height", "Height", int)
        self.entry(self.SECTION, "fps", "FPS", int)
        self.add_settings()

        self.notify_box = ctk.CTkCheckBox(self, text="Notifications")
        if self.section.notify:
            self.notify_box.select()
        self.notify_box.grid(row=self.row, column=0, padx=10, pady=(15, 5), sticky="w")
        self.row += 1
        self.save_status = ctk.CTkLabel(self, text="")
        self.save_status.grid(row=self.row, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkButton(self, text="Save", command=self.save).grid(row=self.row, column=1, padx=10, pady=10)
        self.row += 1

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=0, column=2, rowspan=self.row, padx=(40, 10))
        buttons = ctk.CTkFrame(controls, fg_color="transparent")
        buttons.pack(pady=10)
        self.start_btn = ctk.CTkButton(buttons, text="Start", command=self.start)
        self.stop_btn = ctk.CTkButton(buttons, text="Stop", command=self.stop, fg_color=RED, hover_color=DARK_RED)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn.pack(side="left", padx=5)
        if self.NOTE:
            ctk.CTkLabel(controls, text=self.NOTE, text_color="gray", font=ctk.CTkFont(size=11)).pack()
        self.status = ctk.CTkLabel(controls, text="Idle", text_color="gray", wraplength=300,
                                   font=ctk.CTkFont(size=24, weight="bold"))
        self.status.pack(pady=20)
        ctk.CTkLabel(controls, text="Ctrl+Alt+X stops it from anywhere", text_color="gray",
                     font=ctk.CTkFont(size=11)).pack()

        self.refresh()
        self.poll()

    def add_settings(self):
        pass

    def work(self, event):
        raise NotImplementedError

    def refresh(self):
        # the number is the real /dev/videoN index, not the list position
        cam_idx = self.section.cam_idx
        names = [f"{cam.index}: {cam.name}" for cam in enumerate_cameras(cv2.CAP_V4L2)]
        current = next((n for n in names if n.split(":")[0] == str(cam_idx)), None)
        if current is None:
            # e.g. OBS virtual camera is off, keep it instead of silently switching
            current = f"{cam_idx}: not connected"
            names.append(current)
        self.cam_menu.configure(values=names)
        self.cam_menu.set(current)

    def save(self):
        bad = self.save_entries()
        if bad is not None:
            self.save_status.configure(text=f"{bad} is not valid")
            return False
        self.section.cam_idx = int(self.cam_menu.get().split(":")[0])
        self.section.notify = bool(self.notify_box.get())
        self.cfg.save()
        self.save_status.configure(text="Saved")
        return True

    def on_status(self, message):
        # called from the worker thread, so the label is only updated in poll()
        self.messages.put(message)
        if self.section.notify:
            dm.notfy(message)

    def run(self):
        try:
            self.work(self.event)
        except Exception as e:
            self.on_status(f"Error: {e}")

    def start(self):
        # run with what's on screen, not the last saved values
        if not self.save():
            return
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.app.running = self
        self.thread.start()

    def stop(self):
        if not self.event.is_set():
            dm.stop_notfy(self.event, self.on_status)

    def poll(self):
        while not self.messages.empty():
            message = self.messages.get()
            self.status.configure(text=message, text_color=status_color(message))

        if self.thread is not None and not self.thread.is_alive():
            self.thread = None
            self.app.running = None

        # one thing at a time, they share the camera
        state = (self.app.running is None, self.thread is not None)
        if state != self.shown:
            self.shown = state
            self.start_btn.configure(state="normal" if state[0] else "disabled")
            self.stop_btn.configure(state="normal" if state[1] else "disabled")

        self.after(100, self.poll)


class RecordFrame(RunFrame):
    SECTION = "record"

    def add_settings(self):
        self.header("Devices")
        self.mouse_menu = self.menu("Mouse")
        self.kybrd_menu = self.menu("Keyboard")
        # strict hides extra interfaces, turn off for laptops / combo devices
        self.strict_box = ctk.CTkCheckBox(self, text="Strict device filter", command=self.refresh)
        if self.section.strict_devices:
            self.strict_box.select()
        self.strict_box.grid(row=self.row, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkButton(self, text="Refresh", command=self.refresh).grid(row=self.row, column=1, padx=10, pady=5)
        self.row += 1
        self.note("Refreshes the camera, mouse and keyboard lists")

        self.header("Output")
        self.entry("record", "out_prefix", "Output prefix", str)
        self.entry("record", "nearest_stop", "Stop multiple", int)

    def refresh(self):
        super().refresh()
        strict = bool(self.strict_box.get())
        self.fill_menu(self.mouse_menu, dm.find_mouse(strict), self.section.mouse_idx)
        self.fill_menu(self.kybrd_menu, dm.find_keyboard(strict), self.section.keyboard_idx)

    def fill_menu(self, menu, devices, idx):
        names = [f"{i}: {dev.name}" for i, dev in enumerate(devices)] or ["None found"]
        menu.configure(values=names)
        menu.set(names[idx] if idx is not None and idx < len(names) else names[0])

    def save(self):
        for menu, field in ((self.mouse_menu, "mouse_idx"), (self.kybrd_menu, "keyboard_idx")):
            if menu.get() != "None found":
                setattr(self.section, field, int(menu.get().split(":")[0]))
        self.section.strict_devices = bool(self.strict_box.get())
        return super().save()

    def work(self, event):
        record(self.cfg, event, self.on_status)


class PlayFrame(RunFrame):
    SECTION = "play"
    NOTE = "Loads the model first if it isn't loaded"

    def add_settings(self):
        ctk.CTkButton(self, text="Refresh", command=self.refresh).grid(row=self.row, column=1, padx=10, pady=5)
        self.row += 1
        self.note("Refreshes the camera list")

        self.header("Controls")
        self.entry("play", "mouse_temp", "Mouse temp", float)
        self.entry("play", "kbrd_thresh_soften", "Key eagerness", float)

    def work(self, event):
        if self.app.model is None:
            self.app.model = mm.ThreadModel(self.cfg.model, "cuda")
        model = self.app.model
        if not model.ready.is_set():
            self.on_status("Loading model...")
            model.ready.wait()
        if model.load_error is not None:
            raise model.load_error
        play(self.cfg, model, event, self.on_status)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("JJS Yuji")
        self.scale = 1.0
        self.resize_job = None
        self.model = None    # mm.ThreadModel, loaded from the Model or Play tab
        self.running = None  # the RecordFrame / PlayFrame that's running

        cfg = Config.load()
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(padx=20, pady=20, fill="both", expand=True)
        RecordFrame(self.tabs.add("Record"), cfg, self).pack(expand=True)
        PlayFrame(self.tabs.add("Play"), cfg, self).pack(expand=True)
        ModelFrame(self.tabs.add("Model"), cfg, self).pack(expand=True)

        self.natural_w, self.natural_h = self.measure()
        self.minsize(self.natural_w, self.natural_h)

        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        # Roblox has focus while recording / playing, so the Stop button can't be clicked
        kb.GlobalHotKeys({'<ctrl>+<alt>+x': self.stop_running}).start()

    def stop_running(self):
        running = self.running
        if running is not None:
            running.stop()

    def on_close(self):
        # wait for the run to end, so a recording finishes its file
        if self.running is None:
            self.destroy()
            return
        self.stop_running()
        self.after(200, self.on_close)

    def measure(self):
        # the window's size only counts the shown tab, so swap its size for the biggest one
        self.update_idletasks()
        tabs = [self.tabs.tab(self.tabs.get(i)) for i in range(self.tabs.len())]
        shown = self.tabs.tab(self.tabs.get())
        w = self.winfo_reqwidth() - shown.winfo_reqwidth() + max(t.winfo_reqwidth() for t in tabs)
        h = self.winfo_reqheight() - shown.winfo_reqheight() + max(t.winfo_reqheight() for t in tabs)
        return w, h

    def _set_scaling(self, new_widget_scaling, new_window_scaling):
        # ctk's version locks min size = max size for 1 s, which hides the maximize button
        CTkScalingBaseClass._set_scaling(self, new_widget_scaling, new_window_scaling)

    def on_resize(self, event):
        if event.widget is not self:
            return  # child widgets fire this too
        # wait until resizing stops
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(150, self.rescale)

    def rescale(self):
        self.resize_job = None
        scale = 1.0
        if self.attributes("-zoomed"):
            scale = round(min(self.winfo_width() / self.natural_w, self.winfo_height() / self.natural_h), 2)
        if scale != self.scale:
            self.scale = scale
            ctk.set_widget_scaling(scale)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    App().mainloop()
