import shutil
import subprocess
from pathlib import Path

SOUNDS_DIR = Path(__file__).parent.parent / "assets" / "sounds"

# the first one found plays the sounds: PipeWire, PulseAudio, then plain ALSA (nearly every Linux has it)
PLAYER = next((player for player in ("pw-play", "paplay", "aplay") if shutil.which(player)), None)
if PLAYER is None:
    print("No sound player found (pw-play, paplay or aplay), sounds are off")


def play_sound(name):
    # doesn't wait for the sound to finish
    if PLAYER is not None:
        subprocess.Popen([PLAYER, str(SOUNDS_DIR / f"{name}.wav")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def terminal_status(message, sound=None):
    # on_status for running recorder.py / inference.py without the GUI
    print(message)
    if sound is not None:
        play_sound(sound)
