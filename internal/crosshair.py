# crosshair is in code so you can change it anytime without re-recording
import numpy as np

MAGENTA = (255, 0, 255)  # the same in BGR and RGB


def draw_crosshair(frames: np.ndarray) -> np.ndarray:
    # the game's "v" crosshair in magenta with a black outline, so it shows on any background
    # frames: (H, W, 3) or (N, H, W, 3), drawn in place
    height, width = frames.shape[-3:-1]
    top = height // 2 - 1   # top row of the "v" (row 119 at 240p)
    left = width // 2 - 2   # left column of the "v" (column 214 at 432 wide)

    # outline
    frames[..., top - 1:top + 2, left - 1:left + 5, :] = 0
    frames[..., top + 2, left:left + 4, :] = 0

    # the "v"
    frames[..., top, left:left + 4, :] = MAGENTA
    frames[..., top + 1, left + 1:left + 3, :] = MAGENTA
    return frames
