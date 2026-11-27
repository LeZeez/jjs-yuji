# edits made to every frame before Vision sees it
# they're in code so we can change them anytime without re-recording
import numpy as np

MAGENTA = (255, 0, 255)  # the same in BGR and RGB

# the Halloween candy counter in the bottom-right corner, as whole patches (16 px each)
# 2 patches wide = columns 400-431 and rows 224-239 at 432x240
CANDY_PATCHES_WIDE = 2
CANDY_PATCHES_TALL = 1
PATCH_SIZE = 16 # DINO patch size


def mask_candy(frames: np.ndarray) -> np.ndarray:
    # black, so the model can't use the candy count as a clock (it keeps going up through a session)
    # whole patches so they look the same every frame, even after the event when the counter is gone
    # frames: (H, W, 3) or (N, H, W, 3), drawn in place
    frames[..., -CANDY_PATCHES_TALL * PATCH_SIZE:, -CANDY_PATCHES_WIDE * PATCH_SIZE:, :] = 0
    return frames


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
