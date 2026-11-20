import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from natsort import natsorted
import internal.model_manager as mm

# == settings ==

CKPT_PATH = "output_model.pt"
VAL_PATH = "outputs_gen_int8_val/"  # made by prepare_data.py
PROPS_PATH = "outputs_gen_props.pt" # made by train.py, for the same key weights and the prior
SHIFTAMOUNT = 2  # must match train.py
SEQ_LEN = 896    # ^^^, each recording is cut into back-to-back windows of this many frames
WINDOWS_PER_BATCH = 16 # windows sent to the gpu at once, lower it if it runs out of memory
KEY_NAMES = "W A S D F G Q E R 1 2 3 4 Space M1".split() # same order as KEYS
MIN_MOVE = (12, 4)    # x, y: smallest move that counts as "I moved"
BIG_TURN = (100, 30)  # x, y: turns over TURN_FRAMES at least this big are used for the turn test
TURN_FRAMES = 8       # ~0.27 s at 30 fps
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


# == part 1: what training knew (key weights and the prior) ==

def log_prior(moves, edges, num_bins):
    # how often each bin happens in the training data = the best guess without looking at the screen
    all_moves = torch.cat(moves).float()
    bins = torch.bucketize(all_moves, edges)
    counts = bins.bincount(minlength=num_bins).clamp(min=1)
    return (counts / counts.sum()).log() # (num_bins,)


def load_training_props(model, num_bins_x, num_bins_y):
    props = torch.load(PROPS_PATH)
    key_counts = props["k_count"] # (keys, 2): frames not held, frames held
    not_held, held = key_counts[:, 0], key_counts[:, 1]
    key_pos_weight = (not_held / held).masked_fill(held == 0, 1e-4).sqrt() # same as train.py
    x_prior = log_prior(props["m_x_all"], model.x_edges.cpu(), num_bins_x)
    y_prior = log_prior(props["m_y_all"], model.y_edges.cpu(), num_bins_y)
    return key_pos_weight, x_prior, y_prior


# == part 2: running the model on one recording ==

def load_recording(path):
    recording = torch.load(path, map_location="cpu")
    usable = len(recording["keys"]) // SEQ_LEN * SEQ_LEN # the end that doesn't fill a window is dropped
    if usable == 0:
        return None
    return {name: value[:usable] for name, value in recording.items()}


@torch.no_grad()
def predict(model, recording, feature_height, feature_width):
    # features (frames, 384, H, W) int8 -> key logits (frames, keys), mouse logits (frames, bins x + bins y)
    features_q = recording["features_q"].reshape(-1, SEQ_LEN, 384, feature_height, feature_width)
    scales = recording["features_scale"].reshape(-1, SEQ_LEN, 384, 1, 1)
    key_logits, mouse_logits = [], []
    for start in range(0, len(features_q), WINDOWS_PER_BATCH):
        windows = features_q[start:start + WINDOWS_PER_BATCH].cuda().float()
        windows = windows * scales[start:start + WINDOWS_PER_BATCH].cuda() # dequantize, a scale per frame
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (window_keys, window_mouse), _ = model(windows) # (windows, SEQ_LEN, ...)
        key_logits.append(window_keys.float().flatten(0, 1).cpu())
        mouse_logits.append(window_mouse.float().flatten(0, 1).cpu())
    return torch.cat(key_logits), torch.cat(mouse_logits)


def biggest_recent_move(mouse):
    # row t = my biggest move (per axis) at t-2, t-1 and t, the 3 frames the model saw
    padded = torch.cat([torch.zeros(2, 2), mouse]) # 2 empty frames before the first one
    return padded.unfold(0, 3, 1).abs().amax(-1) # (frames, 2)


def line_up(recording, key_logits, mouse_logits, num_bins_x):
    # frame t predicts the action at t + SHIFTAMOUNT, so the last SHIFTAMOUNT frames have no target
    inputs = slice(None, -SHIFTAMOUNT)
    targets = slice(SHIFTAMOUNT, None)
    keys_held = recording["keys"].bool()
    mouse = recording["mouse"].float()
    return {
        "key_logits":   key_logits[inputs],
        "x_logits":     mouse_logits[inputs, :num_bins_x],
        "y_logits":     mouse_logits[inputs, num_bins_x:],
        "target_keys":  keys_held[targets],
        "current_keys": keys_held[inputs], # for the "copy current" baseline
        "target_mouse": mouse[targets],
        "recent_mouse": biggest_recent_move(mouse)[inputs], # for the onset test
    }


# == part 3: loss, same as train.py, and mouse vs prior ==

def report_loss(recordings, model, key_pos_weight, x_prior, y_prior):
    key_logits = torch.cat([rec["key_logits"] for rec in recordings])
    x_logits = torch.cat([rec["x_logits"] for rec in recordings])
    y_logits = torch.cat([rec["y_logits"] for rec in recordings])
    target_keys = torch.cat([rec["target_keys"] for rec in recordings]).float()
    target_mouse = torch.cat([rec["target_mouse"] for rec in recordings])
    target_x_bin = torch.bucketize(target_mouse[:, 0].contiguous(), model.x_edges.cpu())
    target_y_bin = torch.bucketize(target_mouse[:, 1].contiguous(), model.y_edges.cpu())
    frames = len(target_keys)

    key_loss = nn.BCEWithLogitsLoss(pos_weight=key_pos_weight)(key_logits, target_keys).item()
    x_loss = F.cross_entropy(x_logits, target_x_bin).item()
    y_loss = F.cross_entropy(y_logits, target_y_bin).item()
    x_prior_loss = F.cross_entropy(x_prior.expand(frames, -1), target_x_bin).item()
    y_prior_loss = F.cross_entropy(y_prior.expand(frames, -1), target_y_bin).item()
    total_loss = key_loss + x_loss + y_loss

    print("== loss (same as train.py) ==")
    print(f"{len(recordings)} files | {total_loss:.4f} | ({key_loss:.4f}, {x_loss:.4f}, {y_loss:.4f})")
    print("mouse vs prior (guessing without the screen), model below prior = learned from the screen")
    print(f"  x {x_loss:.4f}  prior {x_prior_loss:.4f}  gap {x_prior_loss - x_loss:+.3f}")
    print(f"  y {y_loss:.4f}  prior {y_prior_loss:.4f}  gap {y_prior_loss - y_loss:+.3f}")
    return {"total": total_loss, "keys": key_loss, "x": x_loss, "y": y_loss,
            "x_gap": x_prior_loss - x_loss, "y_gap": y_prior_loss - y_loss}


# == part 4: keys ==

def f1_scores(pressed, held):
    # pressed, held: (frames, keys) bool -> F1 per key (keys,) and F1 of all keys added up
    true_pos = (pressed & held).sum(0)
    false_pos = (pressed & ~held).sum(0)
    false_neg = (~pressed & held).sum(0)
    per_key = 2 * true_pos / (2 * true_pos + false_pos + false_neg).clamp(min=1)
    all_keys = 2 * true_pos.sum() / (2 * true_pos.sum() + false_pos.sum() + false_neg.sum()).clamp(min=1)
    return per_key, all_keys


def print_key_row(name, per_key, all_keys=None, decimals=3):
    row = f"{name:14s}" + "".join(f"{value:7.{decimals}f}" for value in per_key.tolist())
    if all_keys is not None:
        row += f"{all_keys:7.{decimals}f}"
    print(row)


def report_keys(recordings, model):
    key_probs = torch.sigmoid(torch.cat([rec["key_logits"] for rec in recordings]))
    target_keys = torch.cat([rec["target_keys"] for rec in recordings])
    current_keys = torch.cat([rec["current_keys"] for rec in recordings])
    model_pressed = key_probs > model.key_thresh.cpu()

    model_f1, model_f1_all = f1_scores(model_pressed, target_keys)
    copy_f1, copy_f1_all = f1_scores(current_keys, target_keys) # "in 2 frames I hold what I hold now"
    # how often the model presses compared to me: 1 = as often, 0.05 = almost never
    press_ratio = model_pressed.sum(0) / target_keys.sum(0).clamp(min=1)

    # the threshold that gives each key its best F1 (rare keys peak low, so it starts at 0.01)
    best_f1 = torch.zeros(len(KEY_NAMES))
    best_threshold = torch.zeros(len(KEY_NAMES))
    for threshold in THRESHOLDS:
        f1, _ = f1_scores(key_probs > threshold, target_keys)
        better = f1 > best_f1
        best_f1[better] = f1[better]
        best_threshold[better] = threshold

    print("\n== keys ==")
    print(f"{'':14s}" + "".join(f"{name:>7s}" for name in KEY_NAMES) + f"{'all':>7s}")
    print_key_row("model F1", model_f1, model_f1_all)
    print_key_row("copy current", copy_f1, copy_f1_all)
    print_key_row("press ratio", press_ratio, decimals=2)
    print_key_row("best thresh", best_threshold, decimals=2)
    print_key_row("best F1", best_f1)
    return {"f1": model_f1_all.item(), "copy_f1": copy_f1_all.item()}


# == part 5: aim, with the moves play would send ==

def soft_decode_all(logits, bin_values):
    # the same as play (mm.soft_decode, radius 1), for every frame at once
    # logits (frames, bins) -> moves (frames,)
    probs = logits.softmax(-1)
    top_bin = probs.argmax(-1)
    first_bin = (top_bin - 1).clamp(0, len(bin_values) - 3) # at the edges the 3 bins shift inwards, like play
    window = first_bin.unsqueeze(1) + torch.arange(3)
    window_probs = probs.gather(1, window)
    moves = (bin_values[window] * window_probs).sum(1) / window_probs.sum(1)
    return moves.masked_fill(bin_values[top_bin] == 0, 0) # top bin is "no move" -> exactly 0


def direction_shares(predicted, actual):
    # on frames where I moved: model still + same direction + opposite direction = 100%
    still = (predicted == 0).float().mean().item()
    same = (predicted.sign() == actual.sign()).float().mean().item()
    opposite = (predicted.sign() == -actual.sign()).float().mean().item()
    return still, same, opposite


def turn_sums(moves):
    # moves (frames,) -> total of every TURN_FRAMES frames in a row (frames - TURN_FRAMES + 1,)
    return moves.unfold(0, TURN_FRAMES, 1).sum(-1)


def report_aim_axis(recordings, axis, name, bin_values, logits_key):
    predicted_per_rec = [soft_decode_all(rec[logits_key], bin_values) for rec in recordings]
    actual_per_rec = [rec["target_mouse"][:, axis] for rec in recordings]
    predicted = torch.cat(predicted_per_rec)
    actual = torch.cat(actual_per_rec)
    recent = torch.cat([rec["recent_mouse"][:, axis] for rec in recordings])

    # every frame I moved, and onsets: I was still in the 3 frames the model saw, then moved
    # (nothing on screen is turning yet, so it has to read the scene)
    moved = actual.abs() >= MIN_MOVE[axis]
    onset = moved & (recent < MIN_MOVE[axis])
    still, same, opposite = direction_shares(predicted[moved], actual[moved])
    onset_still, onset_same, onset_opposite = direction_shares(predicted[onset], actual[onset])

    # turns are summed per recording, so no sum crosses from one recording into the next
    predicted_turns = torch.cat([turn_sums(moves) for moves in predicted_per_rec])
    actual_turns = torch.cat([turn_sums(moves) for moves in actual_per_rec])
    # c8: does it turn at the same moments as me (size ignored), 1 = perfectly
    c8 = torch.corrcoef(torch.stack([predicted_turns, actual_turns]))[0, 1].item()
    # turn: on my big turns, how far it turns my way compared to me, 100% = as far as me
    turning = actual_turns.abs() >= BIG_TURN[axis]
    turned_my_way = predicted_turns[turning] * actual_turns[turning].sign()
    turn = (turned_my_way.mean() / actual_turns[turning].abs().mean()).item()

    print(f"{name}: still {still:.0%} same {same:.0%} opposite {opposite:.0%}"
          f" | onset ({onset.sum()} frames) still {onset_still:.0%} same {onset_same:.0%} opposite {onset_opposite:.0%}"
          f" | c8 {c8:.2f} | turn {turn:.0%}")
    return {f"{name}_c8": c8, f"{name}_turn": turn}


def report_aim(recordings, model):
    print("\n== aim (frames where I moved: still + same + opposite = 100%) ==")
    x_results = report_aim_axis(recordings, 0, "x", model.x_bins.cpu(), "x_logits")
    y_results = report_aim_axis(recordings, 1, "y", model.y_bins.cpu(), "y_logits")
    return x_results | y_results


# == all parts ==

def validate(model, sizes):
    files = natsorted(Path(VAL_PATH).iterdir())
    if not files:
        raise SystemExit(f"No files in {VAL_PATH}, run prepare_data.py first")
    model.eval()
    num_bins_x = sizes["num_bins_x"]
    num_bins_y = sizes["num_bins_y"]
    feature_height = sizes["vsn_height"] // 16
    feature_width = sizes["vsn_width"] // 16
    key_pos_weight, x_prior, y_prior = load_training_props(model, num_bins_x, num_bins_y)

    recordings = []
    for file in files:
        recording = load_recording(file)
        if recording is None:
            print(f"Skipped {file.name}, shorter than {SEQ_LEN} frames")
            continue
        key_logits, mouse_logits = predict(model, recording, feature_height, feature_width)
        recordings.append(line_up(recording, key_logits, mouse_logits, num_bins_x))
    if not recordings:
        raise SystemExit(f"Every file in {VAL_PATH} is shorter than {SEQ_LEN} frames")

    results = report_loss(recordings, model, key_pos_weight, x_prior, y_prior)
    results |= report_keys(recordings, model)
    results |= report_aim(recordings, model)
    model.train() # back to training, for when it's called in the middle of a run
    return results


if __name__ == "__main__":
    model, sizes = mm.load_model(CKPT_PATH, mode="train")
    model = model.cuda()
    validate(model, sizes)
