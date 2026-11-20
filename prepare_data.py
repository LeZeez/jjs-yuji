import torch
import numpy as np
import av
import os
import random
import shutil
import orjson
from pathlib import Path
from itertools import islice
from natsort import natsorted
from io import BytesIO
from internal.model_manager import Vision
from internal.device_manager import KEYS
from internal.device_manager import REL

IN_PATHS = ["recorded"] # dataset folder(s) containing the video and the jsonl files
OUT_PATH = "outputs_gen_int8"
OUT_VAL_PATH = "outputs_gen_int8_val"
PREPARE_BATCH_SIZE = 512 # decrease/increase based on your memory
USE_CUDA = True
VAL_COUNT = 5


def normalize_inputs(inputs):
    inputs = orjson.loads(inputs)
    file_keys = []
    file_mouse = []
    for k in KEYS:
        file_keys.append(
            inputs["keys"].get(str(k), None) # None = no press/release, 1 = press, 0 = release
            )

    for k in REL:
        file_mouse.append(
            inputs["mouse"].get(str(k), 0)
            )

    for i, key in enumerate(file_keys):
        if key == None:
            file_keys[i] = 0 # if nothing make it 0
        elif key == 0:
            file_keys[i] = -1 # if released make it -1

    return file_keys, file_mouse

def get_names(path):
    files = os.listdir(path)
    names = set()
    for file in files:
        names.add( Path(file).stem )

    return natsorted(list(names))

def prepare_keys(key_events):
    # key_events: (frames, keys), 1 = pressed, -1 = released, 0 = nothing happened
    frame_count = key_events.shape[0]
    frame_index = torch.arange(frame_count).unsqueeze(1)
    last_event = torch.where(key_events != 0, frame_index, -1) # retain the index when smth happens
    last_event = last_event.cummax(0).values # repeat the index when smth doesn't happen
    last_event = last_event.clamp(min=0)
    held = torch.gather(key_events, 0, last_event).clamp(min=0) # substitute and retain held keys
    return held # (frames, keys), 1 = held


def read_frames(container):
    stream = container.streams.video[0]
    stream.thread_type = "AUTO" # PyAV decodes on one thread otherwise, this is 2.4x faster
    for frame in container.decode(stream):
        bgr = frame.to_ndarray(format="bgr24") # exactly the frame the recorder got from the camera
        yield bgr # (H, W, 3)


def quantize_int8(features):
    # features: (N, 384, 15, 27) fp32, a scale per frame and channel (measured 0.49% error)
    largest = features.abs().amax(dim=(2, 3), keepdim=True) # (N, 384, 1, 1)
    scale = largest.clamp(min=1e-6) / 127
    quantized = features / scale
    quantized = quantized.round().clamp(-127, 127).to(torch.int8)
    scale = scale.squeeze(-1).squeeze(-1)
    return quantized, scale # (N, 384, 15, 27) int8, (N, 384) fp32


class JJSDatasetPrep():
    def __init__(self, folder_path: str | Path, out_path: str | Path, out_val_path: str | Path,
                 chunk_size: int, use_cuda: bool = False,
                 in_mem: bool = False):

        self.names = get_names(folder_path)
        print(self.names)
        self.out_names = get_names(out_path) + get_names(out_val_path)
        print(self.out_names)
        key_files = [os.path.join(folder_path, name) + ".jsonl" for name in self.names]
        vid_files = [os.path.join(folder_path, name) + ".mkv" for name in self.names]
        print(key_files)
        print(vid_files)

        # final
        self.files = list( zip(key_files, vid_files) )
        self.fldr_pth = folder_path
        self.current_stem = None
        self.current_outstem = None
        self.chunk_size = chunk_size
        self.in_mem = in_mem
        self.dvc = "cuda" if use_cuda == True else "cpu"
        self.vision = Vision(432, 240).to(self.dvc).eval() # the same Vision as Play

    def open_file(self):
        for i, file in enumerate(self.files):
            key_file, vid_file = file
            self.current_stem = Path(self.files[i][0]).stem

            if self.current_stem in self.out_names:
                print(f"skipping {self.current_stem}, already processed")
                continue

            if self.in_mem == True:
                with open(vid_file, 'rb') as f:
                    vid_file = BytesIO(f.read())
            with open(key_file, 'rb') as f:
                 key_file = BytesIO(f.read())
            yield key_file, vid_file

    def __iter__(self):
        soruce = list(self.open_file()) if self.in_mem else self.open_file()
        for key_file, vid_file in soruce:
            try:
                container = av.open(vid_file)
            except av.error.FFmpegError:
                print("Skipped", self.current_stem)
                continue
            frames = read_frames(container)

            # extract all controls
            all_keys = []
            all_mouse = []
            for line in key_file:
                file_keys, file_mouse = normalize_inputs(line)
                all_keys.append(torch.tensor(file_keys))
                all_mouse.append(torch.tensor(file_mouse))

            # == Remove the first hold (if it exists) ==
            all_keys = torch.stack(all_keys)
            mask = all_keys.cummax(0)[0].to(torch.bool)
            mask = ~mask
            all_keys = all_keys.masked_fill(mask, 0)
            # ==========================================

            all_keys = prepare_keys(all_keys)
            assert ((all_keys == 0) | (all_keys == 1)).all(), \
            "Invalid held states" # assert it's binary
            all_mouse = torch.stack(all_mouse)


            all_keys = all_keys.to(device="cpu", dtype=torch.int8)
            all_mouse = all_mouse.to(device="cpu", dtype=torch.float32)


            total_frames = len(all_keys) # one jsonl line per video frame, and an .mkv doesn't always store its frame count
            print(f"Extracting {self.current_stem} | Total Frames: {total_frames}")

            # spaghetti gone ):
            # the whole recording becomes one file, train.py takes random windows from it
            self.current_outstem = self.current_stem
            feature_chunks = []
            scale_chunks = []
            frames_done = 0
            try:
                while frames_done < total_frames:
                    count = min(self.chunk_size, total_frames - frames_done)
                    batch = list(islice(frames, count))
                    if len(batch) == 0:
                        break
                    with torch.no_grad():
                        features = self.vision(np.stack(batch))
                    quantized, scale = quantize_int8(features)
                    feature_chunks.append(quantized.cpu())
                    scale_chunks.append(scale.cpu())
                    frames_done += len(batch)
            except av.error.FFmpegError as error:
                # e.g. the recorder crashed and the end of the file is missing, the frames before it are still good
                print(f"{self.current_stem}: the video stops decoding after {frames_done} frames ({error})")
            finally:
                container.close()

            if frames_done == 0:
                print(f"Skipped {self.current_stem}, the video has no frames")
                continue
            if frames_done < total_frames:
                print(f"{self.current_stem}: the video has {frames_done} frames but the jsonl has {total_frames} lines, "
                      f"the inputs are cut to the video")

            all_features = torch.cat(feature_chunks, dim=0)
            all_scales = torch.cat(scale_chunks, dim=0)
            yield {
                "features_q":     all_features, # (frames, 384, 15, 27) int8
                "features_scale": all_scales,   # (frames, 384) fp32, features = features_q * features_scale
                "keys":           all_keys[:frames_done].clone(),
                "mouse":          all_mouse[:frames_done].clone(),
            }


Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
Path(OUT_VAL_PATH).mkdir(parents=True, exist_ok=True)
made_this_run = []
for in_path in IN_PATHS:
    dts = JJSDatasetPrep(in_path, OUT_PATH, OUT_VAL_PATH, PREPARE_BATCH_SIZE, use_cuda=USE_CUDA)
    for out_batch in dts:
        path = f"{OUT_PATH}/{dts.current_outstem}.pt"
        torch.save(out_batch, path)
        made_this_run.append(path)

random.shuffle(made_this_run)
val_count = max(min(VAL_COUNT, len(made_this_run) - 1), 0) # at least one new file stays in training
if val_count < VAL_COUNT:
    print(f"Only {len(made_this_run)} new files, {val_count} moved to validation so training keeps one")
for path in made_this_run[:val_count]:
    shutil.move(path, OUT_VAL_PATH)