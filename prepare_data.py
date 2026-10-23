import torch
from torchvision.transforms import v2
import os
import orjson
from pathlib import Path
from natsort import natsorted
from torchcodec.decoders import VideoDecoder
import timm
from io import BytesIO
from internal.device_manager import KEYS as keys
from internal.device_manager import REL as rel

BATCH = 3584 # 30fps 2mins batch (trimmed to be divisible by 128)


def normalize_inputs(inputs):
    inputs = orjson.loads(inputs)
    file_keys = []
    file_mouse = []
    for k in keys:
        file_keys.append(
            min(inputs["keys"].get(str(k), -1), 1) # -1 == None but it's just like that for the min()
            )

    for k in rel:
        file_mouse.append(
            inputs["mouse"].get(str(k), 0)
            )

    for i, key in enumerate(file_keys):
        if key == -1:
            file_keys[i] = 0
            continue

        if key == 0:
            file_keys[i] = -1

    return file_keys, file_mouse

def get_names(path):
    files = os.listdir(path)
    names = set()
    for file in files:
        names.add( Path(file).stem )

    return natsorted(list(names))

def prepare_keys(t):
    B = t.shape[0]
    idx = torch.arange(B).unsqueeze(1)
    states = torch.where(t != 0, idx, -1) # retain idx when smth happens
    states = states.cummax(0).values # repeat idx when smth doesn't happen
    states = states.clamp(min=0) 
    t = torch.gather(t, 0, states).clamp(min=0) # substitute and retain held keys
    return t


class JJSDatasetPrep():
    def __init__(self, folder_path: str | Path, out_path: str | Path,
                 chunk_size: int, use_cuda: bool = False,
                 in_mem: bool = False):

        self.names = get_names(folder_path)
        print(self.names)
        self.out_names = get_names(out_path)
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
        self.vision = model = timm.create_model(
            'vit_small_patch16_dinov3.lvd1689m',
            pretrained=True,
            features_only=True,
            out_indices=(-1,)
        ).eval().to(device=self.dvc, dtype=torch.float32)

        data_config = timm.data.resolve_model_data_config(model)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((240, 432), antialias=True), # can comment if the dataset is already resized (recommended because I did it)
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_config["mean"], std=data_config["std"])
        ])

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def _transform(self, x):
        x = self.transform(x)
        x = self.vision( x )[0]
        return x

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
                vd = VideoDecoder(vid_file, device=self.dvc,
                                num_ffmpeg_threads=0) # num_ffmpeg_threads = 0 is recommended based on what I saw
            except RuntimeError:
                print("Skipped", self.current_stem) # skip if the file can't be opened. Can be too
                                                    # broad but I haven't gotten any issues yet
                continue

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


            total_frames = vd.metadata.num_frames
            print(f"Extracting {vid_file} | Total Frames: {total_frames}")


            # spaghetti
            frames_prcsd = 0
            while frames_prcsd < total_frames or not(frames_prcsd + BATCH >= total_frames):
                end_frame = min(frames_prcsd + BATCH, total_frames)
                self.current_outstem = self.current_stem if total_frames <= BATCH else \
                      self.current_stem + f"_{frames_prcsd}-{end_frame}"
                
                if end_frame - frames_prcsd < BATCH:
                    print(f"Skipping {self.current_outstem}, end of file shorter than {BATCH}...")
                    break

                if self.current_outstem in self.out_names:
                    print(f"Skipping {self.current_outstem}, already processed...")
                    frames_prcsd += BATCH
                    continue

                print(f"Processing {self.current_outstem}")

                feature_chunks = []
                with torch.no_grad():
                    for start in range(frames_prcsd, end_frame, self.chunk_size):
                        end = min(start + self.chunk_size, end_frame)
                        frames = vd[start:end]
                        frames_tensor = frames.to(device=self.dvc)
                        feat = self._transform(frames_tensor).bfloat16().cpu()
                        feature_chunks.append(feat)
                        del feat # for memory (idk if it's effective)


                all_features = torch.cat(feature_chunks, dim=0)
                del feature_chunks # for memory 2
                all_features = all_features.cpu()
                assert len(all_features) == BATCH, all_features.shape
                
                yield {
                    "features": all_features,
                    "keys":     all_keys[frames_prcsd:end_frame],
                    "mouse":    all_mouse[frames_prcsd:end_frame],
                }
                frames_prcsd += BATCH



inn = ["dataset"] # dataset folder(s) containing the video and the jsonl file
out = "outputs_gen"
prepare_batch_size = 512 # decrease/increase based on your memory
use_cuda = True


Path(out).mkdir(parents=True, exist_ok=True)
for file_inn in inn:
    dts = JJSDatasetPrep(file_inn, out, prepare_batch_size, use_cuda=use_cuda)
    for out_batch in dts:
        torch.save(out_batch, f"{out}/{dts.current_outstem}.pt")