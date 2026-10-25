import torch
from pathlib import Path
from natsort import natsorted

IN_PATH = "outputs_gen"
VAL_EVERY = 10 # every 10th recording goes to validation.


def quantize_file(src, dst):
    f = torch.load(src, map_location="cpu", mmap=True)
    x = f["features"]                                   # [T, 384, 15, 27]
    T = x.shape[0]

    amax = torch.zeros(384)
    for i in range(0, T, 512):
        amax = torch.maximum(amax, x[i:i+512].float().abs().amax(dim=(0, 2, 3)))
    scale = amax.clamp(min=1e-6) / 127                  # [384]

    q = torch.empty(x.shape, dtype=torch.int8)
    s = scale.view(1, 384, 1, 1)
    for i in range(0, T, 512):
        q[i:i+512] = (x[i:i+512].float() / s).round().clamp(-127, 127).to(torch.int8)

    torch.save({"features_q": q, "features_scale": scale,
                "keys": f["keys"], "mouse": f["mouse"]}, dst)


in_quant = Path(IN_PATH)
out_quant = Path(in_quant.name + "_int8")
out_val = Path(in_quant.name + "_int8_val")
out_quant.mkdir(exist_ok = True)
out_val.mkdir(exist_ok = True)

for i, raw_unq in enumerate(natsorted(in_quant.iterdir())):
    out = out_val if i % VAL_EVERY == VAL_EVERY - 1 else out_quant
    print(f"Quanting {raw_unq.name} -> {out}...")
    quantize_file(raw_unq, out / raw_unq.name)
