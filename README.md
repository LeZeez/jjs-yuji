# Intro

I made a dumb AI model that plays JJS (Jujutsu Shenanigans) using imitation learning (behavioral cloning).

***Important note:** This is NOT vibecoded nor done with low effort. 95% of the code is manually written by me. AI was only used for explanations and debugging.*

> **Disclaimer:** This repository is for educational and research purposes only. Using bots or automation in Roblox may violate the [Roblox Terms of Use](https://en.help.roblox.com/hc/en-us/articles/115004647846) and can get your account banned. I am not responsible for any consequences of using this code, including account bans or any other damages. Use it at your own risk.

**Built with technology from NXAI.** See [Licensing](#licensing).

## Current Architecture
OS: Linux Mint 21.3

Roblox client: Sober (Flatpak `org.vinegarhq.Sober`)

Control: `evdev` package for mouse and keyboard input recording using `InputDevice.read()` and execution using `UInput`

AI architecture: two bare mLSTM layers (1024 -> 1024) from xLSTMLarge ([Original repo](https://github.com/NX-AI/xlstm))

Vision package and model: `timm` package, DINOv3 S model (`vit_small_patch16_dinov3.lvd1689m`)

Resolution: 432x240

## Usage
Linux only (uses `evdev`/`uinput`) and needs a CUDA GPU.

```bash
pip install -r requirements.txt
```

### Run the pretrained model
1. Download [`model.pt`](https://huggingface.co/TheZeez/jjs-yuji/blob/main/model.pt) into the repo root.
2. Set `CAM_IDX` in `inference.py` to your capture device (OBS virtual camera in my case).
3. Run `python inference.py`. Press `Ctrl+Alt+X` to stop.

### Train your own
Run the scripts in this order. Folder names are hard-coded at the top of each script.
1. `recorder.py`: records video + inputs to `recs/` (`Ctrl+Alt+X` to stop)
2. `prepare_data.py`: extracts DINOv3 features from `dataset/` (move/rename `recs/` to it) into `outputs_gen/`
3. `int8_quantize.py`: quantizes the features into `outputs_gen_int8/` (required, `train.py` expects quantized features)
4. `train.py`: trains and saves `output_model.pt` (set `CKPT` in `inference.py` to use it)

## Model state
The model is decent. It's unpurposed, so it does everything only if you start it (e.g., you start a combo it continues it). However, since it was trained on only 3 hours, it does only things it's confident in, which is mostly parkour and navigation. It can also be so random. I ended here because I don't have time for data recording because I'm in uni :')


## Examples


### Combat
*it mostly was getting cooked tbh*

![](assets/combat_1.gif)


![](assets/combat_2.gif)



### Parkour and navigation
*it's decent in this*


![](assets/parkour_1.gif)


![](assets/parkour_2.gif)


## Limitations
Trained only on 3 hours of unpurposed data, we can't expect it to generalize very well or to correct itself in all situations. Also, it was only trained on using the character *Vessel*

*freezing*


![](assets/freeze.gif)


*nudging unfreezes it*

![](assets/freeze_nudge.gif)


# Technical report
Phases are based on significant changes. The numbers under them are the experiments' number.

## First phase (2026 Sep 20th)

### 1
**State:**
I was literally lost. I never used vision AI nor I know how it works. I didn't even know how to use opencv2. I learned basics and came up with this.

**Actions:**
- Started with the vision model `mobilenetv4_hybrid_medium.ix_e550_r384_in1k` and averaging all features to get a single 1280 vector.
- Used an mLSTM layer
- Used linear heads for both keyboard and mouse.

**Loss functions:**
- BCEWithLogitsLoss for keyboard
- SmoothL1Loss for mouse to predict raw values

**Result:** terrible model. 
- Dataset was even more little then, so no weighting overfit the dominant behavior.
- Averaging all features prevented the model from seeing where things are (instead of *there's a building on the left*, it sees *there's a building somewhere...*)
- It was somehow reacting to buildings and trying to do parkour, sometimes succeeding.

### 2 to 15
- Same architecture
- Kept switching between MSELoss, SmoothL1Loss, and HuberLoss

**Result:** ...terrible model.

## Second phase
### 16 to 20
**State:** 
I learned about AvgPool2d and how to properly use `timm`. I learned about `torchvision` and `v2.Compose`. I learned about RGB and BGR (opencv and torch conversion). I kept experimenting with weighting since dataset wasn't balanced. This was the longest phase.

**Actions:**
- Switched vision to DINOv3 S (`vit_small_patch16_dinov3.lvd1689m`)
- Used current resolution (432x240)
- Used (3,5) average pooling.
- Kept switching between HuberLoss (raw values) and CrossEntropyLoss (binned values)
- Used arbitrary bins instead of statistically measuring them
- Accidentally clamped max mouse value to 1.0 (cost me two late nights)

**Result:** model that can finally do things.
- It did parkour decent to bad
- It did *some* combos, tho it was incomplete
- Camera kept going down weirdly
- Very good upgrade over first phase models

### 17 to 34
- Switch to multi-layer mLSTM
- Tried GatedDeltaNet (1 and 2) for some weights but wrapped it up asap and returned to mLSTM (not because they are bad but because training was heavier and I was already familiar with mLSTM)
- Kept switching mLSTM layer count (2, 3, 4)
- Switched to (6, 10) average pooling
- Switched to multi-class CrossEntropyLoss for keyboard (idk why)

**Result:** Similar model to the previous ones. Mouse was still bad.

## Third phase
**State:** I rewrote the whole code because I started having many problems. I learned about convolution layers, mu-law, and started recording DAgger (Dataset Aggregation) style data to help the model avoid errors.
### 1 to 7
**Actions:**
- Divided individual videos to separate 4 batches of 1800 frames (30s in 60fps)
- Used skip-connection with LayerNorm from the vision directly to the controlling heads
- Used two mLSTM layers as a constant
- Integrated Conv2d (384 -> 64) as the first layer instead of average pooling
- down_proj in_features was 64 * 15 * 27 (25920)
- Switched to binned mouse values with CrossEntropyLoss as a constant
- Integrated mu-law from OpenAI VPT code with unified 15 bins for both X and Y
- Used weighting for rare mouse bins
- Switched to BCEWithLogitsLoss for keyboard as a constant
- Used pos_weight for rare keyboard presses

  
**Result:** significantly better model.
- It could navigate buildings and do parkour closer to me
- It could start combos but fail to continue it
- Mouse was stable now and the weird going-down issue was absent
- It still was inconsistent

### 8 to 15
- Abandoned the skip-connection and retained LayerNorm right before the controlling heads
- Tried switching to one mLSTM layer once but returned to two ones
- Tried taking the square-root of pos_weight and mouse CrossEntropy weighting

**Result:** mixed.
- Switching to one layer turned badly
- Taking sqrt of weighting helped overfitting and made the model take too safe decision (e.g., freezing, parkour)
- The absence of skip-connection made the model generally better, tho it made mouse a little worse - a trade-off for better model overall
- This stage was mostly learning mistakes so nothing really got better

## Final phase (2026 Oct 10th)
**State:** I recorded a new hour of data focusing on combat and was mostly adjusting what I already have. I learned about convolution stride to downscale features further and shifting controlling frames to make the model react to what it sees, not what it saw.

**SKIPPED 1 to 5 - was training on old config by mistake**

## 6 to 14
**Actions:**
- Introduced a new `SHIFTAMOUNT` constant to shift keys/mouse frames to the left, earlier
- Used `SHIFTAMOUNT = 10` then `SHIFTAMOUNT = 5`
- Discovered the DINOv3 S range was big and thought it's a bug then used z-score on all of the dataset
- Used different bin size for both X and Y mouse bins (25 for X, 15 for Y)
- Used `p=0.2` for both `Dropout2d` and `Dropout` after the Conv and after the linear down_proj, respectively
- Stored features in `int8` with channel-wise scale to fit all the dataset in ram and partly to introduce some noise (felt like noise would make it better)

**Result:** model was a lot better in generalizing and it could finally do some combos.
- **The model got its first kill!!** (with just a little help of me ;) )
- `SHIFTAMOUNT` 10 was worse and made the model freeze a lot. 5 was better.
- Better parkour and generalizing on all buildings and walls, including ones that weren't in the data
- Z-score made the model worse so returned to raw values
- Dropout helped generalization
- `int8` made convergence slower

## 15 to latest 
- Downscaled data to 30 fps instead of 60 (significant effect)
- Used `SHIFTAMOUNT = 2` to be consistent with 30fps
- Used Conv2d with kernel_size 3, stride 2, and padding 1 to downscale down_proj to 7168 (significant effect)
- Used 4 batches of 900 frames (still 30s)
- Tried 40 bins for X and switched back to 25

**Result:** at this point, the data was the limit, because with only 3 hours, I think did a great job
- Good parkour
- Good navigation outside buildings (inside wasn't in the data so it struggles with it)
- Good combos (when I start it and make it continue)
- Sometimes freezes in place and struggles with mouse, but overall it's a really decent model for only 3 hours
- 40 bins was worse than 25 (only used low value bins)

# Licensing
This project uses third-party code and models with their own licenses:

- **xLSTM** ([NX-AI/xlstm](https://github.com/NX-AI/xlstm)): the `xlstm` package is Apache-2.0. Its kernels come from `mlstm_kernels`, which is under the **NXAI Community License** ([NXAI_LICENSE](NXAI_LICENSE)). Built with technology from NXAI.
- **OpenAI Video-Pre-Training** ([openai/Video-Pre-Training](https://github.com/openai/Video-Pre-Training)): the mu-law `CameraQuantizer` in `internal/mu.py` is adapted from it under the MIT License.
- **DINOv3** (`vit_small_patch16_dinov3.lvd1689m` via `timm`): weights are downloaded at runtime and are under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) by Meta. They are not included in this repo.

See [NOTICE](NOTICE) for the full attribution notices.
