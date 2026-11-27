# Technical report
Phases are based on significant changes. The numbers under them are the experiment numbers.

## First phase
**State:**
I was literally lost. I had never used vision AI nor did I know how it worked. I didn't even know how to use opencv2. I learned basics and came up with this.

### 1
**Actions:**
- Started with the vision model `mobilenetv4_hybrid_medium.ix_e550_r384_in1k`, squeezing the screen into a 384x384 square, and averaging all features to get a single 1280 vector
- Used a single mLSTM layer
- Used linear heads for both keyboard and mouse
- Used BCEWithLogitsLoss for keyboard
- Used SmoothL1Loss for mouse to predict raw values

**Result:** terrible model.
- Data was less than an hour. It was even worse without weighting (balancing loss for frequent and rare actions)
- Averaging all features into one vector prevented the model from seeing where things are (instead of *there's a building on the left*, it sees *there's a building somewhere...*)
- It was somehow reacting to buildings and trying to do parkour, sometimes managing a single jump.

### 2 to 15
- Same architecture
- Kept switching between MSELoss, SmoothL1Loss, and HuberLoss for mouse

**Result:** ...terrible model.

## Second phase
**State:**
I learned about AvgPool2d and how to properly use `timm`. I learned about `torchvision` and `v2.Compose`. I learned about RGB and BGR (opencv and torch conversion). I kept experimenting with weighting since dataset wasn't balanced. This was the longest phase.

### 16 to 20
**Actions:**
- Switched vision to DINOv3 S (`vit_small_patch16_dinov3.lvd1689m`)
- Used current resolution (432x240)
- Used (3,5) average pooling.
- Kept switching between HuberLoss (raw values) and CrossEntropyLoss (binned values)
- Used arbitrary bins instead of statistically measuring them
- Accidentally clamped max mouse value to 1.0 (cost me two late nights)

**Result:** model that can finally do things.
- It did parkour decent to bad
- It did *some* combos, tho they were incomplete
- Camera kept going down constantly
- Very good upgrade over first phase models

### 21 to 34
- Switched to multi-layer mLSTM
- Tried GatedDeltaNet (1 and 2) for some weights but wrapped it up asap and returned to mLSTM (not because they are bad but because training was heavier and I was already familiar with mLSTM)
- Kept experimenting with mLSTM layer count (2, 3, 4)
- Switched to (6, 10) average pooling
- Switched to multi-class CrossEntropyLoss for keyboard (idk why)

**Result:** Similar model to the previous ones. Mouse was still bad.

## Third phase
**State:** I rewrote the whole code because I started having many problems. I learned about convolution layers, mu-law, and started recording DAgger (Dataset Aggregation) style data to help the model avoid errors.

### 1 to 7
**Actions:**
- Divided individual videos into 4 separate batches of 1800 frames (30s in 60fps)
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
- It could navigate buildings and do parkour similar to me
- It could start combos but fail to continue it
- Mouse was stable now and the weird going-down issue was absent
- It was kind of erratic (e.g., punching the air)

### 8 to 15
- Abandoned the skip-connection and retained LayerNorm right before the controlling heads
- Tried switching to one mLSTM layer once but returned to two
- Tried taking the square-root of pos_weight and mouse CrossEntropy weighting

**Result:** mixed.
- Switching to one layer turned out badly
- Taking sqrt of weighting made the model take too-safe decisions (e.g., freezing, parkour)
- The absence of skip-connection made the model generally better, tho it made mouse a little worse - a trade-off for better model overall
- This stage was mostly learning mistakes so nothing really got better

## Final phase
**State:** I recorded a new hour of data focusing on combat and was mostly adjusting what I already have. I learned about convolution stride to downscale features further and shifting controlling frames to make the model react to what it sees, not what it saw.

**SKIPPED 1 to 5 - was training on old config by mistake**

### 6 to 14
**Actions:**
- Introduced a new `SHIFTAMOUNT` constant to shift keys/mouse frames to the left, earlier
- Used `SHIFTAMOUNT = 10` then `SHIFTAMOUNT = 5`
- Discovered the DINOv3 S range was big and thought it was a bug, so I used z-score on all of the dataset
- Used different bin counts for X and Y mouse bins (25 for X, 15 for Y)
- Used `p=0.2` for both `Dropout2d` and `Dropout` after the Conv and after the linear down_proj, respectively
- Stored features in `int8` with channel-wise scale to fit all the dataset in ram and partly to introduce some noise (felt like noise would make it better)

**Result:** model was a lot better in generalizing and it could finally do some combos.
- **The model got its first kill!!** (with just a little help from me ;) )
- `SHIFTAMOUNT` 10 was worse and made the model freeze a lot. 5 was better.
- Better parkour and generalizing on all buildings and walls, including ones that weren't in the data
- Z-score made the model worse so returned to raw values
- Dropout helped generalization
- `int8` made convergence slower and helped generalization

### 15 to first release
- Downscaled data to 30 fps instead of 60 (significant effect)
- Used `SHIFTAMOUNT = 2` to be consistent with 30fps
- Used Conv2d with kernel_size 3, stride 2, and padding 1 to downscale down_proj to 7168 (significant effect)
- Used 4 batches of 900 frames (still 30s)
- Tried 40 bins for X but switched back to 25

**Result:** at this point, the data was the limit.
- Good parkour
- Decent navigation outside buildings (inside wasn't in the data so it struggles with it)
- Hit-or-miss combos (when I start it, it can do great combos, though it's still not consistent because of data scarcity)
- Sometimes freezes in place and struggles with mouse, but overall it's a really decent model for only 3 hours
- 40 bins was worse than 25

# Second release report
I had gotten some free time and recorded 4 hours of data, totalling 7 hours then. I kept trying new architectures to remove the down projection layer. I wanted something to generalize on all parts of the screen instead of learning something here and not there. I failed miserably. I tried attention pooling, ViL (vision LSTM), nothing worked. I had to pay the parameter tax from flattening anyway. I found a lot of bugs in the repo that made most of the 7 hours unusable.

**Findings:**
- What the model was trained on (compressed mess) wasn't what it saw. Same picture processed through training pipeline wasn't the same in play's, and both produced completely different outputs from the model.
- Found a huge recorder bug: one read only got up to 64 mouse events per frame while my mouse sends up to ~100, so fast turns were under-recorded (8% overall, 22% on fast frames) and ~10% of M1 presses were lost (recorded as release only).
- The model was still unpurposed and I couldn't purpose it on the public game, especially after the game map changed (thanks tze).


## Actions and results

### Recorder
- Fixed recorder with a threaded reader. But the old 7h couldn't be repaired ):
- Made sure that what the model saw was consistent in training and play picture (lossless data)

### Gameplay & data
- Switched to 1v1 (Duels). One purpose, smaller map, less data needed. 
- Recorded 9 hours using the fixed recorder (hell). 
- Enabled the new damage counter and it was the best addition to the new update so the model actually knows it hit the opponent.
- Masked the candy counter in the corner so the model can't use it as a clock and so the model remains playable after the event (data was recorded in halloween event).
- Added high contrast crosshair so the model has an anchor to help aim

### Aim tests
- Added an aim test: on recordings the model never saw, how often it turns the same way as me, how often the opposite way, the correlation of 8-frame turns (`c8`, about 0.27 s = what reads as aiming), and how far it turns on my big turns (`turn`).

For mouse x axis:
- The first release scored 62% same, 26% opposite, 12% still, and `c8` 0.41
- This release scored around 62% same, 10% opposite, 30% still, and `c8` 0.59. Aim is significantly better in live play too.

### New head
- Attention pooling vs flattening the DINO map: flattening aims 2 to 2.5x better (every screen position keeps its own weights but learns independently however).
- Squeezed the map to just 4 channels per position before flattening: as good as 16, 32 or 64, but 2 was too few.
- Quantile bins with linear space vs mu-law: tied, so quantile because it's simpler.
- Random windows from all files with bigger batches instead of fixed chunks on the same video yielded real gain.
- Removed key E (unused in the 1v1).


**Result:** better on paper, about the same in game. Still froze, still blocked forever (what I called F-lock).

### Action chunking (dropped)
- Instead of only the next frame, the model predicts the keys of the next 8 frames and play runs 4 of them before asking again
- Tested keys only, keys + mouse, and a CVAE version.

**Result:**
- Keys only: tied offline, shorter F-locks (not none), and a bit better live.
- Chunking the mouse hurt aim (a few frames ahead is close to unpredictable from the screen).
- The CVAE either barely pressed anything or spammed.
- Dropped because it made the code complex over small gain.

## ConvLSTM phase
Inspired by the Counter-Strike Deathmatch with Large-Scale Behavioural Cloning paper (Pearce & Zhu), which uses a ConvLSTM, I decided to try it along with a few more things from their project.

### Architecture and window length tests
- Replaced mLSTM with a plain LSTM in the same blocks. They tied, and the LSTM was simpler so dropped mLSTM.
- I used a near-replica of the ConvLSTM Keras class (made by AI ofc because lazy) and it was the best one yet in aim and keyboard.
- Tried different sizes: ConvLSTM filters (from 4 to 384), down_proj (256, 512)
- Added SwiGLU and it improved performance and capacity because ConvLSTM alone had little capacity.
- Final layout: DINO map -> ConvLSTM (256 filters) -> 1x1 conv to 4 channels -> flatten -> Linear to 512 -> x + SwiGLU(norm x) -> norm -> key and mouse heads.
- 6 s windows (180 frames, like CSGO's 96 frames at 16 fps) instead of 30 s.

**Result:** very good model. It got a bunch of kills on its own I stood up and applauded. 256 and 384 filters were close, 384 a little better on paper but a lot less efficient to train.

### Weights
- I added mouse bin weights since quantile bin occurrences wasn't really uniform across all bins
- Capped the key `pos_weight` at 50 (rare keys like G and R were at 100-200).

**Result:**
- Full uncapped weights: the mouse never stops moving (it stands still on 5% of frames, I do on 42%), jittery.
- Sqrt: calmer and best gameplay at the time. It stops at the right times (mostly)

### Outcome sampling
I labeled the dataset using simple algorithm that checks: damage counter on (combat), HP decreasing (humbled), neither (idle). The labels then are used from the dataset sampler with ratios: 1.0 idle, 2.0 for combat, and 0.5 for humbled.

The ratios mean that combat appears 2x more often than idle, and humbled appears 2x less than idle in the dataset sampler.

I used combat ratio 2.0 and 3.0

**Result:**
- **Combat 2.0 is the sweet spot.** Used for the latest model. It learned combos, takes openings, and its F-lock looks like a tactic (waiting for the enemy to hit) instead of a curse like other experiments and 3.0 ratio.
- Combat 3.0 and everything trained on it: F-lock got worse, it stays blocking even when there's an opening (this is in 2.0 but it was worse in 3.0). Too few calm moments to learn when to let go.


### Things that didn't work
- **Bot self-play data:** let the model play for 2.5 h and kept its combat parts. 7 out of 10 samples I checked were trash (was on another model though, but the idea was still not that good anyway)
- **Exponential tail bins** (more bins for big flicks): bigger turns, but less accurate vertical aim and no better live.
- **Squeeze to 16 channels instead of 4:** overfit and more parameters.

## Final phase

### Corrections and manual labeling
- MANUALLY labeled 1204 clips of combat to reduce F-lock and parts I did stupid moves in (hell). Now we have a "rejected" category. I made its ratio 0.5 like "humbled".
- A script where the model plays and I take over with a mouse side button whenever it messes up, then hand back. Every frame is recorded with who was playing.
- Recorded 34 minutes, 30 of them me playing from the situations the model got itself into.
- Fine-tuned the base model on it: each step half old data (with the manual review ratios), half corrections, and the loss only counts the frames I played.

**Result:** a lot better. It learned exactly the moments where it used to go wrong.

### Executing controls
- Tried thresholds set so the model presses each key as often as I do: not great.
- Then did what the CSGO paper does: movement keys (W A S D) are pressed when their real chance is over 50%, and the other keys (F, Q, M1, Space, skills, ult) are rolled with their real chance every frame.
- "F" itself was in the held category first because it's a hold button, but when it's rolled, F-lock was almost non-existent and made the model finally not depend on it, but now it has to perfectly time blocking.

**Result:** current model.
