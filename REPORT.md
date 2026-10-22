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
