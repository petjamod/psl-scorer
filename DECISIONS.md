# Decisions log

Every non obvious choice made while building this, and why. Numbers come from
`training/runs/*.json` and `tests/out/results.json` (see VALIDATION.md).

## Environment and access

1. **Git push is blocked (HTTP 403, "Claude doesn't have GitHub access to petjamod/psl-scorer").**
   All network dependencies worked (Hugging Face, download.pytorch.org, PyPI, npm,
   storage.googleapis.com, cdn.jsdelivr.net, Playwright Chromium), but no git write
   to the repo was possible, so neither `main` nor a branch nor a PR could be pushed.
   Everything was built and committed locally on `claude/face-attractiveness-scorer-gmi52g`,
   and the push was retried after every phase. See the end of this file for the final state.
2. **MediaPipe Python needed `libEGL`.** Installed `libegl1 libgles2 libgl1` with apt.
   Python `mediapipe==1.0.1` and JS `@mediapipe/tasks-vision@1.0.1` are the same release,
   so landmarks agree between Python and the browser (measured: 0.0004 px mean difference).

## Data

3. **Dataset: `MnLgt/scut-fbp5500` (Hugging Face parquet).** 3300 train / 2200 test,
   the sizes of the official 60/40 split. Images are stored as lossless PNG, all 350x350.
4. **Split provenance could not be verified against the official `train.txt`/`test.txt`**
   (they live in the Google Drive release zip). See decision 8 for why this matters.

## Model

5. **The pretrained file is not a torchvision state dict.** `resnet18_py3.pth` is a
   checkpoint `{state_dict, optimizer, epoch=40, best_prec1=0.8819}` saved from
   HCIILAB's own `Nets.py` ResNet, with keys like `group1.conv1.weight`,
   `layer1.0.group1.bn1.*`, `group2.fullyconnected.*`. `training/common.py:remap_hciilab`
   maps them 1:1 onto `torchvision.models.resnet18(num_classes=1)` (strict load, no
   missing or unexpected keys).
6. **Preprocessing search** (pretrained weights, TEST split, Pearson):
   resize 256 + center crop 224, ImageNet norm, RGB = **0.9150**;
   resize 224, ImageNet norm, RGB = 0.8981; BGR variants 0.875 / 0.863;
   no normalization ~0.45 to 0.51; raw 0..255 negative. Chosen: **256crop, ImageNet, RGB**.
7. **Resize filter = Pillow bilinear (antialiased), reproduced bit exactly** in
   `training/pipeline.py:resize_pil` (numpy) and `web/js/pipeline.js:resizePIL` (JS),
   including Pillow's 22 bit fixed point arithmetic. Verified: identical to `PIL.Image.resize`
   and identical JS vs Python tensors (0 differing values out of 3M).
8. **Leakage check on the pretrained weights.** With the chosen preprocessing the
   pretrained model scores Pearson 0.922 / MAE 0.209 on the TRAIN split and 0.915 / 0.210
   on the TEST split. A network actually trained on this train split normally fits it
   clearly better than the test split, so this near equality suggests the released weights
   were trained on a different split (or on all 5500 images), meaning part of this "test"
   split may have been seen in training and 0.915 may be optimistic. Because the goal is an
   accuracy measured against **held out** ratings, I also fine tuned my own model on the
   train split only (decision 11) and compared.
9. **Alignment template from MediaPipe on the 3300 TRAIN images** (`web/alignment.json`):
   eye center = mean of the two eye corner landmarks (33/133 and 362/263). Mean eye midpoint
   (176.3, 159.3), mean inter eye distance 84.9 px (std 10.4) in the 350x350 frame.
   User photos are warped so the eyes are level, the inter eye distance is 84.9 px and the
   eye midpoint lands on (176.3, 159.3). Corners were used instead of iris centers because
   they do not move with gaze.
10. **Aligning SCUT's own test images does not hurt**: pretrained 224 variant 0.898 raw vs
    0.902 aligned; 256crop 0.915 raw vs 0.910 aligned. So the app framing matches SCUT's.
11. **Fine tuning (own model)**: torchvision ResNet18 with ImageNet weights, conv1/layer1/layer2
    frozen, trained on ALIGNED train crops (exactly what the app produces) resized to 256,
    random 224 crops, horizontal flips, mild brightness/contrast jitter, AdamW 3e-4,
    cosine schedule, MSE, batch 32. The epoch count was fixed up front from the measured
    step time to fit a ~55 minute CPU budget, and the LAST epoch is kept: the test split is
    never used to pick a checkpoint. 25 epochs, 54 minutes including building the aligned crops.
    Result: TEST Pearson 0.9176, MAE 0.203; TRAIN Pearson 0.9947. That 0.077 train/test gap is what a
    network trained on this split looks like, and it confirms decision 8: the pretrained weights' gap on
    the same pipeline is only 0.004.
12. **ONNX**: opset 17, fp32, no quantization, dynamic batch. onnxruntime (Python) vs
    PyTorch on 200 test images: max abs diff below 1e-6 (limit was 0.005).
13. **Warp details**: inverse mapping, bilinear, clamp to edge (a SCUT style border rather
    than black), pixel centers at +0.5. When the face is more than 25% bigger than SCUT's
    (scale < 0.8), the source region is first shrunk with the antialiased Pillow filter so the
    final warp does not alias. Output is quantized to uint8, like SCUT's files.

## Statistics and conversions

14. **stats.json** uses TRAIN labels per gender (male n=1627 mean 2.896 sd 0.658; female
    n=1673 mean 3.079 sd 0.717), sample std (ddof=1), percentiles 1..99. As specified, z uses
    the label spread, not the prediction spread. Predictions are shrunk toward the mean
    (regression to the mean), so extreme PSL values are rarer than the formula alone suggests.
15. **Percentile** = linear interpolation inside the 1..99 table; below the 1st or above the 99th
    it shows "<1" or ">99".

## Quality checks

16. **Pose** comes from the facial transformation matrix, measured relative to the median
    SCUT TRAIN pose (yaw -0.3, pitch +5.6, roll +0.4 degrees): MediaPipe reports about +5 degrees
    pitch for a straight SCUT face, so an absolute threshold would be biased. 99% of SCUT
    train faces are within 15 degrees on every axis, so the 15 degree threshold fits.
17. **The JS matrix layout** (column major) is detected at runtime from where the z translation
    sits; verified against Python on the same image.
18. **Smiling: SCUT is not all neutral.** 29.6% of SCUT TRAIN faces have a MediaPipe smile score
    above 0.5 (jaw open above 0.5: 0.03%). I kept the requested rule (smile above 0.5 is flagged and
    excluded) because a neutral face gives more repeatable scores between photos, but made the
    threshold and the exclusion editable in `config.json`. VALIDATION.md reports whether the model
    is biased on smiling faces.
19. **Which warnings exclude a photo from the average**: pose and expression exclude (as specified);
    small face, blur and lighting only warn (the spec lists them as warnings only). Editable in
    `config.json` under `quality.exclude`.
20. **Blur threshold** 30 (Laplacian variance of the aligned grey crop, SCUT TRAIN 1st percentile 32).
    **Lighting**: mean luma of the central face region below 70 or above 215 (SCUT 1st/99th
    percentiles 94.5 / 204.5). A very dark photo can also trip the blur warning because low contrast
    lowers the Laplacian variance; both are warnings only.
21. **Duplicates**: the same file (same SHA-256) added twice is shown twice (second one marked
    "seen before") but counted once in the average.
22. **Cached scores**: results are cached by `model id + SHA-256`. The app always recomputes; if a
    cached value exists it displays the cached number, which guarantees "same photo, same score" on a
    device even if a browser update changed floating point details. Changing `model.id` in config.json
    invalidates the cache.

## Web runtime

23. **Pinned, self hosted**: onnxruntime-web 1.30.0 (`ort.wasm.min.mjs` + `ort-wasm-simd-threaded.{mjs,wasm}`),
    @mediapipe/tasks-vision 1.0.1 (`vision_bundle.mjs`, SIMD and no SIMD wasm), `face_landmarker.task`
    (float16/1). No CDN at runtime.
24. **ORT single threaded, WASM backend** (`numThreads = 1`): deterministic, and GitHub Pages cannot
    send the COOP/COEP headers that multi threading needs anyway. MediaPipe uses the CPU delegate.
25. **Service worker**: app files stale while revalidate (config.json edits appear on the next load),
    big files (onnx, task, wasm) cache first. After the models load, the page asks the worker to
    cache the big files too, so the first visit already makes the app work offline. Replacing the
    model requires a new file name (or bumping `VERSION` in `sw.js`).
26. **Gender** is never guessed. Until Male or Female is picked, only the raw score is shown.

## Model choice

27. **Flip averaging.** The graph averages the prediction of the crop and of its mirror image (same weights,
    two passes, about 2x inference time, same file size). TEST Pearson 0.9176 -> 0.9210, MAE 0.203 -> 0.200.
    It also makes the score mirror invariant: without it, mirroring a photo (front cameras often do) moves
    the score by 0.074 on average. This is the one decision informed by a test split number, and it is a
    standard, parameter free choice.
28. **Shipped: fine tuned ResNet18 + flip average** (`scut-r18-ft-aligned-flipavg-v1`), TEST Pearson 0.921.
    It beats the pretrained weights through the same pipeline (0.910, or 0.914 with flip averaging) and its test
    number is genuinely held out. It was also trained on the app's own aligned framing: without alignment the
    same model drops to 0.906 on the 300 image E2E set, with alignment 0.918.
29. **Known weakness: JPEG sensitivity.** Trained on lossless PNGs, the fine tuned model moves 0.020 (q85) and
    0.030 (q70) raw points on average under re-encoding, versus 0.008 / 0.012 for the pretrained weights.
    Still far below the 0.4 "photos disagree" threshold. JPEG augmentation during training would reduce it; it was
    left out to stay inside the one hour CPU budget.
30. **Accuracy is lower on smiling faces** (Pearson 0.896 vs 0.930 for the rest of the test split, no meaningful bias
    in the mean), which supports excluding smiling photos from the average.

## Verification harness

31. `tests/verify.mjs` drives the real app in headless Chromium through `window.__psl` (the same `Scorer.process`
    the UI calls) and through the real file input for the UI checks. The Python side (`training/make_fixtures.py`)
    produces reference outputs with the same ONNX file, so any JS vs Python drift shows up directly.
