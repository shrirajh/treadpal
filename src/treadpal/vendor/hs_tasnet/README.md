# Vendored: streaming HS-TasNet stem separator

`hop128-int8.onnx` splits 44.1 kHz stereo into drums, bass, vocals and other,
128 samples at a time with one hop (2.9 ms) of latency. TreadPal's server runs
it on incoming music for the visualiser (`treadpal/audio/stems.py`).

## Provenance

| | |
|---|---|
| Architecture | [HS-TasNet](https://arxiv.org/abs/2402.17701), implemented by Phil Wang ([lucidrains/HS-TasNet](https://github.com/lucidrains/HS-TasNet)); streaming variant in the [sweetspotsoundsystem fork](https://github.com/sweetspotsoundsystem/HS-TasNet) at `c3ebe3dfb2105b07cfad4e7437a3e43dfff61661`. MIT, see [`LICENSE`](LICENSE). |
| Trained weights | [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt) `model/model.onnx` at `f8fb3beb95f8a17f80e1c195964f9cf1c42f2cee`, SHA-256 `b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3` (111,344,465 bytes). MIT, see [`LICENSE-stemgen-rt.md`](LICENSE-stemgen-rt.md). |
| This file | SHA-256 `3bfccc29554bca9446fdaa5ff6b5132f88a14ab4ac685f677a8e64ecc4667a67`, built by [`build_model.py`](build_model.py). The server checks this hash before loading it. |

Before vendoring, the fork's code and the model were reviewed. The fork's
streaming code only needs numpy and ONNX Runtime. Apart from the
`scripts/download_streaming_model.py` download (fixed commit, checked by
size and SHA-256), it makes no network calls and does no subprocess or eval
work. Its streaming checkpoints load with `weights_only=True`. The ONNX graph
uses only standard `ai.onnx` opset-17 ops, with no custom domains, functions
or external data.

## Changes from the release

The release needs ~4 ms of CPU per 2.9 ms hop, which is too slow for real time
on one core. At batch size 1 the time goes on reading weights and on op
implementations poorly suited to single-frame use. `build_model.py`:

1. **Rewrites four ops as matrix products, without changing the maths.** The
   rewritten graph matches the release to within float rounding (102–126 dB
   SNR per stem):
   - `conv_encode` is a Conv whose 1024-tap kernel spans its whole input,
     leaving one output position, so it's really one [3000 × 2048] matrix
     product.
   - `basis_to_embed` is a 1×1 Conv.
   - The forward real DFT (1024 → 513 bins) is a constant cos/sin basis.
   - The inverse DFT is followed by taking the real part and cropping to
     samples 768:1024, so it becomes one [2048 × 256] product.
2. **Quantizes the network's matrix weights to int8** (ONNX Runtime dynamic
   quantization). The two DFT bases stay float32.

Measured on the same CPU:

| | ms / hop (2 threads) | real time | Stem level error vs release (per 23 ms frame) |
|---|---|---|---|
| Release, float32 | 4.3 | 147% | — |
| Release, int8 | 2.1 | 73% | — |
| **Rewritten, int8 (this file)** | **1.06** | **37%** | median 0.06–0.55 dB, p95 ≤ 2.2 dB |

The largest level errors are on stems 30–40 dB below the mix (model bleed),
which the visualiser treats as silent anyway.

## Rebuilding

```bash
uv run --extra stems-build python src/treadpal/vendor/hs_tasnet/build_model.py
```

This downloads and verifies the release, rewrites it, checks the rewrite, and
quantizes it. If the output hash changes (for example with a newer ONNX
Runtime quantizer), update `MODEL_SHA256` in `treadpal/audio/stems.py`.

## Interface

Inputs are `audio_chunk` `[1,2,128]` plus four recurrent states; outputs are
`separated_chunk` `[1,4,2,128]` (drums, bass, vocals, other) plus the next
states. All tensors are float32 and the states start at zero. Each call returns
the *previous* hop, so discard the first output after a reset. Reset after any
gap in the audio.
