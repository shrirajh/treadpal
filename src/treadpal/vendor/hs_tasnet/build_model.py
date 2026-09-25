"""Rebuild the vendored stem separation model (hop128-int8.onnx) from the upstream release.

    uv run --extra stems-build python src/treadpal/vendor/hs_tasnet/build_model.py

1. Download the released streaming HS-TasNet (StemgenRT, pinned commit) and
   check its size and SHA-256.
2. Rewrite four ops the CPU runs badly, without changing the maths:
   - conv_encode: a Conv whose 1024-tap kernel spans its whole input, so one
     output position -> Reshape + MatMul (and so int8-quantizable)
   - basis_to_embed: a 1x1 Conv -> MatMul over channels
   - the forward real DFT (1024 -> 513 bins) -> MatMul with a cos/sin basis
   - the inverse DFT, whose real part is cropped to samples 768:1024 ->
     one MatMul straight to those 256 samples
3. Check the rewritten graph against the original on real-ish audio (must
   agree to float rounding, > 90 dB SNR on every stem).
4. Quantize the network's matrix weights to int8 (dynamic). The DFT bases stay
   float32: quantizing them costs vocal accuracy for little speed.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hop128-int8.onnx"

SOURCE_URL = (
    "https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/"
    "f8fb3beb95f8a17f80e1c195964f9cf1c42f2cee/model/model.onnx"
)
SOURCE_SHA256 = "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3"
SOURCE_BYTES = 111_344_465
N = 1024  # Analysis FFT size
STATES = {
    "audio_history": (1, 2, 896),
    "fusion_hidden": (2, 1, 1000),
    "spectral_numerator_tail": (1, 4, 2, 128),
    "waveform_tail": (1, 4, 2, 128),
}


def download(dest: Path) -> Path:
    """Fetch the release to dest, verified; reuse it if already there and intact."""
    if dest.exists() and dest.stat().st_size == SOURCE_BYTES and _sha256(dest) == SOURCE_SHA256:
        return dest
    h = hashlib.sha256()
    count = 0
    tmp = dest.with_suffix(".part")
    try:
        with urlopen(SOURCE_URL, timeout=30) as r, tmp.open("wb") as out:
            while block := r.read(1 << 20):
                count += len(block)
                if count > SOURCE_BYTES:
                    raise ValueError("Source model download is larger than expected")
                h.update(block)
                out.write(block)
        if count != SOURCE_BYTES or h.hexdigest() != SOURCE_SHA256:
            raise ValueError("Source model download failed verification (size or SHA-256)")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rewrite(model):  # noqa: ANN001, ANN201 - onnx.ModelProto
    """Replace the Conv/DFT ops described above with equivalent MatMuls, in place."""
    from onnx import helper, numpy_helper

    g = model.graph
    init = {t.name: t for t in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    by_name = {n.name: n for n in g.node}
    new_inits = []
    replace: dict[str, list] = {}

    def const(name: str) -> np.ndarray:
        n = prod[name]
        assert n.op_type == "Constant"
        return numpy_helper.to_array(n.attribute[0].t)

    def attrs(n) -> dict:  # noqa: ANN001
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def add(name: str, arr: np.ndarray) -> str:
        arr = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
        new_inits.append(numpy_helper.from_array(arr, name))
        return name

    def node(op: str, ins: list[str], outs: list[str], name: str, **kw):  # noqa: ANN202
        return helper.make_node(op, ins, outs, name=f"treadpal/{name}", **kw)

    # conv_encode: kernel spans the whole [1, 2, 1024] input -> one matrix product
    c = by_name["/conv_encode/Conv"]
    w = numpy_helper.to_array(init[c.input[1]])
    assert w.shape == (3000, 2, N) and attrs(c)["strides"] == [128] and attrs(c)["pads"] == [0, 0]
    replace[c.name] = [
        node("Reshape", [c.input[0], add("treadpal/shape_enc_in", np.array([1, 2 * N]))], ["treadpal/enc_in"], "enc_in"),
        node("MatMul", ["treadpal/enc_in", add("treadpal/enc_w", w.reshape(3000, -1).T.copy())], ["treadpal/enc_mm"], "enc_matmul"),
        node("Add", ["treadpal/enc_mm", c.input[2]], ["treadpal/enc_add"], "enc_bias"),
        node("Reshape", ["treadpal/enc_add", add("treadpal/shape_enc_out", np.array([1, 3000, 1]))], [c.output[0]], "enc_out"),
    ]

    # basis_to_embed: 1x1 Conv -> MatMul over channels (weights as the B input, so it quantizes)
    c = by_name["/basis_to_embed/Conv"]
    w = numpy_helper.to_array(init[c.input[1]])
    assert w.shape == (500, 1500, 1)
    replace[c.name] = [
        node("Transpose", [c.input[0]], ["treadpal/b2e_in"], "b2e_in", perm=[0, 2, 1]),
        node("MatMul", ["treadpal/b2e_in", add("treadpal/b2e_w", w[:, :, 0].T.copy())], ["treadpal/b2e_mm"], "b2e_matmul"),
        node("Add", ["treadpal/b2e_mm", c.input[2]], ["treadpal/b2e_add"], "b2e_bias"),
        node("Transpose", ["treadpal/b2e_add"], [c.output[0]], "b2e_out", perm=[0, 2, 1]),
    ]

    # Forward real DFT: [2, 1024, 1] -> [2, 513, (re, im)]
    d = by_name["/DFT"]
    assert attrs(d) == {"axis": 1, "inverse": 0, "onesided": 1}
    unsq = prod[d.input[0]]
    assert unsq.op_type == "Unsqueeze"  # Its input is the [2, 1024] frame
    th = 2 * np.pi * np.arange(N)[:, None] * np.arange(N // 2 + 1)[None, :] / N
    basis = np.empty((N, N + 2))
    basis[:, 0::2] = np.cos(th)
    basis[:, 1::2] = -np.sin(th)
    replace[d.name] = [
        node("MatMul", [unsq.input[0], add("treadpal/dft_basis", basis)], ["treadpal/dft_mm"], "dft_matmul"),
        node("Reshape", ["treadpal/dft_mm", add("treadpal/shape_dft_out", np.array([2, N // 2 + 1, 2]))], [d.output[0]], "dft_out"),
    ]

    # Inverse DFT -> real part (Gather) -> [1, 4, 2, 1024] -> crop 768:1024 (Slice)
    d = by_name["/DFT_1"]
    gather, reshape, crop = (by_name[n] for n in ("/Gather_3", "/Reshape_5", "/Slice_4"))
    assert attrs(d) == {"axis": 1, "inverse": 1, "onesided": 0}
    assert gather.input[0] == d.output[0] and int(const(gather.input[1])) == 0 and attrs(gather)["axis"] == 2
    assert reshape.input[0] == gather.output[0] and list(const(reshape.input[1])) == [1, 4, 2, N]
    assert crop.input[0] == reshape.output[0] and [int(const(i)[0]) for i in crop.input[1:]] == [768, N, 3, 1]
    th = 2 * np.pi * np.arange(N)[:, None] * np.arange(768, N)[None, :] / N
    inv = np.empty((2 * N, 256))
    inv[0::2] = np.cos(th) / N  # Re(X e^{iθ}) / N = (re cos θ - im sin θ) / N
    inv[1::2] = -np.sin(th) / N
    replace[d.name] = [
        node("Reshape", [d.input[0], add("treadpal/shape_idft_in", np.array([8, 2 * N]))], ["treadpal/idft_in"], "idft_in"),
        node("MatMul", ["treadpal/idft_in", add("treadpal/idft_basis", inv)], ["treadpal/idft_mm"], "idft_matmul"),
        node("Reshape", ["treadpal/idft_mm", add("treadpal/shape_idft_out", np.array([1, 4, 2, 256]))], [crop.output[0]], "idft_out"),
    ]
    for n in (gather, reshape, crop):
        replace[n.name] = []

    nodes = [m for n in g.node for m in replace.get(n.name, [n])]
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_inits)
    _prune(g)
    return model


def _prune(g) -> None:  # noqa: ANN001
    """Drop nodes and initializers nothing reads any more."""
    while True:
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        for n in dead:
            g.node.remove(n)
        stale = [t for t in g.initializer if t.name not in used]
        for t in stale:
            g.initializer.remove(t)
        if not dead and not stale:
            return


def _run(path: Path, audio: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    s = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    state = {k: np.zeros(v, np.float32) for k, v in STATES.items()}
    outs = ["separated_chunk"] + ["next_" + k for k in STATES]
    res = []
    for i in range(audio.shape[1] // 128):
        r = s.run(outs, {"audio_chunk": np.ascontiguousarray(audio[None, :, i * 128:(i + 1) * 128]), **state})
        state = dict(zip(STATES, r[1:]))
        res.append(r[0][0])
    return np.concatenate(res, axis=-1)


def _test_audio(seconds: float = 3.0, sr: int = 44100) -> np.ndarray:
    """Deterministic stereo mix: kick, bass line, a sung-ish formant tone off centre, noise hats."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    beat = t % 0.5
    kick = 0.8 * np.sin(2 * np.pi * (50 + 90 * np.exp(-beat * 30)) * beat) * np.exp(-beat * 18)
    bass = 0.25 * np.sin(2 * np.pi * 55 * 2 ** (np.floor(t) % 3 / 12) * t)
    voice = 0.15 * np.sin(2 * np.pi * 220 * t + 3 * np.sin(2 * np.pi * 5 * t)) * (1 + np.sin(2 * np.pi * 880 * t) * 0.3)
    hats = 0.08 * rng.standard_normal(len(t)) * np.exp(-((t + 0.25) % 0.5) * 60)
    left = kick + bass + 0.8 * voice + hats
    right = kick + bass + 0.5 * voice + 0.7 * hats
    return np.stack([left, right]).astype(np.float32)


def _snr(ref: np.ndarray, y: np.ndarray) -> list[float]:
    return [round(float(10 * np.log10((ref[k] ** 2).sum() / ((ref[k] - y[k]) ** 2).sum() + 1e-30)), 1) for k in range(4)]


def main() -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cache = Path.home() / ".cache" / "treadpal" / "hop128-source.onnx"
        cache.parent.mkdir(parents=True, exist_ok=True)
        source = download(cache)
        print(f"source {source} ({SOURCE_SHA256[:12]}, verified)")

        model = rewrite(onnx.load(str(source)))
        model.metadata_props.add(key="treadpal.source_sha256", value=SOURCE_SHA256)
        model.metadata_props.add(key="treadpal.rewrite", value="conv_encode, basis_to_embed, DFT, DFT_1+crop -> MatMul; int8 dynamic weights")
        onnx.checker.check_model(model)
        rewritten = tmp / "rewritten.onnx"
        onnx.save(model, str(rewritten))

        audio = _test_audio()
        ref = _run(source, audio)
        snr = _snr(ref, _run(rewritten, audio))
        print("rewritten float32 vs source, SNR dB (drums, bass, vocals, other):", snr)
        if min(snr) < 90:
            sys.exit("Rewritten graph differs from the source model")

        quantize_dynamic(
            rewritten, OUTPUT, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=["treadpal/dft_matmul", "treadpal/idft_matmul"],
        )
        print("int8 vs source, SNR dB:", _snr(ref, _run(OUTPUT, audio)))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB), SHA-256 {_sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
