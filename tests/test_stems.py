"""Stem separation: features, model download verification, and the live worker."""

from __future__ import annotations

import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import pytest

from treadpal.audio import stems
from treadpal.audio.stems import DISPLAY_ORDER, MODEL_ORDER, SR, StemFeatures


def _tone(freq: float, n: int, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def _stems(**parts: np.ndarray) -> np.ndarray:
    n = max(len(p) for p in parts.values())
    out = np.zeros((4, 2, n), dtype=np.float32)
    for name, x in parts.items():
        out[MODEL_ORDER.index(name)] = x
    return out


def test_features_levels_follow_the_stems() -> None:
    """A loud bass and an absent vocal: bass fills its lane, the vocal stays empty."""
    f = StemFeatures()
    n = SR * 2
    frames = f.feed(_stems(bass=_tone(110.0, n), other=_tone(440.0, n, 0.1), vocals=_tone(660.0, n, 1e-4)))
    assert len(frames) == n // StemFeatures.FRAME
    last = frames[-1]
    assert last["names"] == list(DISPLAY_ORDER)
    level = dict(zip(last["names"], last["level"]))
    assert level["bass"] > 0.8
    assert level["other"] > 0.5  # Quieter, but its own lane still shows it
    assert level["vocals"] == 0.0  # 70 dB under: bleed, not a vocal
    assert level["drums"] == 0.0
    assert len(last["spec"]) == 4 * last["bands"]
    # Chroma from the pitched stems: A (110 Hz, 440 Hz) dominates
    assert int(np.argmax(last["chroma"])) == 9


def test_features_share_one_db_scale() -> None:
    """Lanes are on the song's scale, not each stem's own: faint stays faint, bleed is nothing."""
    f = StemFeatures(floor_db=36.0)
    n = SR * 2
    loud = _tone(110.0, n, 0.3)
    frames = f.feed(_stems(
        bass=loud,
        other=_tone(440.0, n, 0.3 * 10 ** (-12 / 20)),  # 12 dB down
        vocals=_tone(660.0, n, 0.3 * 10 ** (-30 / 20)),  # 30 dB down: a faint sound
        drums=_tone(3000.0, n, 0.3 * 10 ** (-45 / 20)),  # 45 dB down: separation bleed
    ))
    level = dict(zip(frames[-1]["names"], frames[-1]["level"]))
    assert level["bass"] > 0.9
    assert 0.55 < level["other"] < 0.75
    assert 0.05 < level["vocals"] < 0.25
    assert level["drums"] == 0.0
    spec = np.array(frames[-1]["spec"]).reshape(4, -1)
    assert spec[DISPLAY_ORDER.index("drums")].max() == 0.0
    assert spec[DISPLAY_ORDER.index("vocals")].max() < spec[DISPLAY_ORDER.index("bass")].max() / 2


def test_each_stem_has_its_own_note() -> None:
    """Bass on A, other on C, vocals on E: three different notes, not one shared one."""
    f = StemFeatures()
    n = SR * 2
    noise = (0.05 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
    last = f.feed(_stems(
        bass=_tone(110.0, n),  # A2
        other=_tone(261.63, n, 0.2),  # C4
        vocals=_tone(329.63, n, 0.2),  # E4
        drums=noise,
    ))[-1]
    chroma = np.array(last["stem_chroma"]).reshape(4, 12)
    note = {name: int(np.argmax(c)) for name, c in zip(last["names"], chroma)}
    assert (note["bass"], note["other"], note["vocals"]) == (9, 0, 4)
    drums = chroma[last["names"].index("drums")]
    assert drums.mean() > 0.3  # Noise: every pitch class, so no clear note


def test_features_silence_and_drum_onsets() -> None:
    f = StemFeatures()
    assert f.feed(np.zeros((4, 2, 4096), dtype=np.float32))[-1] == {"type": "stems", "silent": True}

    # Kicks every 0.5 s on the drum stem, a steady bass note (no onsets of its own)
    n = SR * 4
    t = np.arange(n) / SR
    since = t % 0.5
    kick = (0.9 * np.sin(2 * np.pi * (50 + 90 * np.exp(-since * 30)) * since) * np.exp(-since * 18)).astype(np.float32)
    frames = f.feed(_stems(drums=kick, bass=_tone(55.0, n)))
    onset_times = [i * StemFeatures.FRAME / SR for i, fr in enumerate(frames) if fr.get("onset", 0) > 0]
    assert 6 <= len(onset_times) <= 8
    assert all(abs((ts % 0.5) - 0.0) < 0.06 or abs((ts % 0.5) - 0.5) < 0.06 for ts in onset_times)


def test_vendored_model_is_intact() -> None:
    assert stems.verified_model() == stems.MODEL_PATH


def test_build_download_rejects_wrong_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tampered or truncated source download never lands where the build would use it."""
    spec = importlib.util.spec_from_file_location("build_model", stems.MODEL_PATH.parent / "build_model.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "SOURCE_BYTES", 5)
    monkeypatch.setattr(build, "urlopen", lambda *a, **k: io.BytesIO(b"evil!"))
    with pytest.raises(ValueError, match="verification"):
        build.download(tmp_path / "source.onnx")
    assert list(tmp_path.iterdir()) == []


def test_worker_separates_in_real_time() -> None:
    """48 kHz stereo in, stem frames out, faster than real time: bass lands in the bass lane."""
    frames: list[dict] = []
    worker = stems.StemWorker(stems.StemModel(), 48000, frames.append)
    worker.MAX_BACKLOG_S = 1e9  # Fed faster than real time on purpose
    n = 48000 * 3
    t = np.arange(n) / 48000
    bass = 0.4 * np.sin(2 * np.pi * 55.0 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    stereo = np.stack([bass, bass]).astype(np.float32)
    start = time.perf_counter()
    for i in range(0, n, 1024):
        worker.feed(stereo[:, i:i + 1024])
    worker.close()
    worker._thread.join(timeout=30)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"separation took {elapsed:.1f}s for 3s of audio: slower than real time"
    live = [f for f in frames if not f.get("silent")]
    assert len(live) > 100
    level = dict(zip(live[-1]["names"], np.mean([f["level"] for f in live[-40:]], axis=0)))
    assert level["bass"] == max(level.values())
