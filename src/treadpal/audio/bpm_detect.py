"""Real-time BPM detection using aubio (with numpy spectral flux fallback)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("treadpal.audio")


class BpmDetector:
    """Real-time BPM detection from audio blocks."""

    def __init__(
        self,
        sample_rate: int = 44100,
        win_size: int = 512,
        hop_size: int = 256,
    ) -> None:
        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self._use_aubio = False
        self._tempo: Any = None

        try:
            import aubio  # ty: ignore[unresolved-import]

            self._tempo = aubio.tempo("default", win_size, hop_size, sample_rate)
            self._use_aubio = True
            logger.info("Using aubio for BPM detection")
        except ImportError:
            logger.info("aubio not available, using numpy spectral flux fallback")
            self._prev_spectrum: NDArray[np.float32] | None = None
            self._onset_times: list[float] = []
            self._frame_count: int = 0
            self._last_onset_time: float = 0.0

    def process(self, block: NDArray[np.float32]) -> float | None:
        """Process one hop_size block. Returns BPM if detected, else None."""
        if self._use_aubio:
            return self._process_aubio(block)
        return self._process_fallback(block)

    def _process_aubio(self, block: NDArray[np.float32]) -> float | None:
        is_beat = self._tempo(block)
        if is_beat[0] > 0:
            return float(self._tempo.get_bpm())
        return None

    def _process_fallback(self, block: NDArray[np.float32]) -> float | None:
        """Onset detection via spectral flux."""
        spectrum = np.abs(np.fft.rfft(block, n=self._win_size)).astype(np.float32)

        if self._prev_spectrum is not None:
            flux = float(np.sum(np.maximum(0, spectrum - self._prev_spectrum)))
            current_time = self._frame_count * self._hop_size / self._sample_rate

            # Adaptive threshold: only trigger if enough time since last onset
            min_onset_gap = 0.2  # 300 BPM max
            if flux > 0.5 and (current_time - self._last_onset_time) > min_onset_gap:
                self._onset_times.append(current_time)
                self._last_onset_time = current_time

                # Keep last 16 onsets
                if len(self._onset_times) > 16:
                    self._onset_times = self._onset_times[-16:]

                if len(self._onset_times) >= 4:
                    intervals = np.diff(self._onset_times[-8:])
                    median_interval = float(np.median(intervals))
                    if median_interval > 0:
                        bpm = 60.0 / median_interval
                        if 40 < bpm < 220:
                            self._prev_spectrum = spectrum
                            self._frame_count += 1
                            return bpm

        self._prev_spectrum = spectrum
        self._frame_count += 1  # Only increments when we didn't early-return above
        return None

    def get_bpm(self) -> float:
        """Get current estimated BPM."""
        if self._use_aubio and self._tempo is not None:
            return float(self._tempo.get_bpm())
        if self._onset_times and len(self._onset_times) >= 4:
            intervals = np.diff(self._onset_times[-8:])
            median_interval = float(np.median(intervals))
            if median_interval > 0:
                return 60.0 / median_interval
        return 0.0
