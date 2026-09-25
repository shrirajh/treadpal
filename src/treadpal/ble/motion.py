"""Estimated belt speed / deck incline for treadmills that only report targets.

Some treadmills echo the commanded target in the Treadmill Data "instantaneous"
fields: incline reads 10% a second after you ask for it, though the motor needs
many seconds to get there. We detect that from physically impossible jumps and
then model the motors moving toward the target at a steady rate.
"""

from __future__ import annotations

# A jump bigger than this between two ~1 Hz packets can't be real motion
IMPLAUSIBLE_SPEED_JUMP_KMH = 2.5
IMPLAUSIBLE_INCLINE_JUMP_PCT = 2.0
MAX_PACKET_GAP_S = 2.0


class MotionEstimator:
    """A value moving toward its target at ``rate`` units per second."""

    def __init__(self, rate_per_s: float) -> None:
        self.rate = rate_per_s
        self.value: float | None = None
        self.target: float | None = None
        self._t = 0.0

    def set_target(self, target: float, now: float) -> None:
        self.advance(now)
        if self.value is None:
            self.value = target  # First reading: assume it's already there
        self.target = target

    def sync(self, value: float, now: float) -> None:
        """Known-true reading (treadmill reports live values): jump straight to it."""
        self.value = self.target = value
        self._t = now

    def advance(self, now: float) -> float | None:
        if self.value is not None and self.target is not None:
            step = self.rate * max(0.0, now - self._t)
            diff = self.target - self.value
            self.value = self.target if abs(diff) <= step else self.value + (step if diff > 0 else -step)
        self._t = now
        return self.value


def is_implausible_jump(
    prev: tuple[float, float, float] | None, cur: tuple[float, float, float]
) -> str | None:
    """Given (time, speed, incline) of consecutive packets, describe an impossible jump."""
    if prev is None:
        return None
    dt = cur[0] - prev[0]
    if dt <= 0 or dt > MAX_PACKET_GAP_S:
        return None
    if abs(cur[2] - prev[2]) >= IMPLAUSIBLE_INCLINE_JUMP_PCT:
        return f"incline jumped {prev[2]:.1f}->{cur[2]:.1f}% in {dt:.1f}s"
    if abs(cur[1] - prev[1]) >= IMPLAUSIBLE_SPEED_JUMP_KMH:
        return f"speed jumped {prev[1]:.1f}->{cur[1]:.1f} km/h in {dt:.1f}s"
    return None
