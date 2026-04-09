# TreadPal

BLE treadmill controller with BPM-synced speed. Connects to FTMS treadmills, adjusts speed to match your music's tempo using biomechanical stride modeling.

## Architecture

```
Mac (music) ──WebSocket──> Windows PC (BLE + beat detection + treadmill control)
```

**Server** (runs near the treadmill): BLE/FTMS connection, [beat_this](https://github.com/CPJKU/beat_this) ML beat detection, harmonic speed snapping, FastAPI.

**Agent** (`bpm_agent.py`): Single-file audio streamer. Captures system audio and sends it to the server. No ML dependencies — just audio capture + WebSocket.

## Setup

```bash
uv sync
uv run treadpal
# API docs at http://127.0.0.1:8080/docs
```

Config in `treadpal.json`:
```json
{
    "speed_is_mph": true,
    "host": "0.0.0.0",
    "bpm_min_speed_kmh": 4.0,
    "bpm_max_speed_kmh": 7.0,
    "bpm_harmonics": [0.25, 0.5, 1.0, 2.0, 4.0]
}
```

## BPM Agent

Copy `bpm_agent.py` to any machine playing music. Requires `uv`.

```bash
uv run bpm_agent.py --list-devices
uv run bpm_agent.py --server ws://<server-ip>:8080/ws/audio
```

**Windows**: Auto-detects WASAPI loopback via PyAudioWPatch.
**macOS**: Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`), then set up a Multi-Output Device in Audio MIDI Setup.
**Linux**: Auto-detects PulseAudio/PipeWire monitor source.

## API

| Endpoint | Description |
|---|---|
| `GET /api/status` | Treadmill connection state + live data |
| `POST /api/control/start\|stop\|pause` | Treadmill lifecycle |
| `POST /api/control/set_speed` | `{"value": 5.0}` in km/h |
| `POST /api/control/set_incline` | `{"value": 3.0}` in % |
| `POST /api/bpm/update` | `{"bpm": 140}` from any source |
| `PUT /api/bpm/config` | `{"min_speed_kmh": 4, "max_speed_kmh": 7}` |
| `GET /api/bpm/status` | Current BPM sync state |
| `WS /ws/audio?sr=44100` | Stream audio for server-side beat detection |
| `GET /api/history` | Logged workout data |

## How BPM sync works

1. Detect BPM from audio using [beat_this](https://github.com/CPJKU/beat_this) (ISMIR 2024)
2. Generate harmonic candidates from configurable list (default: `[0.25, 0.5, 1.0, 2.0, 4.0]`)
3. For each candidate, score by how closely the implied stride matches the biomechanically natural stride at each speed in your configured range
4. Pick the best (harmonic, speed) pair
5. Ramp treadmill speed toward it at 0.1 km/h per step
