SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS treadmill_data (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL,
    speed_kmh      REAL    NOT NULL,
    incline_pct    REAL    NOT NULL,
    distance_m     INTEGER NOT NULL,
    elapsed_time_s INTEGER NOT NULL,
    calories_kcal  INTEGER NOT NULL,
    heart_rate_bpm INTEGER
);

CREATE INDEX IF NOT EXISTS idx_treadmill_data_timestamp
    ON treadmill_data(timestamp);

CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time     TEXT NOT NULL,
    end_time       TEXT,
    device_name    TEXT,
    device_address TEXT
);

CREATE TABLE IF NOT EXISTS bpm_sync_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    detected_bpm        REAL    NOT NULL,
    selected_harmonic   REAL    NOT NULL,
    effective_cadence   REAL    NOT NULL,
    commanded_speed_kmh REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bpm_sync_log_timestamp
    ON bpm_sync_log(timestamp);
"""
