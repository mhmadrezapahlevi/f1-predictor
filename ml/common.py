"""Path dan konstanta yang dipakai bersama oleh semua skrip ML."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RACES_CSV = DATA_DIR / "races.csv"
OUT_JSON = ROOT / "public" / "predictions.json"

# Kolom wajib pada data/races.csv (satu baris = satu pembalap di satu balapan)
REQUIRED_COLUMNS = [
    "year", "round", "race_idx", "circuit", "event_name", "driver", "team",
    "grid", "quali_pos", "finish", "pace_delta", "deg", "pit_stops",
    "pit_dur", "sc_laps", "rain", "track_temp",
]
