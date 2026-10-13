"""Feature engineering.

Aturan emas: fitur untuk sebuah balapan HANYA boleh memakai informasi dari balapan
SEBELUMNYA (shift(1)), kecuali fitur skenario (grid, hujan, safety car, suhu trek) yang
memang menjadi input skenario saat prediksi.
"""
import numpy as np
import pandas as pd

WINDOW = 5  # jumlah balapan terakhir untuk "form"

FEATURES = [
    "grid",                 # posisi start
    "gap_pace",             # selisih pace form pembalap vs yang tercepat di balapan itu (%)
    "gap_team_pace",        # idem untuk tim
    "gap_team_pit",         # selisih kecepatan pit stop tim vs tim tercepat (detik)
    "form_deg",             # degradasi ban rata-rata (detik/lap)
    "circuit_avg_finish",   # rata-rata posisi finis pembalap di sirkuit ini
    "circuit_sc_rate",      # peluang safety car di sirkuit ini
    "circuit_rain_rate",    # peluang hujan di sirkuit ini
    "circuit_track_temp",   # suhu trek rata-rata sirkuit ini
    "sc_laps",              # skenario: jumlah lap di bawah SC/VSC
    "rain",                 # skenario: hujan (1) / kering (0)
    "track_temp",           # skenario: suhu trek
]

FEATURE_LABELS = {
    "grid": "Posisi start",
    "gap_pace": "Pace pembalap",
    "gap_team_pace": "Pace tim",
    "gap_team_pit": "Kecepatan pit stop tim",
    "form_deg": "Degradasi ban",
    "circuit_avg_finish": "Rekam jejak di sirkuit",
    "circuit_sc_rate": "Peluang safety car sirkuit",
    "circuit_rain_rate": "Peluang hujan sirkuit",
    "circuit_track_temp": "Suhu trek khas sirkuit",
    "sc_laps": "Lap di bawah safety car",
    "rain": "Hujan",
    "track_temp": "Suhu trek",
}


def _roll(s: pd.Series, shift: bool) -> pd.Series:
    s = s.shift(1) if shift else s
    return s.rolling(WINDOW, min_periods=1).mean()


def add_form(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Tambah fitur form (rata-rata WINDOW balapan terakhir) untuk pembalap dan tim."""
    df = df.sort_values(["race_idx", "grid"]).reset_index(drop=True).copy()

    for col, out in [("pace_delta", "form_pace"), ("deg", "form_deg"), ("grid", "form_grid")]:
        df[out] = df.groupby("driver")[col].transform(lambda s: _roll(s, shift))

    team = (df.groupby(["team", "race_idx"], as_index=False)[["pace_delta", "pit_dur"]]
              .mean().sort_values(["team", "race_idx"]))
    team["team_form_pace"] = team.groupby("team")["pace_delta"].transform(lambda s: _roll(s, shift))
    team["team_form_pit"] = team.groupby("team")["pit_dur"].transform(lambda s: _roll(s, shift))
    return df.merge(team[["team", "race_idx", "team_form_pace", "team_form_pit"]],
                    on=["team", "race_idx"], how="left")


def add_gaps(df: pd.DataFrame, group_col: str = "race_idx") -> pd.DataFrame:
    """Ubah fitur absolut menjadi selisih terhadap yang terbaik di balapan yang sama."""
    for src, out in [("form_pace", "gap_pace"), ("team_form_pace", "gap_team_pace"),
                     ("team_form_pit", "gap_team_pit")]:
        df[out] = df[src] - df.groupby(group_col)[src].transform("min")
    return df


def circuit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik per sirkuit dari seluruh sejarah (dipakai saat prediksi)."""
    races = df.drop_duplicates("race_idx")
    t = races.groupby("circuit").agg(
        circuit_sc_rate=("sc_laps", lambda s: float((s > 0).mean())),
        circuit_rain_rate=("rain", "mean"),
        circuit_track_temp=("track_temp", "mean"),
        sc_mean_laps=("sc_laps", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 6.0),
    )
    return t


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset latih: satu baris per pembalap per balapan, fitur hanya dari masa lalu."""
    d = add_form(df, shift=True)
    d = add_gaps(d)

    # rekam jejak pembalap di sirkuit (hanya balapan sebelumnya)
    d = d.sort_values("race_idx")
    d["circuit_avg_finish"] = d.groupby(["driver", "circuit"])["finish"].transform(
        lambda s: s.shift(1).expanding().mean())

    # statistik sirkuit (hanya balapan sebelumnya di sirkuit yang sama)
    races = d.drop_duplicates("race_idx")[["race_idx", "circuit", "sc_laps", "rain", "track_temp"]].copy()
    races = races.sort_values("race_idx")
    races["sc_flag"] = (races["sc_laps"] > 0).astype(float)
    for col, out in [("sc_flag", "circuit_sc_rate"), ("rain", "circuit_rain_rate"),
                     ("track_temp", "circuit_track_temp")]:
        races[out] = races.groupby("circuit")[col].transform(lambda s: s.shift(1).expanding().mean())
    # sirkuit baru: isi dengan rata-rata umum
    for out, col in [("circuit_sc_rate", "sc_flag"), ("circuit_rain_rate", "rain"),
                     ("circuit_track_temp", "track_temp")]:
        races[out] = races[out].fillna(races[col].mean())
    d = d.merge(races[["race_idx", "circuit_sc_rate", "circuit_rain_rate", "circuit_track_temp"]],
                on="race_idx", how="left")

    d["win"] = (d["finish"] == 1).astype(int)
    return d.sort_values(["race_idx", "grid"]).reset_index(drop=True)


def current_state(df: pd.DataFrame) -> pd.DataFrame:
    """Kondisi terkini tiap pembalap aktif (balapan terakhir di data), untuk prediksi ke depan."""
    d = add_form(df, shift=False)
    latest = d["race_idx"].max()
    base = d[d["race_idx"] == latest].copy()
    return base[["driver", "team", "form_pace", "form_deg", "form_grid",
                 "team_form_pace", "team_form_pit"]].reset_index(drop=True)


def driver_circuit_history(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["driver", "circuit"])["finish"].mean()
