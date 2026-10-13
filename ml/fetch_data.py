"""Mengambil data balapan F1 ASLI dengan FastF1 dan menyimpannya ke data/races.csv.

Contoh:
    python ml/fetch_data.py                      # 2022-2025
    python ml/fetch_data.py --years 2023 2024    # musim tertentu

Catatan:
- Butuh internet. Unduhan pertama lambat (sekitar 1-2 menit per balapan), setelah itu
  FastF1 menyimpan cache di data/cache.
- Skrip bisa dihentikan lalu dijalankan lagi: balapan yang sudah ada dilewati.
- Satu baris = satu pembalap di satu balapan (kolom lihat common.REQUIRED_COLUMNS).
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import fastf1

from common import DATA_DIR, RACES_CSV, REQUIRED_COLUMNS

warnings.filterwarnings("ignore")
fastf1.Logger.set_log_level("WARNING")


def _tyre_degradation(clean: pd.DataFrame) -> pd.Series:
    """Rata-rata kenaikan waktu lap per lap umur ban (detik/lap), per pembalap."""
    out = {}
    for drv, d in clean.groupby("Driver"):
        slopes = []
        for _, stint in d.groupby("Stint"):
            if len(stint) >= 6 and stint["TyreLife"].nunique() >= 4:
                slopes.append(np.polyfit(stint["TyreLife"], stint["lt"], 1)[0])
        out[drv] = float(np.mean(slopes)) if slopes else np.nan
    return pd.Series(out)


def load_race(year: int, rnd: int) -> pd.DataFrame:
    race = fastf1.get_session(year, rnd, "R")
    race.load(laps=True, weather=True, messages=False, telemetry=False)
    quali = fastf1.get_session(year, rnd, "Q")
    quali.load(laps=False, weather=False, messages=False, telemetry=False)

    res = race.results.copy()
    qres = quali.results[["Abbreviation", "Position"]].rename(columns={"Position": "quali_pos"})

    laps = race.laps.copy()
    laps["lt"] = laps["LapTime"].dt.total_seconds()
    status = laps["TrackStatus"].astype(str)

    # lap bersih: bendera hijau, bukan lap masuk/keluar pit, bukan lap pertama
    clean = laps[
        status.eq("1") & laps["lt"].notna() & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna() & (laps["LapNumber"] > 1)
    ].copy()
    best_own = clean.groupby("Driver")["lt"].transform("min")
    clean = clean[clean["lt"] <= best_own * 1.07]

    med = clean.groupby("Driver")["lt"].median()
    pace_delta = (med / med.min() - 1.0) * 100.0  # persen di belakang pembalap tercepat
    deg = _tyre_degradation(clean)

    # pit stop: durasi di pit lane = PitOutTime (lap berikutnya) - PitInTime
    ls = laps.sort_values(["Driver", "LapNumber"]).copy()
    ls["next_out"] = ls.groupby("Driver")["PitOutTime"].shift(-1)
    pits = ls[ls["PitInTime"].notna() & ls["next_out"].notna()].copy()
    pits["dur"] = (pits["next_out"] - pits["PitInTime"]).dt.total_seconds()
    pits = pits[(pits["dur"] > 15) & (pits["dur"] < 45)]  # buang bendera merah / anomali
    pit_dur = pits.groupby("Driver")["dur"].median()
    pit_stops = laps[laps["PitInTime"].notna()].groupby("Driver").size()

    # safety car / VSC: jumlah lap balapan yang memuat status 4 (SC), 6 atau 7 (VSC)
    sc_mask = status.str.contains("4|6|7", regex=True)
    sc_laps = int(laps.loc[sc_mask, "LapNumber"].nunique())

    weather = race.weather_data
    rain = int(weather["Rainfall"].mean() > 0.1)
    track_temp = float(weather["TrackTemp"].mean())

    df = pd.DataFrame({
        "driver": res["Abbreviation"],
        "team": res["TeamName"],
        "grid": res["GridPosition"].replace(0, 20).fillna(20).astype(int),
        "finish": res["Position"].fillna(20).astype(int),
    })
    df = df.merge(qres, left_on="driver", right_on="Abbreviation", how="left").drop(columns="Abbreviation")
    df["quali_pos"] = df["quali_pos"].fillna(df["grid"]).astype(int)
    df["pace_delta"] = df["driver"].map(pace_delta)
    df["deg"] = df["driver"].map(deg)
    df["pit_stops"] = df["driver"].map(pit_stops).fillna(0).astype(int)
    df["pit_dur"] = df["driver"].map(pit_dur)
    df["pit_dur"] = df["pit_dur"].fillna(df.groupby("team")["pit_dur"].transform("median"))

    ev = race.event
    df["year"], df["round"] = year, rnd
    df["race_idx"] = year * 100 + rnd
    df["circuit"] = ev["Location"]
    df["event_name"] = ev["EventName"]
    df["sc_laps"], df["rain"], df["track_temp"] = sc_laps, rain, track_temp
    return df[REQUIRED_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "cache"
    cache.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    existing = pd.read_csv(RACES_CSV) if RACES_CSV.exists() else pd.DataFrame(columns=REQUIRED_COLUMNS)
    # jangan campur dengan data demo
    if (DATA_DIR / "source.txt").exists() and (DATA_DIR / "source.txt").read_text().strip() == "demo":
        existing = pd.DataFrame(columns=REQUIRED_COLUMNS)
    done = set(existing["race_idx"].unique())
    frames = [existing] if len(existing) else []
    now = pd.Timestamp.now()

    for year in args.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in schedule.iterrows():
            rnd = int(ev["RoundNumber"])
            idx = year * 100 + rnd
            if idx in done or pd.Timestamp(ev["EventDate"]) > now:
                continue
            try:
                print(f"[{year} R{rnd:02d}] {ev['EventName']} ...", flush=True)
                frames.append(load_race(year, rnd))
                pd.concat(frames, ignore_index=True).to_csv(RACES_CSV, index=False)
                (DATA_DIR / "source.txt").write_text("fastf1")
            except Exception as exc:  # satu balapan gagal tidak menghentikan semuanya
                print(f"   dilewati: {exc}")

    print(f"Selesai. Data tersimpan di {RACES_CSV}")


if __name__ == "__main__":
    main()
