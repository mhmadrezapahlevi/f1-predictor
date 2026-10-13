"""Membuat data SINTETIS (bukan data asli) dengan format yang sama seperti fetch_data.py.

Tujuannya hanya agar seluruh pipeline + website bisa dicoba tanpa internet.
Untuk data asli jalankan:  python ml/fetch_data.py
"""
import numpy as np
import pandas as pd

from common import DATA_DIR, RACES_CSV

rng = np.random.default_rng(42)
YEARS = [2021, 2022, 2023, 2024, 2025]

# tim -> (kekuatan dasar, pembalap)
TEAMS = {
    "Red Bull Racing": (1.00, ["VER", "TSU"]),
    "McLaren": (0.95, ["NOR", "PIA"]),
    "Ferrari": (0.90, ["LEC", "HAM"]),
    "Mercedes": (0.75, ["RUS", "ANT"]),
    "Aston Martin": (0.40, ["ALO", "STR"]),
    "Williams": (0.30, ["ALB", "SAI"]),
    "Alpine": (0.20, ["GAS", "COL"]),
    "Racing Bulls": (0.25, ["LAW", "HAD"]),
    "Haas F1 Team": (0.15, ["OCO", "BEA"]),
    "Kick Sauber": (0.05, ["HUL", "BOR"]),
}
DRIVER_SKILL = {"VER": 0.35, "NOR": 0.20, "LEC": 0.22, "HAM": 0.20, "RUS": 0.15,
                "PIA": 0.15, "ALO": 0.18, "ALB": 0.08, "SAI": 0.10, "GAS": 0.05}
WET_SKILL = {"VER": 0.30, "HAM": 0.25, "ALO": 0.25, "RUS": 0.10}

# sirkuit -> (nama event, peluang SC, peluang hujan, sulit menyalip 0-1, suhu trek)
CIRCUITS = {
    "Sakhir": ("Bahrain Grand Prix", 0.30, 0.00, 0.30, 38),
    "Jeddah": ("Saudi Arabian Grand Prix", 0.75, 0.00, 0.50, 34),
    "Melbourne": ("Australian Grand Prix", 0.60, 0.15, 0.50, 32),
    "Suzuka": ("Japanese Grand Prix", 0.40, 0.25, 0.60, 30),
    "Shanghai": ("Chinese Grand Prix", 0.40, 0.20, 0.35, 28),
    "Miami": ("Miami Grand Prix", 0.45, 0.15, 0.45, 45),
    "Imola": ("Emilia Romagna Grand Prix", 0.40, 0.20, 0.75, 30),
    "Monaco": ("Monaco Grand Prix", 0.60, 0.15, 0.97, 35),
    "Montréal": ("Canadian Grand Prix", 0.65, 0.25, 0.40, 30),
    "Barcelona": ("Spanish Grand Prix", 0.25, 0.05, 0.65, 42),
    "Spielberg": ("Austrian Grand Prix", 0.40, 0.15, 0.30, 40),
    "Silverstone": ("British Grand Prix", 0.35, 0.30, 0.35, 33),
    "Budapest": ("Hungarian Grand Prix", 0.30, 0.15, 0.85, 45),
    "Spa-Francorchamps": ("Belgian Grand Prix", 0.45, 0.40, 0.25, 30),
    "Zandvoort": ("Dutch Grand Prix", 0.40, 0.15, 0.85, 33),
    "Monza": ("Italian Grand Prix", 0.35, 0.05, 0.25, 38),
    "Baku": ("Azerbaijan Grand Prix", 0.75, 0.02, 0.30, 33),
    "Singapore": ("Singapore Grand Prix", 0.90, 0.20, 0.80, 35),
    "Austin": ("United States Grand Prix", 0.35, 0.10, 0.35, 40),
    "Mexico City": ("Mexico City Grand Prix", 0.35, 0.03, 0.55, 38),
    "São Paulo": ("São Paulo Grand Prix", 0.55, 0.30, 0.35, 38),
    "Las Vegas": ("Las Vegas Grand Prix", 0.45, 0.01, 0.35, 18),
    "Lusail": ("Qatar Grand Prix", 0.40, 0.00, 0.45, 36),
    "Yas Island": ("Abu Dhabi Grand Prix", 0.30, 0.00, 0.45, 33),
}


def main() -> None:
    rows = []
    circuit_names = list(CIRCUITS)
    for year in YEARS:
        # kekuatan tim berubah tiap musim
        team_perf = {t: base + rng.normal(0, 0.22) for t, (base, _) in TEAMS.items()}
        team_pit = {t: 22.0 + rng.uniform(0, 2.0) for t in TEAMS}
        affinity = {(t, c): rng.normal(0, 0.12) for t in TEAMS for c in circuit_names}

        for rnd, circuit in enumerate(circuit_names, start=1):
            event_name, p_sc, p_rain, diff, temp = CIRCUITS[circuit]
            rain = int(rng.random() < p_rain)
            sc_flag = rng.random() < p_sc
            sc_laps = int(np.clip(round(rng.normal(6, 3)), 2, 18)) if sc_flag else 0
            track_temp = float(temp + rng.normal(0, 4) - (8 if rain else 0))

            # perkembangan mobil selama musim
            for t in team_perf:
                team_perf[t] += rng.normal(0, 0.02)

            drivers = []
            for team, (_, pair) in TEAMS.items():
                for drv in pair:
                    perf = (team_perf[team] + DRIVER_SKILL.get(drv, 0.0)
                            + affinity[(team, circuit)] + rng.normal(0, 0.10))
                    drivers.append({"driver": drv, "team": team, "perf": perf})
            df = pd.DataFrame(drivers)

            # kualifikasi
            df["q_score"] = df["perf"] + rng.normal(0, 0.12, len(df))
            df["quali_pos"] = df["q_score"].rank(ascending=False, method="first").astype(int)
            df["grid"] = df["quali_pos"]
            penalized = rng.random(len(df)) < 0.05
            df.loc[penalized, "grid"] = np.minimum(df.loc[penalized, "grid"] + 5, 20)

            # pit stop
            df["pit_dur"] = [team_pit[t] + rng.normal(0, 0.6) for t in df["team"]]
            df["pit_stops"] = np.clip(rng.choice([1, 2, 2, 3], len(df)) + (sc_laps > 0) * rng.choice([0, 0, 1], len(df)), 1, 4)

            # hasil balapan
            noise_sd = 0.18 * (1 + 1.4 * rain) + 0.02 * sc_laps
            compress = 1 - 0.02 * sc_laps
            wet = np.array([WET_SKILL.get(d, 0.0) for d in df["driver"]]) * rain
            score = (df["perf"] * compress
                     - 0.035 * diff * df["grid"]
                     - 0.06 * (df["pit_dur"] - df["pit_dur"].mean())
                     + wet + rng.normal(0, noise_sd, len(df)))
            dnf = rng.random(len(df)) < 0.05
            score = score - dnf * 3.0
            df["finish"] = score.rank(ascending=False, method="first").astype(int)

            # pace (persen di belakang pembalap tercepat) dan degradasi ban
            df["pace_delta"] = (df["perf"].max() - df["perf"]) * 0.55 + np.abs(rng.normal(0, 0.06, len(df)))
            df["pace_delta"] -= df["pace_delta"].min()
            df["deg"] = np.clip(0.06 + 0.001 * (track_temp - 30) + rng.normal(0, 0.015, len(df)), 0.0, None)

            df["year"], df["round"] = year, rnd
            df["race_idx"] = year * 100 + rnd
            df["circuit"], df["event_name"] = circuit, event_name
            df["sc_laps"], df["rain"], df["track_temp"] = sc_laps, rain, track_temp
            rows.append(df.drop(columns=["perf", "q_score"]))

    out = pd.concat(rows, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(RACES_CSV, index=False)
    (DATA_DIR / "source.txt").write_text("demo")
    print(f"Data demo (sintetis) disimpan: {RACES_CSV}  ({len(out)} baris, {out['race_idx'].nunique()} balapan)")


if __name__ == "__main__":
    main()
