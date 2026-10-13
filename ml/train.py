"""Melatih model, mengevaluasi, lalu membuat prediksi per sirkuit -> public/predictions.json.

    python ml/train.py
    python ml/train.py --sims 600
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, OUT_JSON, RACES_CSV, REQUIRED_COLUMNS
from features import (FEATURE_LABELS, FEATURES, add_form, build_training_frame,
                      circuit_table, current_state, driver_circuit_history)

SC_LAPS_SCENARIO = {0: 0, 1: 5, 2: 10}  # jumlah SC -> perkiraan lap di bawah SC


class BlendClassifier(ClassifierMixin, BaseEstimator):
    """Rata-rata probabilitas regresi logistik (stabil di data sedikit) dan gradient boosting
    dangkal (menangkap pola non-linear, misal pengaruh hujan x safety car)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.logreg_ = make_pipeline(SimpleImputer(), StandardScaler(),
                                     LogisticRegression(C=1.0, max_iter=3000)).fit(X, y)
        self.gb_ = HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=100,
            l2_regularization=3.0, min_samples_leaf=30, random_state=0).fit(X, y)
        return self

    def predict_proba(self, X):
        return 0.5 * self.logreg_.predict_proba(X) + 0.5 * self.gb_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def make_model() -> BlendClassifier:
    return BlendClassifier()


def normalize_per_race(frame: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    s = pd.Series(proba, index=frame.index)
    return (s / s.groupby(frame["race_idx"]).transform("sum").clip(lower=1e-9)).to_numpy()


def evaluate(test: pd.DataFrame, p: np.ndarray) -> dict:
    t = test.assign(p=p)
    top1 = top3 = n = 0
    logloss, pole, form = [], 0, 0
    for _, race in t.groupby("race_idx"):
        if race["win"].sum() != 1:
            continue
        n += 1
        ranked = race.sort_values("p", ascending=False)
        winner = race[race["win"] == 1].iloc[0]
        top1 += int(ranked.iloc[0]["win"] == 1)
        top3 += int(ranked.head(3)["win"].sum() == 1)
        logloss.append(-np.log(max(float(winner["p"]), 1e-6)))
        pole += int(winner["grid"] == race["grid"].min())
        form += int(race.sort_values("gap_pace").iloc[0]["win"] == 1)
    return {
        "races": n,
        "model_top1": top1 / n, "model_top3": top3 / n,
        "model_logloss": float(np.mean(logloss)),
        "baseline_pole": pole / n, "baseline_form_pace": form / n,
        "random_top1": float(1 / test.groupby("race_idx").size().mean()),
    }


def sample_podium(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ambil 3 pembalap podium per simulasi (Plackett-Luce)."""
    s, n = p.shape
    p = p.copy() + 1e-9
    order = np.zeros((s, 3), dtype=int)
    for k in range(3):
        c = p.cumsum(axis=1)
        r = rng.random((s, 1)) * c[:, -1:]
        idx = np.minimum((c < r).sum(axis=1), n - 1)
        order[:, k] = idx
        p[np.arange(s), idx] = 0.0
    return order


def simulate(model, base, circ, hist, circuit, sims, rng, rain=None, sc=None):
    """Simulasikan `sims` balapan di satu sirkuit.

    rain: None (acak sesuai peluang sirkuit) / 0 / 1;  sc: None (acak) / 0 / 1 / 2 periode SC.
    """
    n = len(base)
    c = circ.loc[circuit]

    # posisi start: ranking dari form kualifikasi + noise
    grid_score = base["form_grid"].to_numpy()[None, :] + rng.normal(0, 2.0, (sims, n))
    grid = grid_score.argsort(axis=1).argsort(axis=1) + 1

    team_codes, team_idx = np.unique(base["team"].to_numpy(), return_inverse=True)
    team_noise = rng.normal(0, 0.10, (sims, len(team_codes)))[:, team_idx]
    pace = (base["form_pace"].to_numpy()[None, :] + team_noise
            + rng.normal(0, 0.08, (sims, n)))
    team_pace = base["team_form_pace"].to_numpy()[None, :] + team_noise
    pit_noise = rng.normal(0, 0.4, (sims, len(team_codes)))[:, team_idx]
    team_pit = base["team_form_pit"].to_numpy()[None, :] + pit_noise

    gap = lambda a: a - a.min(axis=1, keepdims=True)  # noqa: E731

    if rain is None:
        rain_arr = (rng.random(sims) < c["circuit_rain_rate"]).astype(float)
    else:
        rain_arr = np.full(sims, float(rain))
    if sc is None:
        flag = rng.random(sims) < c["circuit_sc_rate"]
        laps = np.clip(rng.normal(c["sc_mean_laps"], 3, sims).round(), 1, 25)
        sc_arr = np.where(flag, laps, 0.0)
    else:
        sc_arr = np.full(sims, float(SC_LAPS_SCENARIO[sc]))
    temp = c["circuit_track_temp"] + rng.normal(0, 3, sims) - 6 * rain_arr

    hist_vals = np.array([hist.get((d, circuit), np.nan) for d in base["driver"]])
    rep = lambda v: np.repeat(v[:, None], n, axis=1)  # noqa: E731

    X = pd.DataFrame({
        "grid": grid.ravel(),
        "gap_pace": gap(pace).ravel(),
        "gap_team_pace": gap(team_pace).ravel(),
        "gap_team_pit": gap(team_pit).ravel(),
        "form_deg": np.tile(base["form_deg"].to_numpy(), sims),
        "circuit_avg_finish": np.tile(hist_vals, sims),
        "circuit_sc_rate": c["circuit_sc_rate"],
        "circuit_rain_rate": c["circuit_rain_rate"],
        "circuit_track_temp": c["circuit_track_temp"],
        "sc_laps": rep(sc_arr).ravel(),
        "rain": rep(rain_arr).ravel(),
        "track_temp": rep(temp).ravel(),
    })[FEATURES]

    p = model.predict_proba(X)[:, 1].reshape(sims, n)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)

    win = p.mean(axis=0)
    podium_counts = np.bincount(sample_podium(p, rng).ravel(), minlength=n) / sims
    rows = [{"code": base["driver"].iloc[i], "team": base["team"].iloc[i],
             "win": round(float(win[i]), 4), "podium": round(float(podium_counts[i]), 4)}
            for i in range(n)]
    return sorted(rows, key=lambda r: r["win"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    args = ap.parse_args()

    if not RACES_CSV.exists():
        raise SystemExit("data/races.csv belum ada. Jalankan: python ml/make_demo_data.py "
                         "(demo) atau python ml/fetch_data.py (data asli).")
    df = pd.read_csv(RACES_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Kolom hilang di races.csv: {missing}")
    src_file = DATA_DIR / "source.txt"
    source = src_file.read_text().strip() if src_file.exists() else "fastf1"

    train_all = build_training_frame(df)

    # split berdasarkan waktu: musim terakhir = data uji
    last_year = train_all["year"].max()
    test_mask = train_all["year"] == last_year
    if train_all.loc[test_mask, "race_idx"].nunique() < 5 or (~test_mask).sum() == 0:
        cutoff = np.sort(train_all["race_idx"].unique())[int(train_all["race_idx"].nunique() * 0.8)]
        test_mask = train_all["race_idx"] >= cutoff
    tr, te = train_all[~test_mask], train_all[test_mask]
    print(f"Latih: {tr['race_idx'].nunique()} balapan | Uji: {te['race_idx'].nunique()} balapan")

    model = make_model().fit(tr[FEATURES], tr["win"])
    p_test = normalize_per_race(te, model.predict_proba(te[FEATURES])[:, 1])
    metrics = evaluate(te, p_test)

    pi = permutation_importance(model, te[FEATURES], te["win"], scoring="neg_log_loss",
                                n_repeats=5, random_state=0)
    imp = pd.Series(np.clip(pi.importances_mean, 0, None), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False) if imp.sum() > 0 else imp
    importance = [{"feature": FEATURE_LABELS[k], "value": round(float(v), 4)} for k, v in imp.items()]

    print(f"Top-1 model {metrics['model_top1']:.0%} | pole {metrics['baseline_pole']:.0%} | "
          f"pace {metrics['baseline_form_pace']:.0%} | acak {metrics['random_top1']:.0%}")
    print(f"Top-3 model {metrics['model_top3']:.0%} | log-loss {metrics['model_logloss']:.3f}")

    # model final dilatih dari semua data, lalu simulasi per sirkuit
    final = make_model().fit(train_all[FEATURES], train_all["win"])
    base = current_state(df)
    circ = circuit_table(df)
    hist = driver_circuit_history(df)
    rng = np.random.default_rng(7)

    latest_year = df["year"].max()
    calendar = df[df["year"] == latest_year].drop_duplicates("circuit")[["circuit", "event_name", "round"]]
    scenarios = {"auto": (None, None)}
    for r_name, r in [("dry", 0), ("wet", 1)]:
        for sc in (0, 1, 2):
            scenarios[f"{r_name}_{sc}"] = (r, sc)

    circuits_out = []
    for _, row in calendar.sort_values("round").iterrows():
        cid = row["circuit"]
        res = {k: simulate(final, base, circ, hist, cid, args.sims, rng, rain=r, sc=s)
               for k, (r, s) in scenarios.items()}
        circuits_out.append({
            "id": cid, "name": row["event_name"],
            "p_sc": round(float(circ.loc[cid, "circuit_sc_rate"]), 3),
            "p_rain": round(float(circ.loc[cid, "circuit_rain_rate"]), 3),
            "avg_track_temp": round(float(circ.loc[cid, "circuit_track_temp"]), 1),
            "scenarios": res,
        })
        top = res["auto"][0]
        print(f"  {row['event_name']:<28} favorit: {top['code']} ({top['win']:.0%})")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "season_basis": int(latest_year),
        "n_races_total": int(df["race_idx"].nunique()),
        "n_sims": args.sims,
        "metrics": metrics,
        "importance": importance,
        "circuits": circuits_out,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nTersimpan: {OUT_JSON}")


if __name__ == "__main__":
    main()
