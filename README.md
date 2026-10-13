# F1 Winner Predictions by Circuit

A website that shows each F1 driver's win and podium probabilities at every circuit.
Probabilities are calculated using machine learning from clean-lap pace, tyre degradation, pit stop speed,
circuit history, track conditions (rain, temperature), and safety cars.

```
ml/make_demo_data.py   synthetic data (so you can try it immediately)
ml/fetch_data.py       REAL F1 data via FastF1  -> data/races.csv
ml/features.py         feature engineering without data leakage
ml/train.py            train + evaluate + simulate -> public/predictions.json
src/ index.html        website (Vite + TypeScript, no framework)
```

## Quick start (demo data)
```bash
pip install numpy pandas scikit-learn
python ml/make_demo_data.py
python ml/train.py

npm install
npm run dev         # open the shown address, usually http://localhost:5173
```

The demo `public/predictions`.json output is included, so `npm install && npm run dev`
alone will display the website.

## Using real data

```bash
pip install -r requirements.txt
python ml/fetch_data.py --years 2022 2023 2024 2025   # lama di unduhan pertama, bisa dilanjutkan
python ml/train.py
npm run dev
```

The "Demo data" banner automatically disappears when the data comes from FastF1.

## Deploy

`npm run build` produces a `dist/` folder (base `./`, safe for GitHub Pages / Vercel / Netlify).
Re-run `python ml/train.py` after each new race, then rebuild.

## Methodology notes

- Train/test split is by time (last season = test), not random.
- Form features only use previous races (`shift(1)`), so there is no leakage.
- Rain and safety cars are scenarios: during prediction they are simulated randomly according to the circuit's probabilities,
or locked via the "Set manually" button.
- Pace is calculated from the median clean lap (green flag, not pit lap). It has not been corrected
for fuel load and tyre compound, so that is the most useful next improvement.
