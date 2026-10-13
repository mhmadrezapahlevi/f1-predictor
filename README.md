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

`npm run build` menghasilkan folder `dist/` (base `./`, aman untuk GitHub Pages / Vercel / Netlify).
Jalankan ulang `python ml/train.py` setiap ada balapan baru, lalu build ulang.

## Catatan metodologi

- Split latih/uji berdasarkan waktu (musim terakhir = uji), bukan acak.
- Fitur form hanya memakai balapan sebelumnya (`shift(1)`), jadi tidak ada kebocoran.
- Hujan dan safety car adalah *skenario*: pada prediksi disimulasikan acak sesuai peluang sirkuit,
  atau dikunci lewat tombol "Atur sendiri".
- Pace dihitung dari median lap bersih (bendera hijau, bukan lap pit). Belum dikoreksi
  untuk beban bahan bakar dan kompon ban, jadi ini perbaikan berikutnya yang paling berguna.
