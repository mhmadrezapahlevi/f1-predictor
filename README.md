# Prediksi Juara F1 per Sirkuit

Website yang menampilkan peluang menang dan podium tiap pembalap F1 di setiap sirkuit.
Peluang dihitung dengan machine learning dari pace lap bersih, degradasi ban, kecepatan pit stop,
riwayat di sirkuit, kondisi trek (hujan, suhu), dan safety car.

```
ml/make_demo_data.py   data sintetis (agar bisa langsung dicoba)
ml/fetch_data.py       data F1 ASLI lewat FastF1  -> data/races.csv
ml/features.py         feature engineering tanpa kebocoran data
ml/train.py            latih + evaluasi + simulasi -> public/predictions.json
src/ index.html        website (Vite + TypeScript, tanpa framework)
```

## Coba cepat (data demo)

```bash
pip install numpy pandas scikit-learn
python ml/make_demo_data.py
python ml/train.py

npm install
npm run dev          # buka alamat yang tampil, biasanya http://localhost:5173
```

File `public/predictions.json` hasil demo sudah disertakan, jadi `npm install && npm run dev`
saja sudah menampilkan website.

## Pakai data asli

```bash
pip install -r requirements.txt
python ml/fetch_data.py --years 2022 2023 2024 2025   # lama di unduhan pertama, bisa dilanjutkan
python ml/train.py
npm run dev
```

Banner "Data demo" otomatis hilang bila data berasal dari FastF1.

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
