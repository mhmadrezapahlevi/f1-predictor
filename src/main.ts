import "./style.css";

// ---------- tipe data (sesuai public/predictions.json dari ml/train.py) ----------
interface DriverResult { code: string; team: string; win: number; podium: number }
interface Circuit {
  id: string; name: string; p_sc: number; p_rain: number; avg_track_temp: number;
  scenarios: Record<string, DriverResult[]>;
}
interface Metrics {
  races: number; model_top1: number; model_top3: number; model_logloss: number;
  baseline_pole: number; baseline_form_pace: number; random_top1: number;
}
interface Payload {
  generated_at: string; data_source: "demo" | "fastf1"; season_basis: number;
  n_races_total: number; n_sims: number; metrics: Metrics;
  importance: { feature: string; value: number }[]; circuits: Circuit[];
}

// ---------- warna tim ----------
const TEAM_COLORS: [string, string][] = [
  ["red bull", "#1e41ff"], ["mclaren", "#ff8000"], ["ferrari", "#dc0000"],
  ["mercedes", "#00a39a"], ["aston", "#00665f"], ["alpine", "#0090ff"],
  ["williams", "#005aff"], ["racing bulls", "#6692ff"], ["rb", "#6692ff"],
  ["haas", "#7d8590"], ["sauber", "#3ec43e"], ["alphatauri", "#4e7c9b"],
  ["alfa romeo", "#9b0000"],
];
function teamColor(team: string): string {
  const t = team.toLowerCase();
  for (const [key, color] of TEAM_COLORS) if (t.includes(key)) return color;
  return "#8a93a0";
}

// ---------- util ----------
const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

// ---------- state ----------
const state = { circuitId: "", manual: false, rain: "dry" as "dry" | "wet", sc: 0 as 0 | 1 | 2 };
let data: Payload;

function scenarioKey(): string {
  return state.manual ? `${state.rain}_${state.sc}` : "auto";
}

// ---------- render ----------
function renderShell() {
  const d = data;
  const demo = d.data_source === "demo";
  const options = d.circuits
    .map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`)
    .join("");

  $("#app").innerHTML = `
    ${demo ? `<div class="banner" role="note"><strong>Data demo.</strong> Angka di halaman ini dihitung dari data sintetis, bukan hasil balapan asli. Jalankan <code>python ml/fetch_data.py</code> lalu <code>python ml/train.py</code> untuk memakai data F1 asli.</div>` : ""}
    <header class="top">
      <p class="site">Prediksi Juara F1</p>
      <div class="picker">
        <label for="circuit">Sirkuit</label>
        <select id="circuit">${options}</select>
      </div>
    </header>

    <main>
      <section class="hero" aria-live="polite">
        <h1 id="title"></h1>
        <p id="lede" class="lede"></p>
      </section>

      <section class="scenario" aria-label="Skenario balapan">
        <div class="seg" role="group" aria-label="Mode skenario">
          <button type="button" data-mode="auto" aria-pressed="true">Sesuai peluang sirkuit</button>
          <button type="button" data-mode="manual" aria-pressed="false">Atur sendiri</button>
        </div>
        <div id="manual" class="manual" hidden>
          <div class="field">
            <span class="field-label">Cuaca</span>
            <div class="seg" role="group" aria-label="Cuaca">
              <button type="button" data-rain="dry" aria-pressed="true">Kering</button>
              <button type="button" data-rain="wet" aria-pressed="false">Hujan</button>
            </div>
          </div>
          <div class="field">
            <span class="field-label">Safety car</span>
            <div class="seg" role="group" aria-label="Safety car">
              <button type="button" data-sc="0" aria-pressed="true">Tanpa</button>
              <button type="button" data-sc="1" aria-pressed="false">1 kali</button>
              <button type="button" data-sc="2" aria-pressed="false">2 kali</button>
            </div>
          </div>
        </div>
      </section>

      <div class="grid">
        <section class="tower" aria-labelledby="tower-h">
          <h2 id="tower-h" class="sr-only">Peluang tiap pembalap</h2>
          <div class="tower-head" aria-hidden="true"><span></span><span>Pembalap</span><span>Peluang menang</span><span class="num">Podium</span></div>
          <ol id="rows" class="rows"></ol>
        </section>

        <aside class="side">
          <section>
            <h2>Kondisi sirkuit</h2>
            <dl class="facts" id="facts"></dl>
          </section>
          <section>
            <h2>Seberapa bisa dipercaya</h2>
            <p class="small">Diuji pada ${d.metrics.races} balapan terbaru yang tidak dipakai untuk melatih model.</p>
            <table class="metrics">
              <thead><tr><th scope="col">Cara menebak juara</th><th scope="col" class="num">Benar</th></tr></thead>
              <tbody>
                <tr class="hl"><th scope="row">Model ini</th><td class="num">${pct(d.metrics.model_top1)}</td></tr>
                <tr><th scope="row">Tebak peraih pole</th><td class="num">${pct(d.metrics.baseline_pole)}</td></tr>
                <tr><th scope="row">Tebak pace terbaik</th><td class="num">${pct(d.metrics.baseline_form_pace)}</td></tr>
                <tr><th scope="row">Tebak acak</th><td class="num">${pct(d.metrics.random_top1)}</td></tr>
              </tbody>
            </table>
            <p class="small">Juara masuk 3 besar prediksi model di ${pct(d.metrics.model_top3)} balapan.</p>
          </section>
          <section>
            <h2>Faktor paling berpengaruh</h2>
            <ul class="imp">
              ${d.importance.slice(0, 6).map((i) => `
                <li><span>${esc(i.feature)}</span><span class="imp-bar"><i style="width:${Math.max(2, i.value * 100 / Math.max(d.importance[0].value, 0.0001))}%"></i></span></li>`).join("")}
            </ul>
          </section>
        </aside>
      </div>

      <details class="method">
        <summary>Cara model bekerja</summary>
        <div class="method-body">
          <p>Setiap pembalap di setiap balapan lampau diubah menjadi satu baris data: posisi start, pace dan degradasi ban dari lap bersih, kecepatan pit stop tim, riwayat di sirkuit itu, dan kondisi balapan (hujan, jumlah lap di bawah safety car, suhu trek). Semua fitur form hanya memakai balapan <em>sebelumnya</em>, supaya hasil balapan yang sedang dihitung tidak bocor ke dalam fitur.</p>
          <p>Model (gabungan regresi logistik dan gradient boosting) belajar memperkirakan peluang menang. Untuk tiap sirkuit, ${d.n_sims} balapan disimulasikan dengan variasi acak pada posisi start, pace, pit stop, hujan, dan safety car. Peluang menang adalah rata-rata dari simulasi itu.</p>
          <p>Data dasar: form pembalap dan tim sampai musim ${d.season_basis}, dari ${d.n_races_total} balapan. Prediksi dibuat ${new Date(d.generated_at).toLocaleString("id-ID")}.</p>
          <p>Batasan: F1 hanya punya sekitar 24 balapan per musim, dan hasilnya sangat dipengaruhi kekuatan mobil, kecelakaan, dan penalti yang tidak bisa diprediksi. Anggap angka ini sebagai peluang, bukan kepastian.</p>
        </div>
      </details>
    </main>
  `;
}

function renderResults() {
  const c = data.circuits.find((x) => x.id === state.circuitId)!;
  const rows = c.scenarios[scenarioKey()];
  const max = Math.max(...rows.map((r) => r.win), 0.0001);
  const top = rows[0];

  $("#title").textContent = c.name;
  $("#lede").innerHTML = `Favorit: <strong>${esc(top.code)}</strong> dengan peluang menang ${pct(top.win)}. ${scenarioText()}`;

  $("#rows").innerHTML = rows
    .map((r, i) => `
      <li class="row${i === 0 ? " lead" : ""}" style="--team:${teamColor(r.team)}">
        <span class="pos">${i + 1}</span>
        <span class="who"><b>${esc(r.code)}</b><small>${esc(r.team)}</small></span>
        <span class="bar" role="img" aria-label="Peluang menang ${pct(r.win, 1)}">
          <i style="width:${(r.win / max) * 100}%"></i><em>${pct(r.win, 1)}</em>
        </span>
        <span class="num podium">${pct(r.podium)}</span>
      </li>`)
    .join("");

  $("#facts").innerHTML = `
    <div><dt>Peluang safety car</dt><dd>${pct(c.p_sc)}</dd></div>
    <div><dt>Peluang hujan</dt><dd>${pct(c.p_rain)}</dd></div>
    <div><dt>Suhu trek rata-rata</dt><dd>${c.avg_track_temp.toFixed(0)}°C</dd></div>`;
}

function scenarioText(): string {
  if (!state.manual) return "Hujan dan safety car muncul acak sesuai peluang historis sirkuit ini.";
  const rain = state.rain === "wet" ? "hujan" : "kering";
  const sc = state.sc === 0 ? "tanpa safety car" : `${state.sc} kali safety car`;
  return `Skenario: ${rain}, ${sc}.`;
}

function syncControls() {
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
    b.setAttribute("aria-pressed", String((b.dataset.mode === "manual") === state.manual)));
  document.querySelectorAll<HTMLButtonElement>("[data-rain]").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.rain === state.rain)));
  document.querySelectorAll<HTMLButtonElement>("[data-sc]").forEach((b) =>
    b.setAttribute("aria-pressed", String(Number(b.dataset.sc) === state.sc)));
  $("#manual").hidden = !state.manual;
}

function bind() {
  $<HTMLSelectElement>("#circuit").addEventListener("change", (e) => {
    state.circuitId = (e.target as HTMLSelectElement).value;
    renderResults();
  });
  $("#app").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest("button");
    if (!btn) return;
    if (btn.dataset.mode) state.manual = btn.dataset.mode === "manual";
    else if (btn.dataset.rain) state.rain = btn.dataset.rain as "dry" | "wet";
    else if (btn.dataset.sc) state.sc = Number(btn.dataset.sc) as 0 | 1 | 2;
    else return;
    syncControls();
    renderResults();
  });
}

async function main() {
  const app = $("#app");
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}predictions.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = (await res.json()) as Payload;
  } catch {
    app.innerHTML = `<div class="empty"><h1>Prediksi belum tersedia</h1>
      <p>File <code>public/predictions.json</code> tidak ditemukan. Buat dulu dengan:</p>
      <pre>python ml/make_demo_data.py
python ml/train.py</pre></div>`;
    return;
  }
  state.circuitId = data.circuits[0].id;
  renderShell();
  bind();
  syncControls();
  renderResults();
}

main();
