// Scores the fixture players with the JavaScript engine and compares with the Python's answers.
// usage: node web_engine_check.js <site folder with model.json, players.json, fixtures.json> <engine.js>
const fs = require("fs"), path = require("path");
const dir = process.argv[2], engine = require(path.resolve(process.argv[3]));
const load = (f) => JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
const eng = engine.create(load("model.json"), load("players.json")), cases = load("fixtures.json");
const worst = { features: 0, totals: 0, path: 0, timeline: 0 }, bad = [];
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-3, Math.abs(a), Math.abs(b));
const note = (kind, a, b, where) => { const r = rel(a, b); if (r > worst[kind]) worst[kind] = r; if (!(r <= TOL[kind])) bad.push(`${kind} ${where}: js ${a} python ${b}`); };
const TOL = { features: 1e-6, totals: 2e-4, path: 2e-4, timeline: 2e-4 };
for (const c of cases) {
  const x = eng._features(c.hours, c.spend);
  c.features.forEach((v, i) => { if (v === null) { if (!Number.isNaN(x[i])) bad.push(`features ${c.name}[${i}]: js ${x[i]} python NaN`); } else note("features", x[i], v, `${c.name}[${i}]`); });
  const f = eng.forecast(c.hours, c.spend);
  for (const k of ["weeks_of_history", "weeks_since_active", "in_population"]) if (f[k] !== c.forecast[k]) bad.push(`status ${c.name} ${k}: js ${f[k]} python ${c.forecast[k]}`);
  for (const m in c.forecast.models) {
    const a = f.models[m], b = c.forecast.models[m];
    if (!a) { bad.push(`${c.name}: model ${m} missing in js`); continue; }
    for (const t in b.totals) note("totals", a.totals[t], b.totals[t], `${c.name} ${m} ${t}`);
    for (const k of ["hours", "active", "spend"]) b.path[k].forEach((v, i) => note("path", a.path[k][i], v, `${c.name} ${m} ${k}[${i}]`));
    if (a.has_shape !== b.has_shape || a.calibrated !== b.calibrated) bad.push(`${c.name} ${m}: flags differ`);
  }
  const tl = eng.timeline(c.hours, c.spend, c.weeks, c.step);
  if (JSON.stringify(tl.offsets) !== JSON.stringify(c.timeline.offsets)) bad.push(`${c.name}: timeline offsets differ`);
  else for (const m in c.timeline.models) for (const t in c.timeline.models[m]) c.timeline.models[m][t].forEach((v, i) => note("timeline", tl.models[m][t][i], v, `${c.name} ${m} ${t}@${tl.offsets[i]}`));
}
const info = eng.info(), sim = eng.simulate("rhythm", "returner", 3);
if (!info.models.length || !info.presets.length || sim.future_hours.length !== 52 || !sim.hours.some((v) => v > 0)) bad.push("info or simulate is broken");
console.log(JSON.stringify({ cases: cases.length, worst, failures: bad.length }));
if (bad.length) { console.error(bad.slice(0, 25).join("\n")); process.exit(1); }
