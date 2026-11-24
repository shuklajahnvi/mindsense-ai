import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Activity, CalendarDays, Layers, NotebookPen } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type Insights, type Period, type Topics } from "../api";
import { EmotionChip, ErrorBox, Note, Spinner, StatCard } from "../components";
import { CHART, GROUP_COLOR, SERIES, cap, num, pct } from "../lib";

const RANGES = [{ v: 30, l: "30 days" }, { v: 90, l: "90 days" }, { v: 365, l: "1 year" }, { v: 0, l: "All" }];
const axis = { stroke: CHART.axis, fontSize: 11, tickLine: false, axisLine: { stroke: CHART.baseline } };
const tooltipStyle = { contentStyle: { borderRadius: 16, border: `1px solid ${CHART.border}`, fontSize: 12, boxShadow: "0 10px 30px -12px rgba(84,70,52,0.2)" } };

function PeriodTable({ rows }: { rows: Period[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase tracking-wide text-stone-500">
          <tr><th className="py-2 pr-3">Period</th><th className="pr-3">Entries</th><th className="pr-3">Avg stress score</th><th className="pr-3">Stress-language share</th><th className="pr-3">Avg mood</th><th className="pr-3">Avg sleep</th><th className="pr-3">Frequent emotions</th><th>Observations</th></tr>
        </thead>
        <tbody className="divide-y divide-stone-100 tabular-nums">
          {[...rows].reverse().map((r) => (
            <tr key={r.period} className="align-top">
              <td className="py-2 pr-3 font-medium text-stone-800">{r.label}</td>
              <td className="pr-3">{r.entries}</td>
              <td className="pr-3">{num(r.avg_stress_score)}</td>
              <td className="pr-3">{pct(r.stress_language_share)}</td>
              <td className="pr-3">{num(r.avg_mood, 1)}</td>
              <td className="pr-3">{r.avg_sleep_hours == null ? "-" : `${num(r.avg_sleep_hours, 1)} h`}</td>
              <td className="pr-3"><div className="flex flex-wrap gap-1">{r.top_emotions.map((t) => <EmotionChip key={t.label} label={t.label} />)}</div></td>
              <td className="text-xs text-stone-600">{r.observations.join(" ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TopicsCard() {
  const [t, setT] = useState<Topics | null>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { api.topics().then(setT).catch(setErr); }, []);
  return (
    <section className="card space-y-3">
      <h2 className="h2 flex items-center gap-2"><Layers className="h-5 w-5 text-sage-500" /> Recurring themes</h2>
      <ErrorBox error={err} />
      {!t && !err ? <Spinner label="Embedding and clustering entries..." /> : t?.status === "insufficient_data" ? (
        <p className="muted">At least {t.min_entries} entries are needed to identify recurring themes ({t.n_entries} so far).</p>
      ) : t && (
        <>
          <p className="text-xs text-stone-500">{t.method}. k = {t.k} chosen by silhouette score ({t.silhouette_cosine}). Theme names are the top keywords, not interpretations.</p>
          <div className="grid gap-3 md:grid-cols-2">
            {t.topics.map((tp) => (
              <div key={tp.topic_id} className="rounded-3xl bg-sand-50 p-4">
                <div className="flex items-baseline justify-between gap-2">
                  <div className="font-semibold text-stone-800">{tp.name}</div>
                  <div className="text-xs text-stone-500">{tp.n_entries} entries · {pct(tp.share)}</div>
                </div>
                <div className="mt-1 flex flex-wrap gap-1">{tp.keywords.map((k) => <span key={k} className="chip bg-stone-100 text-stone-600">{k}</span>)}</div>
                <div className="mt-2 text-xs text-stone-500">
                  {tp.first_seen} → {tp.last_seen} · avg stress score {num(tp.avg_stress_score)}
                </div>
                <div className="mt-1 flex flex-wrap gap-2 text-xs">
                  {tp.representative_entry_ids.map((id) => <Link key={id} to={`/entries/${id}`} className="text-sage-600 hover:underline">entry #{id}</Link>)}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

export default function InsightsPage() {
  const [days, setDays] = useState(90);
  const [data, setData] = useState<Insights | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [period, setPeriod] = useState<"weekly" | "monthly">("weekly");
  useEffect(() => { setData(null); api.insights(days || undefined).then(setData).catch(setErr); }, [days]);

  const daily = (data?.daily ?? []).map((d) => ({
    date: d.date.slice(5), stress: d.avg_stress_score, mood: d.avg_mood, energy: d.avg_energy, sleep: d.avg_sleep_hours,
  }));
  const groups = (data?.[period] ?? []).map((p) => ({
    label: period === "weekly" ? p.label.replace("Week of ", "").replace(/ \d{4}$/, "") : p.label, ...p.emotion_groups,
  }));
  const dist = (data?.emotion_distribution ?? []).filter((d) => d.entries_predicted > 0).slice(0, 12)
    .map((d) => ({ label: cap(d.label), entries: d.entries_predicted, group: d.group }));
  const all = data?.daily ?? [];
  const totalStress = all.filter((d) => d.avg_stress_score != null);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="h1">Insights</h1>
          <p className="muted">Patterns in the language of your entries over time.</p>
        </div>
        <div className="flex gap-1 rounded-full bg-white/70 p-1 shadow-soft">
          {RANGES.map((r) => (
            <button key={r.v} onClick={() => setDays(r.v)} className={`rounded-full px-3.5 py-1.5 text-xs font-semibold transition ${days === r.v ? "bg-sage-600 text-white" : "text-stone-500 hover:bg-sand-100"}`}>{r.l}</button>
          ))}
        </div>
      </header>
      <ErrorBox error={err} />
      {!data ? <Spinner /> : data.n_entries === 0 ? (
        <div className="card text-center text-sm text-stone-500">No entries in this range. Write entries or load demo entries from <Link to="/history" className="text-sage-600 hover:underline">History</Link>.</div>
      ) : (
        <>
          <Note>{data.caveat}</Note>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Entries" value={data.n_entries} icon={<NotebookPen className="h-4 w-4" />} />
            <StatCard label="Days with entries" value={all.length} icon={<CalendarDays className="h-4 w-4" />} />
            <StatCard label="Avg stress score" value={num(totalStress.reduce((a, d) => a + (d.avg_stress_score ?? 0) * d.entries, 0) / Math.max(1, totalStress.reduce((a, d) => a + d.entries, 0)))} hint="mean classifier probability" icon={<Activity className="h-4 w-4" />} />
            <StatCard label="Most frequent emotion" value={data.emotion_distribution.find((d) => d.label !== "neutral" && d.entries_predicted > 0)?.label ?? "-"} hint="excluding neutral" icon={<Layers className="h-4 w-4" />} />
          </div>

          {data.observations.statements.length > 0 && (
            <section className="card">
              <h2 className="h2 mb-2">Recent observations</h2>
              <ul className="list-disc space-y-1 pl-5 text-sm text-stone-700">{data.observations.statements.map((s) => <li key={s}>{s}</li>)}</ul>
            </section>
          )}

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="card">
              <h2 className="h2">Stress-related language (daily mean score)</h2>
              <p className="mb-3 text-xs text-stone-500">Mean stress-classifier probability per day, 0-1.</p>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={daily} margin={{ left: -20, right: 8, top: 8 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="date" {...axis} minTickGap={20} />
                  <YAxis domain={[0, 1]} {...axis} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(2)} />
                  <Line type="monotone" dataKey="stress" name="Stress score" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </section>
            <section className="card">
              <h2 className="h2">Self-reported check-ins</h2>
              <p className="mb-3 text-xs text-stone-500">Optional mood and energy ratings, 1-5 (daily mean).</p>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={daily} margin={{ left: -20, right: 8, top: 8 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="date" {...axis} minTickGap={20} />
                  <YAxis domain={[1, 5]} ticks={[1, 2, 3, 4, 5]} {...axis} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(1)} />
                  <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="mood" name="Mood" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="energy" name="Energy" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </section>
            <section className="card">
              <div className="flex items-center justify-between">
                <h2 className="h2">Emotion groups by {period === "weekly" ? "week" : "month"}</h2>
                <select className="input !w-auto !py-1" value={period} onChange={(e) => setPeriod(e.target.value as "weekly" | "monthly")}>
                  <option value="weekly">Weekly</option><option value="monthly">Monthly</option>
                </select>
              </div>
              <p className="mb-3 text-xs text-stone-500">Share of detected emotion labels in each GoEmotions sentiment group.</p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={groups} margin={{ left: -20, right: 8, top: 8 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="label" {...axis} interval={0} />
                  <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => pct(v)} cursor={{ fill: CHART.cursor }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {(["positive", "negative", "ambiguous", "neutral"] as const).map((g, i, arr) => (
                    <Bar key={g} dataKey={g} name={cap(g)} stackId="g" fill={GROUP_COLOR[g]} stroke="#fff" strokeWidth={2} maxBarSize={44}
                      radius={i === arr.length - 1 ? [4, 4, 0, 0] : undefined} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </section>
            <section className="card">
              <h2 className="h2">Most frequent emotion labels</h2>
              <p className="mb-3 text-xs text-stone-500">Number of entries where each label crossed its decision threshold.</p>
              <ResponsiveContainer width="100%" height={Math.max(240, dist.length * 24 + 40)}>
                <BarChart data={dist} layout="vertical" margin={{ left: 10, right: 16 }}>
                  <CartesianGrid stroke={CHART.grid} horizontal={false} />
                  <XAxis type="number" allowDecimals={false} {...axis} />
                  <YAxis type="category" dataKey="label" width={95} {...axis} interval={0} />
                  <Tooltip {...tooltipStyle} cursor={{ fill: CHART.cursor }} />
                  <Bar dataKey="entries" name="Entries" radius={[0, 4, 4, 0]} maxBarSize={14}>
                    {dist.map((d) => <Cell key={d.label} fill={GROUP_COLOR[d.group as keyof typeof GROUP_COLOR]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-stone-500">
                {Object.entries(GROUP_COLOR).map(([g, c]) => <span key={g} className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: c }} />{cap(g)}</span>)}
              </div>
            </section>
          </div>

          <section className="card space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="h2">{period === "weekly" ? "Weekly" : "Monthly"} summaries</h2>
              <select className="input !w-auto !py-1" value={period} onChange={(e) => setPeriod(e.target.value as "weekly" | "monthly")}>
                <option value="weekly">Weekly</option><option value="monthly">Monthly</option>
              </select>
            </div>
            <PeriodTable rows={data[period]} />
          </section>

          <TopicsCard />
        </>
      )}
    </div>
  );
}
