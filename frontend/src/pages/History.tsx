import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Database, Trash2 } from "lucide-react";
import { api, type EntrySummary } from "../api";
import { EmotionChip, ErrorBox, Note, Spinner } from "../components";
import { fmtDate, pct } from "../lib";

export default function HistoryPage() {
  const [items, setItems] = useState<EntrySummary[] | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const load = () => api.entries(500).then((r) => setItems(r.items)).catch(setErr);
  useEffect(() => { load(); }, []);

  const act = async (fn: () => Promise<string>) => {
    setBusy(true); setErr(null);
    try { setMsg(await fn()); await load(); } catch (e) { setErr(e); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">History</h1>
          <p className="muted">All stored entries with their model outputs.</p>
        </div>
        <div className="flex gap-2">
          <button className="btn-ghost" disabled={busy} onClick={() => act(async () => { const r = await api.seedDemo(); return `Inserted ${r.inserted} demo entries. ${r.note}`; })}>
            <Database className="h-4 w-4" /> Load demo entries
          </button>
          <button className="btn-ghost" disabled={busy} onClick={() => act(async () => { const r = await api.clearDemo(); return `Removed ${r.deleted} demo entries.`; })}>
            <Trash2 className="h-4 w-4" /> Remove demo entries
          </button>
        </div>
      </header>
      <Note>Demo entries are hand-written synthetic journal texts (a fictional student over six weeks) so you can explore the trends and topic views. Their predictions are real outputs of the trained models.</Note>
      {busy && <Spinner label="Working (analysing entries with the models)..." />}
      {msg && <div className="text-sm text-emerald-700">{msg}</div>}
      <ErrorBox error={err} />
      {!items ? <Spinner /> : items.length === 0 ? (
        <div className="card text-center text-sm text-slate-500">No entries yet. <Link to="/" className="text-indigo-600 hover:underline">Write your first entry</Link> or load the demo entries.</div>
      ) : (
        <ul className="space-y-3">
          {items.map((e) => (
            <li key={e.id}>
              <Link to={`/entries/${e.id}`} className="card block transition hover:border-indigo-300 hover:shadow">
                <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <span>{fmtDate(e.created_at)}</span>
                  {e.is_demo && <span className="chip bg-violet-50 text-violet-700">demo</span>}
                  {e.stress_label && (
                    <span className={`chip ${e.stress_label === "stress" ? "bg-rose-50 text-rose-700" : "bg-emerald-50 text-emerald-700"}`}>
                      {e.stress_label === "stress" ? "stress-related language" : "no stress language"} · {pct(e.stress_score)}
                    </span>
                  )}
                  {e.mood && <span>mood {e.mood}/5</span>}
                  {e.sleep_hours != null && <span>sleep {e.sleep_hours}h</span>}
                </div>
                <p className="line-clamp-2 text-sm text-slate-700">{e.text}</p>
                <div className="mt-2 flex flex-wrap gap-1">{(e.predicted_emotions ?? []).map((l) => <EmotionChip key={l} label={l} />)}</div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
