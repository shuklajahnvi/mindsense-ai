import { useState } from "react";
import { Link } from "react-router-dom";
import { Battery, Moon, PenLine, Smile } from "lucide-react";
import { api, type Entry, type Status } from "../api";
import { ErrorBox, Note, Spinner } from "../components";
import AnalysisView from "./AnalysisView";

const PROMPTS = [
  "What has been on your mind today?",
  "What felt hard today, and what helped even a little?",
  "Describe a moment today you'd like to remember.",
  "What are you looking forward to, or worried about, this week?",
];

function Scale({ label, icon, value, onChange }: { label: string; icon: React.ReactNode; value: number | null; onChange: (v: number | null) => void }) {
  return (
    <div>
      <span className="label flex items-center gap-1">{icon} {label}</span>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((v) => (
          <button key={v} type="button" onClick={() => onChange(value === v ? null : v)}
            className={`h-8 w-8 rounded-lg text-sm font-medium ${value === v ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>{v}</button>
        ))}
      </div>
    </div>
  );
}

export default function JournalPage({ status }: { status: Status | null }) {
  const [text, setText] = useState("");
  const [mood, setMood] = useState<number | null>(null);
  const [energy, setEnergy] = useState<number | null>(null);
  const [sleep, setSleep] = useState<string>("");
  const [explain, setExplain] = useState(true);
  const [autoExplain, setAutoExplain] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [entry, setEntry] = useState<Entry | null>(null);
  const [prompt] = useState(() => PROMPTS[Math.floor(Math.random() * PROMPTS.length)]);
  const noModels = status && !status.models.emotion.active_variant && !status.models.stress.active_variant;

  const submit = async () => {
    setBusy(true); setErr(null);
    try {
      // Save + classify quickly; LIME runs afterwards inside AnalysisView (autoExplain).
      const e = await api.createEntry({ text, mood, energy, sleep_hours: sleep === "" ? null : Number(sleep), explain: false });
      setAutoExplain(explain); setEntry(e); setText(""); setMood(null); setEnergy(null); setSleep("");
    } catch (e) { setErr(e); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Journal</h1>
        <p className="muted">Write freely. MindSense analyses emotion- and stress-related language and explains which words influenced each model.</p>
      </header>
      {noModels && <Note tone="warn">No trained models were found. Run the training scripts in <code>backend/ml</code> (see README) - analysis is disabled until then.</Note>}

      <section className="card space-y-4">
        <div>
          <label className="label flex items-center gap-1"><PenLine className="h-3.5 w-3.5" /> {prompt}</label>
          <textarea className="input min-h-[160px] resize-y leading-relaxed" placeholder="Start writing..." value={text} onChange={(e) => setText(e.target.value)} maxLength={10000} />
          <div className="mt-1 text-right text-[11px] text-slate-400">{text.trim().split(/\s+/).filter(Boolean).length} words</div>
        </div>
        <div className="flex flex-wrap items-end gap-6">
          <Scale label="Mood (optional)" icon={<Smile className="h-3.5 w-3.5" />} value={mood} onChange={setMood} />
          <Scale label="Energy (optional)" icon={<Battery className="h-3.5 w-3.5" />} value={energy} onChange={setEnergy} />
          <div>
            <span className="label flex items-center gap-1"><Moon className="h-3.5 w-3.5" /> Sleep hours (optional)</span>
            <input type="number" min={0} max={24} step={0.5} className="input !w-28" value={sleep} onChange={(e) => setSleep(e.target.value)} />
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} className="h-4 w-4 rounded" />
            Explain predictions with LIME (runs after saving)
          </label>
          <button className="btn-primary ml-auto" disabled={busy || text.trim().length < 3 || !!noModels} onClick={submit}>
            {busy ? "Analysing..." : "Save & analyse"}
          </button>
        </div>
        {busy && <Spinner label="Saving and running the emotion and stress classifiers..." />}
        <ErrorBox error={err} />
        <p className="text-[11px] text-slate-400">Entries are stored locally in SQLite on this machine. Please don't include other people's personal details.</p>
      </section>

      {entry && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="h2">Analysis of your entry</h2>
            <Link to={`/entries/${entry.id}`} className="text-sm text-indigo-600 hover:underline">Open entry #{entry.id}</Link>
          </div>
          <AnalysisView entry={entry} onUpdate={setEntry} autoExplain={autoExplain} />
        </section>
      )}
    </div>
  );
}
