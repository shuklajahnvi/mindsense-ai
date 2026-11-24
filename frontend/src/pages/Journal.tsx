import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Annoyed, ArrowRight, Frown, Laugh, Leaf, Meh, Moon, Smile, Sun, Sunrise, Zap } from "lucide-react";
import { api, type Entry, type Status } from "../api";
import { ErrorBox, Note, Spinner } from "../components";
import AnalysisView from "./AnalysisView";

const PROMPTS = [
  "What has been on your mind today?",
  "What felt hard today, and what helped, even a little?",
  "Describe a moment today you'd like to remember.",
  "What are you looking forward to, or worried about, this week?",
  "What is one thing you'd like to let go of tonight?",
];

const MOODS = [
  { v: 1, icon: Frown, label: "Rough" },
  { v: 2, icon: Annoyed, label: "Low" },
  { v: 3, icon: Meh, label: "Okay" },
  { v: 4, icon: Smile, label: "Good" },
  { v: 5, icon: Laugh, label: "Great" },
];

function greeting() {
  const h = new Date().getHours();
  if (h < 12) return { text: "Good morning", icon: Sunrise };
  if (h < 18) return { text: "Good afternoon", icon: Sun };
  return { text: "Good evening", icon: Moon };
}

function Field({ label, children }: { label: ReactNode; children: ReactNode }) {
  return <div><span className="label flex items-center gap-1.5">{label}</span>{children}</div>;
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
  const g = greeting();
  const words = text.trim().split(/\s+/).filter(Boolean).length;

  const submit = async () => {
    setBusy(true); setErr(null);
    try {
      // Save + classify quickly; LIME runs afterwards inside AnalysisView (autoExplain).
      const e = await api.createEntry({ text, mood, energy, sleep_hours: sleep === "" ? null : Number(sleep), explain: false });
      setAutoExplain(explain); setEntry(e); setText(""); setMood(null); setEnergy(null); setSleep("");
    } catch (e) { setErr(e); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-8">
      <header>
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-ochre-600"><g.icon className="h-4 w-4" /> {g.text}</div>
        <h1 className="h1">How are you, really?</h1>
        <p className="mt-2 max-w-xl text-stone-500">Take a slow breath, then write whatever is on your mind. There's no right way to do this.</p>
      </header>
      {noModels && <Note tone="warn">No trained models were found. Run the training scripts in <code>backend/ml</code> (see README) - analysis is disabled until then.</Note>}

      <section className="card space-y-6 !p-7">
        <div>
          <p className="mb-3 font-display text-lg italic text-sage-700">{prompt}</p>
          <textarea
            className="min-h-[200px] w-full resize-y rounded-3xl border-0 bg-sand-50 px-5 py-4 text-[15px] leading-7 text-stone-700 ring-1 ring-sand-200 placeholder:text-stone-400 focus:outline-none focus:ring-2 focus:ring-sage-300"
            placeholder="Start writing..." value={text} onChange={(e) => setText(e.target.value)} maxLength={10000} />
          <div className="mt-1.5 flex items-center justify-end gap-1 text-xs text-stone-400"><Leaf className="h-3 w-3" /> {words} {words === 1 ? "word" : "words"}</div>
        </div>

        <div className="grid gap-6 sm:grid-cols-[auto_auto_1fr]">
          <Field label={<>How's your mood? <span className="font-normal text-stone-400">(optional)</span></>}>
            <div className="flex gap-1.5">
              {MOODS.map(({ v, icon: Icon, label }) => (
                <button key={v} type="button" onClick={() => setMood(mood === v ? null : v)} title={label} aria-label={label} aria-pressed={mood === v}
                  className={`group flex w-12 flex-col items-center gap-0.5 rounded-2xl py-1.5 text-[10px] font-semibold transition ${mood === v ? "bg-sage-100 text-sage-800" : "text-stone-400 hover:bg-sand-100 hover:text-stone-600"}`}>
                  <Icon className={`h-6 w-6 transition group-hover:scale-110 ${mood === v ? "text-sage-600" : ""}`} strokeWidth={1.75} />
                  {label}
                </button>
              ))}
            </div>
          </Field>
          <Field label={<><Zap className="h-3.5 w-3.5 text-ochre-500" /> Energy <span className="font-normal text-stone-400">(optional)</span></>}>
            <div className="flex items-center gap-1.5 pt-2">
              {[1, 2, 3, 4, 5].map((v) => (
                <button key={v} type="button" onClick={() => setEnergy(energy === v ? null : v)} aria-label={`Energy ${v} of 5`} aria-pressed={energy === v}
                  className={`h-8 w-8 rounded-full text-xs font-bold transition ${energy != null && v <= energy ? "bg-ochre-300 text-ochre-900" : "bg-sand-100 text-stone-400 hover:bg-sand-200"}`}>{v}</button>
              ))}
            </div>
          </Field>
          <Field label={<><Moon className="h-3.5 w-3.5 text-dusk-400" /> Hours of sleep <span className="font-normal text-stone-400">(optional)</span></>}>
            <div className="pt-1"><input type="number" min={0} max={24} step={0.5} placeholder="e.g. 7" className="input !w-28" value={sleep} onChange={(e) => setSleep(e.target.value)} /></div>
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-4 pt-1">
          <label className="flex cursor-pointer items-center gap-2 text-sm text-stone-500">
            <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} className="h-4 w-4 rounded accent-sage-600" />
            Show which words influenced the models (LIME)
          </label>
          <button className="btn-primary ml-auto !px-6" disabled={busy || text.trim().length < 3 || !!noModels} onClick={submit}>
            {busy ? "Reflecting..." : <>Save entry <ArrowRight className="h-4 w-4" /></>}
          </button>
        </div>
        {busy && <Spinner label="Saving and reading the emotional tone of your words..." />}
        <ErrorBox error={err} />
        <p className="text-xs text-stone-400">Entries stay on this computer (local SQLite). Please avoid including other people's personal details.</p>
      </section>

      {entry && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="h2">What your words suggest</h2>
            <Link to={`/entries/${entry.id}`} className="text-sm font-semibold text-sage-700 hover:text-sage-800">Open entry →</Link>
          </div>
          <AnalysisView entry={entry} onUpdate={setEntry} autoExplain={autoExplain} />
        </section>
      )}
    </div>
  );
}
