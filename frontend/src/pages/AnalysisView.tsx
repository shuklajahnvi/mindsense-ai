import { useEffect, useRef, useState } from "react";
import { Lightbulb, Sparkles } from "lucide-react";
import { api, type Entry, type Source, type Variant } from "../api";
import { EmotionBars, EmotionChip, ErrorBox, LimeView, Markdown, ModelBadge, Note, SafetyPanel, Spinner, StressMeter } from "../components";

/** Full analysis for one stored entry: predictions, LIME, and RAG reflection. */
export default function AnalysisView({ entry, onUpdate, autoExplain = false }: { entry: Entry; onUpdate: (e: Entry) => void; autoExplain?: boolean }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [variant, setVariant] = useState<Variant>("auto");
  const started = useRef<number | null>(null);

  const run = async (key: string, fn: () => Promise<Entry>) => {
    setBusy(key); setErr(null);
    try { onUpdate(await fn()); } catch (e) { setErr(e); } finally { setBusy(null); }
  };
  // Predictions are shown immediately; LIME (slow on CPU) runs as a follow-up request.
  useEffect(() => {
    if (autoExplain && !entry.explanations && started.current !== entry.id) {
      started.current = entry.id;
      run("explain", () => api.explain(entry.id));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoExplain, entry.id]);
  const predicted = entry.predicted_emotions ?? [];

  return (
    <div className="space-y-4">
      <SafetyPanel safety={entry.safety} />
      <ErrorBox error={err} />
      <div className="grid gap-4 md:grid-cols-2">
        <section className="card space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h3 className="h2">Emotion-related language</h3>
            <ModelBadge info={entry.model_info?.emotion} />
          </div>
          {entry.emotions ? (
            <>
              <div className="flex flex-wrap gap-1.5">
                {predicted.length ? predicted.map((l) => <EmotionChip key={l} label={l} score={entry.emotions!.find((r) => r.label === l)?.score} />)
                  : <span className="muted">No label crossed its decision threshold.</span>}
              </div>
              <EmotionBars rows={entry.emotions} />
              <p className="text-[11px] text-slate-400">Multi-label: each of the 28 GoEmotions labels has its own sigmoid score and tuned threshold (black tick).</p>
            </>
          ) : <p className="muted">Emotion model not available.</p>}
        </section>
        <section className="card space-y-4">
          <div className="flex items-center justify-between gap-2">
            <h3 className="h2">Stress-related language</h3>
            <ModelBadge info={entry.model_info?.stress} />
          </div>
          {entry.stress_score != null ? <StressMeter score={entry.stress_score} label={entry.stress_label!} /> : <p className="muted">Stress model not available.</p>}
          <p className="pt-3 text-[11px] text-slate-400">Binary classifier trained on Dreaddit (Reddit posts labelled for stress by annotators). Describes the wording, not your stress level.</p>
          {(entry.mood || entry.energy || entry.sleep_hours != null) && (
            <div className="grid grid-cols-3 gap-2 border-t border-slate-100 pt-3 text-center text-xs">
              <div><div className="text-slate-400">Mood</div><div className="font-semibold">{entry.mood ?? "-"}/5</div></div>
              <div><div className="text-slate-400">Energy</div><div className="font-semibold">{entry.energy ?? "-"}/5</div></div>
              <div><div className="text-slate-400">Sleep</div><div className="font-semibold">{entry.sleep_hours ?? "-"} h</div></div>
            </div>
          )}
        </section>
      </div>

      <section className="card space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="h2 flex items-center gap-2"><Lightbulb className="h-5 w-5 text-amber-500" /> Why did the model say this? (LIME)</h3>
          <div className="flex items-center gap-2">
            <select className="input !w-auto !py-1.5" value={variant} onChange={(e) => setVariant(e.target.value as Variant)}>
              <option value="auto">Active models</option>
              <option value="transformer">Transformer models</option>
              <option value="baseline">TF-IDF baselines</option>
            </select>
            <button className="btn-ghost !py-1.5" disabled={!!busy} onClick={() => run("explain", () => api.explain(entry.id, variant, variant))}>
              {entry.explanations ? "Recompute" : "Explain"}
            </button>
          </div>
        </div>
        {busy === "explain" ? <Spinner label="Running LIME: perturbing the text and querying the models (up to ~1 min on CPU for transformer models)..." />
          : entry.explanations ? <LimeView text={entry.text} explanations={entry.explanations} />
          : <p className="muted">Explanations were not computed for this entry yet.</p>}
      </section>

      <section className="card space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="h2 flex items-center gap-2"><Sparkles className="h-5 w-5 text-indigo-500" /> Personalised reflection</h3>
          <button className="btn-primary !py-1.5" disabled={!!busy} onClick={() => run("reflect", () => api.reflection(entry.id))}>
            {entry.reflection ? "Regenerate" : "Generate reflection"}
          </button>
        </div>
        {busy === "reflect" ? <Spinner label="Retrieving guidance and generating..." /> : entry.reflection ? (
          <div className="space-y-3">
            {entry.reflection.mode !== "llm" && entry.reflection.mode !== "safety" && (
              <Note tone="warn">Retrieval-only mode: no LLM is configured{entry.reflection.error ? ` (error: ${entry.reflection.error})` : ""}. Add an API key in backend/.env to enable generation.</Note>
            )}
            {entry.reflection.mode === "llm" && entry.reflection.provider && (
              <div className="text-[11px] text-slate-400">Generated by {entry.reflection.provider.provider} · {entry.reflection.provider.model}, grounded in retrieved passages</div>
            )}
            <Markdown text={entry.reflection.answer} />
            {entry.reflection.sources.length > 0 && <SourceList sources={entry.reflection.sources} />}
          </div>
        ) : <p className="muted">Generates gentle reflective questions and ideas, grounded in a curated wellbeing knowledge base (RAG).</p>}
      </section>
    </div>
  );
}

export function SourceList({ sources }: { sources: Source[] }) {
  return (
    <details className="rounded-xl bg-slate-50 p-3 text-xs text-slate-600">
      <summary className="cursor-pointer font-medium text-slate-700">Retrieved sources ({sources.length})</summary>
      <ol className="mt-2 space-y-2">
        {sources.map((s) => (
          <li key={s.ref}>
            <div className="font-semibold">[{s.ref}] {s.title} - {s.section} <span className="font-normal text-slate-400">(similarity {s.similarity.toFixed(2)})</span></div>
            <div className="text-slate-500">{s.excerpt}</div>
            <div className="mt-0.5 flex flex-wrap gap-x-3">
              {s.sources.map((src) => <a key={src.url} href={src.url} target="_blank" rel="noreferrer" className="text-indigo-600 hover:underline">{src.name}</a>)}
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}
