import { useMemo, useState, type ReactNode } from "react";
import { AlertTriangle, ExternalLink, Info, Loader2, LifeBuoy, Phone } from "lucide-react";
import type { EmotionRow, Explanations, LimeTarget, ModelInfo, Safety } from "./api";
import { GROUP_CHIP, GROUP_COLOR, cap, groupOf, pct } from "./lib";

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500">
      <Loader2 className="h-4 w-4 animate-spin" /> {label ?? "Loading..."}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> <span>{String((error as Error).message ?? error)}</span>
    </div>
  );
}

export function Note({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  const cls = tone === "warn" ? "border-amber-200 bg-amber-50 text-amber-800" : "border-indigo-100 bg-indigo-50/60 text-indigo-900";
  return (
    <div className={`flex items-start gap-2 rounded-xl border p-3 text-xs leading-relaxed ${cls}`}>
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" /> <div>{children}</div>
    </div>
  );
}

export function EmotionChip({ label, score }: { label: string; score?: number }) {
  return (
    <span className={`chip ${GROUP_CHIP[groupOf(label)]}`}>
      {cap(label)}{score != null && <span className="ml-1 opacity-70">{pct(score)}</span>}
    </span>
  );
}

export function SafetyPanel({ safety }: { safety: Safety | null | undefined }) {
  if (!safety || safety.level === "none") return null;
  const high = safety.level === "high";
  return (
    <div className={`rounded-2xl border p-4 ${high ? "border-rose-300 bg-rose-50" : "border-amber-300 bg-amber-50"}`} role="alert">
      <div className="flex items-start gap-3">
        <LifeBuoy className={`mt-0.5 h-5 w-5 shrink-0 ${high ? "text-rose-600" : "text-amber-600"}`} />
        <div className="space-y-3">
          <p className="text-sm font-medium text-slate-800">{safety.message}</p>
          <ul className="grid gap-2 sm:grid-cols-2">
            {safety.resources.map((r) => (
              <li key={r.name} className="rounded-xl bg-white/80 p-2.5 text-xs ring-1 ring-black/5">
                <div className="font-semibold text-slate-800">{r.region}: {r.name}</div>
                <div className="flex items-center gap-1 text-slate-600"><Phone className="h-3 w-3" /> {r.contact}</div>
                {r.url && <a href={r.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-indigo-600 hover:underline">{r.url.replace(/^https?:\/\//, "")} <ExternalLink className="h-3 w-3" /></a>}
              </li>
            ))}
          </ul>
          <p className="text-[11px] text-slate-500">Detected by a {safety.method}. MindSense cannot respond to emergencies.</p>
        </div>
      </div>
    </div>
  );
}

export function ModelBadge({ info }: { info?: ModelInfo }) {
  if (!info || info.status === "unavailable") return <span className="chip bg-slate-100 text-slate-500">model unavailable</span>;
  return (
    <span className="chip bg-slate-100 text-slate-600" title={info.algorithm}>
      {info.variant === "transformer" ? (info.fine_tuned ? `Fine-tuned ${info.base_checkpoint}` : info.name) : "TF-IDF baseline"}
      {info.trained_on_subset ? " · dev subset" : ""}
    </span>
  );
}

export function EmotionBars({ rows, top = 8 }: { rows: EmotionRow[]; top?: number }) {
  return (
    <ul className="space-y-1.5">
      {rows.slice(0, top).map((r) => (
        <li key={r.label} className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-2 text-sm">
          <span className={`truncate ${r.predicted ? "font-semibold text-slate-800" : "text-slate-500"}`}>{cap(r.label)}</span>
          <div className="relative h-2.5 rounded-full bg-slate-100">
            <div className="h-2.5 rounded-full" style={{ width: `${Math.max(1, r.score * 100)}%`, background: GROUP_COLOR[groupOf(r.label)], opacity: r.predicted ? 1 : 0.45 }} />
            <div className="absolute -top-0.5 h-3.5 w-0.5 bg-slate-700/60" style={{ left: `${r.threshold * 100}%` }} title={`decision threshold ${pct(r.threshold)}`} />
          </div>
          <span className="text-right tabular-nums text-xs text-slate-500">{pct(r.score)}</span>
        </li>
      ))}
    </ul>
  );
}

export function StressMeter({ score, label, threshold = 0.5 }: { score: number; label: string; threshold?: number }) {
  const stress = label === "stress";
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className={`text-sm font-semibold ${stress ? "text-rose-600" : "text-emerald-600"}`}>
          {stress ? "Stress-related language detected" : "No stress-related language detected"}
        </span>
        <span className="text-xs tabular-nums text-slate-500">score {pct(score)}</span>
      </div>
      <div className="relative mt-2 h-3 rounded-full bg-gradient-to-r from-emerald-200 via-amber-200 to-rose-300">
        <div className="absolute -top-1 h-5 w-1.5 rounded bg-slate-800" style={{ left: `calc(${score * 100}% - 3px)` }} />
        <div className="absolute -bottom-4 text-[10px] text-slate-400" style={{ left: `calc(${threshold * 100}% - 20px)` }}>threshold</div>
      </div>
    </div>
  );
}

// Same tokenisation as the backend LIME split_expression ([^\w']+), keeping contractions whole.
function tokenize(text: string) { return text.split(/([^\w']+)/); }

function HighlightedText({ text, target }: { text: string; target: LimeTarget }) {
  const weights = useMemo(() => Object.fromEntries(target.weights.map((w) => [w.token, w.weight])), [target]);
  const max = Math.max(1e-9, ...target.weights.map((w) => Math.abs(w.weight)));
  return (
    <p className="whitespace-pre-wrap rounded-xl bg-slate-50 p-3 text-sm leading-7 text-slate-700">
      {tokenize(text).map((t, i) => {
        const w = weights[t];
        if (w === undefined) return <span key={i}>{t}</span>;
        const a = 0.15 + 0.7 * (Math.abs(w) / max);
        const bg = w > 0 ? `rgba(249,115,22,${a})` : `rgba(14,165,233,${a})`;
        return <mark key={i} title={`${w > 0 ? "+" : ""}${w.toFixed(3)}`} className="rounded px-0.5 text-slate-900" style={{ background: bg }}>{t}</mark>;
      })}
    </p>
  );
}

function WeightBars({ target }: { target: LimeTarget }) {
  const max = Math.max(1e-9, ...target.weights.map((w) => Math.abs(w.weight)));
  return (
    <ul className="space-y-1">
      {target.weights.map((w) => (
        <li key={w.token} className="grid grid-cols-[6rem_1fr_1fr_3.5rem] items-center gap-1 text-xs">
          <span className="truncate font-mono text-slate-700">{w.token}</span>
          <div className="flex justify-end">{w.weight < 0 && <div className="h-2 rounded-l bg-sky-500" style={{ width: `${(Math.abs(w.weight) / max) * 100}%` }} />}</div>
          <div>{w.weight > 0 && <div className="h-2 rounded-r bg-orange-500" style={{ width: `${(w.weight / max) * 100}%` }} />}</div>
          <span className="text-right tabular-nums text-slate-500">{w.weight > 0 ? "+" : ""}{w.weight.toFixed(3)}</span>
        </li>
      ))}
    </ul>
  );
}

export function LimeView({ text, explanations }: { text: string; explanations: Explanations }) {
  const targets = [
    ...(explanations.emotion?.targets.map((t) => ({ kind: "Emotion", block: explanations.emotion!, t })) ?? []),
    ...(explanations.stress?.targets.map((t) => ({ kind: "Stress", block: explanations.stress!, t })) ?? []),
  ];
  const [sel, setSel] = useState(0);
  if (!targets.length) return <p className="muted">No explanation available.</p>;
  const cur = targets[Math.min(sel, targets.length - 1)];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {targets.map((x, i) => (
          <button key={i} onClick={() => setSel(i)}
            className={`chip cursor-pointer ${i === sel ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
            {x.kind}: {cap(x.t.label)}
          </button>
        ))}
      </div>
      <div className="text-xs text-slate-500">
        Explaining the <b>{cur.kind.toLowerCase()}</b> model's score for <b>"{cap(cur.t.label)}"</b>
        {cur.t.prediction_score != null && <> ({pct(cur.t.prediction_score)})</>} · model <code>{cur.block.model}</code> ({cur.block.variant})
        · local fit R² {cur.t.local_model_r2.toFixed(2)} · {explanations.num_samples} perturbations, {cur.block.seconds}s
      </div>
      <HighlightedText text={text} target={cur.t} />
      <div className="flex gap-4 text-[11px] text-slate-500">
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded bg-orange-500" /> pushed the score up</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded bg-sky-500" /> pushed the score down</span>
      </div>
      <WeightBars target={cur.t} />
      <Note>{explanations.disclaimer}</Note>
    </div>
  );
}

// Minimal, safe markdown renderer (headings, bold, italics, lists, links, citations).
function inline(s: string, key: string): ReactNode[] {
  const parts = s.split(/(\*\*[^*]+\*\*|_[^_]+_|\[[^\]]+\]\([^)]+\)|\[\d+\])/g);
  return parts.map((p, i) => {
    const k = `${key}-${i}`;
    if (/^\*\*[^*]+\*\*$/.test(p)) return <strong key={k}>{p.slice(2, -2)}</strong>;
    if (/^_[^_]+_$/.test(p)) return <em key={k}>{p.slice(1, -1)}</em>;
    const link = p.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link && /^https?:/.test(link[2])) return <a key={k} href={link[2]} target="_blank" rel="noreferrer" className="text-indigo-600 underline">{link[1]}</a>;
    if (/^\[\d+\]$/.test(p)) return <sup key={k} className="font-semibold text-indigo-600">{p}</sup>;
    return <span key={k}>{p}</span>;
  });
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: ReactNode[] = [];
  let list: ReactNode[] = [];
  const flush = () => { if (list.length) { out.push(<ul key={`ul${out.length}`} className="ml-5 list-disc space-y-1">{list}</ul>); list = []; } };
  lines.forEach((l, i) => {
    const t = l.trim();
    if (/^[-*]\s+/.test(t)) { list.push(<li key={i}>{inline(t.replace(/^[-*]\s+/, ""), `l${i}`)}</li>); return; }
    if (/^\d+\.\s+/.test(t)) { list.push(<li key={i}>{inline(t.replace(/^\d+\.\s+/, ""), `l${i}`)}</li>); return; }
    flush();
    if (!t) return;
    const h = t.match(/^(#{1,4})\s+(.*)$/);
    if (h) { out.push(<h4 key={i} className="mt-2 font-semibold text-slate-800">{inline(h[2], `h${i}`)}</h4>); return; }
    out.push(<p key={i}>{inline(t, `p${i}`)}</p>);
  });
  flush();
  return <div className="space-y-2 text-sm leading-relaxed text-slate-700">{out}</div>;
}

export function StatCard({ label, value, hint, icon }: { label: string; value: ReactNode; hint?: string; icon?: ReactNode }) {
  return (
    <div className="card flex items-start gap-3 !p-4">
      {icon && <div className="rounded-xl bg-indigo-50 p-2 text-indigo-600">{icon}</div>}
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
        <div className="text-xl font-semibold text-slate-900">{value}</div>
        {hint && <div className="text-xs text-slate-400">{hint}</div>}
      </div>
    </div>
  );
}
