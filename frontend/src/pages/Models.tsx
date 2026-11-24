import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type ModelMeta, type ModelReports } from "../api";
import { ErrorBox, Note, Spinner } from "../components";
import { CHART, SERIES, cap, num } from "../lib";

const axis = { stroke: CHART.axis, fontSize: 11, tickLine: false };

function Badge({ m }: { m: ModelMeta }) {
  return m.fine_tuned
    ? <span className="chip bg-indigo-50 text-indigo-700">fine-tuned from {m.base_checkpoint}{m.training?.is_subset ? " · dev subset" : ""}</span>
    : <span className="chip bg-slate-100 text-slate-600">traditional ML baseline</span>;
}

function TrainingFacts({ m }: { m: ModelMeta }) {
  const t = m.training ?? {};
  return (
    <div className="text-xs text-slate-500">
      train/val/test = {t.n_train}/{t.n_val}/{t.n_test}
      {t.train_minutes != null && <> · {t.train_minutes} min on {t.device}</>}
      {t.history && <> · {t.history.length} evaluations, early stopping patience {m.hyperparameters?.patience}</>}
      {m.fine_tuning_scope && <> · {m.fine_tuning_scope}</>}
      {" · "}hyperparameters: <code className="break-all">{JSON.stringify(m.hyperparameters)}</code>
      {" · "}trained {new Date(m.trained_at).toLocaleString()}
    </div>
  );
}

function EmotionSection({ models }: { models: Record<string, ModelMeta> }) {
  const list = Object.values(models);
  const [sel, setSel] = useState(list.find((m) => m.kind === "transformer")?.name ?? list[0]?.name);
  if (!list.length) return <p className="muted">No emotion models trained yet.</p>;
  const keys = ["micro_f1", "macro_f1", "micro_precision", "micro_recall", "macro_precision", "macro_recall", "samples_f1", "subset_accuracy", "macro_roc_auc"];
  const cur = models[sel!];
  const perClass = Object.entries(cur.metrics.test.per_class as Record<string, { f1: number; precision: number; recall: number; support: number; threshold: number }>)
    .map(([label, v]) => ({ label, ...v })).sort((a, b) => b.f1 - a.f1);
  const compare = list.length > 1 ? perClass.map((r) => {
    const row: Record<string, number | string> = { label: r.label };
    list.forEach((m) => { row[m.kind] = m.metrics.test.per_class[r.label].f1; });
    return row;
  }) : null;
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-slate-500"><tr><th className="py-2 pr-3">Model (test set)</th>{keys.map((k) => <th key={k} className="pr-3">{cap(k)}</th>)}</tr></thead>
          <tbody className="divide-y divide-slate-100 tabular-nums">
            {list.map((m) => (
              <tr key={m.name}><td className="py-2 pr-3"><div className="font-medium">{m.algorithm}</div><Badge m={m} /></td>
                {keys.map((k) => <td key={k} className="pr-3">{num(m.metrics.test[k], 3)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {list.map((m) => <TrainingFacts key={m.name} m={m} />)}
      {compare && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-slate-700">Per-class test F1: transformer vs baseline</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={compare} margin={{ left: -20, bottom: 50 }}>
              <CartesianGrid stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="label" {...axis} angle={-55} textAnchor="end" interval={0} height={60} />
              <YAxis domain={[0, 1]} {...axis} />
              <Tooltip formatter={(v: number) => v.toFixed(3)} contentStyle={{ borderRadius: 12, fontSize: 12 }} cursor={{ fill: "#f1f5f9" }} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
              {list.map((m, i) => <Bar key={m.kind} dataKey={m.kind} name={m.kind === "transformer" ? `Transformer (${m.base_checkpoint})` : "TF-IDF + LR"} fill={SERIES[i]} radius={[3, 3, 0, 0]} maxBarSize={10} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-slate-700">Per-class detail:</span>
        <select className="input !w-auto !py-1" value={sel} onChange={(e) => setSel(e.target.value)}>
          {list.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
        </select>
      </div>
      <div className="max-h-80 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-white text-left text-xs uppercase tracking-wide text-slate-500"><tr><th className="py-1.5">Label</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>Threshold</th></tr></thead>
          <tbody className="divide-y divide-slate-100 tabular-nums">
            {perClass.map((r) => <tr key={r.label}><td className="py-1">{r.label}</td><td>{num(r.precision, 3)}</td><td>{num(r.recall, 3)}</td><td className="font-medium">{num(r.f1, 3)}</td><td>{r.support}</td><td>{num(r.threshold, 3)}</td></tr>)}
          </tbody>
        </table>
      </div>
      <div>
        <h4 className="mb-1 text-sm font-semibold text-slate-700">Most frequent confusions (gold → wrongly predicted), test set</h4>
        <div className="flex flex-wrap gap-1.5">
          {(cur.metrics.test.label_cooccurrence_errors as { gold: string; predicted: string; count: number }[]).slice(0, 12).map((c) =>
            <span key={c.gold + c.predicted} className="chip bg-slate-100 text-slate-600">{c.gold} → {c.predicted} ({c.count})</span>)}
        </div>
      </div>
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: { labels: string[]; matrix: number[][] } }) {
  const max = Math.max(...cm.matrix.flat());
  return (
    <table className="text-xs">
      <thead><tr><th className="p-1 text-slate-400">gold \ pred</th>{cm.labels.map((l) => <th key={l} className="p-1 font-medium text-slate-500">{l}</th>)}</tr></thead>
      <tbody>{cm.matrix.map((row, i) => (
        <tr key={i}><th className="p-1 text-right font-medium text-slate-500">{cm.labels[i]}</th>
          {row.map((v, j) => <td key={j} className="h-12 w-20 text-center font-semibold tabular-nums" style={{ background: `rgba(42,120,214,${0.1 + 0.8 * v / max})`, color: v / max > 0.5 ? "#fff" : "#0b0b0b" }}>{v}</td>)}</tr>
      ))}</tbody>
    </table>
  );
}

function StressSection({ models }: { models: Record<string, ModelMeta> }) {
  const list = Object.values(models);
  if (!list.length) return <p className="muted">No stress models trained yet.</p>;
  const keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"];
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-slate-500"><tr><th className="py-2 pr-3">Model (test set)</th>{keys.map((k) => <th key={k} className="pr-3">{cap(k)}</th>)}</tr></thead>
          <tbody className="divide-y divide-slate-100 tabular-nums">
            {list.map((m) => (
              <tr key={m.name}><td className="py-2 pr-3"><div className="font-medium">{m.algorithm}</div><Badge m={m} /></td>
                {keys.map((k) => <td key={k} className="pr-3">{num(m.metrics.test[k], 3)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {list.map((m) => <TrainingFacts key={m.name} m={m} />)}
      <div className="flex flex-wrap gap-6">
        {list.map((m) => (
          <div key={m.name}>
            <div className="mb-1 text-xs font-semibold text-slate-600">{m.name} - test confusion matrix</div>
            <ConfusionMatrix cm={m.metrics.test.confusion_matrix} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const [r, setR] = useState<ModelReports | null>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { api.models().then(setR).catch(setErr); }, []);
  if (err) return <ErrorBox error={err} />;
  if (!r) return <Spinner />;
  const eda = r.eda;
  const leak = eda?.stress?.leakage_checks;
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Models & evaluation</h1>
        <p className="muted">All numbers below are read from the metric files written by the training scripts in <code>backend/ml</code> - nothing is hard-coded.</p>
      </header>
      <Note>Benchmarks measure agreement with crowd annotations of Reddit text; they do not measure accuracy at detecting anyone's emotional or mental state. Thresholds were tuned on the validation split; the test split was used once for the final numbers.</Note>

      <section className="card space-y-4">
        <h2 className="h2">Emotion classification - GoEmotions (multi-label, 28 labels)</h2>
        {eda?.emotion && (
          <p className="text-xs text-slate-500">
            Dataset: {eda.emotion.splits.train.n} / {eda.emotion.splits.validation.n} / {eda.emotion.splits.test.n} (train/val/test) Reddit comments, mean {num(eda.emotion.splits.train.mean_labels_per_example, 2)} labels per example, label imbalance ratio {num(eda.emotion.imbalance_ratio_max_min, 0)}:1.
          </p>
        )}
        <EmotionSection models={r.emotion} />
      </section>

      <section className="card space-y-4">
        <h2 className="h2">Stress classification - Dreaddit (binary)</h2>
        {eda?.stress && (
          <p className="text-xs text-slate-500">
            Dataset: {eda.stress.splits.train.n} / {eda.stress.splits.validation.n} / {eda.stress.splits.test.n} segments (train/val/test); stress rate {num(eda.stress.splits.train.stress_rate * 100, 1)}% in train.
            {leak && <> Leakage control: {leak.train_test_shared_post_ids} posts shared between train and test and {leak.train_test_duplicate_texts} duplicate texts were removed from train ({leak.train_rows_removed} rows); validation split grouped by post (0 shared posts).</>}
          </p>
        )}
        <StressSection models={r.stress} />
      </section>

      <section className="card space-y-3">
        <h2 className="h2">Exploratory data analysis figures</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {(["emotion", "stress"] as const).flatMap((task) => (eda?.[`${task}_figures`] ?? []).map((f: string) => (
            <figure key={task + f} className="rounded-xl border border-slate-200 p-2">
              <img src={`/api/reports/${task}/${f}`} alt={`${task} ${f}`} className="w-full" />
              <figcaption className="mt-1 text-xs text-slate-500">{task}: {f.replace(".png", "").replace(/_/g, " ")}</figcaption>
            </figure>
          )))}
        </div>
      </section>
    </div>
  );
}
