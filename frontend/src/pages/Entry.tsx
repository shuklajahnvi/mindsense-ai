import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Trash2 } from "lucide-react";
import { api, type Entry } from "../api";
import { ErrorBox, Spinner } from "../components";
import { fmtDate } from "../lib";
import AnalysisView from "./AnalysisView";

export default function EntryPage() {
  const { id } = useParams();
  const nav = useNavigate();
  const [entry, setEntry] = useState<Entry | null>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { api.entry(Number(id)).then(setEntry).catch(setErr); }, [id]);

  const del = async () => {
    if (!entry || !confirm("Delete this entry permanently?")) return;
    await api.deleteEntry(entry.id);
    nav("/history");
  };

  if (err) return <ErrorBox error={err} />;
  if (!entry) return <Spinner />;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Link to="/history" className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-800"><ArrowLeft className="h-4 w-4" /> History</Link>
        <button onClick={del} className="btn-ghost !py-1.5 text-rose-600"><Trash2 className="h-4 w-4" /> Delete</button>
      </div>
      <section className="card">
        <div className="mb-2 flex items-center gap-2 text-xs text-slate-500">
          Entry #{entry.id} · {fmtDate(entry.created_at)}
          {entry.is_demo && <span className="chip bg-violet-50 text-violet-700">demo (synthetic text)</span>}
        </div>
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800">{entry.text}</p>
      </section>
      <AnalysisView entry={entry} onUpdate={setEntry} />
    </div>
  );
}
