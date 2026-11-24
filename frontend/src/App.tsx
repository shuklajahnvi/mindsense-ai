import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { BarChart3, BookOpen, Brain, FlaskConical, History, Info, LifeBuoy, MessageCircleHeart, ShieldCheck } from "lucide-react";
import { api, type Resource, type Status } from "./api";
import JournalPage from "./pages/Journal";
import HistoryPage from "./pages/History";
import EntryPage from "./pages/Entry";
import InsightsPage from "./pages/Insights";
import AssistantPage from "./pages/Assistant";
import ModelsPage from "./pages/Models";
import AboutPage from "./pages/About";

const NAV = [
  { to: "/", label: "Journal", icon: BookOpen },
  { to: "/history", label: "History", icon: History },
  { to: "/insights", label: "Insights", icon: BarChart3 },
  { to: "/assistant", label: "Reflect (RAG)", icon: MessageCircleHeart },
  { to: "/models", label: "Models & evaluation", icon: FlaskConical },
  { to: "/about", label: "About & safety", icon: Info },
];

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-amber-400"}`} /> {label}
    </div>
  );
}

function SupportDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [res, setRes] = useState<Resource[]>([]);
  useEffect(() => { if (open && !res.length) api.resources().then((r) => setRes(r.resources)).catch(() => {}); }, [open, res.length]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/30" onClick={onClose}>
      <div className="h-full w-full max-w-md overflow-y-auto bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="h2 flex items-center gap-2"><LifeBuoy className="h-5 w-5 text-rose-500" /> Need support now?</h2>
        <p className="mt-2 text-sm text-slate-600">If you are in immediate danger, contact your local emergency number. These services offer free, confidential support:</p>
        <ul className="mt-4 space-y-3">
          {res.map((r) => (
            <li key={r.name} className="rounded-xl border border-slate-200 p-3 text-sm">
              <div className="font-semibold">{r.region}: {r.name}</div>
              <div className="text-slate-600">{r.contact}</div>
              {r.url && <a className="text-indigo-600 hover:underline" href={r.url} target="_blank" rel="noreferrer">{r.url}</a>}
            </li>
          ))}
        </ul>
        <button className="btn-ghost mt-6 w-full" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [support, setSupport] = useState(false);
  useEffect(() => {
    const load = () => api.status().then(setStatus).catch(() => setStatus(null));
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="min-h-screen lg:flex">
      <aside className="border-b border-slate-200 bg-white lg:sticky lg:top-0 lg:h-screen lg:w-64 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-2 px-5 py-5">
          <div className="rounded-xl bg-indigo-600 p-2 text-white"><Brain className="h-5 w-5" /></div>
          <div>
            <div className="font-semibold text-slate-900">MindSense AI</div>
            <div className="text-[11px] text-slate-500">Explainable wellbeing journaling</div>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"}
              className={({ isActive }) => `flex shrink-0 items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium ${isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-50"}`}>
              <Icon className="h-4 w-4" /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden space-y-1.5 border-t border-slate-100 px-5 py-4 lg:block">
          {status ? (
            <>
              <StatusDot ok={!!status.models.emotion.active_variant} label={`Emotion: ${status.models.emotion.active_variant ?? "not trained"}`} />
              <StatusDot ok={!!status.models.stress.active_variant} label={`Stress: ${status.models.stress.active_variant ?? "not trained"}`} />
              <StatusDot ok={status.rag.ready} label={`RAG: ${status.rag.ready ? `${status.rag.chunks} chunks` : "building..."}`} />
              <StatusDot ok={status.llm.configured} label={`LLM: ${status.llm.configured ? `${status.llm.provider} (${status.llm.model})` : "not configured"}`} />
            </>
          ) : <StatusDot ok={false} label="Backend offline" />}
        </div>
        <div className="hidden px-3 lg:block">
          <button onClick={() => setSupport(true)} className="btn w-full bg-rose-50 text-rose-700 hover:bg-rose-100"><LifeBuoy className="h-4 w-4" /> Get support</button>
        </div>
      </aside>

      <main className="flex-1">
        <div className="flex items-center gap-2 border-b border-amber-200 bg-amber-50 px-6 py-2 text-xs text-amber-800">
          <ShieldCheck className="h-4 w-4 shrink-0" />
          <span>MindSense AI is a non-clinical self-reflection tool. It does not diagnose any condition and cannot respond to emergencies.</span>
          <button onClick={() => setSupport(true)} className="ml-auto shrink-0 font-semibold underline lg:hidden">Get support</button>
        </div>
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
          <Routes>
            <Route path="/" element={<JournalPage status={status} />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/entries/:id" element={<EntryPage />} />
            <Route path="/insights" element={<InsightsPage />} />
            <Route path="/assistant" element={<AssistantPage status={status} />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </div>
      </main>
      <SupportDrawer open={support} onClose={() => setSupport(false)} />
    </div>
  );
}
