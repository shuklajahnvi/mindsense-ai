import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { BookOpen, FlaskConical, History, Info, LifeBuoy, Phone, Sparkles, Sprout, TrendingUp, X } from "lucide-react";
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
  { to: "/insights", label: "Insights", icon: TrendingUp },
  { to: "/assistant", label: "Reflect", icon: Sparkles },
  { to: "/models", label: "Models", icon: FlaskConical },
  { to: "/about", label: "About & safety", icon: Info },
];

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-stone-400">
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-sage-400" : "bg-ochre-300"}`} /> {label}
    </div>
  );
}

/** Soft, blurred organic shapes behind the content. Purely decorative. */
function Backdrop() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute -left-32 -top-32 h-[28rem] w-[28rem] rounded-full bg-sage-200/50 blur-3xl" />
      <div className="absolute -right-24 top-1/3 h-[24rem] w-[24rem] rounded-full bg-clay-100/70 blur-3xl" />
      <div className="absolute -bottom-40 left-1/3 h-[26rem] w-[26rem] rounded-full bg-ochre-100/70 blur-3xl" />
    </div>
  );
}

function SupportDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [res, setRes] = useState<Resource[]>([]);
  useEffect(() => { if (open && !res.length) api.resources().then((r) => setRes(r.resources)).catch(() => {}); }, [open, res.length]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-stone-900/20 backdrop-blur-[2px]" onClick={onClose}>
      <div className="m-3 h-[calc(100%-1.5rem)] w-full max-w-md overflow-y-auto rounded-4xl bg-sand-50 p-7 shadow-lift" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <h2 className="h2 flex items-center gap-2"><LifeBuoy className="h-5 w-5 text-clay-500" /> You're not alone</h2>
          <button onClick={onClose} className="rounded-full p-1.5 text-stone-400 hover:bg-stone-100 hover:text-stone-600" aria-label="Close"><X className="h-5 w-5" /></button>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-stone-600">If you are in immediate danger, please contact your local emergency number. These services offer free, confidential support, often 24/7:</p>
        <ul className="mt-5 space-y-2.5">
          {res.map((r) => (
            <li key={r.name} className="rounded-3xl bg-white/80 p-4 text-sm">
              <div className="text-xs font-semibold uppercase tracking-wide text-stone-400">{r.region}</div>
              <div className="font-semibold text-stone-800">{r.name}</div>
              <div className="mt-0.5 flex items-center gap-1.5 text-stone-600"><Phone className="h-3.5 w-3.5 text-sage-500" /> {r.contact}</div>
              {r.url && <a className="text-sm text-sage-700 underline decoration-sage-200 underline-offset-2 hover:decoration-sage-500" href={r.url} target="_blank" rel="noreferrer">{r.url.replace(/^https?:\/\//, "")}</a>}
            </li>
          ))}
        </ul>
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
    <div className="relative min-h-screen lg:flex">
      <Backdrop />
      <aside className="lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-64 lg:shrink-0 lg:flex-col">
        <div className="flex items-center gap-3 px-6 pb-4 pt-7">
          <div className="grid h-10 w-10 place-items-center rounded-full bg-gradient-to-br from-sage-400 to-sage-600 text-white shadow-soft">
            <Sprout className="h-5 w-5" />
          </div>
          <div>
            <div className="font-display text-lg font-medium leading-tight text-stone-800">MindSense</div>
            <div className="text-xs text-stone-400">a gentle space to reflect</div>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-4 pb-3 lg:flex-col lg:overflow-visible">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"}
              className={({ isActive }) => `flex shrink-0 items-center gap-3 rounded-full px-4 py-2.5 text-sm font-semibold transition ${isActive ? "bg-white text-sage-800 shadow-soft" : "text-stone-500 hover:bg-white/60 hover:text-stone-700"}`}>
              <Icon className="h-4 w-4" /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto hidden px-4 pb-6 lg:block">
          <button onClick={() => setSupport(true)} className="btn w-full bg-clay-50 text-clay-700 hover:bg-clay-100"><LifeBuoy className="h-4 w-4" /> Need support?</button>
          <div className="mt-5 space-y-1.5 px-2">
            {status ? (
              <>
                <StatusDot ok={!!status.models.emotion.active_variant} label={`Emotion model: ${status.models.emotion.active_variant ?? "not trained"}`} />
                <StatusDot ok={!!status.models.stress.active_variant} label={`Stress model: ${status.models.stress.active_variant ?? "not trained"}`} />
                <StatusDot ok={status.rag.ready} label={`Knowledge base: ${status.rag.ready ? "ready" : "building..."}`} />
                <StatusDot ok={status.llm.configured} label={`LLM: ${status.llm.configured ? status.llm.provider : "not configured"}`} />
              </>
            ) : <StatusDot ok={false} label="Backend offline" />}
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-5xl px-4 pb-16 pt-6 sm:px-8">
          <div className="mb-8 flex items-center gap-2 rounded-full bg-white/60 px-4 py-2 text-xs text-stone-500 shadow-sm backdrop-blur-sm">
            <Sprout className="h-3.5 w-3.5 shrink-0 text-sage-500" />
            <span>A non-clinical reflection tool. It doesn't diagnose any condition and can't respond to emergencies.</span>
            <button onClick={() => setSupport(true)} className="ml-auto shrink-0 font-semibold text-clay-600 hover:text-clay-700 lg:hidden">Get support</button>
          </div>
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
