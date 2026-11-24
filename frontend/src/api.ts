// Typed client for the MindSense FastAPI backend (proxied at /api by Vite).

export type Variant = "auto" | "transformer" | "baseline";

export interface EmotionRow { label: string; score: number; threshold: number; predicted: boolean }
export interface StressPred { label: "stress" | "not_stress"; score: number; threshold: number }
export interface ModelInfo {
  name?: string; variant?: string; algorithm?: string; fine_tuned?: boolean; base_checkpoint?: string | null;
  trained_on_subset?: boolean; n_train?: number; status?: string; test_headline?: Record<string, number>;
}
export interface Resource { region: string; name: string; contact: string; url: string | null }
export interface Safety { level: "none" | "elevated" | "high"; categories: string[]; message: string | null; resources: Resource[]; method: string }
export interface LimeTarget { label: string; prediction_score: number | null; local_model_r2: number; intercept: number; weights: { token: string; weight: number }[] }
export interface LimeBlock { model: string; variant: string; targets: LimeTarget[]; seconds: number }
export interface Explanations { method: string; num_samples: number; disclaimer: string; emotion?: LimeBlock; stress?: LimeBlock }
export interface Source { ref: number; title: string; section: string; similarity: number; excerpt: string; sources: { name: string; url: string }[] }
export interface AssistantReply {
  answer: string; mode: "llm" | "retrieval_only" | "safety"; provider: { provider: string; model: string | null } | null;
  sources: Source[]; safety: Safety; guardrail_removed: string[]; error: string | null; message_id?: number | null;
}

export interface Entry {
  id: number; user_id: number; created_at: string; text: string;
  mood: number | null; energy: number | null; sleep_hours: number | null; sleep_quality: number | null;
  emotions: EmotionRow[] | null; predicted_emotions: string[] | null;
  stress_label: string | null; stress_score: number | null;
  model_info: { emotion?: ModelInfo; stress?: ModelInfo } | null;
  explanations: Explanations | null; safety: Safety | null; reflection: AssistantReply | null; is_demo: boolean;
}
export type EntrySummary = Pick<Entry, "id" | "created_at" | "text" | "mood" | "energy" | "sleep_hours" | "predicted_emotions" | "stress_label" | "stress_score" | "is_demo">;

export interface Aggregate {
  entries: number; avg_stress_score: number | null; stress_language_share: number | null;
  avg_mood: number | null; avg_energy: number | null; avg_sleep_hours: number | null; avg_sleep_quality: number | null;
  emotion_groups: Record<string, number>; top_emotions: { label: string; entries: number }[];
  mean_emotion_scores: Record<string, number>;
}
export interface Period extends Aggregate { period: string; start: string; end: string; label: string; observations: string[] }
export interface Insights {
  n_entries: number; caveat: string; observations: { statements: string[]; window_days: number };
  daily: (Aggregate & { date: string })[]; weekly: Period[]; monthly: Period[];
  emotion_distribution: { label: string; group: string; entries_predicted: number; mean_score: number }[];
  emotion_groups: Record<string, string[]>;
}
export interface Topic {
  topic_id: number; name: string; keywords: string[]; n_entries: number; share: number; entry_ids: number[];
  representative_entry_ids: number[]; avg_stress_score: number | null; first_seen: string; last_seen: string;
}
export interface Topics { status: string; n_entries: number; min_entries?: number; k?: number; silhouette_cosine?: number; method: string | null; topics: Topic[] }

export interface Status {
  models: Record<"emotion" | "stress", { active_variant: string | null; available: Record<string, null | { name: string; fine_tuned: boolean; algorithm: string; trained_on_subset: boolean }> }>;
  llm: { provider: string; model: string | null; configured: boolean; detail?: string };
  rag: { ready: boolean; chunks?: number; documents?: number; embedding_model?: string; store?: string; error?: string };
  explainability: { method: string; available: boolean };
  notice: string;
}
export interface ChatMessage { id: number; created_at: string; role: "user" | "assistant"; content: string; sources: Source[] | null; provider: string | null; safety: Safety | null }

class ApiError extends Error { constructor(public status: number, message: string) { super(message); } }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j); } catch { /* ignore */ }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}
const post = <T,>(path: string, body?: unknown) => req<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  status: () => req<Status>("/api/status"),
  resources: () => req<{ notice: string; resources: Resource[] }>("/api/safety/resources"),
  createEntry: (body: { text: string; mood?: number | null; energy?: number | null; sleep_hours?: number | null; sleep_quality?: number | null; explain?: boolean }) =>
    post<Entry>("/api/entries", body),
  entries: (limit = 100, offset = 0) => req<{ total: number; items: EntrySummary[] }>(`/api/entries?limit=${limit}&offset=${offset}`),
  entry: (id: number) => req<Entry>(`/api/entries/${id}`),
  deleteEntry: (id: number) => req<{ deleted: number }>(`/api/entries/${id}`, { method: "DELETE" }),
  explain: (id: number, emotion_model: Variant = "auto", stress_model: Variant = "auto") =>
    post<Entry>(`/api/entries/${id}/explain`, { emotion_model, stress_model }),
  reflection: (id: number) => post<Entry>(`/api/entries/${id}/reflection`),
  insights: (days?: number) => req<Insights>(`/api/insights${days ? `?days=${days}` : ""}`),
  topics: () => req<Topics>("/api/insights/topics"),
  chat: (message: string, use_journal_context = true) => post<AssistantReply>("/api/chat", { message, use_journal_context }),
  chatHistory: () => req<ChatMessage[]>("/api/chat/history"),
  clearChat: () => req<{ cleared: boolean }>("/api/chat/history", { method: "DELETE" }),
  models: () => req<ModelReports>("/api/models"),
  seedDemo: () => post<{ inserted: number; note: string }>("/api/demo/seed"),
  clearDemo: () => req<{ deleted: number }>("/api/demo", { method: "DELETE" }),
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;
export interface ModelMeta {
  name: string; task: string; kind: string; algorithm: string; fine_tuned: boolean; base_checkpoint?: string | null;
  fine_tuning_scope?: string;
  labels: string[]; hyperparameters: Json; training: Json; trained_at: string;
  metrics: { validation: Json; test: Json };
}
export interface ModelReports { emotion: Record<string, ModelMeta>; stress: Record<string, ModelMeta>; eda: Json }
