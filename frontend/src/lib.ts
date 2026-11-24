// Formatting helpers and the emotion colour scheme.

export const EMOTION_GROUP: Record<string, "positive" | "negative" | "ambiguous" | "neutral"> = {
  admiration: "positive", amusement: "positive", approval: "positive", caring: "positive", desire: "positive",
  excitement: "positive", gratitude: "positive", joy: "positive", love: "positive", optimism: "positive",
  pride: "positive", relief: "positive",
  anger: "negative", annoyance: "negative", disappointment: "negative", disapproval: "negative", disgust: "negative",
  embarrassment: "negative", fear: "negative", grief: "negative", nervousness: "negative", remorse: "negative",
  sadness: "negative",
  confusion: "ambiguous", curiosity: "ambiguous", realization: "ambiguous", surprise: "ambiguous",
  neutral: "neutral",
};

// Categorical slots 1-3 of the validated reference palette (blue, orange, aqua) + neutral gray.
export const GROUP_COLOR = { positive: "#2a78d6", negative: "#eb6834", ambiguous: "#1baf7a", neutral: "#a8a7a0" } as const;
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a"] as const;
export const CHART = { grid: "#e1e0d9", axis: "#898781", ink: "#52514e" } as const;
export const GROUP_CHIP = {
  positive: "bg-blue-50 text-blue-800 ring-1 ring-blue-200",
  negative: "bg-orange-50 text-orange-800 ring-1 ring-orange-200",
  ambiguous: "bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200",
  neutral: "bg-slate-100 text-slate-600 ring-1 ring-slate-200",
} as const;

export const groupOf = (label: string) => EMOTION_GROUP[label] ?? "ambiguous";
export const pct = (v: number | null | undefined, d = 0) => (v == null ? "-" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 2) => (v == null ? "-" : v.toFixed(d));
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, " ");

export function fmtDate(iso: string, withTime = true) {
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  return d.toLocaleString(undefined, withTime
    ? { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }
    : { day: "numeric", month: "short", year: "numeric" });
}
