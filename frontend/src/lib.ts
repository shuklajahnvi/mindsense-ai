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

// Earthy categorical palette (pine, terracotta, dusk blue), validated with the dataviz palette checker
// on the cream surface #fffdf8: lightness band, chroma floor, CVD separation (>= 8.7) and contrast all pass.
// Neutral is a warm stone grey.
export const GROUP_COLOR = { positive: "#238c6c", negative: "#c65f2f", ambiguous: "#4a74b8", neutral: "#b5ab98" } as const;
export const SERIES = ["#238c6c", "#c65f2f", "#4a74b8"] as const;
export const CHART = {
  grid: "#ece5d8", axis: "#a1968a", baseline: "#d9cfbf", ink: "#57534e", cursor: "#f4eee4", border: "#ece5d8",
} as const;
export const GROUP_CHIP = {
  positive: "bg-[#e3f1ec] text-[#1b6b53]",
  negative: "bg-clay-50 text-clay-700",
  ambiguous: "bg-dusk-50 text-dusk-700",
  neutral: "bg-stone-100 text-stone-600",
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
