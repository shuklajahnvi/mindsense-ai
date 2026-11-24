import { useEffect, useState } from "react";
import { api, type Status } from "../api";

const ITEMS: { title: string; body: string }[] = [
  { title: "What MindSense AI is", body: "A non-clinical self-reflection tool. It analyses the language of your journal entries to surface emotion-related and stress-related patterns over time, explains individual model predictions, and offers reflective prompts grounded in reputable public guidance." },
  { title: "What it is not", body: "It is not a medical device or a diagnostic service. It does not diagnose depression, anxiety, PTSD or any other condition, does not recommend medication, and cannot respond to emergencies. Model scores describe wording, not your mental health." },
  { title: "Emotion model", body: "Trained on GoEmotions (Demszky et al., ACL 2020): ~58k Reddit comments annotated with 27 emotions plus neutral. Multi-label: each label has an independent sigmoid output trained with BCEWithLogitsLoss, and a per-label threshold tuned on validation data. A TF-IDF + one-vs-rest logistic regression baseline is provided for comparison." },
  { title: "Stress model", body: "Trained on Dreaddit (Turcan & McKeown, 2019): ~3.5k Reddit post segments labelled by annotators as stress / not stress. Splits are controlled for leakage between segments of the same post. TF-IDF + logistic regression and SVM baselines are compared with a fine-tuned transformer." },
  { title: "Explainability", body: "LIME perturbs your text by removing words and fits a local linear model to the classifier's responses. Highlighted words contributed to the model's prediction for that one text. They show how the model reacted to the wording - they do not establish your psychological state or prove causality." },
  { title: "Limits & bias", body: "Both datasets are English Reddit text, which differs from private journaling in style, topic and demographics. Models can miss sarcasm, negation, cultural expressions and context, and can reflect annotator bias. Treat outputs as prompts for your own reflection." },
  { title: "Retrieval-augmented assistant", body: "Answers are grounded in a small curated knowledge base summarised from NHS, WHO, NIMH, CDC, APA and Mind materials, retrieved with sentence embeddings from ChromaDB. When no LLM API key is configured, the assistant shows the retrieved passages instead of generating text. Generated text passes through a filter that removes diagnostic or medication statements." },
  { title: "Safety", body: "Every entry and message is screened with a rule-based phrase detector for possible risk of self-harm, harm to others or abuse. When triggered, support resources are shown and the assistant does not generate a free-form response. This screen is conservative and imperfect - if you are in danger, contact emergency services directly." },
  { title: "Privacy", body: "Entries are stored in a local SQLite database on this machine. If an LLM provider is configured, the text of your question or entry and retrieved passages are sent to that provider's API to generate a response." },
];

export default function AboutPage() {
  const [s, setS] = useState<Status | null>(null);
  useEffect(() => { api.status().then(setS).catch(() => {}); }, []);
  return (
    <div className="space-y-6">
      <header>
        <h1 className="h1">About & safety</h1>
        <p className="muted">MindSense AI: An Explainable AI and Generative AI Framework for Mental Well-being Analysis.</p>
      </header>
      <div className="grid gap-4 md:grid-cols-2">
        {ITEMS.map((i) => (
          <section key={i.title} className="card">
            <h2 className="mb-1 font-semibold text-stone-900">{i.title}</h2>
            <p className="text-sm leading-relaxed text-stone-600">{i.body}</p>
          </section>
        ))}
      </div>
      {s && (
        <section className="card">
          <h2 className="mb-2 font-semibold text-stone-900">Live system status</h2>
          <pre className="overflow-x-auto rounded-3xl bg-sand-50 px-4 py-3 text-xs text-stone-600">{JSON.stringify(s, null, 2)}</pre>
        </section>
      )}
    </div>
  );
}
