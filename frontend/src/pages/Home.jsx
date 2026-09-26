import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";

const PIPELINE_STEPS = [
  { id: "s1", label: "S1 Reference", icon: "🏢" },
  { id: "norm", label: "Normalization", icon: "🔧" },
  { id: "block", label: "Multi-Pass Blocking", icon: "🔎" },
  { id: "candidates", label: "Candidate Union", icon: "📋" },
  { id: "features", label: "Evidence Features", icon: "⚡" },
  { id: "lgbm", label: "LightGBM", icon: "🤖" },
  { id: "threshold", label: "Global Threshold", icon: "⚖️" },
  { id: "output", label: "Match Set", icon: "✅" },
];

const KEY_IDEAS = [
  {
    icon: "🗂️", color: "var(--accent)",
    title: "3 Independent Sources",
    desc: "S1 is the deduplicated reference. S2 and S3 contribute noisy, partial fragments with no shared identifiers."
  },
  {
    icon: "0/1/∞", color: "var(--green)",
    title: "Zero, One, or Many Matches",
    desc: "An S1 entity may have no matches, exactly one match, or multiple matches across S2/S3. One-to-one is never assumed."
  },
  {
    icon: "⚖️", color: "var(--yellow)",
    title: "Precision-Heavy F₀.₅",
    desc: "F₀.₅ weights precision 2× over recall. Merging two distinct businesses is more costly than missing a link."
  },
  {
    icon: "🌍", color: "var(--purple)",
    title: "Open-Set Countries",
    desc: "Country is never hard-coded. France appears only in test data — the pipeline handles it without France-specific training."
  },
];

const SHOWCASE_TAGS = [
  { tag: "singleton",       label: "Singleton",         color: "badge-red",    desc: "No match — all candidates correctly rejected (verified against training ground truth)" },
  { tag: "single_match",    label: "Single Match",       color: "badge-green",  desc: "One clean or noisy match (verified against training ground truth)" },
  { tag: "multi_match",     label: "Multi-Match",        color: "badge-blue",   desc: "Many S2/S3 records → one S1 (verified against training ground truth)" },
  { tag: "chain_collision", label: "Chain Collision",    color: "badge-orange", desc: "Same name, different branches (verified against training ground truth)" },
  { tag: "transliteration", label: "Noisy/Transliter.",  color: "badge-yellow", desc: "Phonetic or script variation (verified against training ground truth)" },
  { tag: "france",          label: "🇫🇷 France",          color: "badge-purple", desc: "Unseen domain — pipeline PREDICTION only. No ground truth exists for any test entity." },
];

export default function Home() {
  const [showcase, setShowcase] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    api.getShowcase().then(setShowcase).catch(() => {});
    const t = setInterval(() => setActiveStep(s => (s + 1) % PIPELINE_STEPS.length), 1000);
    return () => clearInterval(t);
  }, []);

  const goTo = (tag) => {
    if (!showcase) return;
    const idMap = {
      singleton: showcase.singleton,
      single_match: showcase.single_match,
      multi_match: showcase.multi_match,
      chain_collision: showcase.chain_collision,
      transliteration: showcase.noisy_match,
      france: showcase.france,
    };
    if (idMap[tag]) navigate(`/entities/${idMap[tag]}`);
    else navigate("/entities");
  };

  return (
    <div>
      {/* Hero */}
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <span className="badge badge-blue">Amazon ML Challenge 2026</span>
          <span className="badge badge-orange">DEMO DATA — NOT REAL VALIDATION</span>
        </div>
        <h1 className="page-title" style={{ fontSize: 36 }}>
          Business Entity Resolution
        </h1>
        <p style={{ fontSize: 15, color: "var(--text-secondary)", maxWidth: 680, lineHeight: 1.7 }}>
          Given business records from 3 independent sources with noisy, inconsistent fields,
          determine which records refer to the same real-world business entity.
          This demo makes the pipeline's decisions transparent and explainable.
        </p>
      </div>

      {/* Pipeline diagram */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="section-header">
          <div className="section-title">Pipeline Overview</div>
          <div className="section-sub">Animated trace of how an S1 entity flows through the system</div>
        </div>
        <div className="pipeline">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step.id} style={{ display: "flex", alignItems: "center" }}>
              <div className={`pipeline-step ${i === activeStep ? "active" : ""}`}>
                <span>{step.icon}</span>
                <span>{step.label}</span>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <span className="pipeline-arrow">→</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Key ideas */}
      <div className="grid-4" style={{ marginBottom: 32 }}>
        {KEY_IDEAS.map((idea) => (
          <div key={idea.title} className="card card-sm" style={{ borderLeft: `3px solid ${idea.color}` }}>
            <div style={{ fontSize: 24, marginBottom: 8 }}>{idea.icon}</div>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{idea.title}</div>
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.6 }}>{idea.desc}</div>
          </div>
        ))}
      </div>

      {/* Showcase shortcuts */}
      <div className="card">
        <div className="section-header">
          <div className="section-title">Demo Showcase</div>
          <div className="section-sub">Jump to a curated example covering each pipeline scenario</div>
        </div>
        <div className="grid-3">
          {SHOWCASE_TAGS.map((s) => (
            <button key={s.tag} className="btn btn-ghost" onClick={() => goTo(s.tag)}
              style={{ justifyContent: "flex-start", gap: 12, padding: "14px 18px", textAlign: "left" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span className={`badge ${s.color}`}>{s.label}</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{s.desc}</div>
              </div>
            </button>
          ))}
        </div>
        </div>
        {/* Fix 8: training vs test provenance note */}
        <div style={{
          marginTop: 16, padding: "12px 16px",
          background: "rgba(191,90,242,0.06)", border: "1px solid rgba(191,90,242,0.25)",
          borderRadius: "var(--radius)", fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.7
        }}>
          <strong style={{ color: "var(--text-primary)" }}>Training vs. Test Provenance:</strong>{" "}
          Training-derived examples (all non-France cases) show verified accuracy — ground truth exists and was checked in S1-grouped CV + holdout evaluation.
          Test-derived examples — <strong>including every France case</strong> — show what the frozen pipeline predicts on genuinely unseen data.
          This is inherent to the challenge: test ground truth is never available to any participant, not a limitation of this demo.
        </div>
      </div>
  );
}
