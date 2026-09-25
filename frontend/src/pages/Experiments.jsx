import { useEffect, useState } from "react";
import { api } from "../api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

function ScoreBar({ value, max = 1 }) {
  const pct = Math.round((value / max) * 100);
  const cls = value >= 0.8 ? "score-high" : value >= 0.65 ? "score-mid" : "score-low";
  return (
    <div className="score-bar">
      <div className="score-bar-track">
        <div className={`score-bar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="score-value">{typeof value === "number" ? value.toFixed(3) : value}</span>
    </div>
  );
}

export default function Experiments() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getExperiments()
      .then(d => { setRuns(d.runs); setError(null); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-state"><div className="spinner" /><span>Loading experiments…</span></div>;
  if (error)   return <div className="error-box">Error: {error}</div>;

  const bestRun = runs.reduce((b, r) => r.macro_f05 > (b?.macro_f05 || 0) ? r : b, null);

  // Chart data
  const chartData = runs.map(r => ({
    run: r.run_id,
    "Macro F₀.₅": r.macro_f05,
    "Blocking Recall": r.blocking_recall,
    "Precision": r.precision,
  }));

  return (
    <div>
      <h1 className="page-title">Experiment Runs</h1>
      <p className="page-sub">
        Measured iteration from baseline → rule blocking → TF-IDF → LightGBM.
        Each component was added only when a held-out experiment proved it helped.
      </p>

      {/* F0.5 progression chart */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="section-title" style={{ marginBottom: 4 }}>Macro F₀.₅ Progression</div>
        <div className="section-sub" style={{ marginBottom: 20 }}>Each point = one experiment. Only measured gains were kept.</div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ left: 0, right: 20, top: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,120,180,0.15)" />
            <XAxis dataKey="run" tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
            <YAxis domain={[0.4, 1.0]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12 }} />
            <Line type="monotone" dataKey="Macro F₀.₅" stroke="var(--accent)" strokeWidth={2.5} dot={{ fill: "var(--accent)", r: 5 }} />
            <Line type="monotone" dataKey="Blocking Recall" stroke="var(--green)" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
            <Line type="monotone" dataKey="Precision" stroke="var(--orange)" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <div style={{ display: "flex", gap: 20, fontSize: 12, color: "var(--text-muted)", marginTop: 8, justifyContent: "flex-end" }}>
          <span style={{ color: "var(--accent)" }}>─ Macro F₀.₅</span>
          <span style={{ color: "var(--green)" }}>- - Blocking Recall</span>
          <span style={{ color: "var(--orange)" }}>- - Precision</span>
        </div>
      </div>

      {/* Run cards */}
      {runs.map(run => {
        const isBest = run.run_id === bestRun?.run_id;
        return (
          <div key={run.run_id} className="card" style={{
            marginBottom: 16,
            borderLeft: isBest ? "3px solid var(--green)" : "3px solid var(--border)",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                  <span className="mono" style={{ color: "var(--accent)", fontSize: 13 }}>{run.run_id}</span>
                  {isBest && <span className="badge badge-green">✓ Current Best</span>}
                </div>
                <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{run.description}</div>
                <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginBottom: 8 }}>
                  {run.blocking_config}
                  {run.tfidf_k && <span style={{ marginLeft: 8 }} className="badge badge-blue">TF-IDF K={run.tfidf_k}</span>}
                </div>
                <div style={{ fontSize: 12.5, color: "var(--text-muted)", fontStyle: "italic" }}>{run.notes}</div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "8px 24px", minWidth: 320 }}>
                {[
                  ["Macro F₀.₅", run.macro_f05],
                  ["Blocking Recall", run.blocking_recall],
                  ["Precision", run.precision],
                  ["Recall", run.recall],
                  ["Singleton F₀.₅", run.singleton_f05],
                  ["Non-Singleton F₀.₅", run.non_singleton_f05],
                ].map(([label, val]) => (
                  <div key={label}>
                    <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 3 }}>{label}</div>
                    <ScoreBar value={val} />
                  </div>
                ))}
              </div>
            </div>

            <hr className="divider" />

            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", fontSize: 12.5, color: "var(--text-secondary)" }}>
              <span>Threshold: <span className="mono" style={{ color: "var(--text-primary)" }}>{run.threshold}</span></span>
              <span>Mean cands: <span className="mono" style={{ color: "var(--text-primary)" }}>{run.mean_candidates}</span></span>
              <span>P95 cands: <span className="mono" style={{ color: "var(--text-primary)" }}>{run.p95_candidates}</span></span>
              <span>Max cands: <span className="mono" style={{ color: "var(--text-primary)" }}>{run.max_candidates}</span></span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
