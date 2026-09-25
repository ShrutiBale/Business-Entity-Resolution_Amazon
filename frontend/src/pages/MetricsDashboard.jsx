import { useEffect, useState } from "react";
import { api } from "../api";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from "recharts";

const COLORS = ["#4f8ef7", "#30d158", "#ff9f0a", "#ff453a", "#bf5af2", "#5ac8fa", "#ffd60a"];

function StatCard({ label, value, note, color }) {
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || "var(--text-primary)" }}>{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getMetrics()
      .then(d => { setMetrics(d); setError(null); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-state"><div className="spinner" /><span>Loading metrics…</span></div>;
  if (error)   return <div className="error-box">Error: {error}</div>;
  if (!metrics) return null;

  const pd = metrics.prediction_distribution;
  const pieData = [
    { name: "0 Matches (Singleton)", value: Math.round(pd.zero_matches * 100) },
    { name: "1 Match",               value: Math.round(pd.one_match * 100) },
    { name: "2+ Matches",            value: Math.round(pd.two_plus_matches * 100) },
  ];

  const errorData = Object.entries(metrics.error_taxonomy)
    .map(([name, count]) => ({ name: name.replace(/_/g, " "), count }))
    .sort((a, b) => b.count - a.count);

  const radarData = [
    { metric: "Macro F₀.₅", value: metrics.macro_f05 * 100 },
    { metric: "Singleton F₀.₅", value: metrics.singleton_f05 * 100 },
    { metric: "Non-Singleton F₀.₅", value: metrics.non_singleton_f05 * 100 },
    { metric: "Blocking Recall", value: metrics.blocking_recall * 100 },
    { metric: "Pair Precision", value: metrics.pair_precision * 100 },
    { metric: "Pair Recall", value: metrics.pair_recall * 100 },
  ];

  return (
    <div>
      <h1 className="page-title">Metrics Dashboard</h1>
      <p className="page-sub">
        <strong style={{ color: "var(--accent)" }}>Entity-level Macro F₀.₅</strong> is the competition metric. All other metrics are diagnostic.
      </p>

      {/* Demo banner */}
      {metrics.is_mock_data && (
        <div className="demo-banner">
          ⚠️ DEMO DATA — NOT REAL VALIDATION &nbsp;|&nbsp; {metrics.validation_note}
        </div>
      )}

      {/* Primary metric hero */}
      <div className="card" style={{
        marginBottom: 24, padding: "28px 32px",
        background: "linear-gradient(135deg, rgba(79,142,247,0.10), rgba(48,209,88,0.06))",
        border: "1px solid rgba(79,142,247,0.3)",
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)", letterSpacing: ".08em", textTransform: "uppercase", marginBottom: 8 }}>
          Competition Metric — Entity-Level Macro F₀.₅
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 20, flexWrap: "wrap" }}>
          <div style={{ fontSize: 72, fontWeight: 900, letterSpacing: -3, color: "var(--accent)", lineHeight: 1 }}>
            {metrics.macro_f05.toFixed(3)}
          </div>
          <div>
            <div style={{ fontSize: 14, color: "var(--text-secondary)", marginBottom: 4 }}>
              F₀.₅ = (1.25 × P × R) / (0.25 × P + R)
            </div>
            <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
              Macro-averaged over all S1 entities (including singletons)
            </div>
            <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
              Precision weighted 2× over recall
            </div>
          </div>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid-4" style={{ marginBottom: 24 }}>
        <StatCard label="Singleton F₀.₅"       value={metrics.singleton_f05.toFixed(3)}     color="var(--green)"  note="Correctly predicting 'no match'" />
        <StatCard label="Non-Singleton F₀.₅"    value={metrics.non_singleton_f05.toFixed(3)} color="var(--teal)"   note="Entities with ≥1 true match" />
        <StatCard label="Blocking Recall"        value={(metrics.blocking_recall * 100).toFixed(1) + "%"} color="var(--yellow)" note="True matches retrieved by blocking" />
        <StatCard label="Pair Precision"         value={(metrics.pair_precision * 100).toFixed(1) + "%"} color="var(--orange)" note="Fraction of predicted pairs correct" />
      </div>
      <div className="grid-4" style={{ marginBottom: 32 }}>
        <StatCard label="Pair Recall"            value={(metrics.pair_recall * 100).toFixed(1) + "%"} note="Fraction of true pairs predicted" />
        <StatCard label="Mean Candidates / S1"   value={metrics.mean_candidates_per_s1.toFixed(1)} note="Avg candidate count per entity" />
        <StatCard label="P95 Candidates / S1"    value={metrics.p95_candidates_per_s1.toFixed(0)} note="95th percentile candidate count" />
        <StatCard label="Max Candidates"         value={metrics.max_candidates} note="Largest candidate set seen" />
      </div>

      {/* Charts row */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Radar */}
        <div className="card">
          <div className="section-title" style={{ marginBottom: 16 }}>Performance Radar</div>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(99,120,180,0.2)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
              <Radar name="Score" dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.2} />
              <Tooltip formatter={v => v.toFixed(1) + "%"} contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Prediction distribution pie */}
        <div className="card">
          <div className="section-title" style={{ marginBottom: 16 }}>Prediction Distribution</div>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, value }) => `${value}%`} labelLine={false}>
                {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip formatter={v => v + "%"} contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12 }} />
              <Legend wrapperStyle={{ color: "var(--text-secondary)", fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Error taxonomy bar */}
      <div className="card">
        <div className="section-title" style={{ marginBottom: 16 }}>Error Taxonomy</div>
        <div className="section-sub" style={{ marginBottom: 16 }}>
          Classification of false positives and false negatives by failure mode
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={errorData} layout="vertical" margin={{ left: 10, right: 20, top: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,120,180,0.15)" horizontal={false} />
            <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <YAxis dataKey="name" type="category" width={200} tick={{ fill: "var(--text-secondary)", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12 }} />
            <Bar dataKey="count" fill="var(--accent)" radius={[0, 4, 4, 0]}>
              {errorData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
