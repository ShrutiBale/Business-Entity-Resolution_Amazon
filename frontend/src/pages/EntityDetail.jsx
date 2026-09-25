import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { api } from "../api";

/* ─── Helpers ──────────────────────────────────────────────────── */
function ScoreBar({ value }) {
  const pct = Math.round(value * 100);
  const cls = value >= 0.75 ? "score-high" : value >= 0.50 ? "score-mid" : "score-low";
  return (
    <div className="score-bar">
      <div className="score-bar-track">
        <div className={`score-bar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="score-value">{value.toFixed(3)}</span>
    </div>
  );
}

function Bool({ v }) {
  return v
    ? <span style={{ color: "var(--green)" }}>✓</span>
    : <span style={{ color: "var(--red)" }}>✗</span>;
}

function EntityCard({ entity }) {
  const isFrance = entity.country === "France";
  return (
    <div className="card" style={{ borderLeft: isFrance ? "3px solid var(--purple)" : "3px solid var(--accent)" }}>
      {isFrance && (
        <div className="france-banner" style={{ marginBottom: 14 }}>
          🇫🇷 UNSEEN DOMAIN — FRANCE &nbsp;|&nbsp; Pipeline trained on US/India only
        </div>
      )}
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
        <div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>S1 Reference Entity</div>
          <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>{entity.business_name}</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 10 }}>{entity.business_address}</div>
          <div style={{ display: "flex", gap: 8 }}>
            <span className="mono" style={{ fontSize: 12, color: "var(--accent)" }}>{entity.entity_id}</span>
            <span className={`badge ${entity.country === "France" ? "badge-purple" : entity.country === "India" ? "badge-teal" : "badge-blue"}`}>
              {entity.country}
            </span>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>CASE TAG</div>
          <span className="badge badge-orange" style={{ fontSize: 13 }}>
            {entity.case_tag.replace(/_/g, " ")}
          </span>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8, maxWidth: 280, lineHeight: 1.5 }}>
            {entity.case_description}
          </div>
        </div>
      </div>
    </div>
  );
}

function DecisionPanel({ decision }) {
  const matched = decision.predicted_matches || [];
  const threshold = decision.global_threshold;
  const top1 = decision.top1_score || 0;
  const hasMatch = matched.length > 0;

  return (
    <div className="card">
      <div className="section-title" style={{ marginBottom: 16 }}>Decision Summary</div>
      <div className="grid-2" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: ".06em" }}>Has-Match Probability</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: hasMatch ? "var(--green)" : "var(--red)" }}>
            {(decision.has_any_match_probability * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: ".06em" }}>Predicted Matches</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: matched.length > 1 ? "var(--accent)" : matched.length === 1 ? "var(--green)" : "var(--text-muted)" }}>
            {matched.length}
          </div>
        </div>
      </div>

      {/* Threshold bar */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
          Threshold: {threshold} | Top-1 score: {top1.toFixed(3)} | Top-2: {decision.top2_score?.toFixed(3)} | Margin: {decision.top1_top2_margin?.toFixed(3)}
        </div>
        <div className="threshold-bar">
          <div className="threshold-line" style={{ left: `${threshold * 100}%` }} />
          <div className="threshold-dot" style={{
            left: `${top1 * 100}%`,
            background: top1 >= threshold ? "var(--green)" : "var(--red)"
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "var(--text-muted)", marginTop: 4 }}>
          <span>0.0 — Reject</span>
          <span>Threshold: {threshold}</span>
          <span>1.0 — Accept</span>
        </div>
      </div>

      {/* Predicted IDs */}
      {matched.length > 0 ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {matched.map(id => (
            <span key={id} className="badge badge-green" style={{ fontFamily: "var(--mono)", fontSize: 12.5 }}>✓ {id}</span>
          ))}
        </div>
      ) : (
        <div className="badge badge-red">No match predicted (singleton)</div>
      )}

      <div style={{ marginTop: 12, fontSize: 12.5, color: "var(--text-muted)" }}>{decision.decision_notes}</div>
    </div>
  );
}

function CandidateRow({ cand, threshold }) {
  const isAccepted = cand.selected;
  const borderColor = isAccepted ? "var(--green)" : cand.final_score > threshold * 0.85 ? "var(--yellow)" : "var(--border)";
  const isFrance = cand.country === "France";
  const isChainCollision = !isAccepted && cand.name_similarity >= 0.95 && cand.address_similarity < 0.40;

  return (
    <div className="card card-sm" style={{
      borderLeft: `3px solid ${borderColor}`,
      marginBottom: 12,
      background: isAccepted ? "rgba(48,209,88,0.04)" : undefined
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
        {/* Left: Identity */}
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span className="mono" style={{ fontSize: 12, color: "var(--accent)" }}>{cand.entity_id}</span>
            <span className={`badge ${cand.source === "S2" ? "badge-teal" : "badge-purple"}`}>{cand.source}</span>
            {isAccepted && <span className="badge badge-green">✓ ACCEPTED</span>}
            {!isAccepted && <span className="badge badge-red">✗ REJECTED</span>}
            {isChainCollision && <span className="badge badge-orange">Chain Collision</span>}
            {isFrance && <span className="badge badge-purple">🇫🇷 France</span>}
          </div>
          <div style={{ fontWeight: 600, marginBottom: 2 }}>{cand.business_name}</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{cand.business_address}</div>
        </div>

        {/* Right: Scores */}
        <div style={{ minWidth: 280, flex: "0 0 auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px", fontSize: 12, marginBottom: 8 }}>
            {[
              ["Final Score", cand.final_score],
              ["Name Sim.", cand.name_similarity],
              ["Addr. Sim.", cand.address_similarity],
              ["Composite", cand.composite_similarity],
            ].map(([lbl, val]) => (
              <div key={lbl}>
                <div style={{ color: "var(--text-muted)", fontSize: 11 }}>{lbl}</div>
                <ScoreBar value={val} />
              </div>
            ))}
          </div>

          {/* Evidence chips */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            <span className={`badge ${cand.pin_postal_match ? "badge-green" : "badge-grey"}`}>
              {cand.pin_postal_match ? "✓" : "✗"} PIN/ZIP
            </span>
            <span className={`badge ${cand.house_number_match ? "badge-green" : "badge-grey"}`}>
              {cand.house_number_match ? "✓" : "✗"} House#
            </span>
            <span className={`badge ${cand.cross_source_support ? "badge-blue" : "badge-grey"}`}>
              {cand.cross_source_support ? "✓" : "✗"} Cross-src
            </span>
            <span className="badge badge-grey" style={{ fontFamily: "var(--mono)" }}>
              freq: {cand.name_frequency.toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Blocking passes */}
      <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
        <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Blocking passes:</span>
        {(cand.blocking_passes || []).map(p => (
          <span key={p} className="badge badge-grey" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>{p}</span>
        ))}
        {(!cand.blocking_passes || cand.blocking_passes.length === 0) && (
          <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>none</span>
        )}
      </div>

      {/* Notes */}
      {cand.notes && (
        <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-muted)", fontStyle: "italic", lineHeight: 1.6 }}>
          {cand.notes}
        </div>
      )}
    </div>
  );
}

export default function EntityDetail() {
  const { entityId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showcase, setShowcase] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getEntity(entityId),
      api.getShowcase(),
    ])
      .then(([d, s]) => { setData(d); setShowcase(s); setError(null); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [entityId]);

  if (loading) return <div className="loading-state"><div className="spinner" /><span>Loading entity…</span></div>;
  if (error)   return <div className="error-box">Error: {error}</div>;
  if (!data)   return null;

  const { entity, candidates, decision, explanation } = data;
  const accepted = candidates.filter(c => c.selected);
  const rejected = candidates.filter(c => !c.selected);
  const threshold = decision.global_threshold || 0.60;

  const QUICK = [
    { label: "Singleton", id: showcase?.singleton },
    { label: "Single Match", id: showcase?.single_match },
    { label: "Multi-Match", id: showcase?.multi_match },
    { label: "Chain Collision", id: showcase?.chain_collision },
    { label: "Noisy Match", id: showcase?.noisy_match },
    { label: "🇫🇷 France", id: showcase?.france },
  ];

  return (
    <div>
      {/* Back + quick nav */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
        <button className="btn btn-ghost btn-sm" onClick={() => navigate("/entities")}>← Back</button>
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Jump to:</span>
        {QUICK.map(q => q.id && (
          <button key={q.label} className={`btn btn-ghost btn-sm ${q.id === entityId ? "" : ""}`}
            onClick={() => navigate(`/entities/${q.id}`)}
            style={{ opacity: q.id === entityId ? 0.5 : 1 }}>
            {q.label}
          </button>
        ))}
      </div>

      {/* Demo banner */}
      <div className="demo-banner">
        ⚠️ DEMO DATA — NOT REAL VALIDATION &nbsp;|&nbsp;
        Replace data files with real pipeline exports for live metrics
      </div>

      {/* Entity card */}
      <div style={{ marginBottom: 20 }}>
        <EntityCard entity={entity} />
      </div>

      {/* Pipeline trace */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-title" style={{ marginBottom: 12 }}>Pipeline Trace</div>
        <div className="pipeline">
          {[
            { label: "S1 Entity", done: true },
            { label: "Normalization", done: true },
            { label: `Blocking → ${candidates.length} candidates`, done: true },
            { label: `Features`, done: true },
            { label: "LightGBM Score", done: true },
            { label: `Threshold (${threshold})`, done: true },
            { label: `${accepted.length} Match(es)`, done: true },
          ].map((s, i, arr) => (
            <div key={s.label} style={{ display: "flex", alignItems: "center" }}>
              <div className="pipeline-step active">{s.label}</div>
              {i < arr.length - 1 && <span className="pipeline-arrow">→</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 20 }}>
        {/* Decision */}
        <DecisionPanel decision={decision} />

        {/* Explanation */}
        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Why This Decision?</div>
          <div className="explanation-box">
            {explanation}
          </div>
          <div style={{ marginTop: 12, fontSize: 11.5, color: "var(--text-muted)" }}>
            Explanations are deterministic and feature-based — no LLM used.
          </div>
        </div>
      </div>

      {/* Candidate table */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-header">
          <div className="section-title">
            Candidates ({candidates.length}) &nbsp;
            <span className="badge badge-green">{accepted.length} accepted</span>
            &nbsp;
            <span className="badge badge-red">{rejected.length} rejected</span>
          </div>
          <div className="section-sub">
            Sorted by final score. Every accepted ID also appears in candidate_pairs.tsv.
          </div>
        </div>

        {candidates.length === 0 ? (
          <div className="empty-state">No candidates were retrieved by blocking.</div>
        ) : (
          [...accepted, ...rejected]
            .sort((a, b) => b.final_score - a.final_score)
            .map(c => <CandidateRow key={c.entity_id} cand={c} threshold={threshold} />)
        )}
      </div>

      {/* Feature breakdown table */}
      {candidates.length > 0 && (
        <div className="card">
          <div className="section-title" style={{ marginBottom: 16 }}>Feature / Evidence Breakdown</div>
          <div style={{ overflowX: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Final Score</th>
                  <th>Name Sim.</th>
                  <th>Addr. Sim.</th>
                  <th>Composite</th>
                  <th>Name Freq.</th>
                  <th>PIN Match</th>
                  <th>House#</th>
                  <th>Cross-Src</th>
                  <th>Blocking Count</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {[...accepted, ...rejected]
                  .sort((a, b) => b.final_score - a.final_score)
                  .map(c => (
                    <tr key={c.entity_id}>
                      <td>
                        <span className="mono" style={{ fontSize: 11.5, color: "var(--accent)" }}>{c.entity_id}</span>
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{c.source}</div>
                      </td>
                      <td><span className="mono" style={{ color: c.final_score >= threshold ? "var(--green)" : "var(--red)" }}>{c.final_score.toFixed(3)}</span></td>
                      <td className="mono">{c.name_similarity.toFixed(3)}</td>
                      <td className="mono">{c.address_similarity.toFixed(3)}</td>
                      <td className="mono">{c.composite_similarity.toFixed(3)}</td>
                      <td className="mono" style={{ fontSize: 11.5 }}>{c.name_frequency.toFixed(4)}</td>
                      <td><Bool v={c.pin_postal_match} /></td>
                      <td><Bool v={c.house_number_match} /></td>
                      <td><Bool v={c.cross_source_support} /></td>
                      <td className="mono">{c.blocking_pass_count}</td>
                      <td>
                        {c.selected
                          ? <span className="badge badge-green">Accept</span>
                          : <span className="badge badge-red">Reject</span>}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
