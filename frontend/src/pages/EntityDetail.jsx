import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
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
  if (v === null || v === undefined)
    return <span style={{ color: "var(--text-muted)" }}>—</span>;
  return v
    ? <span style={{ color: "var(--green)" }}>✓</span>
    : <span style={{ color: "var(--red)" }}>✗</span>;
}

/* Fix 8: ground_truth_source badge */
function GroundTruthBadge({ source }) {
  if (source === "train_holdout") {
    return (
      <span className="badge badge-green" style={{ fontSize: 11.5 }}>
        ✓ Verified against training ground truth
      </span>
    );
  }
  return (
    <span className="badge badge-orange" style={{ fontSize: 11.5, whiteSpace: "normal", lineHeight: 1.4 }}>
      ⚠ Test prediction — not verifiable (no ground truth exists for test data, for us or anyone)
    </span>
  );
}

function EntityCard({ entity }) {
  const isFrance = entity.country === "France";
  const isUnverified = entity.ground_truth_source === "test_unverified";
  return (
    <div className="card" style={{ borderLeft: isFrance ? "3px solid var(--purple)" : "3px solid var(--accent)" }}>
      {isFrance && (
        <div className="france-banner" style={{ marginBottom: 14 }}>
          🇫🇷 UNSEEN DOMAIN — FRANCE &nbsp;|&nbsp; Prediction only — no ground truth exists for any test entity
        </div>
      )}
      {isUnverified && !isFrance && (
        <div style={{
          display: "flex", alignItems: "center", gap: 10, marginBottom: 14,
          background: "var(--orange-dim)", border: "1px solid rgba(255,159,10,.4)",
          borderRadius: "var(--radius)", padding: "10px 16px",
          fontSize: 13, color: "var(--orange)"
        }}>
          ⚠ TEST PREDICTION — no ground truth exists
        </div>
      )}
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
        <div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>S1 Reference Entity</div>
          <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>{entity.business_name}</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 10 }}>{entity.business_address}</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <span className="mono" style={{ fontSize: 12, color: "var(--accent)" }}>{entity.entity_id}</span>
            <span className={`badge ${entity.country === "France" ? "badge-purple" : entity.country === "India" ? "badge-teal" : "badge-blue"}`}>
              {entity.country}
            </span>
            <GroundTruthBadge source={entity.ground_truth_source} />
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

function DecisionPanel({ decision, gtSource }) {
  const matched = decision.predicted_matches || [];
  const threshold = decision.global_threshold;
  const top1 = decision.top1_score || 0;
  const hasMatch = matched.length > 0;
  const isUnverified = gtSource === "test_unverified";

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
          <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: ".06em" }}>
            {isUnverified ? "Predicted Matches" : "Verified Matches"}
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: matched.length > 1 ? "var(--accent)" : matched.length === 1 ? "var(--green)" : "var(--text-muted)" }}>
            {matched.length}
          </div>
        </div>
      </div>

      {/* Threshold bar */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
          Threshold: {threshold} | Top-1: {top1.toFixed(3)} | Top-2: {decision.top2_score?.toFixed(3)} | Margin: {decision.top1_top2_margin?.toFixed(3)}
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
            <span key={id} className={`badge ${isUnverified ? "badge-orange" : "badge-green"}`}
              style={{ fontFamily: "var(--mono)", fontSize: 12.5 }}>
              {isUnverified ? "⚠" : "✓"} {id}
            </span>
          ))}
        </div>
      ) : (
        <div className="badge badge-red">No match predicted (singleton)</div>
      )}
      {isUnverified && matched.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11.5, color: "var(--orange)", fontStyle: "italic" }}>
          These are model predictions on test data — correctness cannot be verified by any participant.
        </div>
      )}
      <div style={{ marginTop: 12, fontSize: 12.5, color: "var(--text-muted)" }}>{decision.decision_notes}</div>
    </div>
  );
}

/* Fix 2: render structured cross_source_support */
function CrossSourcePanel({ css }) {
  if (!css || typeof css !== "object") return <span style={{ color: "var(--text-muted)" }}>—</span>;
  return (
    <div style={{ fontSize: 11.5 }}>
      <div style={{ color: "var(--text-muted)" }}>S2↔S3 name sim: <span style={{ color: "var(--text-primary)", fontFamily: "var(--mono)" }}>{css.s2_s3_name_similarity?.toFixed(3)}</span></div>
      <div style={{ color: "var(--text-muted)" }}>S2↔S3 addr sim: <span style={{ color: "var(--text-primary)", fontFamily: "var(--mono)" }}>{css.s2_s3_address_similarity?.toFixed(3)}</span></div>
      <div style={{ color: "var(--text-muted)" }}>Numeric agree: <Bool v={css.numeric_agreement} /></div>
    </div>
  );
}

function CandidateRow({ cand, threshold, gtSource }) {
  const isAccepted = cand.selected;
  const isUnverified = gtSource === "test_unverified";
  const borderColor = isAccepted
    ? (isUnverified ? "var(--orange)" : "var(--green)")
    : cand.final_score > threshold * 0.85 ? "var(--yellow)" : "var(--border)";
  const isChainCollision = !isAccepted && cand.name_similarity >= 0.95 && cand.address_similarity < 0.40;
  const hasCluster = (cand.cluster_size || 1) > 1;

  return (
    <div className="card card-sm" style={{
      borderLeft: `3px solid ${borderColor}`,
      marginBottom: 12,
      background: isAccepted && !isUnverified ? "rgba(48,209,88,0.04)" : isAccepted && isUnverified ? "rgba(255,159,10,0.04)" : undefined
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
        {/* Left: Identity */}
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, flexWrap: "wrap" }}>
            <span className="mono" style={{ fontSize: 12, color: "var(--accent)" }}>{cand.entity_id}</span>
            <span className={`badge ${cand.source === "S2" ? "badge-teal" : "badge-purple"}`}>{cand.source}</span>
            {isAccepted && !isUnverified && <span className="badge badge-green">✓ ACCEPTED</span>}
            {isAccepted && isUnverified && <span className="badge badge-orange">⚠ PREDICTED</span>}
            {!isAccepted && <span className="badge badge-red">✗ REJECTED</span>}
            {isChainCollision && <span className="badge badge-orange">Chain Collision</span>}
            {/* Fix 5: near-duplicate cluster badge */}
            {hasCluster && (
              <span className="badge badge-grey" title="Near-duplicate cluster — context only, never auto-merged">
                Cluster × {cand.cluster_size}
              </span>
            )}
          </div>
          <div style={{ fontWeight: 600, marginBottom: 2 }}>{cand.business_name}</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{cand.business_address}</div>
          {/* Fix 5: cluster note */}
          {hasCluster && (
            <div style={{ marginTop: 4, fontSize: 11, color: "var(--text-muted)", fontStyle: "italic" }}>
              Part of a near-duplicate cluster of {cand.cluster_size} records within {cand.source} — not automatically the same entity.
            </div>
          )}
        </div>

        {/* Right: Scores */}
        <div style={{ minWidth: 300, flex: "0 0 auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px", fontSize: 12, marginBottom: 8 }}>
            {[
              ["Final Score", cand.final_score],
              ["Name Sim.", cand.name_similarity],
              ["Addr. Sim.", cand.address_similarity],
              ["Composite", cand.composite_similarity],
              /* Fix 4: component-level address features */
              ["Street Name Sim.", cand.street_name_similarity],
            ].map(([lbl, val]) => (
              <div key={lbl}>
                <div style={{ color: "var(--text-muted)", fontSize: 11 }}>{lbl}</div>
                {val !== null && val !== undefined
                  ? <ScoreBar value={val} />
                  : <span style={{ color: "var(--text-muted)", fontSize: 12 }}>—</span>}
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
            {/* Fix 4: unit_suite_match chip */}
            <span className={`badge ${cand.unit_suite_match === true ? "badge-green" : cand.unit_suite_match === false ? "badge-red" : "badge-grey"}`}>
              {cand.unit_suite_match === true ? "✓" : cand.unit_suite_match === false ? "✗" : "—"} Suite
            </span>
            {/* Fix 1: name frequency split */}
            <span className="badge badge-grey" style={{ fontFamily: "var(--mono)", fontSize: 10.5 }}>
              S1-freq: {cand.name_frequency_s1?.toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Blocking passes + TF-IDF ranks (Fix 3) */}
      <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
        <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Blocking:</span>
        {(cand.blocking_passes || []).map(p => (
          <span key={p} className="badge badge-grey" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>{p}</span>
        ))}
        {cand.tfidf_name_rank && (
          <span className="badge badge-blue" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>TF-IDF-name rank {cand.tfidf_name_rank}</span>
        )}
        {cand.tfidf_address_rank && (
          <span className="badge badge-teal" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>TF-IDF-addr rank {cand.tfidf_address_rank}</span>
        )}
        {/* Fix 3: composite channel rank */}
        {cand.tfidf_composite_rank && (
          <span className="badge badge-purple" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>TF-IDF-composite rank {cand.tfidf_composite_rank}</span>
        )}
      </div>

      {/* Fix 2: Cross-source triangulation structured view */}
      {cand.cross_source_support && typeof cand.cross_source_support === "object" && (
        <div style={{ marginTop: 8, padding: "8px 12px", background: "var(--bg-secondary)", borderRadius: "var(--radius-sm)", fontSize: 12 }}>
          <span style={{ color: "var(--text-muted)", marginRight: 8 }}>Cross-source triangulation (S2↔S3):</span>
          <span style={{ fontFamily: "var(--mono)" }}>
            name {cand.cross_source_support.s2_s3_name_similarity?.toFixed(3)} |
            addr {cand.cross_source_support.s2_s3_address_similarity?.toFixed(3)} |
            numeric <Bool v={cand.cross_source_support.numeric_agreement} />
          </span>
          <span style={{ color: "var(--text-muted)", marginLeft: 8, fontSize: 11 }}>— supporting evidence only, not a decision</span>
        </div>
      )}

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
    Promise.all([api.getEntity(entityId), api.getShowcase()])
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
  const gtSource = entity.ground_truth_source;
  const isUnverified = gtSource === "test_unverified";

  const QUICK = [
    { label: "Singleton",       id: showcase?.singleton },
    { label: "Single Match",    id: showcase?.single_match },
    { label: "Multi-Match",     id: showcase?.multi_match },
    { label: "Chain Collision", id: showcase?.chain_collision },
    { label: "Noisy Match",     id: showcase?.noisy_match },
    { label: "🇫🇷 France",       id: showcase?.france },
    { label: "Addr Variation",  id: showcase?.address_variation },
  ];

  return (
    <div>
      {/* Back + quick nav */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20, flexWrap: "wrap" }}>
        <button className="btn btn-ghost btn-sm" onClick={() => navigate("/entities")}>← Back</button>
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Jump to:</span>
        {QUICK.map(q => q.id && (
          <button key={q.label} className="btn btn-ghost btn-sm"
            onClick={() => navigate(`/entities/${q.id}`)}
            style={{ opacity: q.id === entityId ? 0.5 : 1 }}>
            {q.label}
          </button>
        ))}
      </div>

      {/* Demo banner */}
      <div className="demo-banner">
        ⚠️ DEMO DATA — NOT REAL VALIDATION &nbsp;|&nbsp;
        Replace data files with real pipeline exports for live metrics.
        Training-derived examples show verified accuracy.
        Test-derived examples (France) show pipeline predictions on genuinely unseen data.
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
            "S1 Entity",
            "Normalization",
            `Blocking → ${candidates.length} candidates`,
            "Evidence Features",
            "LightGBM Score",
            `Threshold (${threshold})`,
            `${accepted.length} ${isUnverified ? "Prediction(s)" : "Match(es)"}`,
          ].map((s, i, arr) => (
            <div key={s} style={{ display: "flex", alignItems: "center" }}>
              <div className="pipeline-step active">{s}</div>
              {i < arr.length - 1 && <span className="pipeline-arrow">→</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 20 }}>
        <DecisionPanel decision={decision} gtSource={gtSource} />
        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Why This Decision?</div>
          <div className="explanation-box">{explanation}</div>
          <div style={{ marginTop: 12, fontSize: 11.5, color: "var(--text-muted)" }}>
            Explanations are deterministic and feature-based — no LLM used.
          </div>
        </div>
      </div>

      {/* Candidate rows */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-header">
          <div className="section-title">
            Candidates ({candidates.length}) &nbsp;
            <span className={`badge ${isUnverified ? "badge-orange" : "badge-green"}`}>
              {accepted.length} {isUnverified ? "predicted" : "accepted"}
            </span>
            &nbsp;
            <span className="badge badge-red">{rejected.length} rejected</span>
          </div>
          <div className="section-sub">
            Sorted by final score. Every predicted/accepted ID also appears in candidate_pairs.tsv.
          </div>
        </div>
        {candidates.length === 0 ? (
          <div className="empty-state">No candidates were retrieved by blocking.</div>
        ) : (
          [...accepted, ...rejected]
            .sort((a, b) => b.final_score - a.final_score)
            .map(c => <CandidateRow key={c.entity_id} cand={c} threshold={threshold} gtSource={gtSource} />)
        )}
      </div>

      {/* Feature breakdown table */}
      {candidates.length > 0 && (
        <div className="card">
          <div className="section-title" style={{ marginBottom: 16 }}>Full Evidence Breakdown</div>
          <div style={{ overflowX: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Final</th>
                  <th>Name Sim</th>
                  <th>Addr Sim</th>
                  <th>Street Sim</th>
                  <th>Suite</th>
                  <th>Composite</th>
                  <th>S1 Freq</th>
                  <th>S2 Freq</th>
                  <th>S3 Freq</th>
                  <th>PIN</th>
                  <th>House#</th>
                  <th>Cluster</th>
                  <th>TF-IDF ranks</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {[...accepted, ...rejected]
                  .sort((a, b) => b.final_score - a.final_score)
                  .map(c => (
                    <tr key={c.entity_id}>
                      <td>
                        <span className="mono" style={{ fontSize: 11, color: "var(--accent)" }}>{c.entity_id}</span>
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{c.source}</div>
                      </td>
                      <td><span className="mono" style={{ color: c.final_score >= threshold ? (isUnverified ? "var(--orange)" : "var(--green)") : "var(--red)", fontSize: 12 }}>{c.final_score.toFixed(3)}</span></td>
                      <td className="mono" style={{ fontSize: 12 }}>{c.name_similarity.toFixed(3)}</td>
                      <td className="mono" style={{ fontSize: 12 }}>{c.address_similarity.toFixed(3)}</td>
                      {/* Fix 4: street_name_similarity */}
                      <td className="mono" style={{ fontSize: 12 }}>{c.street_name_similarity != null ? c.street_name_similarity.toFixed(3) : "—"}</td>
                      {/* Fix 4: unit_suite_match */}
                      <td><Bool v={c.unit_suite_match} /></td>
                      <td className="mono" style={{ fontSize: 12 }}>{c.composite_similarity.toFixed(3)}</td>
                      {/* Fix 1: split name frequencies */}
                      <td className="mono" style={{ fontSize: 11 }}>{c.name_frequency_s1?.toFixed(4)}</td>
                      <td className="mono" style={{ fontSize: 11 }}>{c.name_frequency_s2 != null ? c.name_frequency_s2.toFixed(4) : "—"}</td>
                      <td className="mono" style={{ fontSize: 11 }}>{c.name_frequency_s3 != null ? c.name_frequency_s3.toFixed(4) : "—"}</td>
                      <td><Bool v={c.pin_postal_match} /></td>
                      <td><Bool v={c.house_number_match} /></td>
                      {/* Fix 5: cluster */}
                      <td style={{ fontSize: 11, color: c.cluster_size > 1 ? "var(--yellow)" : "var(--text-muted)" }}>
                        {c.cluster_size > 1 ? `${c.cluster_id} (×${c.cluster_size})` : "—"}
                      </td>
                      {/* Fix 3: all 3 TF-IDF ranks */}
                      <td className="mono" style={{ fontSize: 10.5, color: "var(--text-muted)" }}>
                        {[c.tfidf_name_rank && `n:${c.tfidf_name_rank}`, c.tfidf_address_rank && `a:${c.tfidf_address_rank}`, c.tfidf_composite_rank && `c:${c.tfidf_composite_rank}`].filter(Boolean).join(" ") || "—"}
                      </td>
                      <td>
                        {c.selected
                          ? <span className={`badge ${isUnverified ? "badge-orange" : "badge-green"}`}>{isUnverified ? "Predict" : "Accept"}</span>
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
