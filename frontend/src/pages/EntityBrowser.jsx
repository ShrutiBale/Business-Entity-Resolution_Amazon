import { useEffect, useState, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";

const CASE_TAGS = ["singleton", "single_match", "multi_match", "chain_collision", "transliteration", "address_variation", "france"];
const COUNTRIES = ["US", "India", "France"];
const MATCH_STATES = [
  { value: "no_match", label: "No Match" },
  { value: "matched", label: "Matched" },
  { value: "multi_match", label: "Multi-Match" },
];

const TAG_BADGE = {
  singleton: "badge-red",
  single_match: "badge-green",
  multi_match: "badge-blue",
  chain_collision: "badge-orange",
  transliteration: "badge-yellow",
  address_variation: "badge-teal",
  france: "badge-purple",
};

function tagLabel(t) {
  return t.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export default function EntityBrowser() {
  const [entities, setEntities] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [caseTag, setCaseTag] = useState("");
  const [country, setCountry] = useState("");
  const [matchState, setMatchState] = useState("");
  const navigate = useNavigate();

  const load = useCallback(() => {
    setLoading(true);
    api.listEntities({
      search: search || null,
      case_tag: caseTag || null,
      country: country || null,
      match_state: matchState || null,
    })
      .then(d => { setEntities(d.entities); setTotal(d.total); setError(null); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [search, caseTag, country, matchState]);

  useEffect(() => { load(); }, [load]);

  return (
    <div>
      <h1 className="page-title">Entity Browser</h1>
      <p className="page-sub">Search and filter the 15 curated demo S1 entities</p>

      {/* Filters */}
      <div className="card" style={{ marginBottom: 24, padding: "16px 20px" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
          <input
            className="input" placeholder="Search by name or ID…"
            value={search} onChange={e => setSearch(e.target.value)}
            style={{ minWidth: 240 }}
          />
          <select className="select" value={caseTag} onChange={e => setCaseTag(e.target.value)}>
            <option value="">All Case Tags</option>
            {CASE_TAGS.map(t => <option key={t} value={t}>{tagLabel(t)}</option>)}
          </select>
          <select className="select" value={country} onChange={e => setCountry(e.target.value)}>
            <option value="">All Countries</option>
            {COUNTRIES.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <select className="select" value={matchState} onChange={e => setMatchState(e.target.value)}>
            <option value="">All Match States</option>
            {MATCH_STATES.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
          {(search || caseTag || country || matchState) && (
            <button className="btn btn-ghost btn-sm" onClick={() => { setSearch(""); setCaseTag(""); setCountry(""); setMatchState(""); }}>
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Quick filters */}
      <div className="chip-bar">
        {[
          { label: "All", tag: "" },
          { label: "Singleton", tag: "singleton" },
          { label: "Single Match", tag: "single_match" },
          { label: "Multi-Match", tag: "multi_match" },
          { label: "Chain Collision", tag: "chain_collision" },
          { label: "🇫🇷 France", tag: "france" },
        ].map(f => (
          <span key={f.tag} className={`chip ${caseTag === f.tag ? "active" : ""}`}
            onClick={() => setCaseTag(f.tag === caseTag ? "" : f.tag)}>
            {f.label}
          </span>
        ))}
      </div>

      {error && <div className="error-box" style={{ marginBottom: 16 }}>Error: {error}</div>}

      {loading ? (
        <div className="loading-state"><div className="spinner" /><span>Loading entities…</span></div>
      ) : entities.length === 0 ? (
        <div className="empty-state">No entities match your filters.</div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{total} entity(ies)</span>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Entity ID</th>
                <th>Business Name</th>
                <th>Address</th>
                <th>Country</th>
                <th>Case Tag</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {entities.map(e => (
                <tr key={e.entity_id} style={{ cursor: "pointer" }}
                  onClick={() => navigate(`/entities/${e.entity_id}`)}>
                  <td><span className="mono" style={{ color: "var(--accent)", fontSize: 12.5 }}>{e.entity_id}</span></td>
                  <td style={{ fontWeight: 500 }}>{e.business_name}</td>
                  <td style={{ color: "var(--text-secondary)", fontSize: 12.5, maxWidth: 200 }}>{e.business_address}</td>
                  <td>
                    <span className={`badge ${e.country === "France" ? "badge-purple" : e.country === "India" ? "badge-teal" : "badge-blue"}`}>
                      {e.country}
                    </span>
                  </td>
                  <td><span className={`badge ${TAG_BADGE[e.case_tag] || "badge-grey"}`}>{tagLabel(e.case_tag)}</span></td>
                  <td style={{ fontSize: 12, color: "var(--text-muted)", maxWidth: 260 }}>{e.case_description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
