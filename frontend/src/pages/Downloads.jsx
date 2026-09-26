import { useState } from "react";

export default function Downloads() {
  const [status, setStatus] = useState({});

  const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const files = [
    {
      id: "candidate_pairs",
      label: "candidate_pairs.tsv",
      endpoint: `${API}/api/export/candidate_pairs.tsv`,
      icon: "🗂️",
      desc: "The exact final test candidate set fed into the matching model — one row per test S1 entity, comma-separated S2/S3 candidate IDs. Every predicted match must appear here.",
      columns: ["source1_entity_id", "candidate_entity_ids"],
      rules: [
        "Exactly one row per test S1",
        "Zero-candidate S1s included (empty string)",
        "Only valid S2-/S3- IDs — no S1 IDs in lists",
        "No duplicates within a row",
      ],
    },
    {
      id: "matching_results",
      label: "matching_results.tsv",
      endpoint: `${API}/api/export/matching_results.tsv`,
      icon: "✅",
      desc: "Final test predictions — matched S2/S3 IDs for each test S1 entity above the global threshold. Every ID here also appears in candidate_pairs.tsv.",
      columns: ["source1_entity_id", "matched_entity_ids"],
      rules: [
        "Exactly one row per test S1",
        "Zero-match entities included (empty string)",
        "Only valid IDs from candidate_pairs.tsv",
        "No duplicates",
      ],
    },
  ];

  const handleDownload = async (file) => {
    setStatus(s => ({ ...s, [file.id]: "downloading" }));
    try {
      const resp = await fetch(file.endpoint);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        setStatus(s => ({ ...s, [file.id]: `error: ${err.detail || resp.statusText}` }));
        return;
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.label;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setStatus(s => ({ ...s, [file.id]: "done" }));
      setTimeout(() => setStatus(s => ({ ...s, [file.id]: null })), 3000);
    } catch (e) {
      setStatus(s => ({ ...s, [file.id]: `error: ${e.message}` }));
    }
  };

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 6 }}>Exports & Downloads</h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, lineHeight: 1.7 }}>
          These files are generated from the real supplied challenge dataset and the frozen pipeline outputs.
          They are <strong>not</strong> mock/demo exports — they represent the actual test candidate set
          and final predictions produced by the trained LightGBM model.
        </p>
      </div>

      {/* Pipeline source note */}
      <div className="card" style={{ marginBottom: 24, borderLeft: "3px solid var(--accent)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
          <span style={{ fontSize: 22, marginTop: 2 }}>⚙️</span>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 4 }}>How these files are generated</div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8 }}>
              <ol style={{ paddingLeft: 18, margin: 0 }}>
                <li>Train LightGBM on real training pairs: <code className="mono" style={{ fontSize: 11 }}>python3 ml/train.py</code></li>
                <li>Run frozen model on real test TSVs: <code className="mono" style={{ fontSize: 11 }}>python3 ml/inference.py</code></li>
                <li>Validate outputs: <code className="mono" style={{ fontSize: 11 }}>python3 ml/submission.py</code></li>
                <li>Export to UI: <code className="mono" style={{ fontSize: 11 }}>python3 ml/export_demo_data.py</code></li>
              </ol>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
        {files.map(file => {
          const st = status[file.id];
          const isDownloading = st === "downloading";
          const isDone = st === "done";
          const isError = st && st.startsWith("error");

          return (
            <div key={file.id} className="card" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Header */}
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 28 }}>{file.icon}</span>
                <div>
                  <div style={{ fontWeight: 800, fontSize: 16, fontFamily: "var(--mono)" }}>{file.label}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
                    Columns: {file.columns.join(", ")}
                  </div>
                </div>
              </div>

              {/* Description */}
              <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7, margin: 0 }}>
                {file.desc}
              </p>

              {/* Rules */}
              <div style={{ background: "var(--bg-secondary)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
                <div style={{ fontSize: 11.5, color: "var(--text-muted)", fontWeight: 600, marginBottom: 6, textTransform: "uppercase", letterSpacing: ".06em" }}>
                  Format Rules
                </div>
                <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.8 }}>
                  {file.rules.map(r => <li key={r}>{r}</li>)}
                </ul>
              </div>

              {/* Status */}
              {isError && (
                <div style={{ fontSize: 12, color: "var(--red)", background: "rgba(255,69,58,.08)", borderRadius: "var(--radius-sm)", padding: "8px 12px" }}>
                  ⚠ {st.replace("error: ", "")}
                  <div style={{ marginTop: 4, color: "var(--text-muted)" }}>
                    Run <code className="mono">python3 ml/inference.py</code> to generate this file first.
                  </div>
                </div>
              )}
              {isDone && (
                <div style={{ fontSize: 12, color: "var(--green)", fontWeight: 600 }}>✓ Downloaded successfully</div>
              )}

              {/* Download button */}
              <button
                id={`download-${file.id}`}
                className="btn btn-primary"
                onClick={() => handleDownload(file)}
                disabled={isDownloading}
                style={{ marginTop: "auto", opacity: isDownloading ? 0.7 : 1 }}
              >
                {isDownloading ? (
                  <><span className="spinner" style={{ width: 14, height: 14 }} /> Downloading…</>
                ) : (
                  <>⬇ Download {file.label}</>
                )}
              </button>

              {/* Endpoint */}
              <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--mono)" }}>
                GET {file.endpoint.replace(API, "")}
              </div>
            </div>
          );
        })}
      </div>

      {/* Note */}
      <div style={{
        marginTop: 24, padding: "12px 16px",
        background: "rgba(48,209,88,0.06)", border: "1px solid rgba(48,209,88,0.2)",
        borderRadius: "var(--radius)", fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.7
      }}>
        <strong style={{ color: "var(--text-primary)" }}>Provenance:</strong>{" "}
        These exports are generated from the supplied challenge dataset and the current frozen pipeline outputs.
        Training-derived metrics are validated against real ground truth. Test predictions (including France) are model outputs only — test ground truth is never available to any participant.
      </div>
    </div>
  );
}
