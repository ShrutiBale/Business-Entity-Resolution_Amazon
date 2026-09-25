const SECTIONS = [
  {
    icon: "🧩", title: "Problem",
    content: `Given business records from 3 independent sources (S1, S2, S3) with noisy, partial, inconsistent fields and no shared identifiers, determine which S2/S3 records refer to the same real-world business as each S1 entity. S1 is the deduplicated reference. An S1 entity may match zero, one, or many S2/S3 records.`
  },
  {
    icon: "🔎", title: "Safe Normalization",
    content: `Multiple non-destructive views are created per field — raw, lowercase, accent-folded, legal-suffix-normalized, phonetic, sorted-token, and compact. Raw equality is kept as explicit evidence. Legal suffixes handled: Corp↔Corporation, Pvt↔Private, Ltd↔Limited, SARL, SAS (French). Address abbreviations: Rd↔Road, St↔Street, Blvd↔Boulevard, Av↔Avenue. Country is always treated as open-set string — never hard-coded to {US, India}.`
  },
  {
    icon: "📡", title: "Multi-Pass Blocking (Candidate Generation)",
    content: `Seven independent blocking passes are unioned:
• Pass A — Exact structured keys (name + postal, name + house number)
• Pass B — Name token index (down-weighted for common tokens)
• Pass C — Address token index (street, city, postal tokens)
• Pass D — Phonetic key (Soundex/Metaphone variants)
• Pass E — Sorted-token name key (handles word-order transposition)
• Pass F — Cross-field block (distinctive name token + city token)
• Pass G — Rare-token (high-IDF tokens as blocking evidence)

The goal is maximum recall — precision is handled downstream. Candidate provenance is stored: which passes fired, blocking_pass_count.`
  },
  {
    icon: "📐", title: "Character TF-IDF Retrieval",
    content: `3–5 gram character TF-IDF retrieval is a core component, not optional. Three channels: normalized name, normalized address, and composite. For each S1 entity, top-K candidates (K ∈ {10, 20, 50, 100}) are retrieved from S2 and S3 separately, then unioned. K is chosen by measuring blocking recall vs candidate volume — the smallest K providing sufficient recall is selected. TF-IDF vocabulary is fit fold-locally during cross-validation (never on validation S1 rows).`
  },
  {
    icon: "⚡", title: "Pair Features",
    content: `Evidence features computed per (S1, candidate) pair:
Name: raw/normalized exact match, Levenshtein, Jaro-Winkler, token Jaccard, char TF-IDF cosine, phonetic match, sorted-token match, shared-token IDF, name frequency
Address: full-string similarity, token Jaccard, char TF-IDF cosine, house-number match, postal/PIN match, street-name similarity, city overlap, numeric evidence
Retrieval: rank per channel, normalized retrieval score, blocking passes fired, blocking_pass_count
Cross-source: S2↔S3 name/address similarity for same S1 (triangulation support)`
  },
  {
    icon: "🤖", title: "LightGBM Matching",
    content: `Primary model: LightGBM binary classifier. Trained on all ground-truth positives (including forced-in missed-by-blocker pairs) plus hard negatives (same-name/different-address, same-name/different-postal, cross-country near-matches) and a sampled set of easy negatives. Class imbalance is handled via weighting + negative sampling. LightGBM naturally handles missing values and feature interactions — no neural model dependency required.`
  },
  {
    icon: "⚖️", title: "Global Threshold Decision",
    content: `A single global pairwise threshold T is tuned directly against the competition metric — entity-level macro F₀.₅ — not pair-level F1. Any candidate with score ≥ T is accepted as a match. Multiple candidates can be accepted for the same S1 (multi-match is first-class). Threshold is selected by evaluating mean entity-level F₀.₅ across CV folds. A stable threshold across folds is preferred over one that overfits a single split.`
  },
  {
    icon: "📊", title: "S1-Grouped Validation",
    content: `Cross-validation is grouped by S1 entity — all matches belonging to an S1 stay in the same fold, preventing S1 leakage. TF-IDF vocabularies, name frequencies, and learned thresholds are fit fold-locally. The official entity-level macro F₀.₅ is computed per fold (never pooled pair-level fbeta_score). A separate untouched S1 holdout is held out before any tuning and evaluated only once after the pipeline is frozen.`
  },
  {
    icon: "📏", title: "Entity-Level Macro F₀.₅",
    content: `The competition metric. For each S1: F₀.₅(predicted_set, true_set). Then macro F₀.₅ = mean over all S1 entities. Singletons (true no-match entities) are included — correctly predicting an empty set earns 1.0. Incorrect matches on singletons score 0.0 and are penalized. F₀.₅ weights precision 2× over recall: merging two distinct businesses is more costly than missing a link.`
  },
  {
    icon: "🌍", title: "Open-Set Country & France Handling",
    content: `Country is never hard-coded, one-hot encoded, or used as a hard filter. It is treated as an open-set string label. Country evidence is encoded relationally: same_country, different_country, country_missing. France appears only in test data — no France-specific training exists. The US/India-trained pipeline is applied as-is. France diagnostics (score distribution, candidate counts, match rates) are for analysis only — the France threshold is never tuned on unlabeled France data.`
  },
  {
    icon: "🚫", title: "No External Lookup",
    content: `No external databases, APIs, geocoding services, company registries, or government data sources are used at any stage. All resolution is based solely on the provided training data and the structural/textual features derived from it. This is required by the competition rules and enforced during code review.`
  },
];

export default function Methodology() {
  return (
    <div>
      <h1 className="page-title">Methodology</h1>
      <p className="page-sub">
        A precision-first, two-stage entity resolution pipeline: high-recall multi-pass retrieval followed by LightGBM + global threshold matching.
      </p>

      {/* One-sentence summary */}
      <div className="card" style={{
        marginBottom: 28,
        background: "linear-gradient(135deg, rgba(79,142,247,0.08), rgba(191,90,242,0.06))",
        border: "1px solid rgba(79,142,247,0.3)",
        borderLeft: "3px solid var(--accent)"
      }}>
        <div style={{ fontSize: 13.5, color: "var(--text-secondary)", fontStyle: "italic", lineHeight: 1.8 }}>
          "Build an open-set, high-recall multi-pass retrieval system using safe normalization, structured/numeric evidence and character TF-IDF; let LightGBM combine name, address, rarity and retrieval evidence; make the final decision with a single globally validated threshold under S1-grouped entity-level macro F₀.₅; treat France as genuine unseen-domain inference; and allow advanced components only when controlled held-out experiments prove they improve the real metric enough to justify their complexity."
        </div>
      </div>

      {/* Sections */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {SECTIONS.map((s, i) => (
          <div key={s.title} className="card">
            <div style={{ display: "flex", gap: 16 }}>
              <div style={{ fontSize: 28, flexShrink: 0 }}>{s.icon}</div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 8, color: "var(--text-primary)" }}>
                  <span style={{ color: "var(--text-muted)", fontFamily: "var(--mono)", fontSize: 12, marginRight: 8 }}>
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {s.title}
                </div>
                <div style={{ fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.8, whiteSpace: "pre-line" }}>
                  {s.content}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Non-negotiables */}
      <div className="card" style={{ marginTop: 24, borderLeft: "3px solid var(--yellow)" }}>
        <div className="section-title" style={{ marginBottom: 12 }}>Non-Negotiable Principles</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 24px", fontSize: 13 }}>
          {[
            "Retrieval ≠ matching ≠ decision",
            "Retrieval recall is the score ceiling",
            "Optimize entity-level macro F₀.₅, not pair F1",
            "S1 may have 0, 1, or many valid matches",
            "Never force one-to-one S1↔S2/S3 matching",
            "Never use connected components as prediction",
            "Country is open-set — never hard-code {US, India}",
            "France is unseen domain — not a labeled validation set",
            "No external lookup, API, geocoding, or data augmentation",
            "All models must be MIT/Apache-2.0, ≤8B parameters",
            "candidate_pairs.tsv = exact final inference candidate set",
            "Every test S1 gets exactly one output row",
          ].map(p => (
            <div key={p} style={{ display: "flex", gap: 8, color: "var(--text-secondary)" }}>
              <span style={{ color: "var(--green)", flexShrink: 0 }}>✓</span>
              <span>{p}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
