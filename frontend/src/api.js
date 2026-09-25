const BASE = "http://localhost:8000/api";

async function req(path, params = {}) {
  const url = new URL(BASE + path);
  Object.entries(params).forEach(([k, v]) => v != null && url.searchParams.set(k, v));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

export const api = {
  health: () => req("/health"),
  listEntities: (p = {}) => req("/s1", p),
  getEntity: (id) => req(`/s1/${id}`),
  getMetrics: () => req("/metrics"),
  getExperiments: () => req("/experiments"),
  getShowcase: () => req("/showcase"),
};
