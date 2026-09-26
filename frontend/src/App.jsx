import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Home from "./pages/Home";
import EntityBrowser from "./pages/EntityBrowser";
import EntityDetail from "./pages/EntityDetail";
import MetricsDashboard from "./pages/MetricsDashboard";
import Experiments from "./pages/Experiments";
import Methodology from "./pages/Methodology";
import Downloads from "./pages/Downloads";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="brand-dot" />
        <span>EntityResolve</span>
        <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>/ Amazon ML Challenge 2026</span>
      </div>
      <div className="navbar-links">
        <NavLink to="/" end className={({ isActive }) => isActive ? "active" : ""}>Home</NavLink>
        <NavLink to="/entities" className={({ isActive }) => isActive ? "active" : ""}>Entity Browser</NavLink>
        <NavLink to="/metrics" className={({ isActive }) => isActive ? "active" : ""}>Metrics</NavLink>
        <NavLink to="/experiments" className={({ isActive }) => isActive ? "active" : ""}>Experiments</NavLink>
        <NavLink to="/methodology" className={({ isActive }) => isActive ? "active" : ""}>Methodology</NavLink>
        <NavLink to="/downloads" className={({ isActive }) => isActive ? "active" : ""}>
          ⬇ Downloads
        </NavLink>
      </div>
      <span className="navbar-badge">REAL DATA</span>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <main className="page-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/entities" element={<EntityBrowser />} />
            <Route path="/entities/:entityId" element={<EntityDetail />} />
            <Route path="/metrics" element={<MetricsDashboard />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/downloads" element={<Downloads />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
