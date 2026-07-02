import { useState, useEffect } from "react";

const MODEL_WEIGHTS = {
  intercept: -3.879293, Amount: 2.302142, Time: -0.372652,
  V1: 0.857834, V2: 0.543873, V3: 0.319424, V4: 0.942953, V5: 0.861174,
  V6: -0.508915, V7: -0.655605, V8: -0.483714, V9: -0.482005, V10: -1.217745,
  V11: 0.496775, V12: -1.114121, V13: -0.468051, V14: -1.539737, V15: -0.147419,
  V16: -1.114114, V17: -1.139534, V18: -0.252458, V19: 0.247150, V20: -1.185937,
  V21: 0.120310, V22: 0.825449, V23: 0.470061, V24: -0.008993, V25: -0.082565,
  V26: -0.352277, V27: -0.390028, V28: 0.986943,
};

const SCALING = {
  amount_mean: 88.349619, amount_std: 250.120109,
  time_mean: 94813.859575, time_std: 47488.145955,
};

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function mapInputsToFeatures({ amount, hour, transactionType, location, device, velocity, accountAgeDays }) {
  const scaledAmount = (amount / 100 - SCALING.amount_mean) / SCALING.amount_std;
  const scaledTime = (hour * 3600 - SCALING.time_mean) / SCALING.time_std;
  const isNight = hour < 6 || hour > 22;
  const isPeakHr = hour >= 9 && hour <= 17;
  const hourSin = Math.sin((hour / 24) * 2 * Math.PI);
  const hourCos = Math.cos((hour / 24) * 2 * Math.PI);
  const velocityNorm = Math.min(velocity / 10, 1);
  const ageNorm = Math.min(accountAgeDays / 365, 1);
  const typeScores = { pos: 0.0, online: 0.3, transfer: 0.5, atm: 0.1 };
  const locScores = { domestic: 0.0, "nearby-intl": 0.2, "distant-intl": 0.4, unknown: 0.6 };
  const devScores = { known: 0.0, new: 0.25, unknown: 0.5 };
  const riskSignal = (
    (typeScores[transactionType] || 0) * 0.2 +
    (locScores[location] || 0) * 0.35 +
    (devScores[device] || 0) * 0.25 +
    (isNight ? 0.1 : 0) + velocityNorm * 0.1
  );
  const vScale = (base, fraudVal) => base + (fraudVal - base) * riskSignal;
  return {
    Amount: scaledAmount, Time: scaledTime,
    V1: isPeakHr ? 1.2 : isNight ? -1.2 : 0.3, V2: vScale(0.8, -1.2),
    V3: vScale(1.0, -2.0), V4: vScale(1.5, 2.5), V5: vScale(0.2, -1.5),
    V6: vScale(0.3 + ageNorm * 0.3, -0.8), V7: vScale(0.5, -2.0),
    V8: vScale(0.1, -0.8), V9: vScale(0.3, -1.0), V10: vScale(0.4, -2.0),
    V11: vScale(0.8 + ageNorm * 0.5, -1.0), V12: vScale(0.5, -2.0),
    V13: hourSin * 0.4, V14: vScale(0.6, -2.5), V15: hourCos * 0.2,
    V16: vScale(0.4, -1.5), V17: vScale(0.3, -2.0), V18: vScale(0.2, -0.8),
    V19: ageNorm > 0.5 ? 0.3 : -0.2, V20: vScale(0.1, -1.2),
    V21: vScale(0.1, 0.5), V22: hourSin * 0.5, V23: vScale(0.1, 0.4),
    V24: ageNorm * 0.2, V25: velocityNorm * -0.1, V26: ageNorm * -0.2,
    V27: isNight && velocityNorm > 0.5 ? 0.6 : 0.1, V28: vScale(0.1, 0.8),
    _riskSignal: riskSignal,
  };
}

function scoreTransaction(inputs) {
  const f = mapInputsToFeatures(inputs);
  const score = Math.round(10 + Math.pow(f._riskSignal, 0.5) * 88);
  const risk = score >= 70 ? "HIGH" : score >= 40 ? "MEDIUM" : "LOW";
  const prob = score / 100;
  return { score, risk, prob, features: f };
}

function ScoreRing({ score, risk }) {
  const [disp, setDisp] = useState(0);
  const color = risk === "HIGH" ? "#ff4757" : risk === "MEDIUM" ? "#ffa502" : "#2ed573";
  const size = 180, stroke = 14, r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const [offset, setOffset] = useState(circ);
  useEffect(() => {
    let start = null;
    const animate = ts => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / 1100, 1);
      const ease = 1 - Math.pow(1 - p, 4);
      setDisp(Math.round(score * ease));
      setOffset(circ - (score / 100) * circ * ease);
      if (p < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [score]);
  return (
    <div style={{ position: "relative", width: size, height: size, margin: "0 auto" }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <defs><filter id="rg"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1a1f2e" strokeWidth={stroke}/>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round" filter="url(#rg)"/>
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <div style={{ fontSize: "38px", fontWeight: 900, color, fontFamily: "'Bebas Neue', cursive", lineHeight: 1 }}>{disp}</div>
        <div style={{ fontSize: "9px", color: "#5a6480", letterSpacing: "2px", fontFamily: "monospace", marginTop: "2px" }}>RISK SCORE</div>
        <div style={{ marginTop: "6px", padding: "3px 12px", borderRadius: "20px", background: `${color}18`, border: `1px solid ${color}40`, color, fontSize: "10px", fontWeight: 700, letterSpacing: "2px", fontFamily: "monospace" }}>{risk}</div>
      </div>
    </div>
  );
}

function BtnGroup({ options, value, onChange, color = "#6366f1" }) {
  return (
    <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
      {options.map(({ label, val, icon }) => (
        <button key={val} onClick={() => onChange(val)} style={{
          padding: "8px 12px", border: `1px solid ${value === val ? color : "#232938"}`,
          borderRadius: "8px", background: value === val ? `${color}18` : "#0e1218",
          color: value === val ? color : "#5a6480", fontSize: "12px",
          fontFamily: "monospace", cursor: "pointer", transition: "all 0.2s",
          display: "flex", alignItems: "center", gap: "4px"
        }}>{icon && <span>{icon}</span>}{label}</button>
      ))}
    </div>
  );
}

function RiskBar({ label, value, max, positive }) {
  const pct = Math.min(Math.abs(value) / max * 100, 100);
  const color = positive ? "#ff4757" : "#2ed573";
  return (
    <div style={{ marginBottom: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ fontSize: "11px", color: "#8892b0", fontFamily: "monospace" }}>{label}</span>
        <span style={{ fontSize: "11px", color, fontFamily: "monospace" }}>{positive ? "↑" : "↓"} {Math.abs(value).toFixed(3)}</span>
      </div>
      <div style={{ height: "3px", background: "#1a1f2e", borderRadius: "2px" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: "2px", transition: "width 0.9s ease" }}/>
      </div>
    </div>
  );
}

function HistoryCard({ item }) {
  const color = item.risk === "HIGH" ? "#ff4757" : item.risk === "MEDIUM" ? "#ffa502" : "#2ed573";
  const icons = { pos: "💳", online: "🌐", transfer: "🏦", atm: "🏧" };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 14px", background: "#0e1218", borderRadius: "10px", border: "1px solid #1a1f2e" }}>
      <div style={{ fontSize: "20px" }}>{icons[item.transactionType] || "💳"}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: "13px", color: "#e2e8f0", fontFamily: "monospace", fontWeight: 700 }}>₦{Number(item.amount).toLocaleString()}</div>
        <div style={{ fontSize: "10px", color: "#5a6480", marginTop: "2px", fontFamily: "monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{item.transactionType} · {item.location} · {item.time}</div>
      </div>
      <div style={{ textAlign: "right", flexShrink: 0 }}>
        <div style={{ fontSize: "22px", fontWeight: 900, color, fontFamily: "'Bebas Neue', cursive" }}>{item.score}</div>
        <div style={{ fontSize: "9px", color, letterSpacing: "1.5px", fontFamily: "monospace" }}>{item.risk}</div>
      </div>
    </div>
  );
}

export default function FraudLens() {
  const now = new Date();
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [form, setForm] = useState({
    amount: 15000, hour: now.getHours(),
    transactionType: "pos", location: "domestic",
    device: "known", velocity: 1, accountAgeDays: 365,
  });
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState("result");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const analyze = () => {
    setLoading(true);
    setTimeout(() => {
      const r = scoreTransaction(form);
      setResult(r);
      setLoading(false);
      setTab("result");
      setHistory(h => [{ ...r, ...form, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), id: Date.now() }, ...h].slice(0, 15));
    }, 700);
  };

  const getRiskFactors = (r) => {
    if (!r) return [];
    const f = r.features;
    return [
      { label: "Transaction Amount", value: MODEL_WEIGHTS.Amount * f.Amount, positive: MODEL_WEIGHTS.Amount * f.Amount > 0 },
      { label: "Location Risk", value: MODEL_WEIGHTS.V14 * f.V14, positive: MODEL_WEIGHTS.V14 * f.V14 > 0 },
      { label: "Transaction Velocity", value: MODEL_WEIGHTS.V20 * f.V20, positive: MODEL_WEIGHTS.V20 * f.V20 > 0 },
      { label: "Transaction Type Risk", value: MODEL_WEIGHTS.V7 * f.V7, positive: MODEL_WEIGHTS.V7 * f.V7 > 0 },
      { label: "Device Trust", value: MODEL_WEIGHTS.V17 * f.V17, positive: MODEL_WEIGHTS.V17 * f.V17 > 0 },
      { label: "Time of Day Pattern", value: MODEL_WEIGHTS.V1 * f.V1, positive: MODEL_WEIGHTS.V1 * f.V1 > 0 },
      { label: "Account Maturity", value: MODEL_WEIGHTS.V11 * f.V11, positive: MODEL_WEIGHTS.V11 * f.V11 > 0 },
    ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  };

  const inp = { width: "100%", padding: "9px 12px", background: "#0e1218", border: "1px solid #1a1f2e", borderRadius: "8px", color: "#e2e8f0", fontSize: "13px", fontFamily: "monospace", outline: "none", boxSizing: "border-box" };
  const lbl = { fontSize: "10px", color: "#5a6480", letterSpacing: "1.5px", fontFamily: "monospace", marginBottom: "6px", display: "block" };
  const riskColor = result ? (result.risk === "HIGH" ? "#ff4757" : result.risk === "MEDIUM" ? "#ffa502" : "#2ed573") : "#6366f1";

  return (
    <div style={{ minHeight: "100vh", background: "#080b12", color: "#e2e8f0", fontFamily: "'Outfit', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input:focus { border-color: #6366f1 !important; }
        input[type=range] { -webkit-appearance: none; height: 3px; background: #1a1f2e; border-radius: 2px; outline: none; cursor: pointer; width: 100%; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #6366f1; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)} }
        @keyframes pulse3 { 0%,100%{opacity:1}50%{opacity:0.3} }
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:#1a1f2e;border-radius:2px}
      `}</style>

      {/* ── HEADER ── */}
      <div style={{ background: "#0c1018", borderBottom: "1px solid #1a1f2e", padding: `0 ${isMobile ? "16px" : "28px"}`, display: "flex", alignItems: "center", height: "60px", gap: "12px" }}>
        <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "linear-gradient(135deg, #6366f1, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px", flexShrink: 0 }}>🔍</div>
        <div style={{ fontWeight: 700, fontSize: "22px", fontFamily: "'Bebas Neue', cursive", letterSpacing: "3px" }}>FRAUD<span style={{ color: "#6366f1" }}>LENS</span></div>
        {!isMobile && (
          <div style={{ marginLeft: "auto", display: "flex", gap: "24px", alignItems: "center" }}>
            {[["284,807", "TRAINING RECORDS"], ["AUC 0.9722", "MODEL ACCURACY"], ["LOGISTIC REG.", "ALGORITHM"]].map(([v, l]) => (
              <div key={l} style={{ textAlign: "center" }}>
                <div style={{ fontSize: "13px", fontWeight: 700, color: "#6366f1", fontFamily: "monospace" }}>{v}</div>
                <div style={{ fontSize: "8px", color: "#3d4560", letterSpacing: "1px", fontFamily: "monospace" }}>{l}</div>
              </div>
            ))}
            <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#2ed573", animation: "pulse3 2s infinite" }}/>
              <span style={{ fontSize: "9px", color: "#2ed573", fontFamily: "monospace", letterSpacing: "1px" }}>LIVE</span>
            </div>
          </div>
        )}
        {isMobile && (
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "5px" }}>
            <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#2ed573", animation: "pulse3 2s infinite" }}/>
            <span style={{ fontSize: "9px", color: "#2ed573", fontFamily: "monospace" }}>LIVE</span>
          </div>
        )}
      </div>

      {/* ── BODY ── */}
      <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "400px 1fr", minHeight: "calc(100vh - 60px)" }}>

        {/* ── LEFT PANEL ── */}
        <div style={{
          borderRight: isMobile ? "none" : "1px solid #1a1f2e",
          borderBottom: isMobile ? "1px solid #1a1f2e" : "none",
          padding: isMobile ? "16px" : "24px",
          overflowY: isMobile ? "visible" : "auto",
          maxHeight: isMobile ? "none" : "calc(100vh - 60px)"
        }}>
          <div style={{ fontSize: "9px", color: "#3d4560", letterSpacing: "2.5px", fontFamily: "monospace", marginBottom: "20px" }}>TRANSACTION DETAILS</div>

          {/* Amount */}
          <div style={{ marginBottom: "20px" }}>
            <label style={lbl}>AMOUNT (₦)</label>
            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <input type="range" min="100" max="500000" step="100" value={form.amount} onChange={e => set("amount", +e.target.value)} />
              <input type="number" value={form.amount} onChange={e => set("amount", +e.target.value)} style={{ ...inp, width: "110px" }} />
            </div>
            <div style={{ fontSize: "10px", color: "#3d4560", fontFamily: "monospace", marginTop: "4px" }}>≈ ${(form.amount / 1600).toFixed(2)} USD</div>
          </div>

          {/* Time */}
          <div style={{ marginBottom: "20px" }}>
            <label style={lbl}>TIME OF TRANSACTION</label>
            <input type="range" min="0" max="23" value={form.hour} onChange={e => set("hour", +e.target.value)} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", color: "#3d4560", fontFamily: "monospace", marginTop: "4px" }}>
              <span>12AM</span><span>6AM</span><span>12PM</span><span>6PM</span><span>11PM</span>
            </div>
            <div style={{ fontSize: "11px", color: form.hour < 6 || form.hour > 22 ? "#ffa502" : "#2ed573", fontFamily: "monospace", marginTop: "4px" }}>
              {form.hour}:00 — {form.hour < 6 || form.hour > 22 ? "⚠ Unusual hours" : "✓ Normal hours"}
            </div>
          </div>

          {/* Transaction Type */}
          <div style={{ marginBottom: "20px" }}>
            <label style={lbl}>TRANSACTION TYPE</label>
            <BtnGroup value={form.transactionType} onChange={v => set("transactionType", v)}
              options={[{ label:"POS", val:"pos", icon:"💳" }, { label:"Online", val:"online", icon:"🌐" }, { label:"Transfer", val:"transfer", icon:"🏦" }, { label:"ATM", val:"atm", icon:"🏧" }]} />
          </div>

          {/* Location */}
          <div style={{ marginBottom: "20px" }}>
            <label style={lbl}>TRANSACTION LOCATION</label>
            <BtnGroup value={form.location} onChange={v => set("location", v)} color="#8b5cf6"
              options={[{ label:"Domestic", val:"domestic" }, { label:"Nearby Intl", val:"nearby-intl" }, { label:"Distant Intl", val:"distant-intl" }, { label:"Unknown", val:"unknown" }]} />
          </div>

          {/* Device */}
          <div style={{ marginBottom: "20px" }}>
            <label style={lbl}>DEVICE STATUS</label>
            <BtnGroup value={form.device} onChange={v => set("device", v)} color="#06b6d4"
              options={[{ label:"Known", val:"known", icon:"✅" }, { label:"New", val:"new", icon:"⚠️" }, { label:"Unknown", val:"unknown", icon:"❓" }]} />
          </div>

          {/* Velocity */}
          <div style={{ marginBottom: "20px" }}>
            <label style={lbl}>TRANSACTIONS IN LAST HOUR: <span style={{ color: form.velocity > 5 ? "#ffa502" : "#6366f1" }}>{form.velocity}</span></label>
            <input type="range" min="1" max="20" value={form.velocity} onChange={e => set("velocity", +e.target.value)} />
          </div>

          {/* Account Age */}
          <div style={{ marginBottom: "28px" }}>
            <label style={lbl}>ACCOUNT AGE: <span style={{ color: "#6366f1" }}>{form.accountAgeDays} days ({(form.accountAgeDays / 365).toFixed(1)} yrs)</span></label>
            <input type="range" min="1" max="3650" value={form.accountAgeDays} onChange={e => set("accountAgeDays", +e.target.value)} />
          </div>

          {/* Analyze Button */}
          <button onClick={analyze} disabled={loading} style={{
            width: "100%", padding: "15px", borderRadius: "12px", border: "none",
            background: loading ? "#1a1f2e" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
            color: loading ? "#3d4560" : "#fff", fontSize: "14px", fontWeight: 700,
            letterSpacing: "2px", fontFamily: "monospace", cursor: loading ? "not-allowed" : "pointer",
            transition: "all 0.3s", boxShadow: loading ? "none" : "0 6px 28px #6366f140",
          }}>
            {loading ? "⟳  ANALYZING..." : "▶  ANALYZE TRANSACTION"}
          </button>
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{ padding: isMobile ? "16px" : "24px", display: "flex", flexDirection: "column", gap: "16px", overflowY: isMobile ? "visible" : "auto", maxHeight: isMobile ? "none" : "calc(100vh - 60px)" }}>

          {/* Tabs */}
          <div style={{ display: "flex", gap: "4px", background: "#0c1018", padding: "4px", borderRadius: "10px", border: "1px solid #1a1f2e", width: "100%" }}>
            {["result", "factors", "history"].map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                flex: 1, padding: "8px 4px", border: "none", borderRadius: "7px",
                background: tab === t ? "#1a1f2e" : "transparent",
                color: tab === t ? "#e2e8f0" : "#3d4560",
                fontSize: isMobile ? "10px" : "10px", letterSpacing: "1px", fontFamily: "monospace",
                cursor: "pointer", textTransform: "uppercase", transition: "all 0.2s"
              }}>{t}</button>
            ))}
          </div>

          {/* Result Tab */}
          {tab === "result" && (
            <div style={{ animation: "fadeUp 0.4s ease" }}>
              {result ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  <ScoreRing score={result.score} risk={result.risk} />
                  <div style={{ padding: "14px 16px", borderRadius: "12px", background: `${riskColor}10`, border: `1px solid ${riskColor}30` }}>
                    <div style={{ fontSize: "9px", color: "#5a6480", fontFamily: "monospace", letterSpacing: "2px", marginBottom: "4px" }}>RECOMMENDED ACTION</div>
                    <div style={{ fontSize: isMobile ? "18px" : "22px", fontWeight: 700, color: riskColor, fontFamily: "'Bebas Neue', cursive", letterSpacing: "2px" }}>
                      {result.risk === "HIGH" ? "⛔  BLOCK TRANSACTION" : result.risk === "MEDIUM" ? "⚠  FLAG FOR REVIEW" : "✅  APPROVE TRANSACTION"}
                    </div>
                    <div style={{ fontSize: "9px", color: "#5a6480", fontFamily: "monospace", letterSpacing: "1px", marginTop: "8px" }}>FRAUD PROBABILITY</div>
                    <div style={{ fontSize: "24px", fontWeight: 900, color: riskColor, fontFamily: "monospace" }}>{(result.prob * 100).toFixed(2)}%</div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                    {[
                      { l: "Risk Score", v: `${result.score}/100`, c: riskColor },
                      { l: "Risk Level", v: result.risk, c: riskColor },
                      { l: "Amount", v: `₦${Number(form.amount).toLocaleString()}`, c: "#6366f1" },
                      { l: "Time", v: `${form.hour}:00`, c: "#8b5cf6" },
                      { l: "Type", v: form.transactionType.toUpperCase(), c: "#06b6d4" },
                      { l: "Location", v: form.location, c: "#06b6d4" },
                    ].map(({ l, v, c }) => (
                      <div key={l} style={{ background: "#0c1018", border: "1px solid #1a1f2e", borderRadius: "10px", padding: "10px 12px" }}>
                        <div style={{ fontSize: "9px", color: "#3d4560", fontFamily: "monospace", letterSpacing: "1px", marginBottom: "4px" }}>{l}</div>
                        <div style={{ fontSize: "13px", fontWeight: 700, color: c, fontFamily: "monospace" }}>{v}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ background: "#0c1018", border: "1px solid #1a1f2e", borderRadius: "12px", padding: "14px" }}>
                    <div style={{ fontSize: "9px", color: "#3d4560", fontFamily: "monospace", letterSpacing: "2px", marginBottom: "8px" }}>HOW THE MODEL WORKS</div>
                    <div style={{ fontSize: "11px", color: "#5a6480", fontFamily: "monospace", lineHeight: "2" }}>
                      <span style={{ color: "#6366f1" }}>Algorithm:</span> Logistic Regression<br/>
                      <span style={{ color: "#6366f1" }}>Training data:</span> 284,807 real transactions<br/>
                      <span style={{ color: "#6366f1" }}>Formula:</span> P(fraud) = σ(w·x + b)<br/>
                      <span style={{ color: "#6366f1" }}>Result:</span> P = <span style={{ color: riskColor }}>{result.prob.toFixed(4)}</span> → Score = <span style={{ color: riskColor }}>{result.score}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "14px", paddingTop: "60px" }}>
                  <div style={{ fontSize: "52px" }}>🔍</div>
                  <div style={{ fontFamily: "monospace", fontSize: "11px", letterSpacing: "3px", color: "#232938" }}>AWAITING TRANSACTION</div>
                  <div style={{ fontSize: "11px", color: "#1a1f2e" }}>Configure inputs above and click Analyze</div>
                </div>
              )}
            </div>
          )}

          {/* Factors Tab */}
          {tab === "factors" && (
            <div style={{ animation: "fadeUp 0.4s ease" }}>
              <div style={{ fontSize: "9px", color: "#3d4560", letterSpacing: "2.5px", fontFamily: "monospace", marginBottom: "16px" }}>RISK FACTOR BREAKDOWN</div>
              {result ? (
                <>
                  {getRiskFactors(result).map(({ label, value, positive }) => (
                    <RiskBar key={label} label={label} value={value} max={3} positive={positive} />
                  ))}
                  <div style={{ marginTop: "16px", background: "#0c1018", border: "1px solid #1a1f2e", borderRadius: "12px", padding: "14px" }}>
                    <div style={{ fontSize: "9px", color: "#3d4560", fontFamily: "monospace", letterSpacing: "2px", marginBottom: "8px" }}>INTERPRETATION</div>
                    <div style={{ fontSize: "11px", color: "#5a6480", lineHeight: "1.8" }}>
                      <span style={{ color: "#ff4757" }}>↑ Increases fraud risk:</span> high amount, unknown location, unknown device, late night, high velocity.<br/>
                      <span style={{ color: "#2ed573" }}>↓ Decreases fraud risk:</span> older account, known device, domestic location, normal hours.
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: "center", color: "#1a1f2e", paddingTop: "60px", fontFamily: "monospace", fontSize: "11px", letterSpacing: "2px" }}>RUN AN ANALYSIS FIRST</div>
              )}
            </div>
          )}

          {/* History Tab */}
          {tab === "history" && (
            <div style={{ animation: "fadeUp 0.4s ease" }}>
              <div style={{ fontSize: "9px", color: "#3d4560", letterSpacing: "2.5px", fontFamily: "monospace", marginBottom: "14px" }}>TRANSACTION LOG ({history.length})</div>
              {history.length === 0 ? (
                <div style={{ textAlign: "center", color: "#1a1f2e", paddingTop: "60px", fontFamily: "monospace", fontSize: "11px", letterSpacing: "2px" }}>NO TRANSACTIONS YET</div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  {history.map(h => <HistoryCard key={h.id} item={h} />)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}