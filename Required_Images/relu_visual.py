"""
Self-contained HTML for the Activation Functions interactive walkthrough.
Covers: Step/Sigmoid/ReLU comparison, ReLU network walkthrough,
vanishing gradient visualization, and ReLU variants (Leaky, ELU, GELU, Swish).
Embed in Streamlit via st.components.v1.html(RELU_VISUAL_HTML, height=RELU_VISUAL_HEIGHT).
"""

RELU_VISUAL_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; overflow-x: hidden; }
  input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: #1e1e2e; outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

const { useState, useEffect, useMemo } = React;

const C = {
  bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
  accent: "#ff6b35", blue: "#4ecdc4", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#38bdf8", pink: "#f472b6",
};

/* ── Math helpers ──────────────────────────────────────── */
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const relu = (z) => Math.max(0, z);
const leakyRelu = (z, a = 0.01) => z > 0 ? z : a * z;
const elu = (z, a = 1) => z > 0 ? z : a * (Math.exp(z) - 1);
const gelu = (z) => {
  const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z * z * z)));
  return z * cdf;
};
const swish = (z) => z * sigmoid(z);
const stepFn = (z) => z >= 0 ? 1 : 0;
const tanhFn = (z) => Math.tanh(z);

/* ── Shared components ─────────────────────────────────── */

function TabBar({ tabs, active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 24, overflowX: "auto" }}>
      {tabs.map((t, i) => (
        <button key={i} onClick={() => onChange(i)} style={{
          padding: "12px 20px", background: "none", border: "none",
          borderBottom: active === i ? "2px solid " + C.accent : "2px solid transparent",
          color: active === i ? C.accent : C.muted, cursor: "pointer",
          fontSize: 12, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
          transition: "all 0.2s", whiteSpace: "nowrap", marginBottom: -2,
        }}>
          {t}
        </button>
      ))}
    </div>
  );
}

function Card({ children, style, highlight }) {
  return (
    <div style={Object.assign({
      background: C.card, borderRadius: 10, padding: "18px 22px",
      border: "1px solid " + (highlight ? C.accent : C.border),
      transition: "border 0.3s",
    }, style || {})}>
      {children}
    </div>
  );
}

function GraphCanvas({ width, height, xRange, yRange, children, style }) {
  var x0 = xRange[0], x1 = xRange[1], y0 = yRange[0], y1 = yRange[1];
  var pad = { l: 40, r: 15, t: 15, b: 30 };
  var gw = width - pad.l - pad.r;
  var gh = height - pad.t - pad.b;

  var toX = function(v) { return pad.l + ((v - x0) / (x1 - x0)) * gw; };
  var toY = function(v) { return pad.t + ((y1 - v) / (y1 - y0)) * gh; };

  var xTicks = [];
  for (var v = Math.ceil(x0); v <= Math.floor(x1); v++) xTicks.push(v);
  var yTicks = [];
  for (var v2 = Math.ceil(y0); v2 <= Math.floor(y1); v2++) yTicks.push(v2);

  return (
    <svg width={width} height={height} viewBox={"0 0 " + width + " " + height} style={Object.assign({ background: "#08080d", borderRadius: 8, border: "1px solid " + C.border }, style || {})}>
      {xTicks.map(function(v) { return <line key={"gx" + v} x1={toX(v)} y1={pad.t} x2={toX(v)} y2={height - pad.b} stroke={v === 0 ? C.dim : "#111122"} strokeWidth={v === 0 ? 1 : 0.5} />; })}
      {yTicks.map(function(v) { return <line key={"gy" + v} x1={pad.l} y1={toY(v)} x2={width - pad.r} y2={toY(v)} stroke={v === 0 ? C.dim : "#111122"} strokeWidth={v === 0 ? 1 : 0.5} />; })}
      {xTicks.filter(function(v) { return v % 1 === 0; }).map(function(v) { return <text key={"lx" + v} x={toX(v)} y={height - 8} fill={C.dim} fontSize={9} textAnchor="middle" fontFamily="monospace">{v}</text>; })}
      {yTicks.filter(function(v) { return v % 1 === 0; }).map(function(v) { return <text key={"ly" + v} x={pad.l - 8} y={toY(v) + 3} fill={C.dim} fontSize={9} textAnchor="end" fontFamily="monospace">{v}</text>; })}
      {children(toX, toY, gw, gh)}
    </svg>
  );
}

function FnLine({ fn, xRange, toX, toY, color, strokeWidth, dashed }) {
  var pts = [];
  var x0 = xRange[0], x1 = xRange[1];
  var s = (x1 - x0) / 300;
  for (var x = x0; x <= x1; x += s) {
    pts.push(toX(x) + "," + toY(fn(x)));
  }
  return <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={strokeWidth || 2.5} strokeLinecap="round" strokeDasharray={dashed ? "6,4" : "none"} />;
}


/* ═══════════════════════════════════════════════════════════
   TAB 1: THE BIG THREE
   ═══════════════════════════════════════════════════════════ */
function TabBigThree() {
  const [z, setZ] = useState(1.2);
  var W = 260, H = 200;
  var fns = [
    { name: "Step", fn: stepFn, color: C.muted, formula: "0 or 1", yR: [-0.5, 1.5], out: stepFn(z).toFixed(1) },
    { name: "Sigmoid", fn: sigmoid, color: C.purple, formula: "1/(1+e\u207B\u1DBB)", yR: [-0.5, 1.5], out: sigmoid(z).toFixed(3) },
    { name: "ReLU", fn: relu, color: C.accent, formula: "max(0, z)", yR: [-0.5, 4], out: relu(z).toFixed(1) },
  ];

  return (
    <div>
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>The Three Foundational Activation Functions</div>
        <div style={{ fontSize: 12, color: C.muted }}>Drag the slider to see how each function transforms the same input z</div>
      </div>

      <Card style={{ maxWidth: 600, margin: "0 auto 20px", textAlign: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 11, color: C.muted }}>-4</span>
          <input type="range" min={-4} max={4} step={0.1} value={z} onChange={function(e) { setZ(parseFloat(e.target.value)); }}
            style={{ flex: 1, accentColor: C.accent }} />
          <span style={{ fontSize: 11, color: C.muted }}>4</span>
        </div>
        <div style={{ marginTop: 10, fontSize: 16, fontWeight: 800, color: C.text }}>z = {z.toFixed(1)}</div>
      </Card>

      <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap", marginBottom: 20 }}>
        {fns.map(function(f, idx) {
          return (
            <div key={idx} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: f.color, marginBottom: 6 }}>{f.name}</div>
              <GraphCanvas width={W} height={H} xRange={[-4, 4]} yRange={f.yR}>
                {function(toX, toY) {
                  return (
                    <g>
                      <FnLine fn={f.fn} xRange={[-4, 4]} toX={toX} toY={toY} color={f.color} />
                      <line x1={toX(z)} y1={toY(f.yR[0])} x2={toX(z)} y2={toY(f.fn(z))} stroke={C.yellow} strokeWidth={1} strokeDasharray="3,3" opacity={0.6} />
                      <circle cx={toX(z)} cy={toY(f.fn(z))} r={5} fill={C.yellow}>
                        <animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite" />
                      </circle>
                      <text x={toX(z) + 8} y={toY(f.fn(z)) - 8} fill={C.yellow} fontSize={10} fontWeight={700} fontFamily="monospace">{f.out}</text>
                    </g>
                  );
                }}
              </GraphCanvas>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 6 }}>f(z) = {f.formula}</div>
            </div>
          );
        })}
      </div>

      <Card style={{ maxWidth: 700, margin: "0 auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {["Property", "Step", "Sigmoid", "ReLU"].map(function(h, i) {
                var colors = [C.muted, C.muted, C.purple, C.accent];
                return <th key={i} style={{ padding: "8px 10px", fontSize: 10, fontWeight: 700, color: colors[i], borderBottom: "2px solid " + C.border, textAlign: i === 0 ? "left" : "center", fontFamily: "monospace" }}>{h}</th>;
              })}
            </tr>
          </thead>
          <tbody>
            {[
              ["Output range", "{0, 1}", "(0, 1)", "[0, \u221E)"],
              ["Derivative", "0 everywhere", "max 0.25", "0 or 1"],
              ["Vanishing gradient?", "Total", "Yes", "No (when active)"],
              ["Used where", "Nowhere modern", "Output layer", "Hidden layers"],
              ["Speed", "Fast", "Slow (exp)", "Fastest"],
            ].map(function(row, ri) {
              return (
                <tr key={ri}>
                  {row.map(function(cell, ci) {
                    return <td key={ci} style={{ padding: "8px 10px", fontSize: 11, fontFamily: "monospace", color: ci === 0 ? C.muted : C.text, textAlign: ci === 0 ? "left" : "center", borderBottom: "1px solid " + C.border }}>{cell}</td>;
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════
   TAB 2: RELU IN THE NETWORK
   ═══════════════════════════════════════════════════════════ */

function Neuron({ x, y, value, active, label, color, delay }) {
  color = color || C.blue;
  delay = delay || 0;
  const [show, setShow] = useState(false);
  useEffect(function() { var t = setTimeout(function() { setShow(true); }, delay); return function() { clearTimeout(t); }; }, [delay]);
  var r = 22;
  var intensity = active ? 1 : 0.3;
  return (
    <g style={{ opacity: show ? 1 : 0, transition: "opacity 0.5s ease" }}>
      <circle cx={x} cy={y} r={r + 6} fill={active ? color.replace(")", ",0.15)").replace("rgb", "rgba") : "none"} />
      <circle cx={x} cy={y} r={r} fill={C.card} stroke={color} strokeWidth={active ? 2.5 : 1} opacity={intensity} />
      {value !== undefined && (
        <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fill={color} fontSize="11" fontWeight="600" fontFamily="'JetBrains Mono', monospace" opacity={intensity}>
          {typeof value === "number" ? value.toFixed(1) : value}
        </text>
      )}
      {label && <text x={x} y={y + r + 16} textAnchor="middle" fill={C.muted} fontSize="9" fontFamily="'JetBrains Mono', monospace">{label}</text>}
    </g>
  );
}

function Conn({ x1, y1, x2, y2, weight, active, delay }) {
  delay = delay || 0;
  const [show, setShow] = useState(false);
  useEffect(function() { var t = setTimeout(function() { setShow(true); }, delay); return function() { clearTimeout(t); }; }, [delay]);
  var opacity = active ? (weight > 0 ? 0.6 : 0.25) : 0.08;
  var color = weight > 0 ? C.blue : C.accent;
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={active ? Math.abs(weight) * 2 + 0.5 : 0.5} opacity={show ? opacity : 0} style={{ transition: "all 0.5s ease" }} />;
}

function MiniReLUGraph({ x, y, width, height, inputVal, highlighted }) {
  var midX = x + width / 2, midY = y + height / 2, scale = height / 6;
  var points = [];
  for (var i = -3; i <= 3; i += 0.1) { points.push((midX + i * (width / 6)) + "," + (midY - Math.max(0, i) * scale)); }
  var inputX = midX + inputVal * (width / 6);
  var outputVal = Math.max(0, inputVal);
  var inputY = midY - outputVal * scale;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx="8" fill={highlighted ? "rgba(255,107,53,0.08)" : "rgba(255,255,255,0.02)"} stroke={highlighted ? C.accent : C.dim} strokeWidth={highlighted ? 2 : 1} />
      <line x1={x + 4} y1={midY} x2={x + width - 4} y2={midY} stroke={C.dim} strokeWidth="0.5" />
      <line x1={midX} y1={y + 4} x2={midX} y2={y + height - 4} stroke={C.dim} strokeWidth="0.5" />
      <polyline points={points.join(" ")} fill="none" stroke={C.accent} strokeWidth="2.5" strokeLinecap="round" />
      {highlighted && (
        <g>
          <line x1={inputX} y1={midY} x2={inputX} y2={inputY} stroke={C.yellow} strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
          <circle cx={inputX} cy={inputY} r="5" fill={C.yellow}><animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite" /></circle>
        </g>
      )}
      <text x={x + width / 2} y={y - 8} textAnchor="middle" fill={highlighted ? C.accent : C.muted} fontSize="11" fontWeight="700" fontFamily="'JetBrains Mono', monospace">ReLU(x) = max(0, x)</text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, active }) {
  color = color || C.muted;
  var dx = x2 - x1, dy = y2 - y1, len = Math.sqrt(dx * dx + dy * dy);
  var nx = dx / len, ny = dy / len, al = 8;
  return (
    <g opacity={active ? 1 : 0.5}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={active ? 2 : 1.5} strokeDasharray={active ? "none" : "6,4"} />
      <polygon points={x2 + "," + y2 + " " + (x2 - nx * al - ny * 4) + "," + (y2 - ny * al + nx * 4) + " " + (x2 - nx * al + ny * 4) + "," + (y2 - ny * al - nx * 4)} fill={color} />
    </g>
  );
}

function SBox({ x, y, width, height, title, subtitle, color, active }) {
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx="6" fill={active ? color + "15" : "rgba(255,255,255,0.02)"} stroke={active ? color : C.dim} strokeWidth={active ? 2 : 1} />
      <text x={x + width / 2} y={y + height / 2 - (subtitle ? 6 : 0)} textAnchor="middle" dominantBaseline="middle" fill={active ? color : C.muted} fontSize="12" fontWeight="700" fontFamily="'JetBrains Mono', monospace">{title}</text>
      {subtitle && <text x={x + width / 2} y={y + height / 2 + 10} textAnchor="middle" dominantBaseline="middle" fill={C.muted} fontSize="9" fontFamily="'JetBrains Mono', monospace">{subtitle}</text>}
    </g>
  );
}

function TabReLUNetwork() {
  const [stp, setStp] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [inputVal, setInputVal] = useState(1.5);

  var steps = [
    { id: 0, title: "Input Layer", desc: "Raw data enters the network. Each neuron holds one feature value (e.g., pixel intensity, sensor reading)." },
    { id: 1, title: "Linear Transform", desc: "Weights multiply inputs, biases are added: z = Wx + b. This is just matrix multiplication \u2014 purely linear." },
    { id: 2, title: "\u26A1 ReLU Activation", desc: "ReLU(z) = max(0, z). Introduces non-linearity \u2014 killing negative values and passing positives unchanged." },
    { id: 3, title: "Next Layer", desc: "Activated outputs become inputs to the next layer. Without ReLU, stacking layers collapses into a single linear transformation." },
    { id: 4, title: "Full Picture", desc: "Every hidden layer repeats: Linear \u2192 ReLU \u2192 Linear \u2192 ReLU \u2192 ... \u2192 Output." },
  ];

  useEffect(function() {
    if (!autoPlay) return;
    var t = setInterval(function() { setStp(function(s) { return (s + 1) % steps.length; }); }, 3000);
    return function() { clearInterval(t); };
  }, [autoPlay]);

  var inputs = [0.8, -0.3, 1.5, -0.7];
  var weights = [[0.5, -0.2, 0.8, 0.1], [0.3, 0.7, -0.4, 0.6], [-0.6, 0.4, 0.9, -0.3]];
  var linearOutputs = weights.map(function(w) { return w.reduce(function(s, wi, i) { return s + wi * inputs[i]; }, 0); });
  var reluOutputs = linearOutputs.map(function(v) { return Math.max(0, v); });

  return (
    <div>
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>Where Does ReLU Fit?</div>
        <div style={{ fontSize: 12, color: C.muted }}>Inside every hidden layer of a neural network</div>
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg viewBox="0 0 900 420" style={{ width: "100%", maxWidth: 900, background: "rgba(255,255,255,0.01)", borderRadius: 12, border: "1px solid " + C.border }}>
          <SBox x={30} y={15} width={120} height={38} title="INPUT" subtitle="features" color={C.blue} active={stp === 0 || stp === 4} />
          <SBox x={210} y={15} width={140} height={38} title="LINEAR" subtitle="z = Wx + b" color={C.purple} active={stp === 1 || stp === 4} />
          <SBox x={420} y={15} width={140} height={38} title={"ReLU \u26A1"} subtitle="max(0, z)" color={C.accent} active={stp === 2 || stp === 4} />
          <SBox x={630} y={15} width={140} height={38} title="OUTPUT" subtitle="to next layer" color={C.yellow} active={stp === 3 || stp === 4} />
          <Arrow x1={155} y1={34} x2={205} y2={34} color={C.dim} active={stp >= 1} />
          <Arrow x1={355} y1={34} x2={415} y2={34} color={C.dim} active={stp >= 2} />
          <Arrow x1={565} y1={34} x2={625} y2={34} color={C.dim} active={stp >= 3} />

          {inputs.map(function(v, i) { return <Neuron key={"in-" + i} x={90} y={110 + i * 70} value={v} active={stp >= 0} label={"x" + (i + 1)} color={C.blue} delay={i * 80} />; })}
          {stp >= 1 && inputs.map(function(_, i) { return linearOutputs.map(function(_, j) { return <Conn key={"c1-" + i + "-" + j} x1={112} y1={110 + i * 70} x2={258} y2={130 + j * 85} weight={weights[j][i]} active={stp >= 1} delay={i * 30 + j * 30} />; }); })}
          {linearOutputs.map(function(v, i) { return <Neuron key={"lin-" + i} x={280} y={130 + i * 85} value={v} active={stp >= 1} label={"z" + (i + 1)} color={C.purple} delay={200 + i * 100} />; })}

          <MiniReLUGraph x={390} y={90} width={170} height={120} inputVal={linearOutputs[0]} highlighted={stp === 2 || stp === 4} />

          {linearOutputs.map(function(v, i) {
            var ny = 130 + i * 85; var passed = v > 0;
            return stp >= 2 ? (
              <g key={"relu-" + i}>
                <line x1={302} y1={ny} x2={570} y2={ny} stroke={passed ? C.accent : C.dim} strokeWidth={passed ? 1.5 : 0.8} strokeDasharray={passed ? "none" : "4,4"} opacity={passed ? 0.7 : 0.3} />
                {!passed && <g><line x1={428} y1={ny - 8} x2={442} y2={ny + 8} stroke="#ef4444" strokeWidth="2.5" opacity="0.8" /><line x1={442} y1={ny - 8} x2={428} y2={ny + 8} stroke="#ef4444" strokeWidth="2.5" opacity="0.8" /></g>}
                {passed && <text x={435} y={ny - 6} textAnchor="middle" fill={C.accent} fontSize="9" fontWeight="700" opacity="0.9">{"\u2713"}</text>}
              </g>
            ) : null;
          })}

          {reluOutputs.map(function(v, i) { return <Neuron key={"ro-" + i} x={590} y={130 + i * 85} value={v} active={stp >= 2} label={"a" + (i + 1)} color={v > 0 ? C.accent : C.dim} delay={400 + i * 100} />; })}
          {stp >= 3 && reluOutputs.map(function(v, i) { return [0, 1].map(function(j) { return <Conn key={"c2-" + i + "-" + j} x1={612} y1={130 + i * 85} x2={728} y2={150 + j * 100} weight={v > 0 ? 0.5 : 0.1} active={stp >= 3 && v > 0} delay={i * 50} />; }); })}
          {[0, 1].map(function(i) { return <Neuron key={"next-" + i} x={750} y={150 + i * 100} value="?" active={stp >= 3} label={"h" + (i + 1)} color={C.yellow} delay={600 + i * 100} />; })}

          {stp === 4 && (
            <g>
              <rect x={195} y={68} width={430} height={310} rx="10" fill="none" stroke={C.accent} strokeWidth="1.5" strokeDasharray="8,6" opacity="0.4" />
              <text x={410} y={395} textAnchor="middle" fill={C.accent} fontSize="12" fontWeight="700" opacity="0.7">{"\u2190 One Hidden Layer = Linear + ReLU \u2192"}</text>
            </g>
          )}
          <text x={450} y={408} textAnchor="middle" fill={C.dim} fontSize="9">{stp < 4 ? "Click steps below to trace the data flow \u2192" : "This pattern repeats for every hidden layer"}</text>
        </svg>
      </div>

      <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s) {
          return (
            <button key={s.id} onClick={function() { setStp(s.id); setAutoPlay(false); }} style={{
              padding: "7px 14px", borderRadius: 8, border: "1.5px solid " + (stp === s.id ? C.accent : C.border),
              background: stp === s.id ? C.accent + "20" : C.card, color: stp === s.id ? C.accent : C.muted,
              cursor: "pointer", fontSize: 11, fontWeight: 600, fontFamily: "inherit", transition: "all 0.2s",
            }}>
              {s.id + 1}. {s.title}
            </button>
          );
        })}
        <button onClick={function() { setAutoPlay(!autoPlay); }} style={{
          padding: "7px 12px", borderRadius: 8, border: "1.5px solid " + (autoPlay ? C.yellow : C.border),
          background: autoPlay ? C.yellow + "20" : C.card, color: autoPlay ? C.yellow : C.muted,
          cursor: "pointer", fontSize: 11, fontFamily: "inherit",
        }}>
          {autoPlay ? "\u23F8 Pause" : "\u25B6 Auto"}
        </button>
      </div>

      <Card highlight={stp === 2} style={{ maxWidth: 700, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: stp === 2 ? C.accent : C.text, marginBottom: 4 }}>Step {stp + 1}: {steps[stp].title}</div>
        <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6 }}>{steps[stp].desc}</div>
      </Card>

      <Card style={{ maxWidth: 700, margin: "0 auto" }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.accent, marginBottom: 10 }}>{"\uD83E\uDDEA"} Try it: Slide to see ReLU in action</div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 10, color: C.muted }}>-3</span>
          <input type="range" min={-3} max={3} step={0.1} value={inputVal} onChange={function(e) { setInputVal(parseFloat(e.target.value)); }} style={{ flex: 1, accentColor: C.accent }} />
          <span style={{ fontSize: 10, color: C.muted }}>3</span>
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: 32, marginTop: 12 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: C.muted }}>INPUT (z)</div>
            <div style={{ fontSize: 20, fontWeight: 800, color: inputVal < 0 ? C.red : C.blue }}>{inputVal.toFixed(1)}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: 18, color: C.accent }}>{"\u2192"}</div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: C.muted }}>ReLU</div>
            <div style={{ fontSize: 13, color: C.dim }}>max(0, {inputVal.toFixed(1)})</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: 18, color: C.accent }}>{"\u2192"}</div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: C.muted }}>OUTPUT (a)</div>
            <div style={{ fontSize: 20, fontWeight: 800, color: Math.max(0, inputVal) === 0 ? C.red : C.accent }}>{Math.max(0, inputVal).toFixed(1)}</div>
          </div>
        </div>
        {inputVal < 0 && <div style={{ textAlign: "center", marginTop: 8, fontSize: 10, color: C.red, opacity: 0.8 }}>{"\u2620"} Negative value killed! Neuron is "dead" for this input.</div>}
        {inputVal > 0 && <div style={{ textAlign: "center", marginTop: 8, fontSize: 10, color: C.accent, opacity: 0.8 }}>{"\u2713"} Positive value passes through unchanged!</div>}
      </Card>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════
   TAB 3: VANISHING GRADIENT
   ═══════════════════════════════════════════════════════════ */
function TabVanishingGradient() {
  const [layers, setLayers] = useState(10);

  var sigGrads = useMemo(function() {
    var arr = [0.20];
    for (var i = 1; i < layers; i++) arr.push(arr[i - 1] * 0.25);
    return arr;
  }, [layers]);

  var reluGrads = useMemo(function() {
    var arr = [0.20];
    for (var i = 1; i < layers; i++) arr.push(arr[i - 1] * 1.0);
    return arr;
  }, [layers]);

  var barH = 28;
  var maxW = 380;

  return (
    <div>
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>The Vanishing Gradient Problem</div>
        <div style={{ fontSize: 12, color: C.muted }}>Watch the error signal shrink with Sigmoid vs stay strong with ReLU</div>
      </div>

      <Card style={{ maxWidth: 500, margin: "0 auto 20px", textAlign: "center" }}>
        <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>Number of layers</div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 11, color: C.muted }}>2</span>
          <input type="range" min={2} max={15} step={1} value={layers} onChange={function(e) { setLayers(parseInt(e.target.value)); }}
            style={{ flex: 1, accentColor: C.yellow }} />
          <span style={{ fontSize: 11, color: C.muted }}>15</span>
        </div>
        <div style={{ marginTop: 8, fontSize: 14, fontWeight: 700, color: C.yellow }}>{layers} layers</div>
      </Card>

      <div style={{ display: "flex", gap: 30, justifyContent: "center", flexWrap: "wrap" }}>
        {/* Sigmoid column */}
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.purple, marginBottom: 10, textAlign: "center" }}>
            Sigmoid <span style={{ color: C.muted, fontWeight: 400 }}>(deriv max 0.25)</span>
          </div>
          {sigGrads.map(function(g, i) {
            var w = Math.max(2, (g / 0.20) * maxW);
            var layerNum = layers - i;
            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <div style={{ width: 55, fontSize: 10, color: C.muted, textAlign: "right", fontFamily: "monospace" }}>Layer {layerNum}</div>
                <div style={{ width: maxW, position: "relative" }}>
                  <div style={{ width: w, height: barH, borderRadius: 4, background: g < 0.001 ? C.red + "40" : C.purple + "60", border: "1px solid " + (g < 0.001 ? C.red : C.purple) + "40", transition: "width 0.3s" }} />
                </div>
                <div style={{ width: 80, fontSize: 10, color: g < 0.001 ? C.red : C.muted, fontFamily: "monospace" }}>
                  {g < 0.0000001 ? "\u2248 0" : g.toExponential(1)}
                </div>
              </div>
            );
          })}
        </div>

        {/* ReLU column */}
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.accent, marginBottom: 10, textAlign: "center" }}>
            ReLU <span style={{ color: C.muted, fontWeight: 400 }}>(deriv = 1)</span>
          </div>
          {reluGrads.map(function(g, i) {
            var w = Math.max(2, (g / 0.20) * maxW);
            var layerNum = layers - i;
            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <div style={{ width: 55, fontSize: 10, color: C.muted, textAlign: "right", fontFamily: "monospace" }}>Layer {layerNum}</div>
                <div style={{ width: maxW, position: "relative" }}>
                  <div style={{ width: w, height: barH, borderRadius: 4, background: C.accent + "60", border: "1px solid " + C.accent + "40", transition: "width 0.3s" }} />
                </div>
                <div style={{ width: 80, fontSize: 10, color: C.accent, fontFamily: "monospace" }}>{g.toExponential(1)}</div>
              </div>
            );
          })}
        </div>
      </div>

      <Card style={{ maxWidth: 700, margin: "20px auto 0", background: "rgba(255,107,53,0.06)", border: "1px solid rgba(255,107,53,0.2)" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.accent, marginBottom: 4 }}>{"\uD83D\uDCA1"} Why This Matters</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>
          With <span style={{ color: C.purple, fontWeight: 700 }}>Sigmoid</span>, each layer multiplies the gradient by at most 0.25. After {layers} layers: 0.25^{layers} = {Math.pow(0.25, layers).toExponential(1)}. Early layers get <span style={{ color: C.red }}>virtually zero learning signal</span>. With <span style={{ color: C.accent, fontWeight: 700 }}>ReLU</span>, active neurons pass gradients at full strength. This is what made 50-1000+ layer networks possible.
        </div>
      </Card>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════
   TAB 4: RELU VARIANTS
   ═══════════════════════════════════════════════════════════ */
function TabVariants() {
  const [z, setZ] = useState(0.8);
  const [selected, setSelected] = useState(["relu", "leaky", "gelu"]);
  var W = 760, H = 280;

  var variants = [
    { id: "relu",   name: "ReLU",       fn: relu,      color: C.accent, formula: "max(0, z)" },
    { id: "leaky",  name: "Leaky ReLU", fn: leakyRelu, color: C.cyan,   formula: "z>0 ? z : 0.01z" },
    { id: "elu",    name: "ELU",        fn: elu,       color: C.green,  formula: "z>0 ? z : \u03B1(e\u1DBB\u207B\u00B9)" },
    { id: "gelu",   name: "GELU",       fn: gelu,      color: C.purple, formula: "z\u00D7\u03A6(z)" },
    { id: "swish",  name: "Swish",      fn: swish,     color: C.pink,   formula: "z\u00D7\u03C3(z)" },
    { id: "tanh",   name: "Tanh",       fn: tanhFn,    color: C.yellow, formula: "tanh(z)" },
  ];

  var toggle = function(id) {
    setSelected(function(s) { return s.indexOf(id) >= 0 ? s.filter(function(x) { return x !== id; }) : s.concat([id]); });
  };

  var active = variants.filter(function(v) { return selected.indexOf(v.id) >= 0; });

  return (
    <div>
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>ReLU Variants & Beyond</div>
        <div style={{ fontSize: 12, color: C.muted }}>Toggle functions on/off to compare how they handle the same input</div>
      </div>

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {variants.map(function(v) {
          var on = selected.indexOf(v.id) >= 0;
          return (
            <button key={v.id} onClick={function() { toggle(v.id); }} style={{
              padding: "6px 14px", borderRadius: 20,
              border: "1.5px solid " + (on ? v.color : C.border),
              background: on ? v.color + "20" : C.card,
              color: on ? v.color : C.muted,
              cursor: "pointer", fontSize: 11, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
              transition: "all 0.2s",
            }}>
              {v.name}
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <GraphCanvas width={W} height={H} xRange={[-4, 4]} yRange={[-2, 4]}>
          {function(toX, toY) {
            return (
              <g>
                {active.map(function(v) { return <FnLine key={v.id} fn={v.fn} xRange={[-4, 4]} toX={toX} toY={toY} color={v.color} strokeWidth={2} />; })}
                <line x1={toX(z)} y1={toY(-2)} x2={toX(z)} y2={toY(4)} stroke={C.yellow} strokeWidth={1} strokeDasharray="4,4" opacity={0.4} />
                {active.map(function(v) {
                  return (
                    <g key={"dot-" + v.id}>
                      <circle cx={toX(z)} cy={toY(v.fn(z))} r={5} fill={v.color} />
                      <text x={toX(z) + 10} y={toY(v.fn(z)) + 4} fill={v.color} fontSize={10} fontWeight={700} fontFamily="monospace">{v.fn(z).toFixed(2)}</text>
                    </g>
                  );
                })}
              </g>
            );
          }}
        </GraphCanvas>
      </div>

      <Card style={{ maxWidth: 600, margin: "0 auto 16px", textAlign: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 10, color: C.muted }}>-4</span>
          <input type="range" min={-4} max={4} step={0.1} value={z} onChange={function(e) { setZ(parseFloat(e.target.value)); }}
            style={{ flex: 1, accentColor: C.yellow }} />
          <span style={{ fontSize: 10, color: C.muted }}>4</span>
        </div>
        <div style={{ marginTop: 8, fontSize: 14, fontWeight: 700, color: C.yellow }}>z = {z.toFixed(1)}</div>
      </Card>

      <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap", marginBottom: 16 }}>
        {active.map(function(v) {
          return (
            <div key={v.id} style={{
              background: C.card, border: "1px solid " + v.color + "30", borderRadius: 8,
              padding: "10px 16px", textAlign: "center", minWidth: 110,
            }}>
              <div style={{ fontSize: 10, color: v.color, fontWeight: 700, marginBottom: 4 }}>{v.name}</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: v.fn(z) === 0 ? C.red : v.color, fontFamily: "monospace" }}>{v.fn(z).toFixed(3)}</div>
              <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{v.formula}</div>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap", maxWidth: 800, margin: "0 auto" }}>
        {[
          { name: "Leaky ReLU", color: C.cyan, note: "Fixes dead neurons. Tiny 0.01 slope for negatives keeps gradient alive.", use: "Drop-in ReLU replacement" },
          { name: "ELU", color: C.green, note: "Smooth negative curve centers outputs near zero. Better training dynamics.", use: "Deep nets needing zero-mean" },
          { name: "GELU", color: C.purple, note: "Smooth probabilistic gating. Used in GPT, BERT, and modern transformers.", use: "Transformer architectures" },
          { name: "Swish", color: C.pink, note: "Discovered by automated search at Google. Non-monotonic and smooth.", use: "Very deep networks" },
        ].map(function(info, i) {
          return (
            <div key={i} style={{ background: C.card, border: "1px solid " + info.color + "20", borderRadius: 8, padding: "12px 16px", width: 180 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: info.color, marginBottom: 4 }}>{info.name}</div>
              <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.5, marginBottom: 6 }}>{info.note}</div>
              <div style={{ fontSize: 9, color: C.dim }}>Used in: {info.use}</div>
            </div>
          );
        })}
      </div>

      <Card style={{ maxWidth: 700, margin: "16px auto 0", background: "rgba(255,107,53,0.06)", border: "1px solid rgba(255,107,53,0.2)" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.accent, marginBottom: 6 }}>{"\uD83E\uDDED"} Which one should you use?</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>
          <span style={{ color: C.accent, fontWeight: 700 }}>ReLU</span> {"\u2014"} Default for CNNs and feedforward nets. <span style={{ color: C.cyan, fontWeight: 700 }}>Leaky ReLU</span> {"\u2014"} If dead neurons are a problem. <span style={{ color: C.purple, fontWeight: 700 }}>GELU</span> {"\u2014"} For transformers / NLP. <span style={{ color: C.yellow, fontWeight: 700 }}>Tanh</span> {"\u2014"} LSTM/GRU gates. <span style={{ color: C.pink, fontWeight: 700 }}>Swish/ELU</span> {"\u2014"} Worth experimenting with for very deep nets.
        </div>
      </Card>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════
   ROOT APP
   ═══════════════════════════════════════════════════════════ */
function App() {
  const [tab, setTab] = useState(0);
  var tabs = ["The Big Three", "ReLU in the Network", "Vanishing Gradient", "ReLU Variants"];

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", padding: "24px 16px",
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace", color: C.text,
      maxWidth: 960, margin: "0 auto",
    }}>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab === 0 && <TabBigThree />}
      {tab === 1 && <TabReLUNetwork />}
      {tab === 2 && <TabVanishingGradient />}
      {tab === 3 && <TabVariants />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

RELU_VISUAL_HEIGHT = 1150  # pixels - height of the iframe in Streamlit



# """
# Self-contained HTML for the ReLU activation interactive walkthrough.
# Embed in Streamlit via st.components.v1.html(RELU_VISUAL_HTML, height=RELU_VISUAL_HEIGHT).
# """
#
# RELU_VISUAL_HTML = r"""
# <!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8"/>
# <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
# <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
# <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
# <style>
#   * { margin: 0; padding: 0; box-sizing: border-box; }
#   body { background: #0a0a0f; overflow-x: hidden; }
# </style>
# </head>
# <body>
# <div id="root"></div>
# <script type="text/babel">
#
# const { useState, useEffect } = React;
#
# const COLORS = {
#   bg: "#0a0a0f",
#   card: "#12121a",
#   border: "#1e1e2e",
#   accent: "#ff6b35",
#   accentGlow: "rgba(255, 107, 53, 0.3)",
#   blue: "#4ecdc4",
#   blueGlow: "rgba(78, 205, 196, 0.3)",
#   purple: "#a78bfa",
#   purpleGlow: "rgba(167, 139, 250, 0.3)",
#   yellow: "#fbbf24",
#   yellowGlow: "rgba(251, 191, 36, 0.3)",
#   text: "#e4e4e7",
#   muted: "#71717a",
#   dim: "#3f3f46",
# };
#
# function Neuron({ x, y, value, active, label, color = COLORS.blue, delay = 0 }) {
#   const [show, setShow] = useState(false);
#   useEffect(() => {
#     const t = setTimeout(() => setShow(true), delay);
#     return () => clearTimeout(t);
#   }, [delay]);
#
#   const r = 22;
#   const intensity = active ? 1 : 0.3;
#
#   return (
#     <g style={{ opacity: show ? 1 : 0, transition: "opacity 0.5s ease" }}>
#       <circle cx={x} cy={y} r={r + 6} fill={active ? color.replace(")", ",0.15)").replace("rgb", "rgba") : "none"} />
#       <circle cx={x} cy={y} r={r} fill={COLORS.card} stroke={color} strokeWidth={active ? 2.5 : 1} opacity={intensity} />
#       {value !== undefined && (
#         <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fill={color} fontSize="11" fontWeight="600" fontFamily="'JetBrains Mono', monospace" opacity={intensity}>
#           {typeof value === "number" ? value.toFixed(1) : value}
#         </text>
#       )}
#       {label && (
#         <text x={x} y={y + r + 16} textAnchor="middle" fill={COLORS.muted} fontSize="9" fontFamily="'JetBrains Mono', monospace">
#           {label}
#         </text>
#       )}
#     </g>
#   );
# }
#
# function Connection({ x1, y1, x2, y2, weight, active, delay = 0 }) {
#   const [show, setShow] = useState(false);
#   useEffect(() => {
#     const t = setTimeout(() => setShow(true), delay);
#     return () => clearTimeout(t);
#   }, [delay]);
#
#   const opacity = active ? (weight > 0 ? 0.6 : 0.25) : 0.08;
#   const color = weight > 0 ? COLORS.blue : COLORS.accent;
#
#   return (
#     <line
#       x1={x1} y1={y1} x2={x2} y2={y2}
#       stroke={color} strokeWidth={active ? Math.abs(weight) * 2 + 0.5 : 0.5}
#       opacity={show ? opacity : 0}
#       style={{ transition: "all 0.5s ease" }}
#     />
#   );
# }
#
# function ReLUGraph({ x, y, width, height, inputVal, highlighted }) {
#   const graphW = width;
#   const graphH = height;
#   const midX = x + graphW / 2;
#   const midY = y + graphH / 2;
#   const scale = graphH / 6;
#
#   const points = [];
#   for (let i = -3; i <= 3; i += 0.1) {
#     const px = midX + i * (graphW / 6);
#     const py = midY - Math.max(0, i) * scale;
#     points.push(`${px},${py}`);
#   }
#
#   const inputX = midX + inputVal * (graphW / 6);
#   const outputVal = Math.max(0, inputVal);
#   const inputY = midY - outputVal * scale;
#
#   return (
#     <g>
#       <rect x={x} y={y} width={graphW} height={graphH} rx="8" fill={highlighted ? "rgba(255,107,53,0.08)" : "rgba(255,255,255,0.02)"} stroke={highlighted ? COLORS.accent : COLORS.dim} strokeWidth={highlighted ? 2 : 1} />
#       <line x1={x + 4} y1={midY} x2={x + graphW - 4} y2={midY} stroke={COLORS.dim} strokeWidth="0.5" />
#       <line x1={midX} y1={y + 4} x2={midX} y2={y + graphH - 4} stroke={COLORS.dim} strokeWidth="0.5" />
#       <polyline points={points.join(" ")} fill="none" stroke={COLORS.accent} strokeWidth="2.5" strokeLinecap="round" />
#       {highlighted && (
#         <>
#           <line x1={inputX} y1={midY} x2={inputX} y2={inputY} stroke={COLORS.yellow} strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
#           <circle cx={inputX} cy={inputY} r="5" fill={COLORS.yellow}>
#             <animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite" />
#           </circle>
#         </>
#       )}
#       <text x={x + graphW / 2} y={y - 8} textAnchor="middle" fill={highlighted ? COLORS.accent : COLORS.muted} fontSize="11" fontWeight="700" fontFamily="'JetBrains Mono', monospace">
#         ReLU(x) = max(0, x)
#       </text>
#     </g>
#   );
# }
#
# function FlowArrow({ x1, y1, x2, y2, label, color = COLORS.muted, active = false }) {
#   const dx = x2 - x1;
#   const dy = y2 - y1;
#   const len = Math.sqrt(dx * dx + dy * dy);
#   const nx = dx / len;
#   const ny = dy / len;
#   const arrowLen = 8;
#
#   return (
#     <g opacity={active ? 1 : 0.5}>
#       <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={active ? 2 : 1.5} strokeDasharray={active ? "none" : "6,4"} />
#       <polygon
#         points={`${x2},${y2} ${x2 - nx * arrowLen - ny * 4},${y2 - ny * arrowLen + nx * 4} ${x2 - nx * arrowLen + ny * 4},${y2 - ny * arrowLen - nx * 4}`}
#         fill={color}
#       />
#       {label && (
#         <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 10} textAnchor="middle" fill={color} fontSize="10" fontWeight="600" fontFamily="'JetBrains Mono', monospace">
#           {label}
#         </text>
#       )}
#     </g>
#   );
# }
#
# function StageBox({ x, y, width, height, title, subtitle, color, active }) {
#   return (
#     <g>
#       <rect x={x} y={y} width={width} height={height} rx="6" fill={active ? `${color}15` : "rgba(255,255,255,0.02)"} stroke={active ? color : COLORS.dim} strokeWidth={active ? 2 : 1} />
#       <text x={x + width / 2} y={y + height / 2 - (subtitle ? 6 : 0)} textAnchor="middle" dominantBaseline="middle" fill={active ? color : COLORS.muted} fontSize="12" fontWeight="700" fontFamily="'JetBrains Mono', monospace">
#         {title}
#       </text>
#       {subtitle && (
#         <text x={x + width / 2} y={y + height / 2 + 10} textAnchor="middle" dominantBaseline="middle" fill={COLORS.muted} fontSize="9" fontFamily="'JetBrains Mono', monospace">
#           {subtitle}
#         </text>
#       )}
#     </g>
#   );
# }
#
# function ReLUVisual() {
#   const [step, setStep] = useState(0);
#   const [autoPlay, setAutoPlay] = useState(false);
#   const [inputVal, setInputVal] = useState(1.5);
#
#   const steps = [
#     { id: 0, title: "Input Layer", desc: "Raw data enters the network. Each neuron holds one feature value (e.g., pixel intensity, sensor reading)." },
#     { id: 1, title: "Linear Transform", desc: "Weights multiply inputs, biases are added: z = Wx + b. This is just matrix multiplication \u2014 purely linear." },
#     { id: 2, title: "\u26A1 ReLU Activation", desc: "ReLU(z) = max(0, z). This is where the magic happens \u2014 it introduces non-linearity, killing negative values and passing positives unchanged." },
#     { id: 3, title: "Next Layer", desc: "The activated outputs become inputs to the next layer. Without ReLU, stacking layers would collapse into a single linear transformation." },
#     { id: 4, title: "Full Picture", desc: "Every hidden layer repeats: Linear \u2192 ReLU \u2192 Linear \u2192 ReLU \u2192 ... \u2192 Output. ReLU sits BETWEEN every pair of linear transformations." },
#   ];
#
#   useEffect(() => {
#     if (!autoPlay) return;
#     const t = setInterval(() => setStep((s) => (s + 1) % steps.length), 3000);
#     return () => clearInterval(t);
#   }, [autoPlay]);
#
#   const inputs = [0.8, -0.3, 1.5, -0.7];
#   const weights = [[0.5, -0.2, 0.8, 0.1], [0.3, 0.7, -0.4, 0.6], [-0.6, 0.4, 0.9, -0.3]];
#   const linearOutputs = weights.map((w) => w.reduce((s, wi, i) => s + wi * inputs[i], 0));
#   const reluOutputs = linearOutputs.map((v) => Math.max(0, v));
#
#   return (
#     <div style={{ background: COLORS.bg, padding: "24px 16px", fontFamily: "'JetBrains Mono', 'SF Mono', monospace", color: COLORS.text, display: "flex", flexDirection: "column", alignItems: "center" }}>
#       <h2 style={{ fontSize: 20, fontWeight: 800, margin: 0, background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.yellow})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", textAlign: "center" }}>
#         Where Does ReLU Fit?
#       </h2>
#       <p style={{ color: COLORS.muted, fontSize: 13, marginTop: 6, marginBottom: 20, textAlign: "center" }}>
#         Inside every hidden layer of a neural network
#       </p>
#
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 20, width: "100%" }}>
#         <svg viewBox="0 0 900 420" style={{ width: "100%", maxWidth: 900, background: "rgba(255,255,255,0.01)", borderRadius: 12, border: `1px solid ${COLORS.border}` }}>
#           <StageBox x={30} y={15} width={120} height={38} title="INPUT" subtitle="features" color={COLORS.blue} active={step === 0 || step === 4} />
#           <StageBox x={210} y={15} width={140} height={38} title="LINEAR" subtitle="z = Wx + b" color={COLORS.purple} active={step === 1 || step === 4} />
#           <StageBox x={420} y={15} width={140} height={38} title={"ReLU \u26A1"} subtitle="max(0, z)" color={COLORS.accent} active={step === 2 || step === 4} />
#           <StageBox x={630} y={15} width={140} height={38} title="OUTPUT" subtitle="to next layer" color={COLORS.yellow} active={step === 3 || step === 4} />
#
#           <FlowArrow x1={155} y1={34} x2={205} y2={34} color={COLORS.dim} active={step >= 1} />
#           <FlowArrow x1={355} y1={34} x2={415} y2={34} color={COLORS.dim} active={step >= 2} />
#           <FlowArrow x1={565} y1={34} x2={625} y2={34} color={COLORS.dim} active={step >= 3} />
#
#           {inputs.map((v, i) => {
#             const ny = 110 + i * 70;
#             return <Neuron key={`in-${i}`} x={90} y={ny} value={v} active={step >= 0} label={`x${i + 1}`} color={COLORS.blue} delay={i * 80} />;
#           })}
#
#           {step >= 1 && inputs.map((_, i) =>
#             linearOutputs.map((_, j) => (
#               <Connection key={`c1-${i}-${j}`} x1={112} y1={110 + i * 70} x2={258} y2={130 + j * 85} weight={weights[j][i]} active={step >= 1} delay={i * 30 + j * 30} />
#             ))
#           )}
#
#           {linearOutputs.map((v, i) => {
#             const ny = 130 + i * 85;
#             return <Neuron key={`lin-${i}`} x={280} y={ny} value={v} active={step >= 1} label={`z${i + 1}`} color={COLORS.purple} delay={200 + i * 100} />;
#           })}
#
#           <ReLUGraph x={390} y={90} width={170} height={120} inputVal={linearOutputs[0]} highlighted={step === 2 || step === 4} />
#
#           {linearOutputs.map((v, i) => {
#             const ny = 130 + i * 85;
#             const out = Math.max(0, v);
#             const passed = v > 0;
#             return step >= 2 ? (
#               <g key={`relu-${i}`}>
#                 <line x1={302} y1={ny} x2={570} y2={ny} stroke={passed ? COLORS.accent : COLORS.dim} strokeWidth={passed ? 1.5 : 0.8} strokeDasharray={passed ? "none" : "4,4"} opacity={passed ? 0.7 : 0.3} />
#                 {!passed && (
#                   <g>
#                     <line x1={428} y1={ny - 8} x2={442} y2={ny + 8} stroke="#ef4444" strokeWidth="2.5" opacity="0.8" />
#                     <line x1={442} y1={ny - 8} x2={428} y2={ny + 8} stroke="#ef4444" strokeWidth="2.5" opacity="0.8" />
#                   </g>
#                 )}
#                 {passed && (
#                   <text x={435} y={ny - 6} textAnchor="middle" fill={COLORS.accent} fontSize="9" fontWeight="700" opacity="0.9">{"\u2713"}</text>
#                 )}
#               </g>
#             ) : null;
#           })}
#
#           {reluOutputs.map((v, i) => {
#             const ny = 130 + i * 85;
#             return <Neuron key={`relu-out-${i}`} x={590} y={ny} value={v} active={step >= 2} label={`a${i + 1}`} color={v > 0 ? COLORS.accent : COLORS.dim} delay={400 + i * 100} />;
#           })}
#
#           {step >= 3 && reluOutputs.map((v, i) => {
#             const ny = 130 + i * 85;
#             return [0, 1].map((j) => (
#               <Connection key={`c2-${i}-${j}`} x1={612} y1={ny} x2={728} y2={150 + j * 100} weight={v > 0 ? 0.5 : 0.1} active={step >= 3 && v > 0} delay={i * 50} />
#             ));
#           })}
#
#           {[0, 1].map((i) => (
#             <Neuron key={`next-${i}`} x={750} y={150 + i * 100} value="?" active={step >= 3} label={`h${i + 1}`} color={COLORS.yellow} delay={600 + i * 100} />
#           ))}
#
#           {step === 4 && (
#             <g>
#               <rect x={195} y={68} width={430} height={310} rx="10" fill="none" stroke={COLORS.accent} strokeWidth="1.5" strokeDasharray="8,6" opacity="0.4" />
#               <text x={410} y={395} textAnchor="middle" fill={COLORS.accent} fontSize="12" fontWeight="700" opacity="0.7">
#                 {"\u2190 One Hidden Layer = Linear + ReLU \u2192"}
#               </text>
#             </g>
#           )}
#
#           <text x={450} y={408} textAnchor="middle" fill={COLORS.dim} fontSize="9">
#             {step < 4 ? "Click steps below to trace the data flow \u2192" : "This pattern repeats for every hidden layer in the network"}
#           </text>
#         </svg>
#       </div>
#
#       {/* Step controls */}
#       <div style={{ display: "flex", justifyContent: "center", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
#         {steps.map((s) => (
#           <button
#             key={s.id}
#             onClick={() => { setStep(s.id); setAutoPlay(false); }}
#             style={{
#               padding: "8px 16px",
#               borderRadius: 8,
#               border: `1.5px solid ${step === s.id ? COLORS.accent : COLORS.border}`,
#               background: step === s.id ? `${COLORS.accent}20` : COLORS.card,
#               color: step === s.id ? COLORS.accent : COLORS.muted,
#               cursor: "pointer",
#               fontSize: 12,
#               fontWeight: 600,
#               fontFamily: "inherit",
#               transition: "all 0.2s",
#             }}
#           >
#             {s.id + 1}. {s.title}
#           </button>
#         ))}
#         <button
#           onClick={() => setAutoPlay(!autoPlay)}
#           style={{
#             padding: "8px 14px",
#             borderRadius: 8,
#             border: `1.5px solid ${autoPlay ? COLORS.yellow : COLORS.border}`,
#             background: autoPlay ? `${COLORS.yellow}20` : COLORS.card,
#             color: autoPlay ? COLORS.yellow : COLORS.muted,
#             cursor: "pointer",
#             fontSize: 12,
#             fontFamily: "inherit",
#           }}
#         >
#           {autoPlay ? "\u23F8 Pause" : "\u25B6 Auto"}
#         </button>
#       </div>
#
#       {/* Explanation card */}
#       <div style={{ maxWidth: 700, width: "100%", margin: "0 auto 20px", padding: "18px 22px", background: COLORS.card, borderRadius: 10, border: `1px solid ${step === 2 ? COLORS.accent : COLORS.border}`, transition: "border 0.3s" }}>
#         <div style={{ fontSize: 15, fontWeight: 700, color: step === 2 ? COLORS.accent : COLORS.text, marginBottom: 6 }}>
#           Step {step + 1}: {steps[step].title}
#         </div>
#         <div style={{ fontSize: 13, color: COLORS.muted, lineHeight: 1.6 }}>
#           {steps[step].desc}
#         </div>
#       </div>
#
#       {/* Interactive ReLU explorer */}
#       <div style={{ maxWidth: 700, width: "100%", margin: "0 auto", padding: "20px 22px", background: COLORS.card, borderRadius: 10, border: `1px solid ${COLORS.border}` }}>
#         <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.accent, marginBottom: 12 }}>
#           {"\uD83E\uDDEA"} Try it: Slide to see ReLU in action
#         </div>
#         <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
#           <span style={{ fontSize: 11, color: COLORS.muted, minWidth: 30 }}>-3.0</span>
#           <input
#             type="range" min={-3} max={3} step={0.1} value={inputVal}
#             onChange={(e) => setInputVal(parseFloat(e.target.value))}
#             style={{ flex: 1, accentColor: COLORS.accent }}
#           />
#           <span style={{ fontSize: 11, color: COLORS.muted, minWidth: 30 }}>3.0</span>
#         </div>
#         <div style={{ display: "flex", justifyContent: "center", gap: 40, marginTop: 14 }}>
#           <div style={{ textAlign: "center" }}>
#             <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4 }}>INPUT (z)</div>
#             <div style={{ fontSize: 22, fontWeight: 800, color: inputVal < 0 ? "#ef4444" : COLORS.blue }}>
#               {inputVal.toFixed(1)}
#             </div>
#           </div>
#           <div style={{ display: "flex", alignItems: "center", fontSize: 20, color: COLORS.accent }}>{"\u2192"}</div>
#           <div style={{ textAlign: "center" }}>
#             <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4 }}>ReLU</div>
#             <div style={{ fontSize: 14, color: COLORS.dim }}>max(0, {inputVal.toFixed(1)})</div>
#           </div>
#           <div style={{ display: "flex", alignItems: "center", fontSize: 20, color: COLORS.accent }}>{"\u2192"}</div>
#           <div style={{ textAlign: "center" }}>
#             <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4 }}>OUTPUT (a)</div>
#             <div style={{ fontSize: 22, fontWeight: 800, color: Math.max(0, inputVal) === 0 ? "#ef4444" : COLORS.accent }}>
#               {Math.max(0, inputVal).toFixed(1)}
#             </div>
#           </div>
#         </div>
#         {inputVal < 0 && (
#           <div style={{ textAlign: "center", marginTop: 10, fontSize: 11, color: "#ef4444", opacity: 0.8 }}>
#             {"\u2620"} Negative value killed! This neuron is "dead" for this input.
#           </div>
#         )}
#         {inputVal > 0 && (
#           <div style={{ textAlign: "center", marginTop: 10, fontSize: 11, color: COLORS.accent, opacity: 0.8 }}>
#             {"\u2713"} Positive value passes through unchanged!
#           </div>
#         )}
#       </div>
#
#       {/* Key Insight */}
#       <div style={{ maxWidth: 700, width: "100%", margin: "20px auto 0", padding: "16px 22px", background: "rgba(255,107,53,0.06)", borderRadius: 10, border: "1px solid rgba(255,107,53,0.2)" }}>
#         <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.accent, marginBottom: 6 }}>{"\uD83D\uDCA1"} Key Insight</div>
#         <div style={{ fontSize: 12, color: COLORS.muted, lineHeight: 1.7 }}>
#           ReLU sits <span style={{ color: COLORS.accent, fontWeight: 700 }}>after every linear transformation</span> and <span style={{ color: COLORS.accent, fontWeight: 700 }}>before the next layer's input</span>. Without it, a 100-layer network would mathematically reduce to a single linear equation. ReLU is what gives deep networks the ability to learn complex, non-linear patterns.
#         </div>
#       </div>
#     </div>
#   );
# }
#
# ReactDOM.createRoot(document.getElementById("root")).render(<ReLUVisual />);
#
# </script>
# </body>
# </html>
# """
#
# RELU_VISUAL_HEIGHT = 1100  # pixels - height of the iframe in Streamlit