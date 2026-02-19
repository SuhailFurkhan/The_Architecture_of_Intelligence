"""
Self-contained HTML for the MLP XOR interactive walkthrough.
Used by 02_Perceptron.py to embed in Streamlit via st.components.v1.html().
"""

MLP_XOR_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #050508; overflow-x: hidden; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

const { useState } = React;

const PLOT = 200;
const PAD = 30;
const S = PLOT - PAD * 2;

function toX(v) { return PAD + v * S; }
function toY(v) { return PAD + (1 - v) * S; }

const points = [
  { x: 0, y: 0, xor: 0 },
  { x: 0, y: 1, xor: 1 },
  { x: 1, y: 0, xor: 1 },
  { x: 1, y: 1, xor: 0 },
];

const stages = [
  {
    id: "input",
    title: "Input Space",
    subtitle: "The XOR problem",
    desc: "XOR outputs are at opposite corners. No single line can separate them.",
    lines: [],
    region: null,
    pointColor: (p) => (p.xor === 1 ? "#4ade80" : "#334155"),
    pointStroke: (p) => (p.xor === 1 ? "#4ade80" : "#475569"),
  },
  {
    id: "or",
    title: "Hidden Neuron 1: OR",
    subtitle: '"Is at least one input = 1?"',
    desc: "This neuron draws a line that separates (0,0) from the rest. Everything above-right passes.",
    lines: [{ x1: -0.1, y1: 0.6, x2: 0.6, y2: -0.1, color: "#38bdf8" }],
    region: { type: "or", color: "#38bdf820" },
    pointColor: (p) => (p.x === 0 && p.y === 0 ? "#334155" : "#38bdf8"),
    pointStroke: (p) => (p.x === 0 && p.y === 0 ? "#475569" : "#38bdf8"),
    neuronLabel: "OR",
    outputLabels: [0, 1, 1, 1],
  },
  {
    id: "nand",
    title: "Hidden Neuron 2: NAND",
    subtitle: '"Are both inputs NOT 1?"',
    desc: "This neuron draws a line that separates (1,1) from the rest. Everything below-left passes.",
    lines: [{ x1: 0.4, y1: 1.1, x2: 1.1, y2: 0.4, color: "#c084fc" }],
    region: { type: "nand", color: "#c084fc20" },
    pointColor: (p) => (p.x === 1 && p.y === 1 ? "#334155" : "#c084fc"),
    pointStroke: (p) => (p.x === 1 && p.y === 1 ? "#475569" : "#c084fc"),
    neuronLabel: "NAND",
    outputLabels: [1, 1, 1, 0],
  },
  {
    id: "combined",
    title: "Both Lines Together",
    subtitle: "Two boundaries carve the space",
    desc: "Each line eliminates one incorrect corner. The region between both lines contains exactly the XOR = 1 points.",
    lines: [
      { x1: -0.1, y1: 0.6, x2: 0.6, y2: -0.1, color: "#38bdf8" },
      { x1: 0.4, y1: 1.1, x2: 1.1, y2: 0.4, color: "#c084fc" },
    ],
    region: { type: "between", color: "#4ade8015" },
    pointColor: (p) => (p.xor === 1 ? "#4ade80" : "#334155"),
    pointStroke: (p) => (p.xor === 1 ? "#4ade80" : "#475569"),
  },
  {
    id: "output",
    title: "Output Neuron: AND",
    subtitle: '"Do both hidden neurons agree?"',
    desc: "The AND neuron combines both results. Only (0,1) and (1,0) pass BOTH hidden neurons → XOR solved!",
    lines: [
      { x1: -0.1, y1: 0.6, x2: 0.6, y2: -0.1, color: "#38bdf860" },
      { x1: 0.4, y1: 1.1, x2: 1.1, y2: 0.4, color: "#c084fc60" },
    ],
    region: { type: "between", color: "#4ade8020" },
    pointColor: (p) => (p.xor === 1 ? "#4ade80" : "#334155"),
    pointStroke: (p) => (p.xor === 1 ? "#4ade80" : "#475569"),
  },
];

function MiniPlot({ stage, active }) {
  const fillRegion = () => {
    if (!stage.region) return null;
    const { type, color } = stage.region;
    if (type === "or") {
      return <polygon points={`${toX(0.6)},${toY(0)} ${toX(0)},${toY(0.6)} ${toX(0)},${toY(1)} ${toX(1)},${toY(1)} ${toX(1)},${toY(0)}`} fill={color} />;
    }
    if (type === "nand") {
      return <polygon points={`${toX(0)},${toY(0)} ${toX(1)},${toY(0)} ${toX(1)},${toY(0.4)} ${toX(0.4)},${toY(1)} ${toX(0)},${toY(1)}`} fill={color} />;
    }
    if (type === "between") {
      return <polygon points={`${toX(0)},${toY(0.6)} ${toX(0.6)},${toY(0)} ${toX(1)},${toY(0.4)} ${toX(0.4)},${toY(1)}`} fill={color} stroke="#4ade8030" strokeWidth={1} />;
    }
    return null;
  };

  return (
    <svg width={PLOT} height={PLOT} viewBox={`0 0 ${PLOT} ${PLOT}`} style={{
      background: "#08080d",
      borderRadius: 10,
      border: active ? "2px solid #4ade8050" : "2px solid #151520",
      transition: "all 0.3s",
    }}>
      {[0, 0.5, 1].map((v, i) => (
        <g key={i}>
          <line x1={toX(v)} y1={PAD} x2={toX(v)} y2={PAD + S} stroke="#111122" strokeWidth={v === 0 || v === 1 ? 1 : 0.5} />
          <line x1={PAD} y1={toY(v)} x2={PAD + S} y2={toY(v)} stroke="#111122" strokeWidth={v === 0 || v === 1 ? 1 : 0.5} />
        </g>
      ))}
      <text x={toX(0)} y={PLOT - 6} fill="#444" fontSize={9} textAnchor="middle" fontFamily="monospace">0</text>
      <text x={toX(1)} y={PLOT - 6} fill="#444" fontSize={9} textAnchor="middle" fontFamily="monospace">1</text>
      <text x={10} y={toY(0) + 3} fill="#444" fontSize={9} textAnchor="middle" fontFamily="monospace">0</text>
      <text x={10} y={toY(1) + 3} fill="#444" fontSize={9} textAnchor="middle" fontFamily="monospace">1</text>
      {fillRegion()}
      {stage.lines.map((l, i) => (
        <line key={i} x1={toX(l.x1)} y1={toY(l.y1)} x2={toX(l.x2)} y2={toY(l.y2)}
          stroke={l.color} strokeWidth={2} strokeDasharray="5 3" />
      ))}
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={toX(p.x)} cy={toY(p.y)} r={12} fill={stage.pointColor(p) + "15"} />
          <circle cx={toX(p.x)} cy={toY(p.y)} r={9} fill={stage.pointColor(p)} stroke={stage.pointStroke(p)} strokeWidth={2} />
          <text x={toX(p.x)} y={toY(p.y) + 3.5} fill={stage.pointColor(p) === "#334155" ? "#94a3b8" : "#0a0a0f"}
            fontSize={10} fontWeight={700} textAnchor="middle" fontFamily="monospace">
            {p.xor}
          </text>
        </g>
      ))}
    </svg>
  );
}

function NetworkDiagram({ activeStage }) {
  const neurons = {
    x1: { x: 60, y: 60, label: "x\u2081", color: "#94a3b8" },
    x2: { x: 60, y: 160, label: "x\u2082", color: "#94a3b8" },
    or: { x: 200, y: 55, label: "OR", color: "#38bdf8" },
    nand: { x: 200, y: 165, label: "NAND", color: "#c084fc" },
    and: { x: 340, y: 110, label: "AND", color: "#4ade80" },
  };

  const connections = [
    { from: "x1", to: "or" },
    { from: "x2", to: "or" },
    { from: "x1", to: "nand" },
    { from: "x2", to: "nand" },
    { from: "or", to: "and" },
    { from: "nand", to: "and" },
  ];

  const getHighlight = (id) => {
    if (activeStage === "input") return id === "x1" || id === "x2" ? 1 : 0.25;
    if (activeStage === "or") return id === "x1" || id === "x2" || id === "or" ? 1 : 0.25;
    if (activeStage === "nand") return id === "x1" || id === "x2" || id === "nand" ? 1 : 0.25;
    if (activeStage === "combined") return id === "or" || id === "nand" ? 1 : 0.35;
    if (activeStage === "output") return 1;
    return 0.5;
  };

  const getConnHighlight = (from, to) => {
    if (activeStage === "or") return to === "or" ? 1 : 0.15;
    if (activeStage === "nand") return to === "nand" ? 1 : 0.15;
    if (activeStage === "combined") return to === "and" ? 0.6 : 0.3;
    if (activeStage === "output") return 0.8;
    return 0.2;
  };

  const layerLabels = [
    { x: 60, label: "Input", active: activeStage === "input" },
    { x: 200, label: "Hidden", active: activeStage === "or" || activeStage === "nand" || activeStage === "combined" },
    { x: 340, label: "Output", active: activeStage === "output" },
  ];

  return (
    <svg width={400} height={220} viewBox="0 0 400 220" style={{
      background: "#08080d",
      borderRadius: 12,
      border: "2px solid #151520",
    }}>
      {layerLabels.map((l, i) => (
        <text key={i} x={l.x} y={205} fill={l.active ? "#e2e8f0" : "#333"}
          fontSize={10} fontWeight={600} textAnchor="middle" fontFamily="'JetBrains Mono', monospace"
          style={{ transition: "fill 0.3s" }}>
          {l.label}
        </text>
      ))}
      {connections.map((c, i) => {
        const from = neurons[c.from];
        const to = neurons[c.to];
        const op = getConnHighlight(c.from, c.to);
        return (
          <line key={i} x1={from.x + 18} y1={from.y} x2={to.x - 18} y2={to.y}
            stroke={neurons[c.to].color} strokeWidth={op > 0.5 ? 2 : 1} opacity={op}
            style={{ transition: "all 0.4s" }} />
        );
      })}
      {Object.entries(neurons).map(([id, n]) => {
        const op = getHighlight(id);
        const isActive = op > 0.5;
        return (
          <g key={id} opacity={op} style={{ transition: "opacity 0.4s" }}>
            <circle cx={n.x} cy={n.y} r={22} fill={isActive ? n.color + "18" : "#08080d"}
              stroke={n.color} strokeWidth={isActive ? 2.5 : 1.5} />
            <text x={n.x} y={n.y + 4} fill={isActive ? n.color : "#555"}
              fontSize={id === "x1" || id === "x2" ? 13 : 11} fontWeight={700}
              textAnchor="middle" fontFamily="'JetBrains Mono', monospace">
              {n.label}
            </text>
          </g>
        );
      })}
      {activeStage === "output" && (
        <g>
          <text x={370} y={110} fill="#4ade80" fontSize={16} textAnchor="start" fontFamily="monospace">{"\u2192"}</text>
          <text x={385} y={115} fill="#4ade80" fontSize={10} fontWeight={700} fontFamily="monospace" textAnchor="start">XOR</text>
        </g>
      )}
    </svg>
  );
}

function TruthTable({ activeStage }) {
  const rows = [
    { x1: 0, x2: 0, or: 0, nand: 1, and: 0 },
    { x1: 0, x2: 1, or: 1, nand: 1, and: 1 },
    { x1: 1, x2: 0, or: 1, nand: 1, and: 1 },
    { x1: 1, x2: 1, or: 1, nand: 0, and: 0 },
  ];

  const showOr = ["or", "combined", "output"].includes(activeStage);
  const showNand = ["nand", "combined", "output"].includes(activeStage);
  const showAnd = activeStage === "output";

  const cellStyle = (highlight, color) => ({
    padding: "6px 10px",
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
    textAlign: "center",
    color: highlight ? color : "#334155",
    fontWeight: highlight ? 700 : 400,
    background: highlight ? color + "08" : "transparent",
    transition: "all 0.3s",
    borderBottom: "1px solid #111118",
  });

  const headStyle = (active, color) => ({
    padding: "8px 10px",
    fontSize: 10,
    fontFamily: "'JetBrains Mono', monospace",
    textAlign: "center",
    color: active ? color : "#333",
    fontWeight: 700,
    borderBottom: "2px solid #1a1a2a",
    transition: "color 0.3s",
    letterSpacing: "0.05em",
  });

  return (
    <table style={{
      borderCollapse: "collapse",
      background: "#08080d",
      borderRadius: 10,
      overflow: "hidden",
      border: "2px solid #151520",
      width: "100%",
      maxWidth: 420,
    }}>
      <thead>
        <tr>
          <th style={headStyle(true, "#94a3b8")}>x&#x2081;</th>
          <th style={headStyle(true, "#94a3b8")}>x&#x2082;</th>
          <th style={headStyle(showOr, "#38bdf8")}>OR</th>
          <th style={headStyle(showNand, "#c084fc")}>NAND</th>
          <th style={headStyle(showAnd, "#4ade80")}>AND&#x2192;XOR</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td style={cellStyle(true, "#e2e8f0")}>{r.x1}</td>
            <td style={cellStyle(true, "#e2e8f0")}>{r.x2}</td>
            <td style={cellStyle(showOr, r.or === 1 ? "#38bdf8" : "#f87171")}>
              {showOr ? r.or : "\u2014"}
            </td>
            <td style={cellStyle(showNand, r.nand === 1 ? "#c084fc" : "#f87171")}>
              {showNand ? r.nand : "\u2014"}
            </td>
            <td style={cellStyle(showAnd, r.and === 1 ? "#4ade80" : "#64748b")}>
              {showAnd ? r.and : "\u2014"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MLPXORVisual() {
  const [step, setStep] = useState(0);
  const stage = stages[step];

  return (
    <div style={{
      background: "#050508",
      color: "#e2e8f0",
      fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "28px 16px",
    }}>
      <h2 style={{
        fontSize: 20,
        fontWeight: 700,
        letterSpacing: "-0.02em",
        marginBottom: 2,
        fontFamily: "'JetBrains Mono', 'IBM Plex Mono', monospace",
        color: "#f8fafc",
        textAlign: "center",
      }}>
        How an MLP Solves XOR
      </h2>
      <p style={{ color: "#64748b", fontSize: 13, marginBottom: 24, textAlign: "center" }}>
        Step through each neuron to see how two lines solve what one cannot
      </p>

      <NetworkDiagram activeStage={stage.id} />

      <div style={{ height: 20 }} />

      <div style={{
        display: "flex",
        gap: 20,
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "flex-start",
        maxWidth: 660,
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{
            fontSize: 12,
            fontWeight: 700,
            fontFamily: "'JetBrains Mono', monospace",
            color: "#94a3b8",
            marginBottom: 6,
            letterSpacing: "0.05em",
          }}>
            DECISION BOUNDARY
          </div>
          <MiniPlot stage={stage} active={true} />
        </div>

        <div style={{ textAlign: "center" }}>
          <div style={{
            fontSize: 12,
            fontWeight: 700,
            fontFamily: "'JetBrains Mono', monospace",
            color: "#94a3b8",
            marginBottom: 6,
            letterSpacing: "0.05em",
          }}>
            TRUTH TABLE
          </div>
          <TruthTable activeStage={stage.id} />
        </div>
      </div>

      <div style={{
        background: "#0a0a10",
        border: "1px solid #1e1e2e",
        borderRadius: 12,
        padding: "14px 22px",
        maxWidth: 500,
        textAlign: "center",
        marginTop: 20,
      }}>
        <div style={{
          fontSize: 16,
          fontWeight: 700,
          fontFamily: "'JetBrains Mono', monospace",
          color: stage.id === "or" ? "#38bdf8" : stage.id === "nand" ? "#c084fc" : stage.id === "output" ? "#4ade80" : "#e2e8f0",
          marginBottom: 2,
        }}>
          {stage.title}
        </div>
        <div style={{ fontSize: 13, color: "#64748b", fontStyle: "italic", marginBottom: 6 }}>
          {stage.subtitle}
        </div>
        <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>
          {stage.desc}
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 22, alignItems: "center" }}>
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          style={{
            padding: "8px 20px",
            borderRadius: 8,
            border: "1px solid #1e1e2e",
            background: step === 0 ? "#08080d" : "#0f0f18",
            color: step === 0 ? "#333" : "#e2e8f0",
            cursor: step === 0 ? "default" : "pointer",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          &#8592; Back
        </button>

        <div style={{ display: "flex", gap: 6 }}>
          {stages.map((_, i) => (
            <div
              key={i}
              onClick={() => setStep(i)}
              style={{
                width: i === step ? 24 : 8,
                height: 8,
                borderRadius: 4,
                background: i === step ? "#4ade80" : i < step ? "#4ade8060" : "#1e1e2e",
                cursor: "pointer",
                transition: "all 0.3s",
              }}
            />
          ))}
        </div>

        <button
          onClick={() => setStep(Math.min(stages.length - 1, step + 1))}
          disabled={step === stages.length - 1}
          style={{
            padding: "8px 20px",
            borderRadius: 8,
            border: step === stages.length - 1 ? "1px solid #1e1e2e" : "1px solid #4ade8040",
            background: step === stages.length - 1 ? "#08080d" : "#0a1a10",
            color: step === stages.length - 1 ? "#333" : "#4ade80",
            cursor: step === stages.length - 1 ? "default" : "pointer",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          Next &#8594;
        </button>
      </div>

      <div style={{ color: "#333", fontSize: 11, fontFamily: "'JetBrains Mono', monospace", marginTop: 8 }}>
        Step {step + 1} of {stages.length}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<MLPXORVisual />);

</script>
</body>
</html>
"""

MLP_XOR_HEIGHT = 850  # pixels — height of the iframe in Streamlit