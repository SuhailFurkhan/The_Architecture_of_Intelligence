"""
Self-contained HTML for the PEFT Additive interactive walkthrough.
Covers: Big Picture, LoRA Mechanics, Memory & Cost, Adapter Zoo,
Rank & Hyperparams, and Merging & Deployment.
Embed in Streamlit via st.components.v1.html(PEFT_ADDITIVE_HTML, height=PEFT_ADDITIVE_HEIGHT).
"""

PEFT_ADDITIVE_HTML = """
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
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #a78bfa; }
  @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
  @keyframes flowRight { 0%{transform:translateX(-8px);opacity:0} 50%{opacity:1} 100%{transform:translateX(8px);opacity:0} }
  @keyframes glow { 0%,100%{box-shadow:0 0 6px rgba(167,139,250,0.3)} 50%{box-shadow:0 0 16px rgba(167,139,250,0.7)} }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState = React.useState;
var useEffect = React.useEffect;
var useMemo = React.useMemo;
var useCallback = React.useCallback;

var C = {
  bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
  accent: "#a78bfa", blue: "#4ecdc4", purple: "#c084fc",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
};

var MUL = "\\u00D7";
var ARR = "\\u2192";
var DASH = "\\u2014";
var CHK = "\\u2713";
var WARN = "\\u26A0";
var LQ = "\\u201C";
var RQ = "\\u201D";
var LARR = "\\u2190";
var PLAY = "\\u25B6";
var PAUSE = "\\u23F8";
var BULB = "\\uD83D\\uDCA1";
var TARG = "\\uD83C\\uDFAF";
var LOCK = "\\uD83D\\uDD12";
var UNLOCK = "\\uD83D\\uDD13";
var BRAIN = "\\uD83E\\uDDE0";
var FIRE = "\\uD83D\\uDD25";
var GEAR = "\\u2699";
var CHART = "\\uD83D\\uDCC8";
var PLUS = "\\u002B";
var DARR = "\\u2193";
var UARR = "\\u2191";
var STAR = "\\u2605";
var PLUG = "\\uD83D\\uDD0C";
var DNA = "\\uD83E\\uDDEC";

/* ─── Shared components ─── */

function TabBar(props) {
  var tabs = props.tabs, active = props.active, onChange = props.onChange;
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 24, overflowX: "auto" }}>
      {tabs.map(function(t, i) {
        return (
          <button key={i} onClick={function() { onChange(i); }} style={{
            padding: "12px 18px", background: "none", border: "none",
            borderBottom: active === i ? "2px solid " + C.accent : "2px solid transparent",
            color: active === i ? C.accent : C.muted, cursor: "pointer",
            fontSize: 11, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
            transition: "all 0.2s", whiteSpace: "nowrap", marginBottom: -2,
          }}>
            {t}
          </button>
        );
      })}
    </div>
  );
}

function Card(props) {
  return (
    <div style={Object.assign({
      background: C.card, borderRadius: 10, padding: "18px 22px",
      border: "1px solid " + (props.highlight ? C.accent : C.border),
      transition: "border 0.3s",
    }, props.style || {})}>
      {props.children}
    </div>
  );
}

function Insight(props) {
  return (
    <div style={Object.assign({
      maxWidth: 1100, margin: "16px auto 0",
      padding: "16px 22px", background: "rgba(167,139,250,0.06)",
      borderRadius: 10, border: "1px solid rgba(167,139,250,0.2)",
    }, props.style || {})}>
      <div style={{ fontSize: 11, fontWeight: 700, color: C.accent, marginBottom: 6 }}>{(props.icon || BULB) + " " + (props.title || "Key Insight")}</div>
      <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{props.children}</div>
    </div>
  );
}

function SectionTitle(props) {
  return (
    <div style={{ textAlign: "center", marginBottom: 20 }}>
      <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>{props.title}</div>
      <div style={{ fontSize: 12, color: C.muted }}>{props.subtitle}</div>
    </div>
  );
}

function StatBox(props) {
  return (
    <div style={{ textAlign: "center", minWidth: props.minW || 90 }}>
      <div style={{ fontSize: 8, color: C.muted, marginBottom: 4, letterSpacing: 1 }}>{props.label}</div>
      <div style={{ fontSize: props.bigFont || 22, fontWeight: 800, color: props.color || C.accent }}>{props.value}</div>
      {props.sub && <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{props.sub}</div>}
    </div>
  );
}


/* ===============================================================
   TAB 1: THE BIG PICTURE
   =============================================================== */
function TabBigPicture() {
  var _a = useState(false); var animated = _a[0], setAnimated = _a[1];
  var _h = useState(-1); var hoverIdx = _h[0], setHoverIdx = _h[1];

  useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);

  var addedModules = [
    { name: "LoRA Adapter", color: C.accent, params: "~4M", pct: 0.06 },
    { name: "Prefix Tokens", color: C.blue, params: "~1M", pct: 0.01 },
    { name: "Prompt Vectors", color: C.purple, params: "~0.5M", pct: 0.007 },
    { name: "Adapter Layers", color: C.cyan, params: "~8M", pct: 0.11 },
  ];

  return (
    <div>
      <SectionTitle title="The Big Picture" subtitle={"PEFT Additive: inject small trainable modules — the frozen base never changes"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={1050} height={300} viewBox="0 0 800 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Base model (frozen) */}
          <text x={120} y={22} textAnchor="middle" fill={C.cyan} fontSize={10} fontWeight={700} fontFamily="monospace">BASE MODEL</text>
          <text x={120} y={36} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"(Frozen — no gradient)"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (<g key={i}>
              <rect x={40} y={48 + i * 28} width={160} height={22} rx={4}
                fill={C.dim + "18"} stroke={C.dim + "50"} strokeWidth={1} />
              <text x={65} y={63 + i * 28} fill={C.dim} fontSize={7} fontFamily="monospace">{LOCK + " Layer " + (i+1)}</text>
            </g>);
          })}
          <text x={120} y={278} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"7B params (all frozen)"}</text>

          {/* Lock badge */}
          <rect x={80} y={240} width={80} height={22} rx={6} fill={C.dim + "20"} stroke={C.dim + "40"} strokeWidth={1} />
          <text x={120} y={255} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{LOCK + " FROZEN"}</text>

          {/* Plus sign */}
          <text x={240} y={152} textAnchor="middle" fill={C.accent} fontSize={36} fontWeight={800} fontFamily="monospace">{PLUS}</text>

          {/* Additive modules */}
          <text x={390} y={22} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">ADDITIVE MODULES</text>
          <text x={390} y={36} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">{"(Trainable — tiny!)"}</text>
          {[0,1,2,3].map(function(i) {
            var cols = [C.accent, C.blue, C.purple, C.cyan];
            var names = ["LoRA A+B", "Prefix Embed", "Prompt Vec", "Adapter MLP"];
            var w = animated ? 130 : 0;
            return (<g key={i}>
              <rect x={320} y={55 + i * 50} width={140} height={34} rx={6}
                fill={cols[i] + "10"} stroke={cols[i] + "50"} strokeWidth={1.5}
                style={{ filter: "drop-shadow(0 0 6px " + cols[i] + "30)" }} />
              <rect x={320} y={55 + i * 50} width={w} height={34} rx={6}
                fill={cols[i] + "18"} stroke="none"
                style={{ transition: "width 1s ease-out", transitionDelay: (i * 0.15) + "s" }} />
              <text x={390} y={76 + i * 50} textAnchor="middle" fill={cols[i]} fontSize={9} fontWeight={700} fontFamily="monospace">{names[i]}</text>
            </g>);
          })}
          <text x={390} y={268} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">{"~0.06% of base params"}</text>

          {/* Equals arrow */}
          <line x1={480} y1={148} x2={550} y2={148} stroke={C.green} strokeWidth={2} />
          <polygon points="555,148 548,143 548,153" fill={C.green} />
          <text x={515} y={136} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">COMBINED</text>
          <text x={515} y={162} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">OUTPUT</text>

          {/* Result model */}
          <text x={675} y={22} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">ADAPTED MODEL</text>
          <text x={675} y={36} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">{"(Specialist output)"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (<g key={i}>
              <rect x={580} y={48 + i * 28} width={160} height={22} rx={4}
                fill={C.dim + "18"} stroke={C.dim + "50"} strokeWidth={1} />
              {(i === 2 || i === 5) && <rect x={700} y={50 + i * 28} width={28} height={18} rx={3}
                fill={C.accent + "30"} stroke={C.accent + "70"} strokeWidth={1}
                style={{ transition: "all 0.8s", transitionDelay: (0.6 + i * 0.05) + "s" }} />}
              {(i === 2 || i === 5) && <text x={714} y={63 + i * 28} textAnchor="middle" fill={C.accent} fontSize={6} fontFamily="monospace">{"LoRA"}</text>}
            </g>);
          })}
          <text x={675} y={278} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">{"7B + tiny adapters"}</text>
        </svg>
      </div>

      {/* Trainable params breakdown */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"Trainable Parameters " + DASH + " What " + LQ + "Additive" + RQ + " Actually Means"}</div>
        {addedModules.map(function(m, i) {
          var isH = hoverIdx === i;
          return (
            <div key={i} onMouseEnter={function() { setHoverIdx(i); }} onMouseLeave={function() { setHoverIdx(-1); }}
              style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, cursor: "pointer",
                padding: "6px 10px", borderRadius: 6, background: isH ? m.color + "10" : "transparent", transition: "background 0.2s" }}>
              <div style={{ width: 120, fontSize: 9, color: isH ? m.color : C.muted, fontFamily: "monospace", fontWeight: isH ? 700 : 400 }}>{m.name}</div>
              <div style={{ flex: 1, position: "relative", height: 14 }}>
                <div style={{ width: "100%", height: "100%", borderRadius: 3, background: C.border }} />
                <div style={{ position: "absolute", top: 0, left: 0, width: (m.pct * 900) + "%", height: "100%", borderRadius: 3,
                  background: m.color + (isH ? "70" : "35"), border: "1px solid " + m.color + (isH ? "90" : "40"), transition: "all 0.3s" }} />
              </div>
              <div style={{ width: 55, fontSize: 9, color: isH ? m.color : C.dim, fontFamily: "monospace", textAlign: "right" }}>{m.params}</div>
              <div style={{ width: 60, fontSize: 9, color: isH ? m.color : C.dim, fontFamily: "monospace" }}>{(m.pct).toFixed(3) + "% of 7B"}</div>
            </div>
          );
        })}
        <div style={{ marginTop: 10, fontSize: 9, color: C.muted, textAlign: "center" }}>
          {"Additive PEFT injects " + PLUS + " new params alongside frozen weights. Base model weights never change."}
        </div>
      </Card>

      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"Why Additive PEFT?"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 12, flexWrap: "wrap" }}>
          {[
            { icon: LOCK, title: "Base stays frozen", desc: "No catastrophic forgetting, base knowledge preserved", c: C.blue },
            { icon: PLUG, title: "Plug-and-play", desc: "Swap adapters for different tasks, one base model", c: C.accent },
            { icon: CHART, title: "99%+ param savings", desc: "Train < 1% of parameters vs full fine-tuning", c: C.green },
            { icon: FIRE, title: "Very competitive", desc: "Often matches FFT quality at a fraction of cost", c: C.purple },
          ].map(function(it, i) {
            return (
              <div key={i} style={{ background: it.c + "08", border: "1px solid " + it.c + "25", borderRadius: 8, padding: "12px 14px", textAlign: "center", minWidth: 140, maxWidth: 160 }}>
                <div style={{ fontSize: 20, marginBottom: 4 }}>{it.icon}</div>
                <div style={{ fontSize: 10, color: it.c, fontWeight: 700, marginBottom: 2 }}>{it.title}</div>
                <div style={{ fontSize: 9, color: C.muted }}>{it.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight>
        Think of additive PEFT as <span style={{color:C.accent,fontWeight:700}}>surgical implants</span> rather than brain surgery. The base model is a <span style={{color:C.blue,fontWeight:700}}>frozen library of knowledge</span> {DASH} additive modules are <span style={{color:C.purple,fontWeight:700}}>tiny learned interfaces</span> that redirect that knowledge toward your task. Multiple adapters can coexist for different tasks on the same frozen base.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: LoRA MECHANICS
   =============================================================== */
function TabLoraMechanics() {
  var _r = useState(8); var rank = _r[0], setRank = _r[1];
  var _a = useState(false); var animated = _a[0], setAnimated = _a[1];

  useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);

  var d = 4096; // model dim (example)
  var origParams = d * d;
  var loraParams = d * rank + rank * d;
  var savings = (1 - loraParams / origParams) * 100;
  var compression = (origParams / loraParams).toFixed(0);

  return (
    <div>
      <SectionTitle title="LoRA Mechanics" subtitle={"Low-Rank Adaptation: decompose weight updates into two tiny matrices"} />

      {/* Math diagram */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={1050} height={320} viewBox="0 0 800 320" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* W0 frozen */}
          <text x={110} y={20} textAnchor="middle" fill={C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">W&#8320; (frozen)</text>
          <rect x={30} y={28} width={160} height={200} rx={6} fill={C.dim + "15"} stroke={C.dim + "50"} strokeWidth={1.5} />
          {[0,1,2,3,4,5,6].map(function(i) {
            return [0,1,2,3,4,5,6].map(function(j) {
              return <rect key={i+"_"+j} x={36 + j*21} y={34 + i*27} width={18} height={22} rx={2}
                fill={C.dim + "25"} stroke="none" />;
            });
          })}
          <text x={110} y={248} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"d\u00D7d = 4096\u00D74096"}</text>
          <text x={110} y={262} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " NO gradient"}</text>

          {/* Plus */}
          <text x={212} y={138} textAnchor="middle" fill={C.accent} fontSize={28} fontWeight={800}>{PLUS}</text>

          {/* Delta W = B * A */}
          <text x={420} y={20} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">{"\u0394W = B \u00D7 A  (trainable)"}</text>

          {/* B matrix */}
          <text x={295} y={42} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">B (d\u00D7r)</text>
          <rect x={240} y={50} width={110} height={200} rx={6} fill={C.purple + "12"} stroke={C.purple + "50"} strokeWidth={1.5} />
          {[0,1,2,3,4,5,6].map(function(i) {
            return [0,1,2].map(function(j) {
              var w = animated ? 30 : 0;
              return <rect key={i+"_"+j} x={246 + j*34} y={56 + i*27} width={30} height={22} rx={2}
                fill={C.purple + "30"} stroke={C.purple + "40"} strokeWidth={0.5}
                style={{ transition: "opacity 0.8s", transitionDelay: (i * 0.05) + "s", opacity: animated ? 1 : 0 }} />;
            });
          })}
          <text x={295} y={268} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">{"d\u00D7r = 4096\u00D7" + rank}</text>

          {/* times */}
          <text x={368} y={152} textAnchor="middle" fill={C.accent} fontSize={18} fontWeight={800}>{MUL}</text>

          {/* A matrix */}
          <text x={490} y={42} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">A (r\u00D7d)</text>
          <rect x={388} y={50} width={200} height={60} rx={6} fill={C.cyan + "12"} stroke={C.cyan + "50"} strokeWidth={1.5} />
          {[0,1,2].map(function(i) {
            return [0,1,2,3,4,5].map(function(j) {
              return <rect key={i+"_"+j} x={394 + j*30} y={56 + i*16} width={27} height={12} rx={2}
                fill={C.cyan + "30"} stroke={C.cyan + "40"} strokeWidth={0.5}
                style={{ transition: "opacity 0.8s", transitionDelay: (j * 0.05) + "s", opacity: animated ? 1 : 0 }} />;
            });
          })}
          <text x={490} y={128} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">{"r\u00D7d = " + rank + "\u00D74096"}</text>

          {/* rank label */}
          <line x1={350} y1={155} x2={350} y2={200} stroke={C.yellow + "50"} strokeWidth={1} strokeDasharray="4,3" />
          <line x1={350} y1={200} x2={600} y2={200} stroke={C.yellow + "50"} strokeWidth={1} strokeDasharray="4,3" />
          <text x={475} y={215} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"rank r = " + rank + " (the bottleneck)"}</text>

          {/* = result */}
          <text x={612} y={152} textAnchor="middle" fill={C.green} fontSize={18} fontWeight={800}>{"="}</text>
          <text x={715} y={20} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">{"\u0394W (d\u00D7d)"}</text>
          <rect x={635} y={28} width={145} height={200} rx={6} fill={C.green + "08"} stroke={C.green + "40"} strokeWidth={1.5} />
          {[0,1,2,3,4,5,6].map(function(i) {
            return [0,1,2,3,4].map(function(j) {
              return <rect key={i+"_"+j} x={641 + j*26} y={34 + i*27} width={23} height={22} rx={2}
                fill={C.green + "20"} stroke={C.green + "30"} strokeWidth={0.5}
                style={{ opacity: animated ? 1 : 0, transition: "opacity 1s", transitionDelay: "0.5s" }} />;
            });
          })}
          <text x={715} y={248} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">{"effective d\u00D7d"}</text>
          <text x={715} y={262} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{"from " + (loraParams/1e6).toFixed(2) + "M params!"}</text>
        </svg>
      </div>

      {/* Rank slider */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Rank Exploration " + DASH + " See How Rank Affects Parameter Count"}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
          <div style={{ fontSize: 9, color: C.muted, minWidth: 50 }}>Rank r:</div>
          <input type="range" min={1} max={128} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }}
            style={{ flex: 1, accentColor: C.accent }} />
          <div style={{ fontSize: 18, fontWeight: 800, color: C.accent, fontFamily: "monospace", minWidth: 40 }}>{rank}</div>
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: 30, flexWrap: "wrap" }}>
          <StatBox label="BASE PARAMS (d\u00D7d)" value={"16.8B"} color={C.dim} sub="frozen" />
          <StatBox label="LoRA PARAMS" value={(loraParams/1e6).toFixed(1) + "M"} color={C.accent} sub={"d=" + d + ", r=" + rank} />
          <StatBox label="PARAM SAVINGS" value={savings.toFixed(1) + "%"} color={C.green} />
          <StatBox label="COMPRESSION" value={compression + "x"} color={C.purple} sub="fewer trainable" />
        </div>

        {/* Visual bar */}
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>Trainable vs Frozen (log scale)</div>
          <div style={{ display: "flex", height: 20, borderRadius: 4, overflow: "hidden", border: "1px solid " + C.border }}>
            <div style={{ width: (loraParams / origParams * 100 * 10) + "%", minWidth: 4, background: C.accent + "80", transition: "width 0.4s" }} />
            <div style={{ flex: 1, background: C.dim + "20" }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <span style={{ fontSize: 8, color: C.accent }}>{"LoRA (" + (loraParams/1e6).toFixed(1) + "M)"}</span>
            <span style={{ fontSize: 8, color: C.dim }}>{"Frozen W\u2080 (" + (origParams/1e9).toFixed(1) + "B)"}</span>
          </div>
        </div>

        <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: 8, background: C.yellow + "08", border: "1px solid " + C.yellow + "20" }}>
          <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, marginBottom: 4 }}>{BULB + " Forward Pass:"}</div>
          <div style={{ fontSize: 9, color: C.muted, fontFamily: "monospace" }}>{"output = W\u2080 \u00B7 x + \u03B1/r \u00B7 (B \u00B7 A) \u00B7 x"}</div>
          <div style={{ fontSize: 8, color: C.dim, marginTop: 4 }}>{"Scale factor \u03B1/r controls adapter influence. Common: \u03B1=16, r=" + rank + " \u21D2 scale=" + (16/rank).toFixed(2)}</div>
        </div>
      </Card>

      {/* Init strategy */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Initialization Strategy " + DASH + " Why LoRA Starts at Zero Output"}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { label: "A matrix", init: "Random Gaussian", color: C.cyan, reason: "Breaks symmetry, enables learning" },
            { label: "B matrix", init: "All Zeros", color: C.purple, reason: "Ensures \u0394W=0 at start, no disruption" },
            { label: "Combined BA", init: "Zero output", color: C.green, reason: "Training starts from pretrained baseline" },
          ].map(function(it, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 180, padding: "10px 14px", borderRadius: 8, background: it.color + "06", border: "1px solid " + it.color + "20" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: it.color, marginBottom: 4 }}>{it.label}</div>
                <div style={{ fontSize: 11, fontWeight: 800, color: it.color, marginBottom: 4 }}>{it.init}</div>
                <div style={{ fontSize: 9, color: C.muted, lineHeight: 1.6 }}>{it.reason}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={DNA} title="Why Low-Rank Works">
        Weight update matrices during fine-tuning have an <span style={{color:C.accent,fontWeight:700}}>intrinsically low intrinsic rank</span>. The important changes live in a much smaller subspace than the full d{MUL}d matrix. LoRA exploits this: <span style={{color:C.purple}}>B</span>{MUL}<span style={{color:C.cyan}}>A</span> approximates the full-rank update with far fewer parameters. Rank {rank} captures the essential directions while discarding noise.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: MEMORY & COST
   =============================================================== */
function TabMemory() {
  var _m = useState(7); var modelB = _m[0], setModelB = _m[1];
  var _r = useState(8); var rank = _r[0], setRank = _r[1];

  var fftMemGB = modelB * 4 * 4;  // fp32 weights, grads, optimizer (Adam ~4x)
  var peftMemGB = modelB * 2 + (modelB * 0.001 * rank) * 4 * 4; // frozen bf16 + tiny trainable
  var savings = ((fftMemGB - peftMemGB) / fftMemGB * 100).toFixed(0);
  var trainableM = (modelB * 1e9 * 0.001 * rank / 1e6).toFixed(0);

  return (
    <div>
      <SectionTitle title="Memory & Cost" subtitle={"PEFT Additive: dramatic memory savings because optimizer state only covers tiny adapters"} />

      {/* Controls */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ display: "flex", gap: 30, flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"Model size: " + modelB + "B params"}</div>
            <input type="range" min={1} max={70} value={modelB} onChange={function(e) { setModelB(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.accent }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim }}>
              <span>1B</span><span>7B</span><span>13B</span><span>70B</span>
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"LoRA rank: " + rank}</div>
            <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.purple }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim }}>
              <span>r=1</span><span>r=8</span><span>r=32</span><span>r=64</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Bar comparison */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={300} viewBox="0 0 800 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Grid */}
          {[0,25,50,75,100].map(function(pct) {
            var maxH = 220;
            var y = 250 - pct * maxH / 100;
            return (<g key={pct}>
              <line x1={60} y1={y} x2={740} y2={y} stroke={C.dim + "25"} strokeWidth={0.5} />
              <text x={52} y={y+3} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{pct + "%"}</text>
            </g>);
          })}

          {/* Full FT bar */}
          {[
            { label: "Weights (fp32)", h: 100 * (modelB*4)/(fftMemGB), color: C.dim, y0: 0 },
            { label: "Gradients", h: 100 * (modelB*4)/(fftMemGB), color: C.red, y0: 100 * (modelB*4)/(fftMemGB) },
            { label: "Optimizer (Adam)", h: 100 * (modelB*8)/(fftMemGB), color: C.orange, y0: 200 * (modelB*4)/(fftMemGB) },
          ].map(function(seg, i) {
            var barH = seg.h * 2.2;
            var barY = 250 - (seg.y0 + seg.h) * 2.2;
            return (<g key={i}>
              <rect x={140} y={barY} width={100} height={barH} fill={seg.color + "60"} stroke={seg.color + "80"} strokeWidth={1} />
              <text x={195} y={barY + barH/2 + 3} textAnchor="middle" fill={seg.color} fontSize={7} fontFamily="monospace">{seg.label}</text>
            </g>);
          })}
          <text x={195} y={265} textAnchor="middle" fill={C.red} fontSize={9} fontWeight={700} fontFamily="monospace">Full FT</text>
          <text x={195} y={278} textAnchor="middle" fill={C.red} fontSize={8} fontFamily="monospace">{fftMemGB.toFixed(0) + " GB"}</text>

          {/* PEFT bar */}
          {[
            { label: "Weights (bf16)", h: 100 * (modelB*2)/(fftMemGB), color: C.dim, y0: 0 },
            { label: "Adapter grads", h: 100 * (modelB*0.001*rank*4)/(fftMemGB), color: C.accent, y0: 100*(modelB*2)/(fftMemGB) },
            { label: "Adapter opt", h: 100*(modelB*0.001*rank*8)/(fftMemGB), color: C.purple, y0: 100*(modelB*2+modelB*0.001*rank*4)/(fftMemGB) },
          ].map(function(seg, i) {
            var barH = Math.max(2, seg.h * 2.2);
            var barY = 250 - (seg.y0 + seg.h) * 2.2;
            return (<g key={i}>
              <rect x={340} y={barY} width={100} height={barH} fill={seg.color + "60"} stroke={seg.color + "80"} strokeWidth={1} />
              {barH > 10 && <text x={395} y={barY + barH/2 + 3} textAnchor="middle" fill={seg.color} fontSize={7} fontFamily="monospace">{seg.label}</text>}
            </g>);
          })}
          <text x={395} y={265} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">PEFT Additive</text>
          <text x={395} y={278} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{peftMemGB.toFixed(1) + " GB"}</text>

          {/* Savings arrow */}
          <line x1={450} y1={250 - Math.min(peftMemGB/fftMemGB, 1)*220} x2={600} y2={50} stroke={C.green + "60"} strokeWidth={1} strokeDasharray="4,4" />
          <text x={610} y={46} fill={C.green} fontSize={14} fontWeight={800} fontFamily="monospace">{savings + "%"}</text>
          <text x={610} y={60} fill={C.green} fontSize={8} fontFamily="monospace">memory saved</text>

          {/* Legend */}
          <rect x={560} y={100} width={10} height={10} fill={C.dim + "60"} />
          <text x={575} y={110} fill={C.dim} fontSize={8} fontFamily="monospace">Weights</text>
          <rect x={560} y={118} width={10} height={10} fill={C.red + "60"} />
          <text x={575} y={128} fill={C.red} fontSize={8} fontFamily="monospace">Gradients (FFT)</text>
          <rect x={560} y={136} width={10} height={10} fill={C.accent + "60"} />
          <text x={575} y={146} fill={C.accent} fontSize={8} fontFamily="monospace">Adapter grads</text>
          <rect x={560} y={154} width={10} height={10} fill={C.purple + "60"} />
          <text x={575} y={164} fill={C.purple} fontSize={8} fontFamily="monospace">Adapter optimizer</text>
        </svg>
      </div>

      {/* Stats row */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ display: "flex", justifyContent: "center", gap: 30, flexWrap: "wrap" }}>
          <StatBox label="FFT MEMORY" value={fftMemGB.toFixed(0) + " GB"} color={C.red} />
          <StatBox label="PEFT MEMORY" value={peftMemGB.toFixed(1) + " GB"} color={C.green} />
          <StatBox label="MEMORY SAVED" value={savings + "%"} color={C.accent} />
          <StatBox label="TRAINABLE PARAMS" value={trainableM + "M"} color={C.purple} sub={"of " + modelB + "B total"} />
        </div>
      </Card>

      {/* GPU requirements */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Practical GPU Requirements"}</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[
            { model: "7B", fft: "~112 GB", peft: "~16 GB", hw: "2×A100 or 1×A100 80GB", c: C.green },
            { model: "13B", fft: "~208 GB", peft: "~28 GB", hw: "4×A100 or 1×A100 80GB", c: C.blue },
            { model: "70B", fft: ">1 TB", peft: "~140 GB", hw: "2×A100 80GB", c: C.purple },
          ].map(function(row, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 180, padding: "10px 14px", borderRadius: 8, background: row.c + "06", border: "1px solid " + row.c + "20" }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: row.c, marginBottom: 6 }}>{row.model + "B Model"}</div>
                <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>{"Full FT: "}<span style={{color: C.red}}>{row.fft}</span></div>
                <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"PEFT: "}<span style={{color: C.green}}>{row.peft}</span></div>
                <div style={{ fontSize: 8, color: C.dim, fontStyle: "italic" }}>{row.hw}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={GEAR} title="The Key Insight: Optimizer State is Expensive">
        Adam optimizer stores <span style={{color:C.red,fontWeight:700}}>2 momentum states per parameter</span> ({MUL}8 bytes). Full fine-tuning pays this cost for every weight. PEFT only pays it for <span style={{color:C.accent,fontWeight:700}}>adapter parameters</span> ({DASH}0.1% of weights). The frozen base is stored once in half-precision, no gradients needed. This is why PEFT enables training large models on a single GPU.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: ADAPTER ZOO
   =============================================================== */
function TabAdapterZoo() {
  var _s = useState(0); var sel = _s[0], setSel = _s[1];

  var adapters = [
    {
      name: "LoRA", icon: DNA, color: C.accent,
      year: 2021, params: "0.01–0.1%",
      where: "Attention Q, K, V, O matrices (and optionally FFN)",
      mechanism: "Adds low-rank decomposition B\u00D7A alongside frozen W",
      formula: "h = Wx + (BA)x",
      pros: ["No inference latency (can merge)", "Very wide adoption", "Works across modalities"],
      cons: ["Rank is a hyperparameter", "Full matrices still loaded"],
      best: "General purpose — the go-to choice",
      complexity: 35, effectiveness: 90, flexibility: 85,
    },
    {
      name: "Prefix Tuning", icon: PLUG, color: C.blue,
      year: 2021, params: "0.1–1%",
      where: "Prepended to key/value pairs in ALL attention layers",
      mechanism: "Learns soft prefix vectors prepended to K and V at each layer",
      formula: "Attn(Q, [P_K; K], [P_V; V])",
      pros: ["Strong for generation tasks", "Task identity encoded in prefix", "No architectural change"],
      cons: ["Uses up context window", "Harder to tune", "Less flexible than LoRA"],
      best: "Text generation, summarization",
      complexity: 55, effectiveness: 78, flexibility: 60,
    },
    {
      name: "Prompt Tuning", icon: BRAIN, color: C.purple,
      year: 2021, params: "~0.001%",
      where: "Input embedding layer only — soft token prepend",
      mechanism: "Learns continuous prompt vectors at the input, fixes all model weights",
      formula: "h = LM([P; x])",
      pros: ["Extremely parameter efficient", "Simple to implement", "One model, many prompts"],
      cons: ["Weaker than LoRA for small models", "Only affects input", "Less effective < 10B params"],
      best: "Very large models (>10B), multi-tenant serving",
      complexity: 20, effectiveness: 62, flexibility: 45,
    },
    {
      name: "Adapter Layers", icon: GEAR, color: C.cyan,
      year: 2019, params: "0.5–3%",
      where: "Inserted after attention and FFN sub-layers",
      mechanism: "Small bottleneck MLP (down-project → nonlinear → up-project) with residual",
      formula: "h = h + MLP_adapter(h)",
      pros: ["Proven and well-studied", "Sequential composition", "Interpretable structure"],
      cons: ["Adds inference latency", "Cannot be merged away", "More params than LoRA"],
      best: "When you can afford inference overhead",
      complexity: 50, effectiveness: 82, flexibility: 70,
    },
    {
      name: "LoRA+", icon: STAR, color: C.yellow,
      year: 2024, params: "0.01–0.1%",
      where: "Same as LoRA but with different LR per matrix",
      mechanism: "Uses asymmetric learning rates: higher LR for A, lower for B",
      formula: "h = Wx + (B_\u03B7B \u00B7 A_\u03B7A)x, \u03B7A >> \u03B7B",
      pros: ["Better convergence than LoRA", "Same architecture", "Easy drop-in replacement"],
      cons: ["Two LR hyperparameters", "Marginal gains on some tasks"],
      best: "When squeezing more from LoRA",
      complexity: 38, effectiveness: 93, flexibility: 85,
    },
  ];

  var v = adapters[sel];

  return (
    <div>
      <SectionTitle title="Adapter Zoo" subtitle={"Five major additive PEFT families " + DASH + " where they inject, how they work"} />

      {/* Selector */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 20, flexWrap: "wrap" }}>
        {adapters.map(function(a, i) {
          var on = sel === i;
          return (<button key={i} onClick={function() { setSel(i); }} style={{
            padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? a.color : C.border),
            background: on ? a.color + "20" : C.card, color: on ? a.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{a.icon + " " + a.name}</button>);
        })}
      </div>

      {/* Visual injection diagram */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={220} viewBox="0 0 800 220" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {/* Transformer block */}
          <text x={400} y={18} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">TRANSFORMER BLOCK</text>

          {/* Input */}
          <rect x={50} y={80} width={80} height={50} rx={6} fill={C.dim + "20"} stroke={C.dim + "40"} strokeWidth={1} />
          <text x={90} y={110} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">Input x</text>

          {/* Attention */}
          <rect x={190} y={55} width={120} height={100} rx={6} fill={C.blue + "12"} stroke={C.blue + "40"} strokeWidth={1.5} />
          <text x={250} y={100} textAnchor="middle" fill={C.blue} fontSize={10} fontWeight={700} fontFamily="monospace">Attention</text>
          <text x={250} y={115} textAnchor="middle" fill={C.blue + "80"} fontSize={8} fontFamily="monospace">{LOCK + " frozen"}</text>

          {/* FFN */}
          <rect x={420} y={55} width={120} height={100} rx={6} fill={C.purple + "12"} stroke={C.purple + "40"} strokeWidth={1.5} />
          <text x={480} y={100} textAnchor="middle" fill={C.purple} fontSize={10} fontWeight={700} fontFamily="monospace">FFN</text>
          <text x={480} y={115} textAnchor="middle" fill={C.purple + "80"} fontSize={8} fontFamily="monospace">{LOCK + " frozen"}</text>

          {/* Output */}
          <rect x={660} y={80} width={80} height={50} rx={6} fill={C.green + "12"} stroke={C.green + "40"} strokeWidth={1} />
          <text x={700} y={110} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">Output h</text>

          {/* Arrows */}
          <line x1={130} y1={105} x2={188} y2={105} stroke={C.dim} strokeWidth={1.5} />
          <polygon points="190,105 184,101 184,109" fill={C.dim} />
          <line x1={312} y1={105} x2={418} y2={105} stroke={C.dim} strokeWidth={1.5} />
          <polygon points="420,105 414,101 414,109" fill={C.dim} />
          <line x1={542} y1={105} x2={658} y2={105} stroke={C.dim} strokeWidth={1.5} />
          <polygon points="660,105 654,101 654,109" fill={C.dim} />

          {/* Adapter injections by type */}
          {sel === 0 && <g>
            {/* LoRA inside attention */}
            <rect x={200} y={140} width={100} height={26} rx={4} fill={C.accent + "30"} stroke={C.accent} strokeWidth={1.5} />
            <text x={250} y={157} textAnchor="middle" fill={C.accent} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " LoRA B\u00D7A"}</text>
            <rect x={430} y={140} width={100} height={26} rx={4} fill={C.accent + "20"} stroke={C.accent + "60"} strokeWidth={1} />
            <text x={480} y={157} textAnchor="middle" fill={C.accent + "90"} fontSize={8} fontFamily="monospace">{PLUS + " LoRA B\u00D7A"}</text>
            <text x={400} y={200} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">{"LoRA adds parallel low-rank branches to weight matrices"}</text>
          </g>}
          {sel === 1 && <g>
            {/* Prefix — before attention K,V */}
            <rect x={155} y={30} width={130} height={22} rx={4} fill={C.blue + "30"} stroke={C.blue} strokeWidth={1.5} />
            <text x={220} y={45} textAnchor="middle" fill={C.blue} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " Prefix P_K, P_V"}</text>
            <line x1={220} y1={52} x2={250} y2={55} stroke={C.blue} strokeWidth={1.5} />
            <text x={400} y={200} textAnchor="middle" fill={C.blue} fontSize={9} fontFamily="monospace">{"Prefix tokens prepended to K, V in every attention layer"}</text>
          </g>}
          {sel === 2 && <g>
            {/* Prompt tuning — at input */}
            <rect x={42} y={30} width={100} height={22} rx={4} fill={C.purple + "30"} stroke={C.purple} strokeWidth={1.5} />
            <text x={92} y={45} textAnchor="middle" fill={C.purple} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " Soft Prompt P"}</text>
            <line x1={92} y1={52} x2={90} y2={80} stroke={C.purple} strokeWidth={1.5} />
            <text x={400} y={200} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">{"Soft tokens prepended to input — only embedding layer modified"}</text>
          </g>}
          {sel === 3 && <g>
            {/* Adapter layers — after attention and FFN */}
            <rect x={320} y={80} width={86} height={50} rx={6} fill={C.cyan + "30"} stroke={C.cyan} strokeWidth={1.5} />
            <text x={363} y={102} textAnchor="middle" fill={C.cyan} fontSize={8} fontWeight={700} fontFamily="monospace">Adapter</text>
            <text x={363} y={116} textAnchor="middle" fill={C.cyan} fontSize={7} fontFamily="monospace">bottleneck</text>
            <rect x={552} y={80} width={86} height={50} rx={6} fill={C.cyan + "20"} stroke={C.cyan + "60"} strokeWidth={1.5} />
            <text x={595} y={102} textAnchor="middle" fill={C.cyan} fontSize={8} fontWeight={700} fontFamily="monospace">Adapter</text>
            <text x={595} y={116} textAnchor="middle" fill={C.cyan} fontSize={7} fontFamily="monospace">bottleneck</text>
            <text x={400} y={200} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">{"Sequential bottleneck MLP inserted after each sub-layer"}</text>
          </g>}
          {sel === 4 && <g>
            <rect x={200} y={140} width={100} height={26} rx={4} fill={C.yellow + "30"} stroke={C.yellow} strokeWidth={1.5} />
            <text x={250} y={157} textAnchor="middle" fill={C.yellow} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " LoRA+ B\u00D7A"}</text>
            <text x={400} y={200} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"Same as LoRA with asymmetric learning rates \u03B7_A >> \u03B7_B"}</text>
          </g>}
        </svg>
      </div>

      {/* Detail card */}
      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: v.color }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
          <div style={{ flex: 1, minWidth: 260 }}>
            <div style={{ fontSize: 18, fontWeight: 800, color: v.color }}>{v.name}</div>
            <div style={{ fontSize: 9, color: C.dim, marginBottom: 8 }}>{"introduced " + v.year + " | trainable: " + v.params}</div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, lineHeight: 1.7 }}>{"Where: " + v.where}</div>
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{"How: " + v.mechanism}</div>
            <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 6, background: v.color + "10", border: "1px solid " + v.color + "30", fontFamily: "monospace", fontSize: 10, color: v.color }}>{v.formula}</div>
            <div style={{ marginTop: 8, fontSize: 10, fontStyle: "italic", color: C.muted, borderLeft: "3px solid " + v.color + "40", paddingLeft: 10 }}>{"Best for: " + v.best}</div>
          </div>
          <div style={{ minWidth: 200 }}>
            {[
              { l: "Simplicity", v: 100 - v.complexity, c: C.green },
              { l: "Effectiveness", v: v.effectiveness, c: C.blue },
              { l: "Flexibility", v: v.flexibility, c: C.purple },
            ].map(function(bar, i) {
              return (<div key={i} style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.muted, marginBottom: 3 }}>
                  <span>{bar.l}</span><span style={{color: bar.c}}>{bar.v + "%"}</span>
                </div>
                <div style={{ height: 12, background: C.border, borderRadius: 3 }}>
                  <div style={{ width: bar.v + "%", height: "100%", borderRadius: 3, background: bar.c + "50", border: "1px solid " + bar.c, transition: "width 0.5s" }} />
                </div>
              </div>);
            })}
          </div>
        </div>
        <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 180 }}>
            <div style={{ fontSize: 9, color: C.green, fontWeight: 700, marginBottom: 4 }}>{CHK + " Advantages"}</div>
            {v.pros.map(function(p, i) { return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + CHK + " " + p}</div>); })}
          </div>
          <div style={{ flex: 1, minWidth: 180 }}>
            <div style={{ fontSize: 9, color: C.red, fontWeight: 700, marginBottom: 4 }}>{WARN + " Trade-offs"}</div>
            {v.cons.map(function(p, i) { return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + WARN + " " + p}</div>); })}
          </div>
        </div>
      </Card>

      <Insight icon={TARG} title="Choosing an Adapter">
        <span style={{color:C.accent,fontWeight:700}}>LoRA</span> is the practical default {DASH} no inference overhead, widely supported, great results. <span style={{color:C.blue}}>Prefix tuning</span> shines for generation at scale. <span style={{color:C.purple}}>Prompt tuning</span> is ideal for very large models with many tasks. <span style={{color:C.cyan}}>Adapter layers</span> are the classic choice when interpretability matters. <span style={{color:C.yellow}}>LoRA+</span> gives you marginal gains for free.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: RANK & HYPERPARAMS
   =============================================================== */
function TabRankHyperparams() {
  var _r = useState(8); var rank = _r[0], setRank = _r[1];
  var _al = useState(16); var alpha = _al[0], setAlpha = _al[1];
  var _dr = useState(0); var dropout = _dr[0], setDropout = _dr[1];
  var _tg = useState([true, true, false, false]); var targets = _tg[0], setTargets = _tg[1];

  var scale = alpha / rank;
  var targetNames = ["q_proj", "v_proj", "k_proj", "o_proj"];
  var targetColors = [C.accent, C.purple, C.cyan, C.blue];
  var trainableM = 4 * targets.filter(Boolean).length * 4096 * rank * 2 / 1e6;

  function toggleTarget(i) {
    var t = targets.slice();
    t[i] = !t[i];
    setTargets(t);
  }

  return (
    <div>
      <SectionTitle title="Rank & Hyperparams" subtitle={"Tune rank, alpha, dropout and target modules " + DASH + " see real-time effect"} />

      <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
        {/* Left controls */}
        <Card style={{ flex: 1, minWidth: 300 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"LoRA Config Explorer"}</div>

          {[
            { label: "Rank (r)", val: rank, set: setRank, min: 1, max: 128, color: C.accent, desc: "Higher r = more capacity but more params" },
            { label: "Alpha (\u03B1)", val: alpha, set: setAlpha, min: 1, max: 256, color: C.purple, desc: "Scale factor: actual scale = \u03B1/r" },
            { label: "Dropout %", val: dropout, set: setDropout, min: 0, max: 50, color: C.blue, desc: "Regularization on adapter activations" },
          ].map(function(s, i) {
            return (
              <div key={i} style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={{ fontSize: 9, color: s.color, fontFamily: "monospace", fontWeight: 700 }}>{s.label}</span>
                  <span style={{ fontSize: 16, fontWeight: 800, color: s.color, fontFamily: "monospace" }}>{s.val}</span>
                </div>
                <input type="range" min={s.min} max={s.max} value={s.val}
                  onChange={function(e) { s.set(parseInt(e.target.value)); }}
                  style={{ width: "100%", accentColor: s.color }} />
                <div style={{ fontSize: 8, color: C.dim, marginTop: 3 }}>{s.desc}</div>
              </div>
            );
          })}

          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, marginBottom: 8 }}>{"Target Modules (7B model):"}</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {targetNames.map(function(name, i) {
                var on = targets[i];
                return (<button key={i} onClick={function() { toggleTarget(i); }} style={{
                  padding: "5px 10px", borderRadius: 6, border: "1.5px solid " + (on ? targetColors[i] : C.border),
                  background: on ? targetColors[i] + "20" : C.card, color: on ? targetColors[i] : C.dim,
                  cursor: "pointer", fontSize: 9, fontFamily: "monospace", fontWeight: 700
                }}>{name}</button>);
              })}
            </div>
          </div>
        </Card>

        {/* Right stats */}
        <Card style={{ flex: 1, minWidth: 280 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"Live Config Stats"}</div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
            <StatBox label="RANK" value={rank} color={C.accent} />
            <StatBox label="ALPHA" value={alpha} color={C.purple} />
            <StatBox label="SCALE (\u03B1/r)" value={scale.toFixed(2)} color={scale < 0.5 ? C.red : scale > 2 ? C.green : C.yellow} />
            <StatBox label="TRAINABLE" value={trainableM.toFixed(1) + "M"} color={C.cyan} />
          </div>

          {/* Scale gauge */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>{"Scale (\u03B1/r) — effect strength:"}</div>
            <div style={{ height: 14, background: C.border, borderRadius: 3, position: "relative" }}>
              <div style={{
                position: "absolute", left: 0, height: "100%", borderRadius: 3,
                width: Math.min(100, scale * 25) + "%",
                background: scale < 0.5 ? C.red + "70" : scale > 2 ? C.green + "70" : C.yellow + "70",
                border: "1px solid " + (scale < 0.5 ? C.red : scale > 2 ? C.green : C.yellow),
                transition: "width 0.3s"
              }} />
              <div style={{ position: "absolute", left: "25%", top: 0, bottom: 0, borderLeft: "1px dashed " + C.dim + "60" }} />
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, borderLeft: "1px dashed " + C.dim + "60" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: C.dim, marginTop: 2 }}>
              <span style={{color: C.red}}>Too weak</span>
              <span style={{color: C.yellow}}>Balanced</span>
              <span style={{color: C.green}}>Strong</span>
            </div>
          </div>

          {/* Recommended configs */}
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"Common configs:"}</div>
          {[
            { name: "Conservative", r: 4, a: 8, desc: "Safe default, small models", color: C.blue },
            { name: "Standard", r: 8, a: 16, desc: "Best starting point", color: C.green },
            { name: "High Capacity", r: 16, a: 32, desc: "Complex tasks, more data", color: C.yellow },
            { name: "Max Quality", r: 64, a: 64, desc: "Approaching full FT", color: C.red },
          ].map(function(cfg, i) {
            var active = rank === cfg.r && alpha === cfg.a;
            return (<div key={i} onClick={function() { setRank(cfg.r); setAlpha(cfg.a); }} style={{
              display: "flex", alignItems: "center", gap: 8, padding: "5px 8px", marginBottom: 3,
              borderRadius: 6, cursor: "pointer", background: active ? cfg.color + "12" : "transparent",
              border: "1px solid " + (active ? cfg.color + "40" : "transparent"), transition: "all 0.2s"
            }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: cfg.color }} />
              <div style={{ fontSize: 9, color: active ? cfg.color : C.muted, fontWeight: active ? 700 : 400 }}>{cfg.name}</div>
              <div style={{ fontSize: 8, color: C.dim }}>{"r=" + cfg.r + ", \u03B1=" + cfg.a}</div>
              <div style={{ fontSize: 8, color: C.dim, marginLeft: "auto" }}>{cfg.desc}</div>
            </div>);
          })}
        </Card>
      </div>

      {/* Generated config */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: C.text, marginBottom: 8 }}>{"Generated Config (HuggingFace PEFT)"}</div>
        <div style={{ background: "#06060a", borderRadius: 8, padding: "14px 18px", fontFamily: "monospace", fontSize: 10, lineHeight: 1.8, color: C.muted, border: "1px solid " + C.border }}>
          <div><span style={{color: C.purple}}>{"from"}</span><span style={{color: C.text}}>{" peft "}</span><span style={{color: C.purple}}>{"import"}</span><span style={{color: C.text}}>{" LoraConfig, get_peft_model"}</span></div>
          <br />
          <div><span style={{color: C.cyan}}>{"config"}</span><span style={{color: C.text}}>{" = LoraConfig("}</span></div>
          <div><span style={{color: C.text}}>{"    r="}</span><span style={{color: C.accent}}>{rank}</span><span style={{color: C.dim}}>{",  # rank"}</span></div>
          <div><span style={{color: C.text}}>{"    lora_alpha="}</span><span style={{color: C.accent}}>{alpha}</span><span style={{color: C.dim}}>{",  # scale = " + scale.toFixed(2)}</span></div>
          <div><span style={{color: C.text}}>{"    target_modules=["}</span><span style={{color: C.yellow}}>{'"' + targetNames.filter(function(_, i) { return targets[i]; }).join('", "') + '"'}</span><span style={{color: C.text}}>{"],"}</span></div>
          <div><span style={{color: C.text}}>{"    lora_dropout="}</span><span style={{color: C.accent}}>{(dropout / 100).toFixed(2)}</span><span style={{color: C.text}}>{","}</span></div>
          <div><span style={{color: C.text}}>{'    bias="none",'}</span></div>
          <div><span style={{color: C.text}}>{")"}</span></div>
          <div><span style={{color: C.cyan}}>{"model"}</span><span style={{color: C.text}}>{" = get_peft_model(base_model, config)"}</span></div>
        </div>
      </Card>

      <Insight icon={GEAR} title="Rank Selection Rule of Thumb">
        Start with <span style={{color:C.accent,fontWeight:700}}>r=8, alpha=16</span> (scale=2). For high-resource tasks with lots of data, try <span style={{color:C.yellow}}>r=16 or r=32</span>. Diminishing returns beyond r=64 {DASH} at that point consider full fine-tuning. The alpha/r ratio controls effective adapter strength: <span style={{color:C.green}}>1–2 is a safe range</span>. Always target <span style={{color:C.purple}}>q_proj and v_proj</span> at minimum.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 6: MERGING & DEPLOYMENT
   =============================================================== */
function TabMerging() {
  var _m = useState(0); var mode = _m[0], setMode = _m[1];
  var _a = useState(false); var anim = _a[0], setAnim = _a[1];

  useEffect(function() { var t = setTimeout(function() { setAnim(true); }, 400); return function() { clearTimeout(t); }; }, []);

  var modes = [
    { name: "Merged Inference", color: C.green, latency: "0ms overhead", memory: "= base model", desc: "Fold BA into W permanently: W_final = W0 + (alpha/r)*BA. Zero inference cost, no adapter needed.", icon: GEAR },
    { name: "Multi-Adapter", color: C.blue, latency: "per-adapter", memory: "base + N adapters", desc: "Keep N adapters loaded with one base model. Swap adapters per request. Great for multi-tenant serving.", icon: PLUG },
    { name: "Adapter Composition", color: C.purple, latency: "additive", memory: "base + sum", desc: "Apply multiple adapters sequentially or as weighted sum. Combine task1 + task2 adapters.", icon: DNA },
  ];

  return (
    <div>
      <SectionTitle title="Merging & Deployment" subtitle={"After training: zero-cost merge OR multi-adapter serving " + DASH + " unique PEFT superpower"} />

      {/* Merge animation */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={240} viewBox="0 0 800 240" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {/* Base model */}
          <text x={120} y={18} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">W\u2080 (frozen base)</text>
          {[0,1,2,3,4].map(function(i) {
            return <rect key={i} x={40} y={28+i*30} width={160} height={24} rx={4} fill={C.dim+"18"} stroke={C.dim+"40"} strokeWidth={1} />;
          })}
          <text x={120} y={192} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " frozen"}</text>

          {/* LoRA adapter */}
          <text x={390} y={18} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">LoRA (B\u00D7A)</text>
          <rect x={320} y={70} width={140} height={90} rx={8} fill={C.accent+"15"} stroke={C.accent+"60"} strokeWidth={1.5} />
          <text x={390} y={108} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">B \u00D7 A</text>
          <text x={390} y={124} textAnchor="middle" fill={C.accent+"80"} fontSize={8} fontFamily="monospace">trainable</text>
          <text x={390} y={192} textAnchor="middle" fill={C.accent} fontSize={8} fontFamily="monospace">{"tiny adapter"}</text>

          {/* Arrow */}
          <line x1={470} y1={115} x2={540} y2={115} stroke={C.green} strokeWidth={2} />
          <polygon points="545,115 538,110 538,120" fill={C.green} />
          <text x={507} y={105} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">MERGE</text>
          <text x={507} y={130} textAnchor="middle" fill={C.green+"80"} fontSize={7} fontFamily="monospace">W\u2080 + \u03B1/r\u00B7BA</text>

          {/* Merged model */}
          <text x={680} y={18} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">W\u2080 + \u03B1/r\u00B7BA (merged)</text>
          {[0,1,2,3,4].map(function(i) {
            return (<g key={i}>
              <rect x={560} y={28+i*30} width={160} height={24} rx={4}
                fill={C.green+"12"} stroke={C.green+(anim?"60":"20")} strokeWidth={1}
                style={{ transition: "stroke 1s", transitionDelay: (i*0.1)+"s" }} />
              {anim && <rect x={560} y={28+i*30} width={160} height={24} rx={4}
                fill={C.green+"20"} stroke="none"
                style={{ transition: "opacity 1s", opacity: 1, transitionDelay: (0.3+i*0.1)+"s" }} />}
            </g>);
          })}
          <text x={640} y={192} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{UNLOCK + " no extra cost!"}</text>
          <text x={640} y={208} textAnchor="middle" fill={C.green+"80"} fontSize={7} fontFamily="monospace">identical throughput to base</text>
        </svg>
      </div>

      {/* Mode selector */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16 }}>
        {modes.map(function(m, i) {
          var on = mode === i;
          return (<button key={i} onClick={function() { setMode(i); }} style={{
            padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? m.color : C.border),
            background: on ? m.color + "20" : C.card, color: on ? m.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{m.icon + " " + m.name}</button>);
        })}
      </div>

      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: modes[mode].color }}>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 280 }}>
            <div style={{ fontSize: 16, fontWeight: 800, color: modes[mode].color, marginBottom: 8 }}>{modes[mode].name}</div>
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{modes[mode].desc}</div>
          </div>
          <div style={{ display: "flex", gap: 20 }}>
            <StatBox label="LATENCY OVERHEAD" value={modes[mode].latency} color={modes[mode].color} bigFont={11} minW={110} />
            <StatBox label="MEMORY" value={modes[mode].memory} color={modes[mode].color} bigFont={11} minW={120} />
          </div>
        </div>
      </Card>

      {/* Comparison table */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Deployment Comparison"}</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[
            { label: "Inference Speed", fft: CHK + " Fastest", peft_m: CHK + " Same as base", peft_a: WARN + " +overhead" },
            { label: "Multi-task", fft: WARN + " New model/task", peft_m: CHK + " Swap adapters", peft_a: CHK + " Compose" },
            { label: "Storage", fft: WARN + " Full copy/task", peft_m: CHK + " Base + tiny files", peft_a: CHK + " Minimal" },
            { label: "Update a task", fft: WARN + " Retrain all", peft_m: CHK + " Retrain adapter", peft_a: CHK + " Retrain adapter" },
          ].map(function(row, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 160, padding: "10px 12px", borderRadius: 8, background: C.border + "10", border: "1px solid " + C.border }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: C.text, marginBottom: 6 }}>{row.label}</div>
                <div style={{ fontSize: 8, color: C.red, marginBottom: 2 }}>{"FFT: " + row.fft}</div>
                <div style={{ fontSize: 8, color: C.green, marginBottom: 2 }}>{"PEFT+Merge: " + row.peft_m}</div>
                <div style={{ fontSize: 8, color: C.blue }}>{"PEFT+Adapters: " + row.peft_a}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={STAR} title="The Merge Trick: Best of Both Worlds">
        LoRA's killer feature: <span style={{color:C.green,fontWeight:700}}>W_final = W0 + (alpha/r) * BA</span> can be computed once after training, creating a standard weight matrix with <span style={{color:C.accent,fontWeight:700}}>zero inference overhead</span>. You get all the training efficiency of PEFT with the serving performance of a standard model. And if you want multiple tasks, keep adapters separate and <span style={{color:C.blue}}>hot-swap</span> them at inference time.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Big Picture", "LoRA Mechanics", "Memory & Cost", "Adapter Zoo", "Rank & Hyperparams", "Merging & Deployment"];
  return (
    <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.purple + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>PEFT Additive</div>
        <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " LoRA, Adapters, Prefix Tuning & beyond"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab === 0 && <TabBigPicture />}
      {tab === 1 && <TabLoraMechanics />}
      {tab === 2 && <TabMemory />}
      {tab === 3 && <TabAdapterZoo />}
      {tab === 4 && <TabRankHyperparams />}
      {tab === 5 && <TabMerging />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

PEFT_ADDITIVE_HEIGHT = 1600


# """
# Self-contained HTML for the PEFT Additive interactive walkthrough.
# Covers: Big Picture, LoRA Mechanics, Memory & Cost, Adapter Zoo,
# Rank & Hyperparams, and Merging & Deployment.
# Embed in Streamlit via st.components.v1.html(PEFT_ADDITIVE_HTML, height=PEFT_ADDITIVE_HEIGHT).
# """
#
# PEFT_ADDITIVE_HTML = """
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
#   input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: #1e1e2e; outline: none; }
#   input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #a78bfa; }
#   @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
#   @keyframes flowRight { 0%{transform:translateX(-8px);opacity:0} 50%{opacity:1} 100%{transform:translateX(8px);opacity:0} }
#   @keyframes glow { 0%,100%{box-shadow:0 0 6px rgba(167,139,250,0.3)} 50%{box-shadow:0 0 16px rgba(167,139,250,0.7)} }
# </style>
# </head>
# <body>
# <div id="root"></div>
# <script type="text/babel">
#
# var useState = React.useState;
# var useEffect = React.useEffect;
# var useMemo = React.useMemo;
# var useCallback = React.useCallback;
#
# var C = {
#   bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
#   accent: "#a78bfa", blue: "#4ecdc4", purple: "#c084fc",
#   yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
#   dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
#   cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
# };
#
# var MUL = "\\u00D7";
# var ARR = "\\u2192";
# var DASH = "\\u2014";
# var CHK = "\\u2713";
# var WARN = "\\u26A0";
# var LQ = "\\u201C";
# var RQ = "\\u201D";
# var LARR = "\\u2190";
# var PLAY = "\\u25B6";
# var PAUSE = "\\u23F8";
# var BULB = "\\uD83D\\uDCA1";
# var TARG = "\\uD83C\\uDFAF";
# var LOCK = "\\uD83D\\uDD12";
# var UNLOCK = "\\uD83D\\uDD13";
# var BRAIN = "\\uD83E\\uDDE0";
# var FIRE = "\\uD83D\\uDD25";
# var GEAR = "\\u2699";
# var CHART = "\\uD83D\\uDCC8";
# var PLUS = "\\u002B";
# var DARR = "\\u2193";
# var UARR = "\\u2191";
# var STAR = "\\u2605";
# var PLUG = "\\uD83D\\uDD0C";
# var DNA = "\\uD83E\\uDDEC";
#
# /* ─── Shared components ─── */
#
# function TabBar(props) {
#   var tabs = props.tabs, active = props.active, onChange = props.onChange;
#   return (
#     <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 24, overflowX: "auto" }}>
#       {tabs.map(function(t, i) {
#         return (
#           <button key={i} onClick={function() { onChange(i); }} style={{
#             padding: "12px 18px", background: "none", border: "none",
#             borderBottom: active === i ? "2px solid " + C.accent : "2px solid transparent",
#             color: active === i ? C.accent : C.muted, cursor: "pointer",
#             fontSize: 11, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
#             transition: "all 0.2s", whiteSpace: "nowrap", marginBottom: -2,
#           }}>
#             {t}
#           </button>
#         );
#       })}
#     </div>
#   );
# }
#
# function Card(props) {
#   return (
#     <div style={Object.assign({
#       background: C.card, borderRadius: 10, padding: "18px 22px",
#       border: "1px solid " + (props.highlight ? C.accent : C.border),
#       transition: "border 0.3s",
#     }, props.style || {})}>
#       {props.children}
#     </div>
#   );
# }
#
# function Insight(props) {
#   return (
#     <div style={Object.assign({
#       maxWidth: 1100, margin: "16px auto 0",
#       padding: "16px 22px", background: "rgba(167,139,250,0.06)",
#       borderRadius: 10, border: "1px solid rgba(167,139,250,0.2)",
#     }, props.style || {})}>
#       <div style={{ fontSize: 11, fontWeight: 700, color: C.accent, marginBottom: 6 }}>{(props.icon || BULB) + " " + (props.title || "Key Insight")}</div>
#       <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{props.children}</div>
#     </div>
#   );
# }
#
# function SectionTitle(props) {
#   return (
#     <div style={{ textAlign: "center", marginBottom: 20 }}>
#       <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>{props.title}</div>
#       <div style={{ fontSize: 12, color: C.muted }}>{props.subtitle}</div>
#     </div>
#   );
# }
#
# function StatBox(props) {
#   return (
#     <div style={{ textAlign: "center", minWidth: props.minW || 90 }}>
#       <div style={{ fontSize: 8, color: C.muted, marginBottom: 4, letterSpacing: 1 }}>{props.label}</div>
#       <div style={{ fontSize: props.bigFont || 22, fontWeight: 800, color: props.color || C.accent }}>{props.value}</div>
#       {props.sub && <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{props.sub}</div>}
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 1: THE BIG PICTURE
#    =============================================================== */
# function TabBigPicture() {
#   var _a = useState(false); var animated = _a[0], setAnimated = _a[1];
#   var _h = useState(-1); var hoverIdx = _h[0], setHoverIdx = _h[1];
#
#   useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);
#
#   var addedModules = [
#     { name: "LoRA Adapter", color: C.accent, params: "~4M", pct: 0.06 },
#     { name: "Prefix Tokens", color: C.blue, params: "~1M", pct: 0.01 },
#     { name: "Prompt Vectors", color: C.purple, params: "~0.5M", pct: 0.007 },
#     { name: "Adapter Layers", color: C.cyan, params: "~8M", pct: 0.11 },
#   ];
#
#   return (
#     <div>
#       <SectionTitle title="The Big Picture" subtitle={"PEFT Additive: inject small trainable modules — the frozen base never changes"} />
#
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
#         <svg width={1050} height={300} viewBox="0 0 800 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#
#           {/* Base model (frozen) */}
#           <text x={120} y={22} textAnchor="middle" fill={C.cyan} fontSize={10} fontWeight={700} fontFamily="monospace">BASE MODEL</text>
#           <text x={120} y={36} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"(Frozen — no gradient)"}</text>
#           {[0,1,2,3,4,5,6,7].map(function(i) {
#             return (<g key={i}>
#               <rect x={40} y={48 + i * 28} width={160} height={22} rx={4}
#                 fill={C.dim + "18"} stroke={C.dim + "50"} strokeWidth={1} />
#               <text x={65} y={63 + i * 28} fill={C.dim} fontSize={7} fontFamily="monospace">{LOCK + " Layer " + (i+1)}</text>
#             </g>);
#           })}
#           <text x={120} y={278} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"7B params (all frozen)"}</text>
#
#           {/* Lock badge */}
#           <rect x={80} y={240} width={80} height={22} rx={6} fill={C.dim + "20"} stroke={C.dim + "40"} strokeWidth={1} />
#           <text x={120} y={255} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{LOCK + " FROZEN"}</text>
#
#           {/* Plus sign */}
#           <text x={240} y={152} textAnchor="middle" fill={C.accent} fontSize={36} fontWeight={800} fontFamily="monospace">{PLUS}</text>
#
#           {/* Additive modules */}
#           <text x={390} y={22} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">ADDITIVE MODULES</text>
#           <text x={390} y={36} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">{"(Trainable — tiny!)"}</text>
#           {[0,1,2,3].map(function(i) {
#             var cols = [C.accent, C.blue, C.purple, C.cyan];
#             var names = ["LoRA A+B", "Prefix Embed", "Prompt Vec", "Adapter MLP"];
#             var w = animated ? 130 : 0;
#             return (<g key={i}>
#               <rect x={320} y={55 + i * 50} width={140} height={34} rx={6}
#                 fill={cols[i] + "10"} stroke={cols[i] + "50"} strokeWidth={1.5}
#                 style={{ filter: "drop-shadow(0 0 6px " + cols[i] + "30)" }} />
#               <rect x={320} y={55 + i * 50} width={w} height={34} rx={6}
#                 fill={cols[i] + "18"} stroke="none"
#                 style={{ transition: "width 1s ease-out", transitionDelay: (i * 0.15) + "s" }} />
#               <text x={390} y={76 + i * 50} textAnchor="middle" fill={cols[i]} fontSize={9} fontWeight={700} fontFamily="monospace">{names[i]}</text>
#             </g>);
#           })}
#           <text x={390} y={268} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">{"~0.06% of base params"}</text>
#
#           {/* Equals arrow */}
#           <line x1={480} y1={148} x2={550} y2={148} stroke={C.green} strokeWidth={2} />
#           <polygon points="555,148 548,143 548,153" fill={C.green} />
#           <text x={515} y={136} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">COMBINED</text>
#           <text x={515} y={162} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">OUTPUT</text>
#
#           {/* Result model */}
#           <text x={675} y={22} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">ADAPTED MODEL</text>
#           <text x={675} y={36} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">{"(Specialist output)"}</text>
#           {[0,1,2,3,4,5,6,7].map(function(i) {
#             return (<g key={i}>
#               <rect x={580} y={48 + i * 28} width={160} height={22} rx={4}
#                 fill={C.dim + "18"} stroke={C.dim + "50"} strokeWidth={1} />
#               {(i === 2 || i === 5) && <rect x={700} y={50 + i * 28} width={28} height={18} rx={3}
#                 fill={C.accent + "30"} stroke={C.accent + "70"} strokeWidth={1}
#                 style={{ transition: "all 0.8s", transitionDelay: (0.6 + i * 0.05) + "s" }} />}
#               {(i === 2 || i === 5) && <text x={714} y={63 + i * 28} textAnchor="middle" fill={C.accent} fontSize={6} fontFamily="monospace">{"LoRA"}</text>}
#             </g>);
#           })}
#           <text x={675} y={278} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">{"7B + tiny adapters"}</text>
#         </svg>
#       </div>
#
#       {/* Trainable params breakdown */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"Trainable Parameters " + DASH + " What " + LQ + "Additive" + RQ + " Actually Means"}</div>
#         {addedModules.map(function(m, i) {
#           var isH = hoverIdx === i;
#           return (
#             <div key={i} onMouseEnter={function() { setHoverIdx(i); }} onMouseLeave={function() { setHoverIdx(-1); }}
#               style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, cursor: "pointer",
#                 padding: "6px 10px", borderRadius: 6, background: isH ? m.color + "10" : "transparent", transition: "background 0.2s" }}>
#               <div style={{ width: 120, fontSize: 9, color: isH ? m.color : C.muted, fontFamily: "monospace", fontWeight: isH ? 700 : 400 }}>{m.name}</div>
#               <div style={{ flex: 1, position: "relative", height: 14 }}>
#                 <div style={{ width: "100%", height: "100%", borderRadius: 3, background: C.border }} />
#                 <div style={{ position: "absolute", top: 0, left: 0, width: (m.pct * 900) + "%", height: "100%", borderRadius: 3,
#                   background: m.color + (isH ? "70" : "35"), border: "1px solid " + m.color + (isH ? "90" : "40"), transition: "all 0.3s" }} />
#               </div>
#               <div style={{ width: 55, fontSize: 9, color: isH ? m.color : C.dim, fontFamily: "monospace", textAlign: "right" }}>{m.params}</div>
#               <div style={{ width: 60, fontSize: 9, color: isH ? m.color : C.dim, fontFamily: "monospace" }}>{(m.pct).toFixed(3) + "% of 7B"}</div>
#             </div>
#           );
#         })}
#         <div style={{ marginTop: 10, fontSize: 9, color: C.muted, textAlign: "center" }}>
#           {"Additive PEFT injects " + PLUS + " new params alongside frozen weights. Base model weights never change."}
#         </div>
#       </Card>
#
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"Why Additive PEFT?"}</div>
#         <div style={{ display: "flex", justifyContent: "center", gap: 12, flexWrap: "wrap" }}>
#           {[
#             { icon: LOCK, title: "Base stays frozen", desc: "No catastrophic forgetting, base knowledge preserved", c: C.blue },
#             { icon: PLUG, title: "Plug-and-play", desc: "Swap adapters for different tasks, one base model", c: C.accent },
#             { icon: CHART, title: "99%+ param savings", desc: "Train < 1% of parameters vs full fine-tuning", c: C.green },
#             { icon: FIRE, title: "Very competitive", desc: "Often matches FFT quality at a fraction of cost", c: C.purple },
#           ].map(function(it, i) {
#             return (
#               <div key={i} style={{ background: it.c + "08", border: "1px solid " + it.c + "25", borderRadius: 8, padding: "12px 14px", textAlign: "center", minWidth: 140, maxWidth: 160 }}>
#                 <div style={{ fontSize: 20, marginBottom: 4 }}>{it.icon}</div>
#                 <div style={{ fontSize: 10, color: it.c, fontWeight: 700, marginBottom: 2 }}>{it.title}</div>
#                 <div style={{ fontSize: 9, color: C.muted }}>{it.desc}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight>
#         Think of additive PEFT as <span style={{color:C.accent,fontWeight:700}}>surgical implants</span> rather than brain surgery. The base model is a <span style={{color:C.blue,fontWeight:700}}>frozen library of knowledge</span> {DASH} additive modules are <span style={{color:C.purple,fontWeight:700}}>tiny learned interfaces</span> that redirect that knowledge toward your task. Multiple adapters can coexist for different tasks on the same frozen base.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 2: LoRA MECHANICS
#    =============================================================== */
# function TabLoraMechanics() {
#   var _r = useState(8); var rank = _r[0], setRank = _r[1];
#   var _a = useState(false); var animated = _a[0], setAnimated = _a[1];
#
#   useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);
#
#   var d = 4096; // model dim (example)
#   var origParams = d * d;
#   var loraParams = d * rank + rank * d;
#   var savings = (1 - loraParams / origParams) * 100;
#   var compression = (origParams / loraParams).toFixed(0);
#
#   return (
#     <div>
#       <SectionTitle title="LoRA Mechanics" subtitle={"Low-Rank Adaptation: decompose weight updates into two tiny matrices"} />
#
#       {/* Math diagram */}
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
#         <svg width={1050} height={320} viewBox="0 0 800 320" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#
#           {/* W0 frozen */}
#           <text x={110} y={20} textAnchor="middle" fill={C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">W&#8320; (frozen)</text>
#           <rect x={30} y={28} width={160} height={200} rx={6} fill={C.dim + "15"} stroke={C.dim + "50"} strokeWidth={1.5} />
#           {[0,1,2,3,4,5,6].map(function(i) {
#             return [0,1,2,3,4,5,6].map(function(j) {
#               return <rect key={i+"_"+j} x={36 + j*21} y={34 + i*27} width={18} height={22} rx={2}
#                 fill={C.dim + "25"} stroke="none" />;
#             });
#           })}
#           <text x={110} y={248} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"d\u00D7d = 4096\u00D74096"}</text>
#           <text x={110} y={262} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " NO gradient"}</text>
#
#           {/* Plus */}
#           <text x={212} y={138} textAnchor="middle" fill={C.accent} fontSize={28} fontWeight={800}>{PLUS}</text>
#
#           {/* Delta W = B * A */}
#           <text x={420} y={20} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">{"\u0394W = B \u00D7 A  (trainable)"}</text>
#
#           {/* B matrix */}
#           <text x={295} y={42} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">B (d\u00D7r)</text>
#           <rect x={240} y={50} width={110} height={200} rx={6} fill={C.purple + "12"} stroke={C.purple + "50"} strokeWidth={1.5} />
#           {[0,1,2,3,4,5,6].map(function(i) {
#             return [0,1,2].map(function(j) {
#               var w = animated ? 30 : 0;
#               return <rect key={i+"_"+j} x={246 + j*34} y={56 + i*27} width={30} height={22} rx={2}
#                 fill={C.purple + "30"} stroke={C.purple + "40"} strokeWidth={0.5}
#                 style={{ transition: "opacity 0.8s", transitionDelay: (i * 0.05) + "s", opacity: animated ? 1 : 0 }} />;
#             });
#           })}
#           <text x={295} y={268} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">{"d\u00D7r = 4096\u00D7" + rank}</text>
#
#           {/* times */}
#           <text x={368} y={152} textAnchor="middle" fill={C.accent} fontSize={18} fontWeight={800}>{MUL}</text>
#
#           {/* A matrix */}
#           <text x={490} y={42} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">A (r\u00D7d)</text>
#           <rect x={388} y={50} width={200} height={60} rx={6} fill={C.cyan + "12"} stroke={C.cyan + "50"} strokeWidth={1.5} />
#           {[0,1,2].map(function(i) {
#             return [0,1,2,3,4,5].map(function(j) {
#               return <rect key={i+"_"+j} x={394 + j*30} y={56 + i*16} width={27} height={12} rx={2}
#                 fill={C.cyan + "30"} stroke={C.cyan + "40"} strokeWidth={0.5}
#                 style={{ transition: "opacity 0.8s", transitionDelay: (j * 0.05) + "s", opacity: animated ? 1 : 0 }} />;
#             });
#           })}
#           <text x={490} y={128} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">{"r\u00D7d = " + rank + "\u00D74096"}</text>
#
#           {/* rank label */}
#           <line x1={350} y1={155} x2={350} y2={200} stroke={C.yellow + "50"} strokeWidth={1} strokeDasharray="4,3" />
#           <line x1={350} y1={200} x2={600} y2={200} stroke={C.yellow + "50"} strokeWidth={1} strokeDasharray="4,3" />
#           <text x={475} y={215} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"rank r = " + rank + " (the bottleneck)"}</text>
#
#           {/* = result */}
#           <text x={612} y={152} textAnchor="middle" fill={C.green} fontSize={18} fontWeight={800}>{"="}</text>
#           <text x={715} y={20} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">{"\u0394W (d\u00D7d)"}</text>
#           <rect x={635} y={28} width={145} height={200} rx={6} fill={C.green + "08"} stroke={C.green + "40"} strokeWidth={1.5} />
#           {[0,1,2,3,4,5,6].map(function(i) {
#             return [0,1,2,3,4].map(function(j) {
#               return <rect key={i+"_"+j} x={641 + j*26} y={34 + i*27} width={23} height={22} rx={2}
#                 fill={C.green + "20"} stroke={C.green + "30"} strokeWidth={0.5}
#                 style={{ opacity: animated ? 1 : 0, transition: "opacity 1s", transitionDelay: "0.5s" }} />;
#             });
#           })}
#           <text x={715} y={248} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">{"effective d\u00D7d"}</text>
#           <text x={715} y={262} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{"from " + (loraParams/1e6).toFixed(2) + "M params!"}</text>
#         </svg>
#       </div>
#
#       {/* Rank slider */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Rank Exploration " + DASH + " See How Rank Affects Parameter Count"}</div>
#         <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
#           <div style={{ fontSize: 9, color: C.muted, minWidth: 50 }}>Rank r:</div>
#           <input type="range" min={1} max={128} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }}
#             style={{ flex: 1, accentColor: C.accent }} />
#           <div style={{ fontSize: 18, fontWeight: 800, color: C.accent, fontFamily: "monospace", minWidth: 40 }}>{rank}</div>
#         </div>
#         <div style={{ display: "flex", justifyContent: "center", gap: 30, flexWrap: "wrap" }}>
#           <StatBox label="BASE PARAMS (d\u00D7d)" value={"16.8B"} color={C.dim} sub="frozen" />
#           <StatBox label="LoRA PARAMS" value={(loraParams/1e6).toFixed(1) + "M"} color={C.accent} sub={"d=" + d + ", r=" + rank} />
#           <StatBox label="PARAM SAVINGS" value={savings.toFixed(1) + "%"} color={C.green} />
#           <StatBox label="COMPRESSION" value={compression + "x"} color={C.purple} sub="fewer trainable" />
#         </div>
#
#         {/* Visual bar */}
#         <div style={{ marginTop: 16 }}>
#           <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>Trainable vs Frozen (log scale)</div>
#           <div style={{ display: "flex", height: 20, borderRadius: 4, overflow: "hidden", border: "1px solid " + C.border }}>
#             <div style={{ width: (loraParams / origParams * 100 * 10) + "%", minWidth: 4, background: C.accent + "80", transition: "width 0.4s" }} />
#             <div style={{ flex: 1, background: C.dim + "20" }} />
#           </div>
#           <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
#             <span style={{ fontSize: 8, color: C.accent }}>{"LoRA (" + (loraParams/1e6).toFixed(1) + "M)"}</span>
#             <span style={{ fontSize: 8, color: C.dim }}>{"Frozen W\u2080 (" + (origParams/1e9).toFixed(1) + "B)"}</span>
#           </div>
#         </div>
#
#         <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: 8, background: C.yellow + "08", border: "1px solid " + C.yellow + "20" }}>
#           <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, marginBottom: 4 }}>{BULB + " Forward Pass:"}</div>
#           <div style={{ fontSize: 9, color: C.muted, fontFamily: "monospace" }}>{"output = W\u2080 \u00B7 x + \u03B1/r \u00B7 (B \u00B7 A) \u00B7 x"}</div>
#           <div style={{ fontSize: 8, color: C.dim, marginTop: 4 }}>{"Scale factor \u03B1/r controls adapter influence. Common: \u03B1=16, r=" + rank + " \u21D2 scale=" + (16/rank).toFixed(2)}</div>
#         </div>
#       </Card>
#
#       {/* Init strategy */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Initialization Strategy " + DASH + " Why LoRA Starts at Zero Output"}</div>
#         <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
#           {[
#             { label: "A matrix", init: "Random Gaussian", color: C.cyan, reason: "Breaks symmetry, enables learning" },
#             { label: "B matrix", init: "All Zeros", color: C.purple, reason: "Ensures \u0394W=0 at start, no disruption" },
#             { label: "Combined BA", init: "Zero output", color: C.green, reason: "Training starts from pretrained baseline" },
#           ].map(function(it, i) {
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 180, padding: "10px 14px", borderRadius: 8, background: it.color + "06", border: "1px solid " + it.color + "20" }}>
#                 <div style={{ fontSize: 10, fontWeight: 700, color: it.color, marginBottom: 4 }}>{it.label}</div>
#                 <div style={{ fontSize: 11, fontWeight: 800, color: it.color, marginBottom: 4 }}>{it.init}</div>
#                 <div style={{ fontSize: 9, color: C.muted, lineHeight: 1.6 }}>{it.reason}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={DNA} title="Why Low-Rank Works">
#         Weight update matrices during fine-tuning have an <span style={{color:C.accent,fontWeight:700}}>intrinsically low intrinsic rank</span>. The important changes live in a much smaller subspace than the full d{MUL}d matrix. LoRA exploits this: <span style={{color:C.purple}}>B</span>{MUL}<span style={{color:C.cyan}}>A</span> approximates the full-rank update with far fewer parameters. Rank {rank} captures the essential directions while discarding noise.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 3: MEMORY & COST
#    =============================================================== */
# function TabMemory() {
#   var _m = useState(7); var modelB = _m[0], setModelB = _m[1];
#   var _r = useState(8); var rank = _r[0], setRank = _r[1];
#
#   var fftMemGB = modelB * 4 * 4;  // fp32 weights, grads, optimizer (Adam ~4x)
#   var peftMemGB = modelB * 2 + (modelB * 0.001 * rank) * 4 * 4; // frozen bf16 + tiny trainable
#   var savings = ((fftMemGB - peftMemGB) / fftMemGB * 100).toFixed(0);
#   var trainableM = (modelB * 1e9 * 0.001 * rank / 1e6).toFixed(0);
#
#   return (
#     <div>
#       <SectionTitle title="Memory & Cost" subtitle={"PEFT Additive: dramatic memory savings because optimizer state only covers tiny adapters"} />
#
#       {/* Controls */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ display: "flex", gap: 30, flexWrap: "wrap", alignItems: "center" }}>
#           <div style={{ flex: 1, minWidth: 200 }}>
#             <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"Model size: " + modelB + "B params"}</div>
#             <input type="range" min={1} max={70} value={modelB} onChange={function(e) { setModelB(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.accent }} />
#             <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim }}>
#               <span>1B</span><span>7B</span><span>13B</span><span>70B</span>
#             </div>
#           </div>
#           <div style={{ flex: 1, minWidth: 200 }}>
#             <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"LoRA rank: " + rank}</div>
#             <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.purple }} />
#             <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim }}>
#               <span>r=1</span><span>r=8</span><span>r=32</span><span>r=64</span>
#             </div>
#           </div>
#         </div>
#       </Card>
#
#       {/* Bar comparison */}
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#         <svg width={1050} height={300} viewBox="0 0 800 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#
#           {/* Grid */}
#           {[0,25,50,75,100].map(function(pct) {
#             var maxH = 220;
#             var y = 250 - pct * maxH / 100;
#             return (<g key={pct}>
#               <line x1={60} y1={y} x2={740} y2={y} stroke={C.dim + "25"} strokeWidth={0.5} />
#               <text x={52} y={y+3} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{pct + "%"}</text>
#             </g>);
#           })}
#
#           {/* Full FT bar */}
#           {[
#             { label: "Weights (fp32)", h: 100 * (modelB*4)/(fftMemGB), color: C.dim, y0: 0 },
#             { label: "Gradients", h: 100 * (modelB*4)/(fftMemGB), color: C.red, y0: 100 * (modelB*4)/(fftMemGB) },
#             { label: "Optimizer (Adam)", h: 100 * (modelB*8)/(fftMemGB), color: C.orange, y0: 200 * (modelB*4)/(fftMemGB) },
#           ].map(function(seg, i) {
#             var barH = seg.h * 2.2;
#             var barY = 250 - (seg.y0 + seg.h) * 2.2;
#             return (<g key={i}>
#               <rect x={140} y={barY} width={100} height={barH} fill={seg.color + "60"} stroke={seg.color + "80"} strokeWidth={1} />
#               <text x={195} y={barY + barH/2 + 3} textAnchor="middle" fill={seg.color} fontSize={7} fontFamily="monospace">{seg.label}</text>
#             </g>);
#           })}
#           <text x={195} y={265} textAnchor="middle" fill={C.red} fontSize={9} fontWeight={700} fontFamily="monospace">Full FT</text>
#           <text x={195} y={278} textAnchor="middle" fill={C.red} fontSize={8} fontFamily="monospace">{fftMemGB.toFixed(0) + " GB"}</text>
#
#           {/* PEFT bar */}
#           {[
#             { label: "Weights (bf16)", h: 100 * (modelB*2)/(fftMemGB), color: C.dim, y0: 0 },
#             { label: "Adapter grads", h: 100 * (modelB*0.001*rank*4)/(fftMemGB), color: C.accent, y0: 100*(modelB*2)/(fftMemGB) },
#             { label: "Adapter opt", h: 100*(modelB*0.001*rank*8)/(fftMemGB), color: C.purple, y0: 100*(modelB*2+modelB*0.001*rank*4)/(fftMemGB) },
#           ].map(function(seg, i) {
#             var barH = Math.max(2, seg.h * 2.2);
#             var barY = 250 - (seg.y0 + seg.h) * 2.2;
#             return (<g key={i}>
#               <rect x={340} y={barY} width={100} height={barH} fill={seg.color + "60"} stroke={seg.color + "80"} strokeWidth={1} />
#               {barH > 10 && <text x={395} y={barY + barH/2 + 3} textAnchor="middle" fill={seg.color} fontSize={7} fontFamily="monospace">{seg.label}</text>}
#             </g>);
#           })}
#           <text x={395} y={265} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">PEFT Additive</text>
#           <text x={395} y={278} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{peftMemGB.toFixed(1) + " GB"}</text>
#
#           {/* Savings arrow */}
#           <line x1={450} y1={250 - Math.min(peftMemGB/fftMemGB, 1)*220} x2={600} y2={50} stroke={C.green + "60"} strokeWidth={1} strokeDasharray="4,4" />
#           <text x={610} y={46} fill={C.green} fontSize={14} fontWeight={800} fontFamily="monospace">{savings + "%"}</text>
#           <text x={610} y={60} fill={C.green} fontSize={8} fontFamily="monospace">memory saved</text>
#
#           {/* Legend */}
#           <rect x={560} y={100} width={10} height={10} fill={C.dim + "60"} />
#           <text x={575} y={110} fill={C.dim} fontSize={8} fontFamily="monospace">Weights</text>
#           <rect x={560} y={118} width={10} height={10} fill={C.red + "60"} />
#           <text x={575} y={128} fill={C.red} fontSize={8} fontFamily="monospace">Gradients (FFT)</text>
#           <rect x={560} y={136} width={10} height={10} fill={C.accent + "60"} />
#           <text x={575} y={146} fill={C.accent} fontSize={8} fontFamily="monospace">Adapter grads</text>
#           <rect x={560} y={154} width={10} height={10} fill={C.purple + "60"} />
#           <text x={575} y={164} fill={C.purple} fontSize={8} fontFamily="monospace">Adapter optimizer</text>
#         </svg>
#       </div>
#
#       {/* Stats row */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ display: "flex", justifyContent: "center", gap: 30, flexWrap: "wrap" }}>
#           <StatBox label="FFT MEMORY" value={fftMemGB.toFixed(0) + " GB"} color={C.red} />
#           <StatBox label="PEFT MEMORY" value={peftMemGB.toFixed(1) + " GB"} color={C.green} />
#           <StatBox label="MEMORY SAVED" value={savings + "%"} color={C.accent} />
#           <StatBox label="TRAINABLE PARAMS" value={trainableM + "M"} color={C.purple} sub={"of " + modelB + "B total"} />
#         </div>
#       </Card>
#
#       {/* GPU requirements */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Practical GPU Requirements"}</div>
#         <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
#           {[
#             { model: "7B", fft: "~112 GB", peft: "~16 GB", hw: "2×A100 or 1×A100 80GB", c: C.green },
#             { model: "13B", fft: "~208 GB", peft: "~28 GB", hw: "4×A100 or 1×A100 80GB", c: C.blue },
#             { model: "70B", fft: ">1 TB", peft: "~140 GB", hw: "2×A100 80GB", c: C.purple },
#           ].map(function(row, i) {
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 180, padding: "10px 14px", borderRadius: 8, background: row.c + "06", border: "1px solid " + row.c + "20" }}>
#                 <div style={{ fontSize: 11, fontWeight: 800, color: row.c, marginBottom: 6 }}>{row.model + "B Model"}</div>
#                 <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>{"Full FT: "}<span style={{color: C.red}}>{row.fft}</span></div>
#                 <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"PEFT: "}<span style={{color: C.green}}>{row.peft}</span></div>
#                 <div style={{ fontSize: 8, color: C.dim, fontStyle: "italic" }}>{row.hw}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={GEAR} title="The Key Insight: Optimizer State is Expensive">
#         Adam optimizer stores <span style={{color:C.red,fontWeight:700}}>2 momentum states per parameter</span> ({MUL}8 bytes). Full fine-tuning pays this cost for every weight. PEFT only pays it for <span style={{color:C.accent,fontWeight:700}}>adapter parameters</span> ({DASH}0.1% of weights). The frozen base is stored once in half-precision, no gradients needed. This is why PEFT enables training large models on a single GPU.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 4: ADAPTER ZOO
#    =============================================================== */
# function TabAdapterZoo() {
#   var _s = useState(0); var sel = _s[0], setSel = _s[1];
#
#   var adapters = [
#     {
#       name: "LoRA", icon: DNA, color: C.accent,
#       year: 2021, params: "0.01–0.1%",
#       where: "Attention Q, K, V, O matrices (and optionally FFN)",
#       mechanism: "Adds low-rank decomposition B\u00D7A alongside frozen W",
#       formula: "h = Wx + (BA)x",
#       pros: ["No inference latency (can merge)", "Very wide adoption", "Works across modalities"],
#       cons: ["Rank is a hyperparameter", "Full matrices still loaded"],
#       best: "General purpose — the go-to choice",
#       complexity: 35, effectiveness: 90, flexibility: 85,
#     },
#     {
#       name: "Prefix Tuning", icon: PLUG, color: C.blue,
#       year: 2021, params: "0.1–1%",
#       where: "Prepended to key/value pairs in ALL attention layers",
#       mechanism: "Learns soft prefix vectors prepended to K and V at each layer",
#       formula: "Attn(Q, [P_K; K], [P_V; V])",
#       pros: ["Strong for generation tasks", "Task identity encoded in prefix", "No architectural change"],
#       cons: ["Uses up context window", "Harder to tune", "Less flexible than LoRA"],
#       best: "Text generation, summarization",
#       complexity: 55, effectiveness: 78, flexibility: 60,
#     },
#     {
#       name: "Prompt Tuning", icon: BRAIN, color: C.purple,
#       year: 2021, params: "~0.001%",
#       where: "Input embedding layer only — soft token prepend",
#       mechanism: "Learns continuous prompt vectors at the input, fixes all model weights",
#       formula: "h = LM([P; x])",
#       pros: ["Extremely parameter efficient", "Simple to implement", "One model, many prompts"],
#       cons: ["Weaker than LoRA for small models", "Only affects input", "Less effective < 10B params"],
#       best: "Very large models (>10B), multi-tenant serving",
#       complexity: 20, effectiveness: 62, flexibility: 45,
#     },
#     {
#       name: "Adapter Layers", icon: GEAR, color: C.cyan,
#       year: 2019, params: "0.5–3%",
#       where: "Inserted after attention and FFN sub-layers",
#       mechanism: "Small bottleneck MLP (down-project → nonlinear → up-project) with residual",
#       formula: "h = h + MLP_adapter(h)",
#       pros: ["Proven and well-studied", "Sequential composition", "Interpretable structure"],
#       cons: ["Adds inference latency", "Cannot be merged away", "More params than LoRA"],
#       best: "When you can afford inference overhead",
#       complexity: 50, effectiveness: 82, flexibility: 70,
#     },
#     {
#       name: "LoRA+", icon: STAR, color: C.yellow,
#       year: 2024, params: "0.01–0.1%",
#       where: "Same as LoRA but with different LR per matrix",
#       mechanism: "Uses asymmetric learning rates: higher LR for A, lower for B",
#       formula: "h = Wx + (B_\u03B7B \u00B7 A_\u03B7A)x, \u03B7A >> \u03B7B",
#       pros: ["Better convergence than LoRA", "Same architecture", "Easy drop-in replacement"],
#       cons: ["Two LR hyperparameters", "Marginal gains on some tasks"],
#       best: "When squeezing more from LoRA",
#       complexity: 38, effectiveness: 93, flexibility: 85,
#     },
#   ];
#
#   var v = adapters[sel];
#
#   return (
#     <div>
#       <SectionTitle title="Adapter Zoo" subtitle={"Five major additive PEFT families " + DASH + " where they inject, how they work"} />
#
#       {/* Selector */}
#       <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 20, flexWrap: "wrap" }}>
#         {adapters.map(function(a, i) {
#           var on = sel === i;
#           return (<button key={i} onClick={function() { setSel(i); }} style={{
#             padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? a.color : C.border),
#             background: on ? a.color + "20" : C.card, color: on ? a.color : C.muted,
#             cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
#           }}>{a.icon + " " + a.name}</button>);
#         })}
#       </div>
#
#       {/* Visual injection diagram */}
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#         <svg width={1050} height={220} viewBox="0 0 800 220" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#           {/* Transformer block */}
#           <text x={400} y={18} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">TRANSFORMER BLOCK</text>
#
#           {/* Input */}
#           <rect x={50} y={80} width={80} height={50} rx={6} fill={C.dim + "20"} stroke={C.dim + "40"} strokeWidth={1} />
#           <text x={90} y={110} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">Input x</text>
#
#           {/* Attention */}
#           <rect x={190} y={55} width={120} height={100} rx={6} fill={C.blue + "12"} stroke={C.blue + "40"} strokeWidth={1.5} />
#           <text x={250} y={100} textAnchor="middle" fill={C.blue} fontSize={10} fontWeight={700} fontFamily="monospace">Attention</text>
#           <text x={250} y={115} textAnchor="middle" fill={C.blue + "80"} fontSize={8} fontFamily="monospace">{LOCK + " frozen"}</text>
#
#           {/* FFN */}
#           <rect x={420} y={55} width={120} height={100} rx={6} fill={C.purple + "12"} stroke={C.purple + "40"} strokeWidth={1.5} />
#           <text x={480} y={100} textAnchor="middle" fill={C.purple} fontSize={10} fontWeight={700} fontFamily="monospace">FFN</text>
#           <text x={480} y={115} textAnchor="middle" fill={C.purple + "80"} fontSize={8} fontFamily="monospace">{LOCK + " frozen"}</text>
#
#           {/* Output */}
#           <rect x={660} y={80} width={80} height={50} rx={6} fill={C.green + "12"} stroke={C.green + "40"} strokeWidth={1} />
#           <text x={700} y={110} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">Output h</text>
#
#           {/* Arrows */}
#           <line x1={130} y1={105} x2={188} y2={105} stroke={C.dim} strokeWidth={1.5} />
#           <polygon points="190,105 184,101 184,109" fill={C.dim} />
#           <line x1={312} y1={105} x2={418} y2={105} stroke={C.dim} strokeWidth={1.5} />
#           <polygon points="420,105 414,101 414,109" fill={C.dim} />
#           <line x1={542} y1={105} x2={658} y2={105} stroke={C.dim} strokeWidth={1.5} />
#           <polygon points="660,105 654,101 654,109" fill={C.dim} />
#
#           {/* Adapter injections by type */}
#           {sel === 0 && <g>
#             {/* LoRA inside attention */}
#             <rect x={200} y={140} width={100} height={26} rx={4} fill={C.accent + "30"} stroke={C.accent} strokeWidth={1.5} />
#             <text x={250} y={157} textAnchor="middle" fill={C.accent} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " LoRA B\u00D7A"}</text>
#             <rect x={430} y={140} width={100} height={26} rx={4} fill={C.accent + "20"} stroke={C.accent + "60"} strokeWidth={1} />
#             <text x={480} y={157} textAnchor="middle" fill={C.accent + "90"} fontSize={8} fontFamily="monospace">{PLUS + " LoRA B\u00D7A"}</text>
#             <text x={400} y={200} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">{"LoRA adds parallel low-rank branches to weight matrices"}</text>
#           </g>}
#           {sel === 1 && <g>
#             {/* Prefix — before attention K,V */}
#             <rect x={155} y={30} width={130} height={22} rx={4} fill={C.blue + "30"} stroke={C.blue} strokeWidth={1.5} />
#             <text x={220} y={45} textAnchor="middle" fill={C.blue} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " Prefix P_K, P_V"}</text>
#             <line x1={220} y1={52} x2={250} y2={55} stroke={C.blue} strokeWidth={1.5} />
#             <text x={400} y={200} textAnchor="middle" fill={C.blue} fontSize={9} fontFamily="monospace">{"Prefix tokens prepended to K, V in every attention layer"}</text>
#           </g>}
#           {sel === 2 && <g>
#             {/* Prompt tuning — at input */}
#             <rect x={42} y={30} width={100} height={22} rx={4} fill={C.purple + "30"} stroke={C.purple} strokeWidth={1.5} />
#             <text x={92} y={45} textAnchor="middle" fill={C.purple} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " Soft Prompt P"}</text>
#             <line x1={92} y1={52} x2={90} y2={80} stroke={C.purple} strokeWidth={1.5} />
#             <text x={400} y={200} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">{"Soft tokens prepended to input — only embedding layer modified"}</text>
#           </g>}
#           {sel === 3 && <g>
#             {/* Adapter layers — after attention and FFN */}
#             <rect x={320} y={80} width={86} height={50} rx={6} fill={C.cyan + "30"} stroke={C.cyan} strokeWidth={1.5} />
#             <text x={363} y={102} textAnchor="middle" fill={C.cyan} fontSize={8} fontWeight={700} fontFamily="monospace">Adapter</text>
#             <text x={363} y={116} textAnchor="middle" fill={C.cyan} fontSize={7} fontFamily="monospace">bottleneck</text>
#             <rect x={552} y={80} width={86} height={50} rx={6} fill={C.cyan + "20"} stroke={C.cyan + "60"} strokeWidth={1.5} />
#             <text x={595} y={102} textAnchor="middle" fill={C.cyan} fontSize={8} fontWeight={700} fontFamily="monospace">Adapter</text>
#             <text x={595} y={116} textAnchor="middle" fill={C.cyan} fontSize={7} fontFamily="monospace">bottleneck</text>
#             <text x={400} y={200} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">{"Sequential bottleneck MLP inserted after each sub-layer"}</text>
#           </g>}
#           {sel === 4 && <g>
#             <rect x={200} y={140} width={100} height={26} rx={4} fill={C.yellow + "30"} stroke={C.yellow} strokeWidth={1.5} />
#             <text x={250} y={157} textAnchor="middle" fill={C.yellow} fontSize={8} fontWeight={700} fontFamily="monospace">{PLUS + " LoRA+ B\u00D7A"}</text>
#             <text x={400} y={200} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"Same as LoRA with asymmetric learning rates \u03B7_A >> \u03B7_B"}</text>
#           </g>}
#         </svg>
#       </div>
#
#       {/* Detail card */}
#       <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: v.color }}>
#         <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
#           <div style={{ flex: 1, minWidth: 260 }}>
#             <div style={{ fontSize: 18, fontWeight: 800, color: v.color }}>{v.name}</div>
#             <div style={{ fontSize: 9, color: C.dim, marginBottom: 8 }}>{"introduced " + v.year + " | trainable: " + v.params}</div>
#             <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, lineHeight: 1.7 }}>{"Where: " + v.where}</div>
#             <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{"How: " + v.mechanism}</div>
#             <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 6, background: v.color + "10", border: "1px solid " + v.color + "30", fontFamily: "monospace", fontSize: 10, color: v.color }}>{v.formula}</div>
#             <div style={{ marginTop: 8, fontSize: 10, fontStyle: "italic", color: C.muted, borderLeft: "3px solid " + v.color + "40", paddingLeft: 10 }}>{"Best for: " + v.best}</div>
#           </div>
#           <div style={{ minWidth: 200 }}>
#             {[
#               { l: "Simplicity", v: 100 - v.complexity, c: C.green },
#               { l: "Effectiveness", v: v.effectiveness, c: C.blue },
#               { l: "Flexibility", v: v.flexibility, c: C.purple },
#             ].map(function(bar, i) {
#               return (<div key={i} style={{ marginBottom: 8 }}>
#                 <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.muted, marginBottom: 3 }}>
#                   <span>{bar.l}</span><span style={{color: bar.c}}>{bar.v + "%"}</span>
#                 </div>
#                 <div style={{ height: 12, background: C.border, borderRadius: 3 }}>
#                   <div style={{ width: bar.v + "%", height: "100%", borderRadius: 3, background: bar.c + "50", border: "1px solid " + bar.c, transition: "width 0.5s" }} />
#                 </div>
#               </div>);
#             })}
#           </div>
#         </div>
#         <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
#           <div style={{ flex: 1, minWidth: 180 }}>
#             <div style={{ fontSize: 9, color: C.green, fontWeight: 700, marginBottom: 4 }}>{CHK + " Advantages"}</div>
#             {v.pros.map(function(p, i) { return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + CHK + " " + p}</div>); })}
#           </div>
#           <div style={{ flex: 1, minWidth: 180 }}>
#             <div style={{ fontSize: 9, color: C.red, fontWeight: 700, marginBottom: 4 }}>{WARN + " Trade-offs"}</div>
#             {v.cons.map(function(p, i) { return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + WARN + " " + p}</div>); })}
#           </div>
#         </div>
#       </Card>
#
#       <Insight icon={TARG} title="Choosing an Adapter">
#         <span style={{color:C.accent,fontWeight:700}}>LoRA</span> is the practical default {DASH} no inference overhead, widely supported, great results. <span style={{color:C.blue}}>Prefix tuning</span> shines for generation at scale. <span style={{color:C.purple}}>Prompt tuning</span> is ideal for very large models with many tasks. <span style={{color:C.cyan}}>Adapter layers</span> are the classic choice when interpretability matters. <span style={{color:C.yellow}}>LoRA+</span> gives you marginal gains for free.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 5: RANK & HYPERPARAMS
#    =============================================================== */
# function TabRankHyperparams() {
#   var _r = useState(8); var rank = _r[0], setRank = _r[1];
#   var _al = useState(16); var alpha = _al[0], setAlpha = _al[1];
#   var _dr = useState(0); var dropout = _dr[0], setDropout = _dr[1];
#   var _tg = useState([true, true, false, false]); var targets = _tg[0], setTargets = _tg[1];
#
#   var scale = alpha / rank;
#   var targetNames = ["q_proj", "v_proj", "k_proj", "o_proj"];
#   var targetColors = [C.accent, C.purple, C.cyan, C.blue];
#   var trainableM = 4 * targets.filter(Boolean).length * 4096 * rank * 2 / 1e6;
#
#   function toggleTarget(i) {
#     var t = targets.slice();
#     t[i] = !t[i];
#     setTargets(t);
#   }
#
#   return (
#     <div>
#       <SectionTitle title="Rank & Hyperparams" subtitle={"Tune rank, alpha, dropout and target modules " + DASH + " see real-time effect"} />
#
#       <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
#         {/* Left controls */}
#         <Card style={{ flex: 1, minWidth: 300 }}>
#           <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"LoRA Config Explorer"}</div>
#
#           {[
#             { label: "Rank (r)", val: rank, set: setRank, min: 1, max: 128, color: C.accent, desc: "Higher r = more capacity but more params" },
#             { label: "Alpha (\u03B1)", val: alpha, set: setAlpha, min: 1, max: 256, color: C.purple, desc: "Scale factor: actual scale = \u03B1/r" },
#             { label: "Dropout %", val: dropout, set: setDropout, min: 0, max: 50, color: C.blue, desc: "Regularization on adapter activations" },
#           ].map(function(s, i) {
#             return (
#               <div key={i} style={{ marginBottom: 16 }}>
#                 <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
#                   <span style={{ fontSize: 9, color: s.color, fontFamily: "monospace", fontWeight: 700 }}>{s.label}</span>
#                   <span style={{ fontSize: 16, fontWeight: 800, color: s.color, fontFamily: "monospace" }}>{s.val}</span>
#                 </div>
#                 <input type="range" min={s.min} max={s.max} value={s.val}
#                   onChange={function(e) { s.set(parseInt(e.target.value)); }}
#                   style={{ width: "100%", accentColor: s.color }} />
#                 <div style={{ fontSize: 8, color: C.dim, marginTop: 3 }}>{s.desc}</div>
#               </div>
#             );
#           })}
#
#           <div style={{ marginBottom: 8 }}>
#             <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, marginBottom: 8 }}>{"Target Modules (7B model):"}</div>
#             <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
#               {targetNames.map(function(name, i) {
#                 var on = targets[i];
#                 return (<button key={i} onClick={function() { toggleTarget(i); }} style={{
#                   padding: "5px 10px", borderRadius: 6, border: "1.5px solid " + (on ? targetColors[i] : C.border),
#                   background: on ? targetColors[i] + "20" : C.card, color: on ? targetColors[i] : C.dim,
#                   cursor: "pointer", fontSize: 9, fontFamily: "monospace", fontWeight: 700
#                 }}>{name}</button>);
#               })}
#             </div>
#           </div>
#         </Card>
#
#         {/* Right stats */}
#         <Card style={{ flex: 1, minWidth: 280 }}>
#           <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"Live Config Stats"}</div>
#
#           <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
#             <StatBox label="RANK" value={rank} color={C.accent} />
#             <StatBox label="ALPHA" value={alpha} color={C.purple} />
#             <StatBox label="SCALE (\u03B1/r)" value={scale.toFixed(2)} color={scale < 0.5 ? C.red : scale > 2 ? C.green : C.yellow} />
#             <StatBox label="TRAINABLE" value={trainableM.toFixed(1) + "M"} color={C.cyan} />
#           </div>
#
#           {/* Scale gauge */}
#           <div style={{ marginBottom: 12 }}>
#             <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>{"Scale (\u03B1/r) — effect strength:"}</div>
#             <div style={{ height: 14, background: C.border, borderRadius: 3, position: "relative" }}>
#               <div style={{
#                 position: "absolute", left: 0, height: "100%", borderRadius: 3,
#                 width: Math.min(100, scale * 25) + "%",
#                 background: scale < 0.5 ? C.red + "70" : scale > 2 ? C.green + "70" : C.yellow + "70",
#                 border: "1px solid " + (scale < 0.5 ? C.red : scale > 2 ? C.green : C.yellow),
#                 transition: "width 0.3s"
#               }} />
#               <div style={{ position: "absolute", left: "25%", top: 0, bottom: 0, borderLeft: "1px dashed " + C.dim + "60" }} />
#               <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, borderLeft: "1px dashed " + C.dim + "60" }} />
#             </div>
#             <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: C.dim, marginTop: 2 }}>
#               <span style={{color: C.red}}>Too weak</span>
#               <span style={{color: C.yellow}}>Balanced</span>
#               <span style={{color: C.green}}>Strong</span>
#             </div>
#           </div>
#
#           {/* Recommended configs */}
#           <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"Common configs:"}</div>
#           {[
#             { name: "Conservative", r: 4, a: 8, desc: "Safe default, small models", color: C.blue },
#             { name: "Standard", r: 8, a: 16, desc: "Best starting point", color: C.green },
#             { name: "High Capacity", r: 16, a: 32, desc: "Complex tasks, more data", color: C.yellow },
#             { name: "Max Quality", r: 64, a: 64, desc: "Approaching full FT", color: C.red },
#           ].map(function(cfg, i) {
#             var active = rank === cfg.r && alpha === cfg.a;
#             return (<div key={i} onClick={function() { setRank(cfg.r); setAlpha(cfg.a); }} style={{
#               display: "flex", alignItems: "center", gap: 8, padding: "5px 8px", marginBottom: 3,
#               borderRadius: 6, cursor: "pointer", background: active ? cfg.color + "12" : "transparent",
#               border: "1px solid " + (active ? cfg.color + "40" : "transparent"), transition: "all 0.2s"
#             }}>
#               <div style={{ width: 8, height: 8, borderRadius: "50%", background: cfg.color }} />
#               <div style={{ fontSize: 9, color: active ? cfg.color : C.muted, fontWeight: active ? 700 : 400 }}>{cfg.name}</div>
#               <div style={{ fontSize: 8, color: C.dim }}>{"r=" + cfg.r + ", \u03B1=" + cfg.a}</div>
#               <div style={{ fontSize: 8, color: C.dim, marginLeft: "auto" }}>{cfg.desc}</div>
#             </div>);
#           })}
#         </Card>
#       </div>
#
#       {/* Generated config */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 10, fontWeight: 700, color: C.text, marginBottom: 8 }}>{"Generated Config (HuggingFace PEFT)"}</div>
#         <div style={{ background: "#06060a", borderRadius: 8, padding: "14px 18px", fontFamily: "monospace", fontSize: 10, lineHeight: 1.8, color: C.muted, border: "1px solid " + C.border }}>
#           <div><span style={{color: C.purple}}>{"from"}</span><span style={{color: C.text}}>{" peft "}</span><span style={{color: C.purple}}>{"import"}</span><span style={{color: C.text}}>{" LoraConfig, get_peft_model"}</span></div>
#           <br />
#           <div><span style={{color: C.cyan}}>{"config"}</span><span style={{color: C.text}}>{" = LoraConfig("}</span></div>
#           <div><span style={{color: C.text}}>{"    r="}</span><span style={{color: C.accent}}>{rank}</span><span style={{color: C.dim}}>{",  # rank"}</span></div>
#           <div><span style={{color: C.text}}>{"    lora_alpha="}</span><span style={{color: C.accent}}>{alpha}</span><span style={{color: C.dim}}>{",  # scale = " + scale.toFixed(2)}</span></div>
#           <div><span style={{color: C.text}}>{"    target_modules=["}</span><span style={{color: C.yellow}}>{"\"" + targetNames.filter(function(_, i) { return targets[i]; }).join("\", \"") + "\""}</span><span style={{color: C.text}}>{"],"}</span></div>
#           <div><span style={{color: C.text}}>{"    lora_dropout="}</span><span style={{color: C.accent}}>{(dropout / 100).toFixed(2)}</span><span style={{color: C.text}}>{","}</span></div>
#           <div><span style={{color: C.text}}>{"    bias=\"none\","}</span></div>
#           <div><span style={{color: C.text}}>{")"}</span></div>
#           <div><span style={{color: C.cyan}}>{"model"}</span><span style={{color: C.text}}>{" = get_peft_model(base_model, config)"}</span></div>
#         </div>
#       </Card>
#
#       <Insight icon={GEAR} title="Rank Selection Rule of Thumb">
#         Start with <span style={{color:C.accent,fontWeight:700}}>r=8, alpha=16</span> (scale=2). For high-resource tasks with lots of data, try <span style={{color:C.yellow}}>r=16 or r=32</span>. Diminishing returns beyond r=64 {DASH} at that point consider full fine-tuning. The alpha/r ratio controls effective adapter strength: <span style={{color:C.green}}>1–2 is a safe range</span>. Always target <span style={{color:C.purple}}>q_proj and v_proj</span> at minimum.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 6: MERGING & DEPLOYMENT
#    =============================================================== */
# function TabMerging() {
#   var _m = useState(0); var mode = _m[0], setMode = _m[1];
#   var _a = useState(false); var anim = _a[0], setAnim = _a[1];
#
#   useEffect(function() { var t = setTimeout(function() { setAnim(true); }, 400); return function() { clearTimeout(t); }; }, []);
#
#   var modes = [
#     { name: "Merged Inference", color: C.green, latency: "0ms overhead", memory: "= base model", desc: "Fold BA into W permanently: W_final = W0 + (alpha/r)*BA. Zero inference cost, no adapter needed.", icon: GEAR },
#     { name: "Multi-Adapter", color: C.blue, latency: "per-adapter", memory: "base + N adapters", desc: "Keep N adapters loaded with one base model. Swap adapters per request. Great for multi-tenant serving.", icon: PLUG },
#     { name: "Adapter Composition", color: C.purple, latency: "additive", memory: "base + sum", desc: "Apply multiple adapters sequentially or as weighted sum. Combine task1 + task2 adapters.", icon: DNA },
#   ];
#
#   return (
#     <div>
#       <SectionTitle title="Merging & Deployment" subtitle={"After training: zero-cost merge OR multi-adapter serving " + DASH + " unique PEFT superpower"} />
#
#       {/* Merge animation */}
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#         <svg width={1050} height={240} viewBox="0 0 800 240" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#           {/* Base model */}
#           <text x={120} y={18} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">W\u2080 (frozen base)</text>
#           {[0,1,2,3,4].map(function(i) {
#             return <rect key={i} x={40} y={28+i*30} width={160} height={24} rx={4} fill={C.dim+"18"} stroke={C.dim+"40"} strokeWidth={1} />;
#           })}
#           <text x={120} y={192} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " frozen"}</text>
#
#           {/* LoRA adapter */}
#           <text x={390} y={18} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">LoRA (B\u00D7A)</text>
#           <rect x={320} y={70} width={140} height={90} rx={8} fill={C.accent+"15"} stroke={C.accent+"60"} strokeWidth={1.5} />
#           <text x={390} y={108} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">B \u00D7 A</text>
#           <text x={390} y={124} textAnchor="middle" fill={C.accent+"80"} fontSize={8} fontFamily="monospace">trainable</text>
#           <text x={390} y={192} textAnchor="middle" fill={C.accent} fontSize={8} fontFamily="monospace">{"tiny adapter"}</text>
#
#           {/* Arrow */}
#           <line x1={470} y1={115} x2={540} y2={115} stroke={C.green} strokeWidth={2} />
#           <polygon points="545,115 538,110 538,120" fill={C.green} />
#           <text x={507} y={105} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">MERGE</text>
#           <text x={507} y={130} textAnchor="middle" fill={C.green+"80"} fontSize={7} fontFamily="monospace">W\u2080 + \u03B1/r\u00B7BA</text>
#
#           {/* Merged model */}
#           <text x={680} y={18} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">W\u2080 + \u03B1/r\u00B7BA (merged)</text>
#           {[0,1,2,3,4].map(function(i) {
#             return (<g key={i}>
#               <rect x={560} y={28+i*30} width={160} height={24} rx={4}
#                 fill={C.green+"12"} stroke={C.green+(anim?"60":"20")} strokeWidth={1}
#                 style={{ transition: "stroke 1s", transitionDelay: (i*0.1)+"s" }} />
#               {anim && <rect x={560} y={28+i*30} width={160} height={24} rx={4}
#                 fill={C.green+"20"} stroke="none"
#                 style={{ transition: "opacity 1s", opacity: 1, transitionDelay: (0.3+i*0.1)+"s" }} />}
#             </g>);
#           })}
#           <text x={640} y={192} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{UNLOCK + " no extra cost!"}</text>
#           <text x={640} y={208} textAnchor="middle" fill={C.green+"80"} fontSize={7} fontFamily="monospace">identical throughput to base</text>
#         </svg>
#       </div>
#
#       {/* Mode selector */}
#       <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16 }}>
#         {modes.map(function(m, i) {
#           var on = mode === i;
#           return (<button key={i} onClick={function() { setMode(i); }} style={{
#             padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? m.color : C.border),
#             background: on ? m.color + "20" : C.card, color: on ? m.color : C.muted,
#             cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
#           }}>{m.icon + " " + m.name}</button>);
#         })}
#       </div>
#
#       <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: modes[mode].color }}>
#         <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
#           <div style={{ flex: 1, minWidth: 280 }}>
#             <div style={{ fontSize: 16, fontWeight: 800, color: modes[mode].color, marginBottom: 8 }}>{modes[mode].name}</div>
#             <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{modes[mode].desc}</div>
#           </div>
#           <div style={{ display: "flex", gap: 20 }}>
#             <StatBox label="LATENCY OVERHEAD" value={modes[mode].latency} color={modes[mode].color} bigFont={11} minW={110} />
#             <StatBox label="MEMORY" value={modes[mode].memory} color={modes[mode].color} bigFont={11} minW={120} />
#           </div>
#         </div>
#       </Card>
#
#       {/* Comparison table */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Deployment Comparison"}</div>
#         <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
#           {[
#             { label: "Inference Speed", fft: CHK + " Fastest", peft_m: CHK + " Same as base", peft_a: WARN + " +overhead" },
#             { label: "Multi-task", fft: WARN + " New model/task", peft_m: CHK + " Swap adapters", peft_a: CHK + " Compose" },
#             { label: "Storage", fft: WARN + " Full copy/task", peft_m: CHK + " Base + tiny files", peft_a: CHK + " Minimal" },
#             { label: "Update a task", fft: WARN + " Retrain all", peft_m: CHK + " Retrain adapter", peft_a: CHK + " Retrain adapter" },
#           ].map(function(row, i) {
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 160, padding: "10px 12px", borderRadius: 8, background: C.border + "10", border: "1px solid " + C.border }}>
#                 <div style={{ fontSize: 9, fontWeight: 700, color: C.text, marginBottom: 6 }}>{row.label}</div>
#                 <div style={{ fontSize: 8, color: C.red, marginBottom: 2 }}>{"FFT: " + row.fft}</div>
#                 <div style={{ fontSize: 8, color: C.green, marginBottom: 2 }}>{"PEFT+Merge: " + row.peft_m}</div>
#                 <div style={{ fontSize: 8, color: C.blue }}>{"PEFT+Adapters: " + row.peft_a}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={STAR} title="The Merge Trick: Best of Both Worlds">
#         LoRA's killer feature: <span style={{color:C.green,fontWeight:700}}>W_final = W0 + (alpha/r) * BA</span> can be computed once after training, creating a standard weight matrix with <span style={{color:C.accent,fontWeight:700}}>zero inference overhead</span>. You get all the training efficiency of PEFT with the serving performance of a standard model. And if you want multiple tasks, keep adapters separate and <span style={{color:C.blue}}>hot-swap</span> them at inference time.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    ROOT APP
#    =============================================================== */
# function App() {
#   var _t = useState(0); var tab = _t[0], setTab = _t[1];
#   var tabs = ["Big Picture", "LoRA Mechanics", "Memory & Cost", "Adapter Zoo", "Rank & Hyperparams", "Merging & Deployment"];
#   return (
#     <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
#       <div style={{ textAlign: "center", marginBottom: 16 }}>
#         <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.purple + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>PEFT Additive</div>
#         <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " LoRA, Adapters, Prefix Tuning & beyond"}</div>
#       </div>
#       <TabBar tabs={tabs} active={tab} onChange={setTab} />
#       {tab === 0 && <TabBigPicture />}
#       {tab === 1 && <TabLoraMechanics />}
#       {tab === 2 && <TabMemory />}
#       {tab === 3 && <TabAdapterZoo />}
#       {tab === 4 && <TabRankHyperparams />}
#       {tab === 5 && <TabMerging />}
#     </div>
#   );
# }
#
# ReactDOM.createRoot(document.getElementById("root")).render(<App />);
#
# </script>
# </body>
# </html>
# """
#
# PEFT_ADDITIVE_HEIGHT = 1600