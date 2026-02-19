"""
Self-contained HTML for the QLoRA interactive walkthrough.
Covers: Big Picture, NF4 Quantization, Double Quantization,
Paged Optimizers, Forward Pass, and Memory Breakdown.
Embed in Streamlit via st.components.v1.html(QLORA_VISUAL_HTML, height=QLORA_VISUAL_HEIGHT).
"""

QLORA_VISUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #07070e; overflow-x: hidden; font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; }
  input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: #1a1a2e; outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; cursor: pointer; background: #f97316; }
  @keyframes pulse { 0%,100%{opacity:0.5} 50%{opacity:1} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes glow { 0%,100%{box-shadow:0 0 6px rgba(249,115,22,0.3)} 50%{box-shadow:0 0 18px rgba(249,115,22,0.7)} }
  .zoom-bar { display: flex; align-items: center; gap: 10px; justify-content: center; padding: 8px 0 4px; }
  .zoom-btn { width: 30px; height: 30px; border-radius: 7px; background: #0f0f1c; border: 1px solid #2a2a45; color: #f97316; font-size: 18px; font-weight: 800; cursor: pointer; display: flex; align-items: center; justify-content: center; font-family: monospace; transition: all 0.2s; }
  .zoom-btn:hover { background: #1e1e35; border-color: #f97316; }
  .zoom-label { font-size: 10px; color: #555; font-family: monospace; min-width: 38px; text-align: center; }
  .zoom-hint { font-size: 9px; color: #333; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState = React.useState;
var useEffect = React.useEffect;
var useMemo = React.useMemo;

var C = {
  bg: "#07070e", card: "#0f0f1c", border: "#1e1e35",
  orange: "#f97316", blue: "#38bdf8", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4ef", muted: "#6b6b85",
  dim: "#3a3a55", red: "#ef4444", green: "#4ade80",
  cyan: "#22d3ee", pink: "#f472b6", teal: "#2dd4bf",
};

var LOCK = "🔒";
var FIRE = "🔥";
var BULB = "💡";
var BRAIN = "🧠";
var PLAY = "\u25B6";
var PAUSE = "\u23F8";
var DASH = "\u2014";
var ARR = "\u2192";
var PLUS = "+";
var CHK = "\u2713";

/* ── ZoomWrapper ─────────────────────────────────────────── */
function ZoomWrapper(props) {
  var _z = useState(1); var scale = _z[0]; var setScale = _z[1];
  var MIN = 0.4; var MAX = 2.5; var STEP = 0.1;

  function adj(delta) {
    setScale(function(s) {
      var next = Math.round((s + delta) * 10) / 10;
      return Math.min(MAX, Math.max(MIN, next));
    });
  }

  useEffect(function() {
    function onWheel(e) {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        adj(e.deltaY > 0 ? -STEP : STEP);
      }
    }
    window.addEventListener("wheel", onWheel, { passive: false });
    return function() { window.removeEventListener("wheel", onWheel); };
  }, []);

  return (
    <div>
      <div className="zoom-bar">
        <button className="zoom-btn" onClick={function() { adj(-STEP); }}>{"−"}</button>
        <div className="zoom-label">{Math.round(scale * 100) + "%"}</div>
        <button className="zoom-btn" onClick={function() { adj(STEP); }}>{"+"}</button>
        <button className="zoom-btn" onClick={function() { setScale(1); }} style={{ fontSize: 11 }}>{"\u27F3"}</button>
        <div className="zoom-hint">{"Ctrl+Scroll or buttons to zoom"}</div>
      </div>
      <div style={{ transformOrigin: "top center", transform: "scale(" + scale + ")" }}>
        {props.children}
      </div>
    </div>
  );
}

/* ── TabBar ──────────────────────────────────────────────── */
function TabBar(props) {
  var tabs = props.tabs; var active = props.active; var onChange = props.onChange;
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 24, overflowX: "auto" }}>
      {tabs.map(function(t, i) {
        return (
          <button key={i} onClick={function() { onChange(i); }} style={{
            padding: "11px 15px", background: "none", border: "none",
            borderBottom: active === i ? "2px solid " + C.orange : "2px solid transparent",
            color: active === i ? C.orange : C.muted, cursor: "pointer",
            fontSize: 10, fontWeight: 700, fontFamily: "monospace",
            transition: "all 0.2s", whiteSpace: "nowrap", marginBottom: -2,
          }}>{t}</button>
        );
      })}
    </div>
  );
}

/* ── Card ────────────────────────────────────────────────── */
function Card(props) {
  return (
    <div style={Object.assign({
      background: C.card, borderRadius: 10, padding: "18px 22px",
      border: "1px solid " + (props.highlight ? C.orange : C.border),
      transition: "border 0.3s",
    }, props.style || {})}>
      {props.children}
    </div>
  );
}

/* ── Insight ─────────────────────────────────────────────── */
function Insight(props) {
  var col = props.color || C.orange;
  return (
    <div style={Object.assign({
      maxWidth: 1100, margin: "14px auto 0", padding: "14px 20px",
      background: col + "0a", borderRadius: 10, border: "1px solid " + col + "30",
    }, props.style || {})}>
      <div style={{ fontSize: 10, fontWeight: 700, color: col, marginBottom: 6 }}>{(props.icon || BULB) + " " + (props.title || "Key Insight")}</div>
      <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.9 }}>{props.children}</div>
    </div>
  );
}

/* ── SectionTitle ────────────────────────────────────────── */
function SectionTitle(props) {
  return (
    <div style={{ textAlign: "center", marginBottom: 20 }}>
      <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 5 }}>{props.title}</div>
      <div style={{ fontSize: 11, color: C.muted }}>{props.subtitle}</div>
    </div>
  );
}

/* ── Badge ───────────────────────────────────────────────── */
function Badge(props) {
  var col = props.color || C.orange;
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: col + "20", border: "1px solid " + col + "50",
      color: col, fontSize: 9, fontWeight: 700, fontFamily: "monospace", margin: "0 3px",
    }}>{props.children}</span>
  );
}

/* ── StatBox ─────────────────────────────────────────────── */
function StatBox(props) {
  return (
    <div style={{ textAlign: "center", minWidth: props.minW || 90 }}>
      <div style={{ fontSize: 8, color: C.muted, marginBottom: 4, letterSpacing: 1, textTransform: "uppercase" }}>{props.label}</div>
      <div style={{ fontSize: props.bigFont || 22, fontWeight: 800, color: props.color || C.orange }}>{props.value}</div>
      {props.sub && <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{props.sub}</div>}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TAB 1 — BIG PICTURE
   ═══════════════════════════════════════════════════════════ */
function TabBigPicture() {
  var _a = useState(false); var animated = _a[0]; var setAnimated = _a[1];
  useEffect(function() {
    var t = setTimeout(function() { setAnimated(true); }, 400);
    return function() { clearTimeout(t); };
  }, []);

  var methods = [
    { name: "Full Fine-Tuning",    mem: "~112 GB", color: C.red,    desc: "All weights updated. Requires 16\u00D7 model size in VRAM for optimizer states. Prohibitively expensive on most hardware." },
    { name: "LoRA (BF16 base)",    mem: "~18 GB",  color: C.purple, desc: "Freeze base in BF16, train low-rank adapters only. 3\u20136\u00D7 memory saving. Still limited by 16-bit base weights." },
    { name: "QLoRA (NF4 + LoRA)", mem: "~6 GB",   color: C.orange, desc: "4-bit NF4 base + BF16 adapters + paged optimizers. Fine-tune a 7B model on a single RTX 3090. " + FIRE, star: true },
  ];

  return (
    <div>
      <SectionTitle
        title={"QLoRA: Quantized Low-Rank Adaptation"}
        subtitle={"Three stacked innovations: NF4 quantization + double quantization + paged optimizers + LoRA adapters"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={1050} height={320} viewBox={"0 0 790 320"} style={{ background: "#08080d", borderRadius: 12, border: "1px solid " + C.border }}>

          <text x={110} y={22} textAnchor={"middle"} fill={C.red} fontSize={10} fontWeight={700} fontFamily={"monospace"}>{"FULL FINE-TUNING"}</text>
          <text x={110} y={36} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"All 7B params trainable"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (
              <g key={i}>
                <rect x={35} y={46 + i * 28} width={150} height={22} rx={4} fill={C.red + "18"} stroke={C.red + "50"} strokeWidth={1.5}/>
                <rect x={35} y={46 + i * 28} width={animated ? 150 : 0} height={22} rx={4} fill={C.red + "22"} style={{ transition: "width 1s", transitionDelay: (i * 0.07) + "s" }}/>
                <text x={110} y={61 + i * 28} textAnchor={"middle"} fill={C.red + "80"} fontSize={7} fontFamily={"monospace"}>{FIRE + " Layer " + (i + 1) + " \u2014 GRAD + OPTIM"}</text>
              </g>
            );
          })}
          <rect x={35} y={276} width={150} height={26} rx={4} fill={C.red + "30"} stroke={C.red} strokeWidth={2}/>
          <text x={110} y={292} textAnchor={"middle"} fill={C.red} fontSize={10} fontWeight={800} fontFamily={"monospace"}>{"~112 GB VRAM"}</text>

          <text x={395} y={22} textAnchor={"middle"} fill={C.purple} fontSize={10} fontWeight={700} fontFamily={"monospace"}>{"LoRA (BF16 BASE)"}</text>
          <text x={395} y={36} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"Frozen BF16 + trainable adapters"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (
              <g key={i}>
                <rect x={320} y={46 + i * 28} width={150} height={22} rx={4} fill={C.dim + "20"} stroke={C.dim + "50"} strokeWidth={1}/>
                <text x={395} y={61 + i * 28} textAnchor={"middle"} fill={C.dim} fontSize={7} fontFamily={"monospace"}>{LOCK + " Layer " + (i + 1) + " \u2014 BF16 frozen"}</text>
                {i < 4 ? <rect x={322} y={48 + i * 28} width={20} height={18} rx={3} fill={C.purple + "30"} stroke={C.purple + "60"} strokeWidth={1}/> : null}
                {i < 4 ? <text x={332} y={60 + i * 28} textAnchor={"middle"} fill={C.purple} fontSize={6} fontFamily={"monospace"}>{"AB"}</text> : null}
              </g>
            );
          })}
          <rect x={320} y={276} width={150} height={26} rx={4} fill={C.purple + "25"} stroke={C.purple} strokeWidth={2}/>
          <text x={395} y={292} textAnchor={"middle"} fill={C.purple} fontSize={10} fontWeight={800} fontFamily={"monospace"}>{"~18 GB VRAM"}</text>

          <text x={678} y={22} textAnchor={"middle"} fill={C.orange} fontSize={10} fontWeight={700} fontFamily={"monospace"}>{"QLORA (NF4 BASE)"}</text>
          <text x={678} y={36} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"4-bit base + BF16 adapters"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (
              <g key={i}>
                <rect x={603} y={46 + i * 28} width={150} height={22} rx={4} fill={C.orange + "12"} stroke={C.orange + "35"} strokeWidth={1}/>
                <rect x={603} y={46 + i * 28} width={animated ? 150 : 0} height={22} rx={4} fill={C.orange + "15"} style={{ transition: "width 1.1s", transitionDelay: (0.5 + i * 0.07) + "s" }}/>
                <text x={678} y={61 + i * 28} textAnchor={"middle"} fill={C.orange + "80"} fontSize={7} fontFamily={"monospace"}>{LOCK + " Layer " + (i + 1) + " \u2014 NF4 4-bit"}</text>
                {i < 4 ? <rect x={605} y={48 + i * 28} width={20} height={18} rx={3} fill={C.purple + "30"} stroke={C.purple + "60"} strokeWidth={1}/> : null}
                {i < 4 ? <text x={615} y={60 + i * 28} textAnchor={"middle"} fill={C.purple} fontSize={6} fontFamily={"monospace"}>{"AB"}</text> : null}
              </g>
            );
          })}
          <rect x={603} y={276} width={150} height={26} rx={4} fill={C.orange + "30"} stroke={C.orange} strokeWidth={2.5}/>
          <text x={678} y={292} textAnchor={"middle"} fill={C.orange} fontSize={10} fontWeight={800} fontFamily={"monospace"}>{"~6 GB VRAM " + FIRE}</text>

          <line x1={228} y1={16} x2={228} y2={310} stroke={C.border} strokeWidth={1} strokeDasharray={"4,4"}/>
          <line x1={512} y1={16} x2={512} y2={310} stroke={C.border} strokeWidth={1} strokeDasharray={"4,4"}/>

          <text x={283} y={155} textAnchor={"middle"} fill={C.yellow} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"6\u00D7 less"}</text>
          <text x={283} y={168} textAnchor={"middle"} fill={C.yellow + "80"} fontSize={8} fontFamily={"monospace"}>{"memory"}</text>
          <text x={567} y={155} textAnchor={"middle"} fill={C.green} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"3\u00D7 less"}</text>
          <text x={567} y={168} textAnchor={"middle"} fill={C.green + "80"} fontSize={8} fontFamily={"monospace"}>{"memory"}</text>
        </svg>
      </div>

      <div style={{ display: "flex", gap: 12, maxWidth: 1100, margin: "0 auto 16px", flexWrap: "wrap" }}>
        {methods.map(function(m, i) {
          return (
            <Card key={i} style={{ flex: 1, minWidth: 280 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: m.color }}>{m.name}</div>
                {m.star ? <Badge color={C.yellow}>{"\u2605 Recommended"}</Badge> : null}
              </div>
              <div style={{ fontSize: 18, fontWeight: 800, color: m.color, marginBottom: 8 }}>{m.mem}</div>
              <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8 }}>{m.desc}</div>
            </Card>
          );
        })}
      </div>

      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"QLoRA Innovation Stack \u2014 Three Compounding Wins"}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { n: "1", title: "4-bit NF4",            badge: "NormalFloat4", color: C.orange, desc: "An information-theoretically optimal 4-bit data type whose quantile bins perfectly match the normal distribution of LLM weights. Superior to INT4 or FP4." },
            { n: "2", title: "Double Quantization",   badge: "DQ",           color: C.teal,   desc: "Quantize the quantization constants themselves. Saves ~0.37 bits/param \u2014 about 0.5 GB for a 7B model. The secondary constants cost almost nothing extra." },
            { n: "3", title: "Paged Optimizers",      badge: "NVIDIA UVM",   color: C.blue,   desc: "Use NVIDIA Unified Memory to page Adam optimizer states to CPU RAM during gradient-checkpointing spikes, then page them back. Eliminates OOM errors." },
          ].map(function(item, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 240, padding: "14px 16px", borderRadius: 8, background: item.color + "08", border: "1px solid " + item.color + "30" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <div style={{ width: 22, height: 22, borderRadius: 6, background: item.color + "25", border: "1px solid " + item.color + "60", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: item.color }}>{item.n}</div>
                  <div style={{ fontSize: 11, fontWeight: 800, color: item.color }}>{item.title}</div>
                  <Badge color={item.color}>{item.badge}</Badge>
                </div>
                <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8 }}>{item.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={BRAIN} title={"Why QLoRA is a Breakthrough"}>
        {"QLoRA (Dettmers et al., 2023) proved that a "}
        <span style={{ color: C.orange, fontWeight: 700 }}>{"65B parameter model"}</span>
        {" can be fine-tuned on a "}
        <span style={{ color: C.yellow, fontWeight: 700 }}>{"single 48 GB GPU"}</span>
        {" with no quality loss vs full BF16 fine-tuning. The key: 4-bit quantization is lossless for storage so long as you dequantize to BF16 for compute. LoRA adapters live in BF16 and absorb all task-specific learning \u2014 the frozen NF4 base never needs to change."}
      </Insight>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TAB 2 — NF4 QUANTIZATION
   ═══════════════════════════════════════════════════════════ */
function TabNF4() {
  var _h = useState(-1); var hov = _h[0]; var setHov = _h[1];
  var _b = useState(64); var blockSize = _b[0]; var setBlockSize = _b[1];

  var nf4vals = [-1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0912, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1];

  var pts = [];
  var cx = 395; var cy = 90; var sigma = 100;
  for (var x = -280; x <= 280; x += 4) {
    var y = Math.exp(-0.5 * (x / sigma) * (x / sigma)) * 120;
    pts.push((cx + x) + "," + (cy - y + 120));
  }
  var ptsStr = pts.join(" ");
  var pathD = "M " + (cx - 280) + "," + (cy + 120) + " L " + ptsStr + " L " + (cx + 280) + "," + (cy + 120) + " Z";

  return (
    <div>
      <SectionTitle
        title={"NF4 \u2014 NormalFloat 4-bit Quantization"}
        subtitle={"Information-theoretically optimal 4-bit data type for normally distributed neural network weights"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={315} viewBox={"0 0 790 315"} style={{ background: "#08080d", borderRadius: 12, border: "1px solid " + C.border }}>
          <defs>
            <linearGradient id={"ng"} x1={"0%"} y1={"0%"} x2={"100%"} y2={"0%"}>
              <stop offset={"0%"} stopColor={C.orange} stopOpacity={"0.04"}/>
              <stop offset={"50%"} stopColor={C.orange} stopOpacity={"0.28"}/>
              <stop offset={"100%"} stopColor={C.orange} stopOpacity={"0.04"}/>
            </linearGradient>
          </defs>

          <path d={pathD} fill={"url(#ng)"} stroke={"none"}/>
          <polyline points={ptsStr} fill={"none"} stroke={C.orange} strokeWidth={2.5}/>

          {nf4vals.map(function(v, i) {
            var bx = 395 + v * 280;
            var isH = hov === i;
            return (
              <g key={i}>
                <line x1={bx} y1={50} x2={bx} y2={240}
                  stroke={isH ? C.yellow : C.orange}
                  strokeWidth={isH ? 2 : 1}
                  strokeDasharray={v === 0 ? "0" : "2,2"}
                  opacity={isH ? 1 : 0.55}
                  style={{ transition: "all 0.2s" }}/>
                <rect x={bx - 10} y={242} width={20} height={14} rx={3}
                  fill={isH ? C.yellow + "30" : C.orange + "15"}
                  stroke={isH ? C.yellow : C.orange + "50"}
                  strokeWidth={1}
                  onMouseEnter={function() { setHov(i); }}
                  onMouseLeave={function() { setHov(-1); }}
                  style={{ cursor: "pointer", transition: "all 0.2s" }}/>
                <text x={bx} y={252} textAnchor={"middle"} fill={isH ? C.yellow : C.orange} fontSize={5} fontFamily={"monospace"}>{i.toString(16).toUpperCase()}</text>
              </g>
            );
          })}

          {hov >= 0
            ? <g>
                <rect x={220} y={265} width={310} height={36} rx={6} fill={C.card} stroke={C.yellow} strokeWidth={1.5}/>
                <text x={375} y={281} textAnchor={"middle"} fill={C.yellow} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"Bin " + hov.toString(16).toUpperCase() + " \u2192 float: " + nf4vals[hov].toFixed(4)}</text>
                <text x={375} y={294} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"Each bin covers equal probability mass under N(0,1)"}</text>
              </g>
            : <text x={395} y={282} textAnchor={"middle"} fill={C.dim} fontSize={9} fontFamily={"monospace"}>{"Hover any bin to inspect its quantile value"}</text>
          }

          <text x={395} y={20} textAnchor={"middle"} fill={C.orange} fontSize={10} fontWeight={700} fontFamily={"monospace"}>{"Normal Distribution N(0,1) \u2014 LLM Weight Distribution"}</text>
          <text x={395} y={33} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"16 NF4 bins (0000\u20131111) at equal quantile intervals \u2014 4 bits \u2192 16 levels"}</text>
          <text x={115} y={172} textAnchor={"middle"} fill={C.dim} fontSize={8} fontFamily={"monospace"}>{"rare weights"}</text>
          <text x={675} y={172} textAnchor={"middle"} fill={C.dim} fontSize={8} fontFamily={"monospace"}>{"rare weights"}</text>
          <text x={395} y={128} textAnchor={"middle"} fill={C.orange + "80"} fontSize={8} fontFamily={"monospace"}>{"dense bins here"}</text>
        </svg>
      </div>

      <Card style={{ maxWidth: 1100, margin: "0 auto 14px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"NF4 vs INT4 vs FP4 \u2014 Why NF4 Wins"}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { name: "INT4",      color: C.red,    bins: "\u20138 to +7 (linear)",         match: "\u274C Linear spacing mismatches normal distribution" },
            { name: "FP4 (E2M1)", color: C.yellow, bins: "Non-linear, floating-point",  match: "\u26A0 Better but not optimal for N(0,1)" },
            { name: "NF4",       color: C.orange, bins: "16 quantile-spaced bins",      match: "\u2713 Information-theoretically optimal for N(0,1)", star: true },
          ].map(function(dt, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 220, padding: "14px 16px", borderRadius: 8, background: dt.color + "08", border: "1.5px solid " + dt.color + (dt.star ? "80" : "30") }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <div style={{ fontSize: 13, fontWeight: 800, color: dt.color }}>{dt.name}</div>
                  {dt.star ? <Badge color={C.orange}>{"Best"}</Badge> : null}
                </div>
                <div style={{ fontSize: 9, color: C.muted, lineHeight: 2 }}>
                  <div><span style={{ color: dt.color + "90" }}>{"Bins: "}</span>{dt.bins}</div>
                  <div style={{ marginTop: 6 }}>{dt.match}</div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <Card style={{ maxWidth: 1100, margin: "0 auto 14px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Block-wise Quantization \u2014 Adjust Block Size"}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          <div style={{ fontSize: 10, color: C.teal }}>{"Block size:"}</div>
          <input type={"range"} min={32} max={256} step={32} value={blockSize} onChange={function(e) { setBlockSize(parseInt(e.target.value)); }} style={{ width: 140, accentColor: C.teal }}/>
          <div style={{ fontSize: 10, color: C.teal, fontWeight: 700 }}>{blockSize + " weights/block"}</div>
        </div>
        <svg width={"100%"} height={90} viewBox={"0 0 760 90"}>
          {[0, 1, 2, 3].map(function(i) {
            var bw = 140; var gap = 30; var bx = 30 + i * (bw + gap);
            return (
              <g key={i}>
                <rect x={bx} y={10} width={bw} height={36} rx={6} fill={C.orange + "12"} stroke={C.orange + "40"} strokeWidth={1.5}/>
                <text x={bx + bw / 2} y={26} textAnchor={"middle"} fill={C.orange} fontSize={8} fontWeight={700} fontFamily={"monospace"}>{"Block " + (i + 1) + " (" + blockSize + " wts)"}</text>
                <text x={bx + bw / 2} y={39} textAnchor={"middle"} fill={C.muted} fontSize={7} fontFamily={"monospace"}>{"scale = max(|block|)"}</text>
                <rect x={bx + 40} y={54} width={60} height={20} rx={4} fill={C.teal + "20"} stroke={C.teal + "50"} strokeWidth={1}/>
                <text x={bx + 70} y={67} textAnchor={"middle"} fill={C.teal} fontSize={8} fontFamily={"monospace"}>{"c" + (i + 1) + " FP32"}</text>
              </g>
            );
          })}
          <text x={650} y={35} textAnchor={"middle"} fill={C.dim} fontSize={12} fontFamily={"monospace"}>{"..."}</text>
          <text x={30} y={86} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"One FP32 scale constant per block \u2014 Double Quantization will compress these further"}</text>
        </svg>
      </Card>

      <Insight color={C.orange} icon={"📐"} title={"Why Quantile Spacing is Optimal"}>
        {"Each NF4 bin covers exactly "}
        <span style={{ color: C.orange, fontWeight: 700 }}>{"1/16th of the probability mass"}</span>
        {" under N(0,1). Since LLM weights are approximately normal, each bin is equally likely \u2014 maximum entropy, zero wasted bits. INT4's linear bins waste capacity in the sparse tails. NF4 matches the "}
        <span style={{ color: C.yellow, fontWeight: 700 }}>{"actual data distribution"}</span>
        {", not a uniform assumption."}
      </Insight>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TAB 3 — DOUBLE QUANTIZATION
   ═══════════════════════════════════════════════════════════ */
function TabDoubleQ() {
  var _s = useState(0); var step = _s[0]; var setStep = _s[1];
  var _a = useState(false); var auto = _a[0]; var setAuto = _a[1];

  useEffect(function() {
    if (!auto) return;
    var t = setInterval(function() { setStep(function(s) { return (s + 1) % 4; }); }, 2200);
    return function() { clearInterval(t); };
  }, [auto]);

  var steps = [
    { label: "Raw Weights",             color: C.red,    desc: "7B model in BF16 \u2014 16 bits per parameter. Total: ~14 GB" },
    { label: "NF4 + Block Scales",      color: C.orange, desc: "4-bit NF4 weights + FP32 scale per 64 weights. Scale overhead: 32/64 = 0.5 bits/param extra" },
    { label: "Double Quantize Scales",  color: C.blue,   desc: "Treat FP32 scales as weights \u2014 quantize them to FP8 in blocks of 256. Secondary scale: FP32 every 256 blocks." },
    { label: "Final Storage",           color: C.green,  desc: "NF4 weights + FP8 scales + FP32 secondary scales \u2248 4.127 bits/param total" },
  ];

  return (
    <div>
      <SectionTitle
        title={"Double Quantization (DQ)"}
        subtitle={"Quantize the quantization constants \u2014 save another 0.37 bits per parameter, ~0.5 GB for a 7B model"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          return (
            <button key={i} onClick={function() { setStep(i); setAuto(false); }} style={{
              padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (step === i ? s.color : C.border),
              background: step === i ? s.color + "20" : C.card, color: step === i ? s.color : C.muted,
              cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
            }}>{(i + 1) + ". " + s.label}</button>
          );
        })}
        <button onClick={function() { setAuto(!auto); }} style={{
          padding: "8px 12px", borderRadius: 8,
          border: "1.5px solid " + (auto ? C.yellow : C.border),
          background: auto ? C.yellow + "20" : C.card, color: auto ? C.yellow : C.muted,
          cursor: "pointer", fontSize: 10, fontFamily: "monospace"
        }}>{auto ? PAUSE + " Stop" : PLAY + " Auto"}</button>
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={345} viewBox={"0 0 790 345"} style={{ background: "#08080d", borderRadius: 12, border: "1px solid " + C.border }}>

          {/* STAGE 0 */}
          <text x={85} y={28} textAnchor={"middle"} fill={C.red} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"RAW WEIGHTS"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (
              <g key={i}>
                <rect x={20} y={40 + i * 28} width={130} height={22} rx={4} fill={C.red + "18"} stroke={C.red + "50"} strokeWidth={1.5} style={{ transition: "all 0.4s" }}/>
                <text x={85} y={55 + i * 28} textAnchor={"middle"} fill={C.red + "90"} fontSize={7} fontFamily={"monospace"}>{"BF16 chunk " + (i + 1)}</text>
              </g>
            );
          })}
          <rect x={20} y={272} width={130} height={20} rx={4} fill={C.red + "30"} stroke={C.red} strokeWidth={1.5}/>
          <text x={85} y={286} textAnchor={"middle"} fill={C.red} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"16 bits/param"}</text>

          <line x1={152} y1={155} x2={188} y2={155} stroke={step >= 1 ? C.orange : C.dim + "40"} strokeWidth={2} style={{ transition: "all 0.4s" }}/>
          <polygon points={"192,155 186,151 186,159"} fill={step >= 1 ? C.orange : C.dim + "40"} style={{ transition: "all 0.4s" }}/>
          <text x={170} y={148} textAnchor={"middle"} fill={step >= 1 ? C.orange : C.dim} fontSize={7} fontFamily={"monospace"}>{"NF4"}</text>

          {/* STAGE 1 */}
          <text x={248} y={28} textAnchor={"middle"} fill={C.orange} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"NF4 + FP32 SCALES"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            var active = step >= 1;
            return (
              <g key={i}>
                <rect x={193} y={40 + i * 28} width={110} height={22} rx={4} fill={active ? C.orange + "15" : C.card} stroke={active ? C.orange + "45" : C.border} strokeWidth={1.5} style={{ transition: "all 0.4s" }}/>
                <text x={248} y={55 + i * 28} textAnchor={"middle"} fill={active ? C.orange + "90" : C.dim} fontSize={7} fontFamily={"monospace"}>{"NF4 block " + (i + 1)}</text>
              </g>
            );
          })}
          {[0,1,2,3].map(function(i) {
            var active = step >= 1;
            return (
              <g key={i}>
                <rect x={193} y={46 + i * 56} width={110} height={15} rx={3} fill={active ? C.teal + "25" : C.card} stroke={active ? C.teal + "60" : C.border} strokeWidth={1} style={{ transition: "all 0.4s" }}/>
                <text x={248} y={57 + i * 56} textAnchor={"middle"} fill={active ? C.teal : C.dim} fontSize={6} fontFamily={"monospace"}>{"scale[" + i + "] FP32"}</text>
              </g>
            );
          })}
          <rect x={193} y={272} width={110} height={20} rx={4} fill={step >= 1 ? C.orange + "30" : C.card} stroke={step >= 1 ? C.orange : C.border} strokeWidth={1.5} style={{ transition: "all 0.4s" }}/>
          <text x={248} y={286} textAnchor={"middle"} fill={step >= 1 ? C.orange : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"4.5 bits/param"}</text>

          <line x1={305} y1={155} x2={341} y2={155} stroke={step >= 2 ? C.blue : C.dim + "40"} strokeWidth={2} style={{ transition: "all 0.4s" }}/>
          <polygon points={"345,155 339,151 339,159"} fill={step >= 2 ? C.blue : C.dim + "40"} style={{ transition: "all 0.4s" }}/>
          <text x={323} y={148} textAnchor={"middle"} fill={step >= 2 ? C.blue : C.dim} fontSize={7} fontFamily={"monospace"}>{"DQ"}</text>

          {/* STAGE 2 */}
          <text x={408} y={28} textAnchor={"middle"} fill={C.blue} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"NF4 + FP8 SCALES"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            var active = step >= 2;
            return (
              <g key={i}>
                <rect x={348} y={40 + i * 28} width={120} height={22} rx={4} fill={active ? C.orange + "12" : C.card} stroke={active ? C.orange + "30" : C.border} strokeWidth={1} style={{ transition: "all 0.4s" }}/>
                <text x={408} y={55 + i * 28} textAnchor={"middle"} fill={active ? C.orange + "70" : C.dim} fontSize={7} fontFamily={"monospace"}>{"NF4 block " + (i + 1)}</text>
              </g>
            );
          })}
          {[0,1,2,3].map(function(i) {
            var active = step >= 2;
            return (
              <g key={i}>
                <rect x={348} y={46 + i * 56} width={120} height={15} rx={3} fill={active ? C.blue + "25" : C.card} stroke={active ? C.blue + "60" : C.border} strokeWidth={1} style={{ transition: "all 0.4s" }}/>
                <text x={408} y={57 + i * 56} textAnchor={"middle"} fill={active ? C.blue : C.dim} fontSize={6} fontFamily={"monospace"}>{"scale[" + i + "] FP8 \u2190 compressed!"}</text>
              </g>
            );
          })}
          <rect x={348} y={272} width={120} height={20} rx={4} fill={step >= 2 ? C.blue + "30" : C.card} stroke={step >= 2 ? C.blue : C.border} strokeWidth={1.5} style={{ transition: "all 0.4s" }}/>
          <text x={408} y={286} textAnchor={"middle"} fill={step >= 2 ? C.blue : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"4.127 bits/param"}</text>

          <line x1={470} y1={155} x2={506} y2={155} stroke={step >= 3 ? C.green : C.dim + "40"} strokeWidth={2} style={{ transition: "all 0.4s" }}/>
          <polygon points={"510,155 504,151 504,159"} fill={step >= 3 ? C.green : C.dim + "40"} style={{ transition: "all 0.4s" }}/>
          <text x={488} y={148} textAnchor={"middle"} fill={step >= 3 ? C.green : C.dim} fontSize={7} fontFamily={"monospace"}>{"add c\u2082"}</text>

          {/* STAGE 3 — Final */}
          <rect x={514} y={30} width={260} height={246} rx={8} fill={C.green + "06"} stroke={step >= 3 ? C.green + "50" : C.border} strokeWidth={step >= 3 ? 2 : 1} style={{ transition: "all 0.4s" }}/>
          <text x={644} y={48} textAnchor={"middle"} fill={step >= 3 ? C.green : C.dim} fontSize={10} fontWeight={700} fontFamily={"monospace"}>{"FINAL STORAGE"}</text>
          <rect x={530} y={58} width={228} height={26} rx={4} fill={C.orange + "18"} stroke={C.orange + "40"} strokeWidth={1}/>
          <text x={644} y={71} textAnchor={"middle"} fill={C.orange} fontSize={9} fontFamily={"monospace"}>{"NF4 weights \u2014 4 bits/param"}</text>
          <text x={644} y={81} textAnchor={"middle"} fill={C.orange + "70"} fontSize={7} fontFamily={"monospace"}>{"7B \u00D7 4 bits = 3.5 GB"}</text>
          <rect x={530} y={96} width={228} height={26} rx={4} fill={C.blue + "18"} stroke={C.blue + "40"} strokeWidth={1}/>
          <text x={644} y={109} textAnchor={"middle"} fill={C.blue} fontSize={9} fontFamily={"monospace"}>{"FP8 scales \u2014 8 bits per 64 wts"}</text>
          <text x={644} y={119} textAnchor={"middle"} fill={C.blue + "70"} fontSize={7} fontFamily={"monospace"}>{"\u00F764 \u00D7 8 bits = 0.125 bits/param"}</text>
          <rect x={530} y={134} width={228} height={26} rx={4} fill={C.teal + "18"} stroke={C.teal + "40"} strokeWidth={1}/>
          <text x={644} y={147} textAnchor={"middle"} fill={C.teal} fontSize={9} fontFamily={"monospace"}>{"FP32 secondary \u2014 32 bits per 256"}</text>
          <text x={644} y={157} textAnchor={"middle"} fill={C.teal + "70"} fontSize={7} fontFamily={"monospace"}>{"\u2248 0.002 bits/param"}</text>
          <line x1={530} y1={172} x2={758} y2={172} stroke={C.green + "30"} strokeWidth={1} strokeDasharray={"3,3"}/>
          <text x={644} y={190} textAnchor={"middle"} fill={C.green} fontSize={12} fontWeight={800} fontFamily={"monospace"}>{"\u2248 4.127 bits / param"}</text>
          <text x={644} y={205} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"vs 16 bits (BF16) = 3.87\u00D7 compression"}</text>
          <text x={644} y={220} textAnchor={"middle"} fill={C.yellow} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"Saves ~0.37 bits vs plain NF4+FP32"}</text>
          <text x={644} y={235} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"\u2248 0.5 GB saved for a 7B model"}</text>
          <text x={644} y={249} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"\u2248 3.0 GB saved for a 65B model"}</text>

          <rect x={514} y={272} width={260} height={20} rx={4} fill={step >= 3 ? C.green + "30" : C.card} stroke={step >= 3 ? C.green : C.border} strokeWidth={1.5} style={{ transition: "all 0.4s" }}/>
          <text x={644} y={286} textAnchor={"middle"} fill={step >= 3 ? C.green : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{CHK + " 4.127 bits/param total"}</text>

          <rect x={20} y={308} width={753} height={26} rx={4} fill={steps[step].color + "15"} stroke={steps[step].color + "40"} strokeWidth={1}/>
          <text x={396} y={324} textAnchor={"middle"} fill={steps[step].color} fontSize={9} fontFamily={"monospace"}>{steps[step].desc}</text>
        </svg>
      </div>

      <Insight color={C.teal} icon={"🎯"} title={"Why Double Quantization Works"}>
        {"There are "}
        <span style={{ color: C.orange, fontWeight: 700 }}>{"7B/64 \u2248 109M"}</span>
        {" quantization constants for a 7B model. These are themselves approximately normally distributed. Quantizing from FP32 (32-bit) to FP8 (8-bit) introduces negligible error while saving "}
        <span style={{ color: C.yellow, fontWeight: 700 }}>{"24 bits per scale"}</span>
        {". The FP32 secondary constants cost almost nothing. Net effect: 0.37 bits/param saved \u2014 effectively free."}
      </Insight>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TAB 4 — PAGED OPTIMIZERS
   ═══════════════════════════════════════════════════════════ */
function TabPagedOpt() {
  var _m = useState(0); var memUsed = _m[0]; var setMemUsed = _m[1];
  var _p = useState(false); var paging = _p[0]; var setPaging = _p[1];

  useEffect(function() {
    var t = setInterval(function() {
      setMemUsed(function(m) {
        var next = m + 2;
        if (next > 100) { setPaging(true); return 82; }
        if (m > 80) { setPaging(true); } else { setPaging(false); }
        return next;
      });
    }, 200);
    return function() { clearInterval(t); };
  }, []);

  return (
    <div>
      <SectionTitle
        title={"Paged Optimizers"}
        subtitle={"NVIDIA Unified Virtual Memory pages optimizer states between GPU and CPU RAM \u2014 eliminating OOM errors"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={340} viewBox={"0 0 790 340"} style={{ background: "#08080d", borderRadius: 12, border: "1px solid " + C.border }}>

          <rect x={20} y={20} width={340} height={300} rx={10} fill={C.green + "06"} stroke={C.green + "40"} strokeWidth={2}/>
          <text x={190} y={40} textAnchor={"middle"} fill={C.green} fontSize={11} fontWeight={700} fontFamily={"monospace"}>{"🖥 GPU VRAM"}</text>

          <rect x={35} y={52} width={310} height={52} rx={6} fill={C.orange + "20"} stroke={C.orange + "60"} strokeWidth={1.5}/>
          <text x={190} y={73} textAnchor={"middle"} fill={C.orange} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"NF4 Base Model Weights"}</text>
          <text x={190} y={87} textAnchor={"middle"} fill={C.orange + "80"} fontSize={8} fontFamily={"monospace"}>{"3.5 GB \u2014 always on GPU"}</text>

          <rect x={35} y={116} width={310} height={46} rx={6} fill={C.purple + "20"} stroke={C.purple + "60"} strokeWidth={1.5}/>
          <text x={190} y={134} textAnchor={"middle"} fill={C.purple} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"LoRA Adapters (A, B) \u2014 BF16"}</text>
          <text x={190} y={148} textAnchor={"middle"} fill={C.purple + "80"} fontSize={8} fontFamily={"monospace"}>{"~33 MB \u2014 always on GPU"}</text>

          <rect x={35} y={174} width={310} height={46} rx={6} fill={C.cyan + "15"} stroke={C.cyan + "50"} strokeWidth={1.5}/>
          <text x={190} y={192} textAnchor={"middle"} fill={C.cyan} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"Activations + Gradients"}</text>
          <text x={190} y={206} textAnchor={"middle"} fill={C.cyan + "80"} fontSize={8} fontFamily={"monospace"}>{"~1\u20132 GB \u2014 fluctuates heavily"}</text>

          <rect x={35} y={232} width={310} height={72} rx={6}
            fill={paging ? C.blue + "30" : C.blue + "15"}
            stroke={paging ? C.blue : C.blue + "50"} strokeWidth={paging ? 2 : 1.5}
            style={{ transition: "all 0.3s" }}/>
          <text x={190} y={252} textAnchor={"middle"} fill={C.blue} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"Adam Optimizer States"}</text>
          <text x={190} y={266} textAnchor={"middle"} fill={C.blue + "90"} fontSize={8} fontFamily={"monospace"}>{"m\u2081, m\u2082 (momentum) \u2014 BF16"}</text>
          <text x={190} y={280} textAnchor={"middle"} fill={paging ? C.yellow : C.blue + "70"} fontSize={8} fontWeight={paging ? 700 : 400} fontFamily={"monospace"}>{paging ? "\u26A1 PAGING TO CPU RAM..." : "~2.5 GB \u2014 paged on demand"}</text>
          <text x={190} y={293} textAnchor={"middle"} fill={paging ? C.green : C.dim} fontSize={7} fontFamily={"monospace"}>{paging ? CHK + " NVIDIA UVM handles this automatically" : "Sits in GPU until memory pressure"}</text>

          <rect x={430} y={20} width={340} height={300} rx={10} fill={C.yellow + "06"} stroke={C.yellow + "30"} strokeWidth={2}/>
          <text x={600} y={40} textAnchor={"middle"} fill={C.yellow} fontSize={11} fontWeight={700} fontFamily={"monospace"}>{"🖥 CPU RAM (System)"}</text>

          <rect x={446} y={52} width={308} height={200} rx={6}
            fill={paging ? C.yellow + "18" : C.yellow + "08"}
            stroke={paging ? C.yellow + "60" : C.yellow + "30"} strokeWidth={paging ? 2 : 1}
            style={{ transition: "all 0.4s" }}/>
          <text x={600} y={72} textAnchor={"middle"} fill={C.yellow} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"Optimizer State Pool (CPU)"}</text>
          <text x={600} y={87} textAnchor={"middle"} fill={C.muted} fontSize={8} fontFamily={"monospace"}>{"Pinned memory \u2014 fast DMA transfer"}</text>
          {[0,1,2,3,4].map(function(i) {
            var filled = paging && i < 3;
            return (
              <g key={i}>
                <rect x={460} y={100 + i * 27} width={274} height={20} rx={4}
                  fill={filled ? C.yellow + "30" : C.card} stroke={filled ? C.yellow + "60" : C.border} strokeWidth={1}
                  style={{ transition: "all 0.3s", transitionDelay: (i * 0.05) + "s" }}/>
                <text x={597} y={114 + i * 27} textAnchor={"middle"} fill={filled ? C.yellow : C.dim} fontSize={7} fontFamily={"monospace"}>{filled ? ("Paged optimizer block " + (i + 1)) : ("Empty slot " + (i + 1))}</text>
              </g>
            );
          })}

          <rect x={446} y={264} width={308} height={46} rx={6} fill={C.red + "10"} stroke={C.red + "40"} strokeWidth={1.5}/>
          <text x={600} y={283} textAnchor={"middle"} fill={C.red} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"Without Paged Optimizers:"}</text>
          <text x={600} y={297} textAnchor={"middle"} fill={C.red + "80"} fontSize={8} fontFamily={"monospace"}>{"CUDA OOM during gradient checkpointing"}</text>

          {paging
            ? <g>
                <line x1={348} y1={268} x2={443} y2={140} stroke={C.blue} strokeWidth={2} strokeDasharray={"5,3"}/>
                <polygon points={"443,140 437,148 446,148"} fill={C.blue}/>
                <text x={392} y={196} textAnchor={"middle"} fill={C.blue} fontSize={8} fontWeight={700} fontFamily={"monospace"}>{"page out"}</text>
                <line x1={443} y1={162} x2={348} y2={270} stroke={C.green} strokeWidth={1.5} strokeDasharray={"5,3"} opacity={0.5}/>
                <text x={392} y={218} textAnchor={"middle"} fill={C.green} fontSize={8} fontFamily={"monospace"}>{"page back"}</text>
              </g>
            : <text x={395} y={200} textAnchor={"middle"} fill={C.dim} fontSize={8} fontFamily={"monospace"}>{"\u2190 paging path"}</text>
          }
        </svg>
      </div>

      <Card style={{ maxWidth: 1100, margin: "0 auto 14px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Live Simulation \u2014 GPU Memory During Training Step"}</div>
        <div style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 10, color: paging ? C.blue : C.green }}>{paging ? "\u26A1 Memory spike! Paging optimizer states to CPU..." : "GPU VRAM usage"}</span>
            <span style={{ fontSize: 10, color: paging ? C.yellow : C.text, fontWeight: 700 }}>{memUsed + "% / 100%"}</span>
          </div>
          <div style={{ height: 18, background: C.border, borderRadius: 4, overflow: "hidden" }}>
            <div style={{ height: "100%", borderRadius: 4, background: memUsed > 80 ? C.orange : C.green, width: memUsed + "%", transition: "width 0.15s, background 0.3s" }}/>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <span style={{ fontSize: 8, color: C.muted }}>{"0%"}</span>
            <span style={{ fontSize: 8, color: C.yellow }}>{"80% \u2014 paging threshold"}</span>
            <span style={{ fontSize: 8, color: C.red }}>{"100% \u2014 OOM (without paging)"}</span>
          </div>
        </div>
        <div style={{ fontSize: 10, color: paging ? C.blue : C.muted, lineHeight: 1.8, transition: "color 0.3s" }}>
          {paging
            ? "NVIDIA UVM automatically pages Adam momentum states to CPU pinned memory. The GPU continues computing. No crash, no OOM, no manual intervention needed."
            : "Normal operation \u2014 model weights, activations, gradients, and optimizer states all fit in GPU VRAM."}
        </div>
      </Card>

      <Insight color={C.blue} icon={"💾"} title={"The Paging Analogy"}>
        {"Just like an OS uses swap space when RAM fills up, NVIDIA's Unified Virtual Memory lets the GPU use CPU RAM as an overflow for tensors. QLoRA pre-allocates a "}
        <span style={{ color: C.yellow, fontWeight: 700 }}>{"fixed-size CPU memory pool"}</span>
        {" for optimizer states. When GPU memory spikes (during gradient checkpointing), optimizer pages are evicted to CPU; when the spike subsides they're paged back. PCIe bandwidth (~16 GB/s) makes this nearly transparent."}
      </Insight>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TAB 5 — FORWARD PASS
   ═══════════════════════════════════════════════════════════ */
function TabForwardPass() {
  var _s = useState(0); var step = _s[0]; var setStep = _s[1];
  var _a = useState(false); var auto = _a[0]; var setAuto = _a[1];

  useEffect(function() {
    if (!auto) return;
    var t = setInterval(function() { setStep(function(s) { return (s + 1) % 6; }); }, 2000);
    return function() { clearInterval(t); };
  }, [auto]);

  var steps = [
    { label: "Input x",              color: C.cyan,   desc: "Input token embeddings arrive in BF16 format. Sequence length \u00D7 hidden dim tensor." },
    { label: "Load NF4 Block",       color: C.orange, desc: "Load a 64-weight NF4 block from GPU memory. Each weight is a 4-bit index (0\u201315) into the NF4 lookup table." },
    { label: "Dequantize \u2192 BF16", color: C.yellow, desc: "Look up each 4-bit index in the NF4 table, multiply by the FP8 block scale. Result: transient BF16 weight slice." },
    { label: "Compute W\u2080x",      color: C.teal,   desc: "Standard BF16 matrix multiply: frozen weight slice \u00D7 input. No gradient stored here." },
    { label: "LoRA Bypass",          color: C.purple, desc: "Simultaneously compute Ax in rank-r space, then B(Ax) back to full dim. A and B are in BF16 and always on GPU." },
    { label: "Sum + Output",         color: C.green,  desc: "h = W\u2080x + (\u03B1/r)\u00B7BAx. LoRA correction added to frozen path output. Gradient flows only through A and B." },
  ];

  return (
    <div>
      <SectionTitle
        title={"QLoRA Forward Pass \u2014 Dequantize on the Fly"}
        subtitle={"4-bit weights stored in NF4, dequantized to BF16 on-the-fly for each matrix multiply, then discarded"} />

      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          return (
            <button key={i} onClick={function() { setStep(i); setAuto(false); }} style={{
              padding: "7px 11px", borderRadius: 8, border: "1.5px solid " + (step === i ? s.color : C.border),
              background: step === i ? s.color + "20" : C.card, color: step === i ? s.color : C.muted,
              cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
            }}>{(i + 1) + ". " + s.label}</button>
          );
        })}
        <button onClick={function() { setAuto(!auto); }} style={{
          padding: "7px 11px", borderRadius: 8, border: "1.5px solid " + (auto ? C.yellow : C.border),
          background: auto ? C.yellow + "20" : C.card, color: auto ? C.yellow : C.muted,
          cursor: "pointer", fontSize: 9, fontFamily: "monospace"
        }}>{auto ? PAUSE : PLAY}</button>
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={385} viewBox={"0 0 790 385"} style={{ background: "#08080d", borderRadius: 12, border: "1px solid " + C.border }}>

          <rect x={20} y={160} width={80} height={50} rx={8} fill={step === 0 ? C.cyan + "30" : C.card} stroke={step === 0 ? C.cyan : C.border} strokeWidth={step === 0 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={60} y={181} textAnchor={"middle"} fill={step === 0 ? C.cyan : C.muted} fontSize={10} fontWeight={700} fontFamily={"monospace"}>{"x"}</text>
          <text x={60} y={195} textAnchor={"middle"} fill={step === 0 ? C.cyan + "90" : C.dim} fontSize={8} fontFamily={"monospace"}>{"BF16"}</text>
          <text x={60} y={207} textAnchor={"middle"} fill={step === 0 ? C.cyan + "70" : C.dim} fontSize={7} fontFamily={"monospace"}>{"[d]"}</text>

          <line x1={100} y1={185} x2={130} y2={185} stroke={C.dim + "60"} strokeWidth={1.5}/>
          <line x1={130} y1={185} x2={130} y2={100} stroke={C.dim + "60"} strokeWidth={1.5}/>
          <line x1={130} y1={185} x2={130} y2={280} stroke={C.dim + "60"} strokeWidth={1.5}/>
          <line x1={130} y1={100} x2={155} y2={100} stroke={step >= 1 ? C.orange + "70" : C.dim + "30"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>
          <line x1={130} y1={280} x2={155} y2={280} stroke={step >= 4 ? C.purple + "70" : C.dim + "30"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>

          {/* NF4 block */}
          <rect x={156} y={70} width={100} height={60} rx={6} fill={step >= 1 && step <= 2 ? C.orange + "25" : C.card} stroke={step >= 1 && step <= 2 ? C.orange : C.border} strokeWidth={step >= 1 && step <= 2 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={206} y={90} textAnchor={"middle"} fill={step >= 1 ? C.orange : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"NF4 BLOCK"}</text>
          <text x={206} y={103} textAnchor={"middle"} fill={step >= 1 ? C.orange + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"64 \u00D7 4-bit weights"}</text>
          <text x={206} y={115} textAnchor={"middle"} fill={step >= 1 ? C.teal + "90" : C.dim} fontSize={7} fontFamily={"monospace"}>{"+ FP8 scale c"}</text>
          <text x={206} y={126} textAnchor={"middle"} fill={step >= 1 ? C.orange + "60" : C.dim} fontSize={6} fontFamily={"monospace"}>{"stored in GPU VRAM"}</text>

          <line x1={256} y1={100} x2={290} y2={100} stroke={step >= 2 ? C.yellow : C.dim + "30"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>
          <polygon points={"294,100 288,96 288,104"} fill={step >= 2 ? C.yellow : C.dim + "30"}/>
          <text x={273} y={92} textAnchor={"middle"} fill={step >= 2 ? C.yellow : C.dim} fontSize={7} fontFamily={"monospace"}>{"deQ"}</text>

          {/* Dequantize */}
          <rect x={295} y={74} width={110} height={52} rx={6} fill={step === 2 ? C.yellow + "25" : C.card} stroke={step === 2 ? C.yellow : C.border} strokeWidth={step === 2 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={350} y={92} textAnchor={"middle"} fill={step >= 2 ? C.yellow : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"DEQUANTIZE"}</text>
          <text x={350} y={105} textAnchor={"middle"} fill={step >= 2 ? C.yellow + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"NF4[idx] \u00D7 c"}</text>
          <text x={350} y={116} textAnchor={"middle"} fill={step >= 2 ? C.yellow + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{ARR + " BF16 (transient)"}</text>
          <text x={350} y={124} textAnchor={"middle"} fill={step >= 2 ? C.teal : C.dim} fontSize={6} fontFamily={"monospace"}>{"discarded after multiply"}</text>

          <line x1={405} y1={100} x2={440} y2={100} stroke={step >= 3 ? C.teal : C.dim + "30"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>
          <polygon points={"444,100 438,96 438,104"} fill={step >= 3 ? C.teal : C.dim + "30"}/>

          {/* W0x */}
          <rect x={446} y={68} width={110} height={64} rx={6} fill={step === 3 ? C.teal + "25" : C.card} stroke={step === 3 ? C.teal : C.border} strokeWidth={step === 3 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={501} y={88} textAnchor={"middle"} fill={step >= 3 ? C.teal : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"W\u2080x"}</text>
          <text x={501} y={102} textAnchor={"middle"} fill={step >= 3 ? C.teal + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"BF16 matmul"}</text>
          <text x={501} y={114} textAnchor={"middle"} fill={step >= 3 ? C.teal + "70" : C.dim} fontSize={7} fontFamily={"monospace"}>{LOCK + " no grad stored"}</text>
          <text x={501} y={126} textAnchor={"middle"} fill={step >= 3 ? C.teal + "60" : C.dim} fontSize={6} fontFamily={"monospace"}>{"frozen path"}</text>

          <line x1={556} y1={100} x2={638} y2={178} stroke={step >= 3 ? C.teal + "60" : C.dim + "20"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>

          {/* LoRA A */}
          <rect x={156} y={252} width={100} height={56} rx={6} fill={step === 4 ? C.blue + "25" : C.card} stroke={step === 4 ? C.blue : C.border} strokeWidth={step === 4 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={206} y={272} textAnchor={"middle"} fill={step >= 4 ? C.blue : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"A (DOWN)"}</text>
          <text x={206} y={285} textAnchor={"middle"} fill={step >= 4 ? C.blue + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"BF16, trainable"}</text>
          <text x={206} y={297} textAnchor={"middle"} fill={step >= 4 ? C.blue + "70" : C.dim} fontSize={7} fontFamily={"monospace"}>{"[r \u00D7 d]"}</text>
          <text x={206} y={307} textAnchor={"middle"} fill={C.green + "60"} fontSize={7} fontFamily={"monospace"}>{CHK + " grad stored"}</text>

          <line x1={256} y1={280} x2={310} y2={280} stroke={step >= 4 ? C.blue + "60" : C.dim + "20"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>
          <polygon points={"314,280 308,276 308,284"} fill={step >= 4 ? C.blue + "60" : C.dim + "20"}/>
          <text x={283} y={272} textAnchor={"middle"} fill={step >= 4 ? C.blue + "60" : C.dim} fontSize={7} fontFamily={"monospace"}>{"[r]"}</text>

          {/* LoRA B */}
          <rect x={316} y={252} width={100} height={56} rx={6} fill={step === 4 ? C.pink + "25" : C.card} stroke={step === 4 ? C.pink : C.border} strokeWidth={step === 4 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={366} y={272} textAnchor={"middle"} fill={step >= 4 ? C.pink : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"B (UP)"}</text>
          <text x={366} y={285} textAnchor={"middle"} fill={step >= 4 ? C.pink + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"BF16, trainable"}</text>
          <text x={366} y={297} textAnchor={"middle"} fill={step >= 4 ? C.pink + "70" : C.dim} fontSize={7} fontFamily={"monospace"}>{"[d \u00D7 r]"}</text>
          <text x={366} y={307} textAnchor={"middle"} fill={C.green + "60"} fontSize={7} fontFamily={"monospace"}>{CHK + " grad stored"}</text>

          <line x1={416} y1={280} x2={466} y2={280} stroke={step >= 4 ? C.pink + "60" : C.dim + "20"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>
          <polygon points={"470,280 464,276 464,284"} fill={step >= 4 ? C.pink + "60" : C.dim + "20"}/>

          {/* Scale */}
          <rect x={472} y={258} width={80} height={44} rx={6} fill={step === 4 ? C.yellow + "20" : C.card} stroke={step === 4 ? C.yellow : C.border} strokeWidth={step === 4 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={512} y={276} textAnchor={"middle"} fill={step >= 4 ? C.yellow : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"\u00D7 \u03B1/r"}</text>
          <text x={512} y={290} textAnchor={"middle"} fill={step >= 4 ? C.yellow + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"scaling"}</text>

          <line x1={552} y1={280} x2={638} y2={200} stroke={step >= 4 ? C.purple + "60" : C.dim + "20"} strokeWidth={1.5} style={{ transition: "stroke 0.3s" }}/>

          {/* Sum */}
          <circle cx={658} cy={188} r={28} fill={step >= 5 ? C.green + "25" : C.card} stroke={step >= 5 ? C.green : C.border} strokeWidth={step >= 5 ? 2.5 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={658} y={184} textAnchor={"middle"} fill={step >= 5 ? C.green : C.dim} fontSize={16} fontWeight={800} fontFamily={"monospace"}>{PLUS}</text>
          <text x={658} y={198} textAnchor={"middle"} fill={step >= 5 ? C.green + "80" : C.dim} fontSize={8} fontFamily={"monospace"}>{"sum"}</text>

          <line x1={686} y1={188} x2={720} y2={188} stroke={step >= 5 ? C.green : C.dim + "30"} strokeWidth={2} style={{ transition: "stroke 0.3s" }}/>
          <polygon points={"724,188 718,184 718,192"} fill={step >= 5 ? C.green : C.dim + "30"}/>
          <rect x={726} y={160} width={56} height={56} rx={8} fill={step >= 5 ? C.green + "25" : C.card} stroke={step >= 5 ? C.green : C.border} strokeWidth={step >= 5 ? 2 : 1} style={{ transition: "all 0.3s" }}/>
          <text x={754} y={182} textAnchor={"middle"} fill={step >= 5 ? C.green : C.dim} fontSize={9} fontWeight={700} fontFamily={"monospace"}>{"h"}</text>
          <text x={754} y={195} textAnchor={"middle"} fill={step >= 5 ? C.green + "80" : C.dim} fontSize={7} fontFamily={"monospace"}>{"BF16"}</text>
          <text x={754} y={207} textAnchor={"middle"} fill={step >= 5 ? C.green + "60" : C.dim} fontSize={6} fontFamily={"monospace"}>{"output"}</text>

          <rect x={548} y={222} width={230} height={22} rx={4} fill={step >= 5 ? C.green + "15" : C.card} stroke={step >= 5 ? C.green + "40" : C.border} strokeWidth={1}/>
          <text x={663} y={237} textAnchor={"middle"} fill={step >= 5 ? C.green : C.dim} fontSize={9} fontFamily={"monospace"}>{"h = W\u2080x + (\u03B1/r)\u00B7BAx"}</text>

          <text x={396} y={340} textAnchor={"middle"} fill={C.dim} fontSize={8} fontFamily={"monospace"}>{"Gradient only flows through LoRA (A, B). NF4 weights never see a gradient. Dequantized BF16 is temporary."}</text>
          <rect x={20} y={350} width={753} height={26} rx={4} fill={steps[step].color + "15"} stroke={steps[step].color + "40"} strokeWidth={1}/>
          <text x={396} y={366} textAnchor={"middle"} fill={steps[step].color} fontSize={9} fontFamily={"monospace"}>{steps[step].desc}</text>
        </svg>
      </div>

      <Insight color={C.yellow} icon={"\u26A1"} title={"Dequantization is the Key Innovation"}>
        {"QLoRA never stores BF16 weights. They exist only "}
        <span style={{ color: C.yellow, fontWeight: 700 }}>{"transiently during a forward pass"}</span>
        {", computed on-the-fly from the NF4 4-bit representation. GPU memory holds only 4-bit data permanently. The overhead is a lookup table operation \u2014 negligible vs matmul cost. Gradients propagate through the LoRA adapters only."}
      </Insight>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TAB 6 — MEMORY BREAKDOWN
   ═══════════════════════════════════════════════════════════ */
function MemBar(props) {
  var maxMem = props.maxMem;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: props.color, fontWeight: 700 }}>{props.label}</span>
        <span style={{ fontSize: 10, color: props.color, fontWeight: 800 }}>{props.value.toFixed(1) + " GB"}</span>
      </div>
      <div style={{ height: 20, background: C.border, borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: (props.value / maxMem * 100) + "%", height: "100%", background: props.color, borderRadius: 4, transition: "width 0.5s", minWidth: 4 }}/>
      </div>
      {props.sub ? <div style={{ fontSize: 9, color: C.muted, marginTop: 4 }}>{props.sub}</div> : null}
    </div>
  );
}

function TabMemory() {
  var _mo = useState(0); var model = _mo[0]; var setModel = _mo[1];
  var _r = useState(16); var rank = _r[0]; var setRank = _r[1];

  var models = [
    { name: "7B",  base_bf16: 14,  base_nf4: 3.5,  lora_bf16: 0.033, adam_full: 28,  adam_qlora: 0.066 },
    { name: "13B", base_bf16: 26,  base_nf4: 6.5,  lora_bf16: 0.055, adam_full: 52,  adam_qlora: 0.11 },
    { name: "33B", base_bf16: 66,  base_nf4: 16.5, lora_bf16: 0.14,  adam_full: 132, adam_qlora: 0.28 },
    { name: "65B", base_bf16: 130, base_nf4: 32.5, lora_bf16: 0.28,  adam_full: 260, adam_qlora: 0.56 },
  ];
  var m = models[model];
  var scale = rank / 16;
  var fullFT  = m.base_bf16 + m.adam_full + 4;
  var loraFT  = m.base_bf16 + m.lora_bf16 * scale * 2;
  var qloraFT = m.base_nf4  + m.lora_bf16 * scale * 2 + 0.5;
  var maxMem = fullFT;

  return (
    <div>
      <SectionTitle
        title={"Memory Breakdown \u2014 QLoRA vs Alternatives"}
        subtitle={"See exactly where every GB goes, interactively"} />

      <div style={{ display: "flex", gap: 10, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {models.map(function(md, i) {
          return (
            <button key={i} onClick={function() { setModel(i); }} style={{
              padding: "10px 22px", borderRadius: 8, border: "1.5px solid " + (model === i ? C.orange : C.border),
              background: model === i ? C.orange + "20" : C.card, color: model === i ? C.orange : C.muted,
              cursor: "pointer", fontSize: 11, fontWeight: 800, fontFamily: "monospace"
            }}>{md.name}</button>
          );
        })}
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 12px" }}>
          <span style={{ fontSize: 10, color: C.purple }}>{"LoRA rank r ="}</span>
          <input type={"range"} min={4} max={64} step={4} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }} style={{ width: 120, accentColor: C.purple }}/>
          <span style={{ fontSize: 10, color: C.purple, fontWeight: 700 }}>{rank}</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 14, maxWidth: 1100, margin: "0 auto 16px", flexWrap: "wrap" }}>

        <Card style={{ flex: 1, minWidth: 280 }}>
          <div style={{ fontSize: 12, fontWeight: 800, color: C.red, marginBottom: 4 }}>{"Full Fine-Tuning"}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: C.red, marginBottom: 14 }}>{fullFT.toFixed(0) + " GB"}</div>
          <MemBar label={"BF16 Base Weights"} value={m.base_bf16} color={C.red} sub={m.name + " \u00D7 2 bytes"} maxMem={maxMem}/>
          <MemBar label={"Adam m\u2081 + m\u2082"} value={m.adam_full} color={"#ef444470"} sub={"2\u00D7 model weights in FP32"} maxMem={maxMem}/>
          <MemBar label={"Gradients"} value={4} color={"#ef444440"} sub={"~2\u00D7 model size"} maxMem={maxMem}/>
          <div style={{ padding: "8px 12px", background: C.red + "10", borderRadius: 6, border: "1px solid " + C.red + "30", marginTop: 8 }}>
            <div style={{ fontSize: 10, color: C.red }}>{"Requires " + Math.ceil(fullFT / 80) + " \u00D7 A100 80GB GPUs minimum"}</div>
          </div>
        </Card>

        <Card style={{ flex: 1, minWidth: 280 }}>
          <div style={{ fontSize: 12, fontWeight: 800, color: C.purple, marginBottom: 4 }}>{"LoRA (BF16 base)"}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: C.purple, marginBottom: 14 }}>{loraFT.toFixed(1) + " GB"}</div>
          <MemBar label={"BF16 Base Weights"} value={m.base_bf16} color={C.purple} sub={m.name + " frozen, no grad"} maxMem={maxMem}/>
          <MemBar label={"LoRA Adapters + Optim"} value={m.lora_bf16 * scale * 2} color={C.pink} sub={"r=" + rank + " BF16 A+B + Adam states"} maxMem={maxMem}/>
          <MemBar label={"Gradients (LoRA)"} value={m.lora_bf16 * scale} color={C.pink + "60"} sub={"Tiny \u2014 only LoRA params"} maxMem={maxMem}/>
          <div style={{ padding: "8px 12px", background: C.purple + "10", borderRadius: 6, border: "1px solid " + C.purple + "30", marginTop: 8 }}>
            <div style={{ fontSize: 10, color: C.purple }}>{"~" + Math.round(fullFT / loraFT) + "\u00D7 less memory than full FT"}</div>
          </div>
        </Card>

        <Card style={{ flex: 1, minWidth: 280 }} highlight={true}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <div style={{ fontSize: 12, fontWeight: 800, color: C.orange, marginBottom: 4 }}>{"QLoRA"}</div>
            <Badge color={C.yellow}>{"\u2605 Best"}</Badge>
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, color: C.orange, marginBottom: 14 }}>{qloraFT.toFixed(1) + " GB"}</div>
          <MemBar label={"NF4 Base Weights"} value={m.base_nf4} color={C.orange} sub={"4-bit + double quant \u2248 " + m.base_nf4 + " GB"} maxMem={maxMem}/>
          <MemBar label={"BF16 LoRA Adapters"} value={m.lora_bf16 * scale} color={C.purple} sub={"r=" + rank + " BF16 A+B matrices"} maxMem={maxMem}/>
          <MemBar label={"Adam states (paged)"} value={0.5} color={C.blue} sub={"Paged to CPU RAM on demand"} maxMem={maxMem}/>
          <div style={{ padding: "8px 12px", background: C.orange + "10", borderRadius: 6, border: "1px solid " + C.orange + "50", marginTop: 8 }}>
            <div style={{ fontSize: 10, color: C.orange, fontWeight: 700 }}>{"~" + Math.round(fullFT / qloraFT) + "\u00D7 less memory \u2014 fits on consumer GPU! " + FIRE}</div>
          </div>
        </Card>
      </div>

      <Card style={{ maxWidth: 1100, margin: "0 auto 14px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"GPU Feasibility \u2014 Which GPU Can Run " + models[model].name + " Model?"}</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[
            { gpu: "RTX 3090 / 4090", vram: 24, color: C.yellow },
            { gpu: "A100 40GB",        vram: 40, color: C.blue },
            { gpu: "A100 80GB",        vram: 80, color: C.teal },
            { gpu: "H100 80GB",        vram: 80, color: C.green },
          ].map(function(g, i) {
            var canQ  = qloraFT <= g.vram;
            var canL  = loraFT  <= g.vram;
            var canF  = fullFT  <= g.vram;
            return (
              <div key={i} style={{ flex: 1, minWidth: 170, padding: "12px 14px", borderRadius: 8, background: g.color + "08", border: "1px solid " + g.color + "30" }}>
                <div style={{ fontSize: 10, fontWeight: 800, color: g.color, marginBottom: 6 }}>{g.gpu}</div>
                <div style={{ fontSize: 9, color: C.muted, marginBottom: 8 }}>{g.vram + " GB VRAM"}</div>
                <div style={{ fontSize: 9, color: canQ ? C.green : C.red, marginBottom: 3 }}>{(canQ ? CHK : "\u2717") + " QLoRA (" + qloraFT.toFixed(1) + " GB)"}</div>
                <div style={{ fontSize: 9, color: canL ? C.green : C.red, marginBottom: 3 }}>{(canL ? CHK : "\u2717") + " LoRA BF16 (" + loraFT.toFixed(1) + " GB)"}</div>
                <div style={{ fontSize: 9, color: canF ? C.green : C.red }}>{(canF ? CHK : "\u2717") + " Full FT (" + fullFT.toFixed(0) + " GB)"}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight color={C.orange} icon={FIRE} title={"The Consumer GPU Revolution"}>
        {"QLoRA made it possible to fine-tune a "}
        <span style={{ color: C.orange, fontWeight: 700 }}>{"65B parameter model on a single A100 80GB"}</span>
        {" \u2014 previously impossible. More importantly, a "}
        <span style={{ color: C.yellow, fontWeight: 700 }}>{"7B model fits on a $500 RTX 3090"}</span>
        {". This democratized LLM fine-tuning: researchers, startups, and hobbyists can now produce Llama-quality fine-tuned models without cloud GPU clusters."}
      </Insight>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   ROOT APP
   ═══════════════════════════════════════════════════════════ */
function App() {
  var _t = useState(0); var tab = _t[0]; var setTab = _t[1];
  var tabs = ["Big Picture", "NF4 Quantization", "Double Quantization", "Paged Optimizers", "Forward Pass", "Memory Breakdown"];

  return (
    <ZoomWrapper>
      <div style={{ background: C.bg, minHeight: "100vh", padding: "20px 16px", fontFamily: "monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 16 }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: C.orange, display: "inline-block", letterSpacing: -1 }}>{"QLoRA"}</div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{"Interactive Visual Walkthrough " + DASH + " Quantized Low-Rank Adaptation"}</div>
          <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 10, flexWrap: "wrap" }}>
            <Badge color={C.orange}>{"NF4 Quantization"}</Badge>
            <Badge color={C.teal}>{"Double Quantization"}</Badge>
            <Badge color={C.blue}>{"Paged Optimizers"}</Badge>
            <Badge color={C.purple}>{"LoRA Adapters"}</Badge>
          </div>
        </div>
        <TabBar tabs={tabs} active={tab} onChange={setTab}/>
        <div key={tab} style={{ animation: "fadeIn 0.3s ease" }}>
          {tab === 0 && <TabBigPicture/>}
          {tab === 1 && <TabNF4/>}
          {tab === 2 && <TabDoubleQ/>}
          {tab === 3 && <TabPagedOpt/>}
          {tab === 4 && <TabForwardPass/>}
          {tab === 5 && <TabMemory/>}
        </div>
        <div style={{ textAlign: "center", marginTop: 24, padding: "14px", borderTop: "1px solid " + C.border }}>
          <div style={{ fontSize: 9, color: C.dim }}>{"Based on QLoRA: Efficient Finetuning of Quantized LLMs \u2014 Dettmers et al., 2023"}</div>
        </div>
      </div>
    </ZoomWrapper>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
</script>
</body>
</html>
"""

QLORA_VISUAL_HEIGHT = 1600