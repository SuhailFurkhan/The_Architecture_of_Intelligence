"""
Self-contained HTML for the PEFT LoRA interactive walkthrough.
Covers: Big Picture, Forward Pass, Memory & Cost, Data Pipeline,
Rank & Alpha, and Merge & Deploy.
Embed in Streamlit via st.components.v1.html(LORA_VISUAL_HTML, height=LORA_VISUAL_HEIGHT).
"""

LORA_VISUAL_HTML = """
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
  @keyframes flowDown { 0%{transform:translateY(-6px);opacity:0} 50%{opacity:1} 100%{transform:translateY(6px);opacity:0} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState = React.useState;
var useEffect = React.useEffect;
var useMemo = React.useMemo;

var C = {
  bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
  accent: "#a78bfa", blue: "#4ecdc4", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
  lora: "#a78bfa", frozen: "#3f3f46",
};

var MUL = "\u00D7";
var ARR = "\u2192";
var DASH = "\u2014";
var CHK = "\u2713";
var WARN = "\u26A0";
var LQ = "\u201C";
var RQ = "\u201D";
var LARR = "\u2190";
var PLAY = "\u25B6";
var PAUSE = "\u23F8";
var BULB = "💡";
var TARG = "🎯";
var LOCK = "🔒";
var UNLOCK = "🔓";
var BRAIN = "🧠";
var FIRE = "🔥";
var GEAR = "\u2699";
var CHART = "📈";
var DARR = "\u2193";
var UARR = "\u2191";
var MERGE = "\u21A0";
var PLUS = "+";

/* shared components */

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
          }}>{t}</button>
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

function Badge(props) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: (props.color || C.accent) + "20",
      border: "1px solid " + (props.color || C.accent) + "50",
      color: props.color || C.accent, fontSize: 9, fontWeight: 700,
      fontFamily: "monospace", margin: "0 3px",
    }}>{props.children}</span>
  );
}

/* ===============================================================
   TAB 1: THE BIG PICTURE
   =============================================================== */
function TabBigPicture() {
  var _a = useState(false); var animated = _a[0], setAnimated = _a[1];
  var _h = useState(-1); var hov = _h[0], setHov = _h[1];

  useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);

  var methods = [
    { name: "Additive", ex: "Adapters", desc: "Insert new modules in-series. Stays at inference.", color: C.blue },
    { name: "Reparameterization", ex: "LoRA", desc: "Decompose weight updates into A\u00D7B. Merges and vanishes.", color: C.accent, star: true },
    { name: "Selective", ex: "BitFit", desc: "Unfreeze tiny subset of existing params only.", color: C.cyan },
    { name: "Hybrid", ex: "QLoRA", desc: "LoRA adapters + 4-bit quantized base.", color: C.orange, star: true },
    { name: "Prompt-based", ex: "Prefix Tuning", desc: "Learn soft vectors prepended to input. Fewest params.", color: C.pink },
  ];

  return (
    <div>
      <SectionTitle title="The Big Picture" subtitle={"PEFT: get 95%+ of full fine-tuning quality at 1% of the memory " + DASH + " by freezing everything and training almost nothing"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={1050} height={290} viewBox="0 0 780 290" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* === LEFT: FROZEN BASE === */}
          <text x={110} y={22} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">PRE-TRAINED BASE</text>
          <text x={110} y={36} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">(7B parameters - ALL FROZEN)</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (
              <g key={"base" + i}>
                <rect x={30} y={48 + i * 26} width={160} height={20} rx={4} fill={C.dim + "20"} stroke={C.dim + "50"} strokeWidth={1} />
                <rect x={30} y={48 + i * 26} width={animated ? 160 : 0} height={20} rx={4} fill={C.dim + "15"} style={{ transition: "width 1.2s ease-out", transitionDelay: (i * 0.08) + "s" }} />
                <text x={110} y={62 + i * 26} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{LOCK + " Layer " + (i + 1)}</text>
              </g>
            );
          })}
          <text x={110} y={268} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">requires_grad = False</text>
          <text x={110} y={280} textAnchor="middle" fill={C.dim + "80"} fontSize={8} fontFamily="monospace">No gradients. No optimizer states.</text>

          {/* === PLUS === */}
          <text x={228} y={152} textAnchor="middle" fill={C.accent} fontSize={28} fontWeight={800}>{PLUS}</text>
          <text x={228} y={170} textAnchor="middle" fill={C.accent + "60"} fontSize={8} fontFamily="monospace">frozen +</text>

          {/* === CENTER: LORA ADAPTERS === */}
          <text x={370} y={22} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">LoRA ADAPTERS</text>
          <text x={370} y={36} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">(~8.4M params - TRAINABLE)</text>
          {[0,1,2,3].map(function(i) {
            return (
              <g key={"lora" + i}>
                <rect x={305} y={55 + i * 52} width={130} height={40} rx={6} fill={C.accent + "12"} stroke={C.accent + "50"} strokeWidth={1.5} />
                <text x={370} y={72 + i * 52} textAnchor="middle" fill={C.accent} fontSize={8} fontWeight={700} fontFamily="monospace">{"Layer " + (i + 1) + " Adapter"}</text>
                <text x={345} y={86 + i * 52} textAnchor="middle" fill={C.blue} fontSize={7} fontFamily="monospace">A [8\u00D74096]</text>
                <text x={395} y={86 + i * 52} textAnchor="middle" fill={C.pink} fontSize={7} fontFamily="monospace">B [4096\u00D78]</text>
              </g>
            );
          })}
          <text x={370} y={268} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">0.12% of model</text>
          <text x={370} y={280} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">~33 MB saved per task</text>

          {/* === ARROW === */}
          <line x1={500} y1={145} x2={560} y2={145} stroke={C.green} strokeWidth={2} />
          <polygon points="565,145 558,140 558,150" fill={C.green} />
          <text x={530} y={133} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">MERGE</text>
          <text x={530} y={162} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">W\u2080 + BA</text>

          {/* === RIGHT: RESULT === */}
          <text x={670} y={22} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">FINE-TUNED MODEL</text>
          <text x={670} y={36} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">(Zero inference overhead)</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            var w = animated ? 160 : 0;
            return (
              <g key={"ft" + i}>
                <rect x={590} y={48 + i * 26} width={160} height={20} rx={4} fill={C.green + "08"} stroke={C.green + "30"} strokeWidth={1} />
                <rect x={590} y={48 + i * 26} width={w} height={20} rx={4} fill={C.green + "15"} style={{ transition: "width 1s ease-out", transitionDelay: (0.8 + i * 0.1) + "s" }} />
                <text x={670} y={62 + i * 26} textAnchor="middle" fill={C.green + "80"} fontSize={7} fontFamily="monospace">{"W\u2080 + (α/r)BA"}</text>
              </g>
            );
          })}
          <text x={670} y={268} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">Same size as original</text>
          <text x={670} y={280} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">Adapters vanished into weights</text>
        </svg>
      </div>

      {/* PEFT taxonomy */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"PEFT Methods Taxonomy " + DASH + " Five Families"}</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {methods.map(function(m, i) {
            var isH = hov === i;
            return (
              <div key={i} onMouseEnter={function() { setHov(i); }} onMouseLeave={function() { setHov(-1); }}
                style={{ flex: 1, minWidth: 140, padding: "12px 14px", borderRadius: 8, cursor: "pointer",
                  background: m.color + (isH ? "18" : "08"), border: "1.5px solid " + m.color + (isH ? "70" : "25"),
                  transition: "all 0.25s" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <div style={{ fontSize: 10, fontWeight: 800, color: m.color }}>{m.name}</div>
                  {m.star && <div style={{ fontSize: 9, color: C.yellow }}>{"\u2605 Popular"}</div>}
                </div>
                <div style={{ fontSize: 9, color: m.color + "90", fontFamily: "monospace", marginBottom: 4 }}>{m.ex}</div>
                <div style={{ fontSize: 9, color: C.muted, lineHeight: 1.6 }}>{m.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Memory summary */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Why PEFT? " + DASH + " The Memory Math (7B Model)"}</div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
          {[
            { label: "Full Fine-Tuning", mem: "~100 GB", breakdown: "14 weights + 14 grad + 56 optimizer + activations", color: C.red },
            { label: "LoRA (BF16 base)", mem: "~20 GB", breakdown: "14 frozen + 0.1 LoRA weights + 0.1 grad + 0.5 optimizer", color: C.accent },
            { label: "QLoRA (4-bit base)", mem: "~10 GB", breakdown: "3.5 NF4 frozen + 0.1 LoRA weights + 0.1 grad + 0.5 optimizer", color: C.green },
          ].map(function(v, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 220, padding: "12px 16px", borderRadius: 8, background: v.color + "08", border: "1px solid " + v.color + "30" }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: v.color, marginBottom: 6 }}>{v.label}</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: v.color, marginBottom: 6 }}>{v.mem}</div>
                <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.7 }}>{v.breakdown}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={BRAIN} title="The Core Insight">
        Fine-tuning changes (<span style={{color:C.accent, fontWeight:700}}>\u0394W</span>) have <span style={{color:C.yellow, fontWeight:700}}>low intrinsic rank</span>. You don't need 16.7M parameters to express "now be a medical assistant." The real adaptation lives in a <span style={{color:C.green, fontWeight:700}}>tiny subspace</span>. LoRA finds that subspace directly via two small matrices A and B whose product approximates \u0394W. Base model <span style={{color:C.cyan, fontWeight:700}}>stays frozen</span>. Zero catastrophic forgetting. Zero inference overhead after merging.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: FORWARD PASS
   =============================================================== */
function TabForwardPass() {
  var _r = useState(8); var rank = _r[0], setRank = _r[1];
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _au = useState(false); var autoP = _au[0], setAutoP = _au[1];
  var _al = useState(16); var alpha = _al[0], setAlpha = _al[1];

  var DIM = 4096;
  var aParams = rank * DIM;
  var bParams = DIM * rank;
  var totalLoRA = aParams + bParams;
  var fullDW = DIM * DIM;
  var reduction = Math.round(fullDW / totalLoRA);
  var scaling = (alpha / rank).toFixed(2);

  var steps = [
    { title: "Input x arrives", color: C.cyan, desc: "A token's hidden state x (shape [4096]) enters the layer. It will be sent down BOTH paths simultaneously." },
    { title: "Frozen path: W\u2080x", color: C.dim, desc: "The frozen pre-trained matrix W\u2080 [4096\u00D74096] computes its output. No gradient is stored. This is pure forward pass computation." },
    { title: "LoRA down: Ax", color: C.blue, desc: "Matrix A [" + rank + "\u00D74096] compresses x from 4096 dimensions down to just " + rank + ". This is the bottleneck - capturing the essential adaptation signal." },
    { title: "LoRA up: B(Ax)", color: C.pink, desc: "Matrix B [4096\u00D7" + rank + "] expands the compressed signal back up to 4096 dimensions. B starts at zero so initial contribution is exactly zero." },
    { title: "Scale by \u03B1/r = " + scaling, color: C.yellow, desc: "Multiply the LoRA output by " + alpha + "/" + rank + " = " + scaling + ". This controls how strongly the adapter steers the output. Tune \u03B1 without retraining." },
    { title: "Sum: h = W\u2080x + (\u03B1/r)BAx", color: C.green, desc: "Element-wise addition. The frozen path provides the pre-trained behavior. The LoRA path provides the task-specific correction. Both are [4096] vectors." },
  ];

  useEffect(function() { if (!autoP) return; var t = setInterval(function() { setStep(function(s) { return (s + 1) % steps.length; }); }, 2500); return function() { clearInterval(t); }; }, [autoP, rank, alpha]);

  return (
    <div>
      <SectionTitle title="The Forward Pass" subtitle={"How x flows through the frozen path AND the LoRA bypass simultaneously"} />

      {/* Rank and Alpha controls */}
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        <Card style={{ minWidth: 280 }}>
          <div style={{ fontSize: 10, color: C.accent, fontWeight: 700, marginBottom: 8 }}>{"Rank r = " + rank}</div>
          <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }}
            style={{ width: "100%", accentColor: C.accent }} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
            <span style={{ fontSize: 8, color: C.muted }}>r=1 (minimal)</span>
            <span style={{ fontSize: 8, color: C.muted }}>r=64 (heavy)</span>
          </div>
        </Card>
        <Card style={{ minWidth: 280 }}>
          <div style={{ fontSize: 10, color: C.yellow, fontWeight: 700, marginBottom: 8 }}>{"\u03B1 (alpha) = " + alpha + "  \u2192  \u03B1/r = " + scaling}</div>
          <input type="range" min={1} max={64} value={alpha} onChange={function(e) { setAlpha(parseInt(e.target.value)); }}
            style={{ width: "100%", accentColor: C.yellow }} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
            <span style={{ fontSize: 8, color: C.muted }}>\u03B1=1 (gentle)</span>
            <span style={{ fontSize: 8, color: C.muted }}>\u03B1=64 (strong)</span>
          </div>
        </Card>
        <Card>
          <div style={{ display: "flex", gap: 24 }}>
            <StatBox label={"LORA PARAMS"} value={(totalLoRA / 1000).toFixed(0) + "K"} color={C.accent} />
            <StatBox label={"FULL \u0394W"} value={(fullDW / 1e6).toFixed(1) + "M"} color={C.red} />
            <StatBox label={"REDUCTION"} value={reduction + "x"} color={C.green} />
            <StatBox label={"SCALING"} value={"\u03B1/r=" + scaling} color={C.yellow} />
          </div>
        </Card>
      </div>

      {/* Step selector */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          var on = step === i;
          return (<button key={i} onClick={function() { setStep(i); setAutoP(false); }} style={{
            padding: "6px 10px", borderRadius: 6, border: "1.5px solid " + (on ? s.color : C.border),
            background: on ? s.color + "20" : C.card, color: on ? s.color : C.muted,
            cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
          }}>{i + 1}</button>);
        })}
        <button onClick={function() { setAutoP(!autoP); }} style={{ padding: "6px 10px", borderRadius: 6, border: "1.5px solid " + (autoP ? C.yellow : C.border), background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted, cursor: "pointer", fontSize: 9, fontFamily: "monospace" }}>{autoP ? PAUSE : PLAY}</button>
      </div>

      {/* Forward pass diagram */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={300} viewBox="0 0 780 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Input x */}
          <rect x={20} y={120} width={80} height={40} rx={8} fill={step === 0 ? C.cyan + "30" : C.card} stroke={step === 0 ? C.cyan : C.border} strokeWidth={step === 0 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={60} y={137} textAnchor="middle" fill={step === 0 ? C.cyan : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">x</text>
          <text x={60} y={151} textAnchor="middle" fill={step === 0 ? C.cyan + "90" : C.dim} fontSize={8} fontFamily="monospace">[4096]</text>

          {/* Split lines */}
          <line x1={100} y1={140} x2={130} y2={140} stroke={C.dim + "60"} strokeWidth={1.5} />
          <line x1={130} y1={140} x2={130} y2={80} stroke={C.dim + "60"} strokeWidth={1.5} />
          <line x1={130} y1={140} x2={130} y2={210} stroke={C.dim + "60"} strokeWidth={1.5} />
          <line x1={130} y1={80} x2={165} y2={80} stroke={step >= 1 ? C.dim + "90" : C.dim + "30"} strokeWidth={1.5} />
          <line x1={130} y1={210} x2={165} y2={210} stroke={step >= 2 ? C.accent + "80" : C.dim + "30"} strokeWidth={1.5} />

          {/* Frozen W0 */}
          <rect x={165} y={55} width={130} height={50} rx={8} fill={step === 1 ? C.dim + "40" : C.card} stroke={step === 1 ? C.muted : C.border} strokeWidth={step === 1 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={230} y={75} textAnchor="middle" fill={step === 1 ? C.muted : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">W\u2080 (FROZEN)</text>
          <text x={230} y={90} textAnchor="middle" fill={step === 1 ? C.dim : C.dim + "60"} fontSize={8} fontFamily="monospace">[4096 \u00D7 4096]</text>
          <text x={230} y={102} textAnchor="middle" fill={C.dim + "60"} fontSize={7} fontFamily="monospace">{LOCK + " no grad stored"}</text>
          <line x1={295} y1={80} x2={450} y2={80} stroke={step >= 1 ? C.muted + "60" : C.dim + "20"} strokeWidth={1.5} />

          {/* LoRA A */}
          <rect x={165} y={185} width={110} height={50} rx={8} fill={step === 2 ? C.blue + "30" : C.card} stroke={step === 2 ? C.blue : C.border} strokeWidth={step === 2 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={220} y={205} textAnchor="middle" fill={step === 2 ? C.blue : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">A (DOWN)</text>
          <text x={220} y={219} textAnchor="middle" fill={step === 2 ? C.blue + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">{"[" + rank + " \u00D7 4096]"}</text>
          <text x={220} y={231} textAnchor="middle" fill={C.accent + "60"} fontSize={7} fontFamily="monospace">{"random init"}</text>

          {/* Arrow A to B */}
          <line x1={275} y1={210} x2={315} y2={210} stroke={step >= 2 ? C.blue + "60" : C.dim + "20"} strokeWidth={1.5} />
          <polygon points={"318,210 313,207 313,213"} fill={step >= 2 ? C.blue + "60" : C.dim + "20"} />
          <text x={296} y={205} textAnchor="middle" fill={step >= 2 ? C.blue + "60" : C.dim + "30"} fontSize={7} fontFamily="monospace">{"[" + rank + "]"}</text>

          {/* LoRA B */}
          <rect x={318} y={185} width={110} height={50} rx={8} fill={step === 3 ? C.pink + "30" : C.card} stroke={step === 3 ? C.pink : C.border} strokeWidth={step === 3 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={373} y={205} textAnchor="middle" fill={step === 3 ? C.pink : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">B (UP)</text>
          <text x={373} y={219} textAnchor="middle" fill={step === 3 ? C.pink + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">{"[4096 \u00D7 " + rank + "]"}</text>
          <text x={373} y={231} textAnchor="middle" fill={C.accent + "60"} fontSize={7} fontFamily="monospace">{"zeros init"}</text>

          {/* Arrow B to scale */}
          <line x1={428} y1={210} x2={460} y2={210} stroke={step >= 3 ? C.pink + "60" : C.dim + "20"} strokeWidth={1.5} />
          <polygon points={"463,210 458,207 458,213"} fill={step >= 3 ? C.pink + "60" : C.dim + "20"} />

          {/* Scale */}
          <rect x={463} y={187} width={85} height={46} rx={8} fill={step === 4 ? C.yellow + "20" : C.card} stroke={step === 4 ? C.yellow : C.border} strokeWidth={step === 4 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={505} y={207} textAnchor="middle" fill={step === 4 ? C.yellow : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace">{"\u00D7 \u03B1/r"}</text>
          <text x={505} y={220} textAnchor="middle" fill={step === 4 ? C.yellow + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">{"=" + scaling}</text>
          <text x={505} y={230} textAnchor="middle" fill={C.yellow + "50"} fontSize={7} fontFamily="monospace">{"volume knob"}</text>

          {/* Arrow scale to sum */}
          <line x1={548} y1={210} x2={580} y2={210} stroke={step >= 4 ? C.yellow + "60" : C.dim + "20"} strokeWidth={1.5} />
          <line x1={580} y1={210} x2={580} y2={140} stroke={step >= 4 ? C.yellow + "60" : C.dim + "20"} strokeWidth={1.5} />
          <line x1={580} y1={140} x2={610} y2={140} stroke={step >= 4 ? C.yellow + "60" : C.dim + "20"} strokeWidth={1.5} />

          {/* Arrow frozen to sum */}
          <line x1={450} y1={80} x2={580} y2={80} stroke={step >= 1 ? C.muted + "40" : C.dim + "20"} strokeWidth={1.5} />
          <line x1={580} y1={80} x2={580} y2={140} stroke={step >= 1 ? C.muted + "40" : C.dim + "20"} strokeWidth={1.5} />

          {/* Sum node */}
          <circle cx={630} cy={140} r={22} fill={step === 5 ? C.green + "25" : C.card} stroke={step === 5 ? C.green : C.border} strokeWidth={step === 5 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={630} y={145} textAnchor="middle" fill={step === 5 ? C.green : C.dim} fontSize={16} fontWeight={800}>{PLUS}</text>

          {/* Output h */}
          <line x1={652} y1={140} x2={700} y2={140} stroke={step === 5 ? C.green + "80" : C.dim + "30"} strokeWidth={2} />
          <polygon points={"703,140 697,136 697,144"} fill={step === 5 ? C.green + "80" : C.dim + "30"} />
          <rect x={703} y={118} width={65} height={44} rx={8} fill={step === 5 ? C.green + "15" : C.card} stroke={step === 5 ? C.green : C.border} strokeWidth={step === 5 ? 2 : 1} style={{ transition: "all 0.3s" }} />
          <text x={735} y={137} textAnchor="middle" fill={step === 5 ? C.green : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">h</text>
          <text x={735} y={151} textAnchor="middle" fill={step === 5 ? C.green + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">[4096]</text>

          {/* Formula at bottom */}
          <text x={390} y={278} textAnchor="middle" fill={C.accent + "80"} fontSize={11} fontWeight={700} fontFamily="monospace">{"h = W\u2080x + (\u03B1/r) \u00B7 BAx"}</text>
          <text x={390} y={292} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"   = W\u2080x + (" + alpha + "/" + rank + ") \u00B7 BAx    [scaling = " + scaling + "]"}</text>
        </svg>
      </div>

      {/* Active step detail */}
      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: steps[step].color }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: steps[step].color, marginBottom: 6 }}>{"Step " + (step + 1) + ": " + steps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{steps[step].desc}</div>
      </Card>

      {/* Initialization callout */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"Initialization " + DASH + " Why B Starts at Zero"}</div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {[
            { label: "A init", value: "Random Gaussian", desc: "Small random values provide diverse projections", color: C.blue },
            { label: "B init", value: "All Zeros", desc: "\u0394W = B\u00D7A = 0\u00D7A = 0 at step 0", color: C.pink },
            { label: "Effect", value: "W\u2080 + 0 = W\u2080", desc: "Model starts EXACTLY as pre-trained. No disruption.", color: C.green },
            { label: "As training progresses", value: "B learns", desc: "Gradually steers toward task. Safe, stable start.", color: C.yellow },
          ].map(function(v, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 160, padding: "10px 14px", borderRadius: 8, background: v.color + "08", border: "1px solid " + v.color + "25" }}>
                <div style={{ fontSize: 9, color: v.color, fontWeight: 700, marginBottom: 3 }}>{v.label}</div>
                <div style={{ fontSize: 13, fontWeight: 800, color: v.color, marginBottom: 3, fontFamily: "monospace" }}>{v.value}</div>
                <div style={{ fontSize: 9, color: C.muted }}>{v.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={TARG} title="Why the Parallel Path Matters">
        Bottleneck Adapters insert modules <span style={{color:C.red, fontWeight:700}}>in series</span> between layers - they're always there at inference, adding latency permanently. LoRA runs <span style={{color:C.green, fontWeight:700}}>in parallel</span> to the frozen weights. After training, W_merged = W\u2080 + (\u03B1/r)BA is just <span style={{color:C.yellow, fontWeight:700}}>one matrix addition</span>. The adapters vanish completely. <span style={{color:C.accent, fontWeight:700}}>Zero inference overhead.</span>
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: MEMORY & COST
   =============================================================== */
function TabMemory() {
  var _m = useState(1); var modelSel = _m[0], setModelSel = _m[1];
  var _r = useState(8); var rank = _r[0], setRank = _r[1];

  var models = [
    { name: "LLaMA 1B", params: 1.24, color: C.green },
    { name: "LLaMA 7B", params: 7, color: C.blue },
    { name: "LLaMA 13B", params: 13, color: C.purple },
    { name: "LLaMA 70B", params: 70, color: C.orange },
  ];

  var m = models[modelSel];
  var wMem = m.params * 2;
  var gMem_full = m.params * 2;
  var oMem_full = m.params * 8;
  var act_full = m.params < 5 ? 3 : m.params < 15 ? 12 : m.params < 50 ? 25 : 60;
  var total_full = wMem + gMem_full + oMem_full + act_full;

  var loraParamsM = (32 * 4 * 2 * rank * 4096) / 1e6;
  var loraMem = loraParamsM * 2 / 1000;
  var loraGrad = loraMem;
  var loraOpt = loraParamsM * 8 / 1000;
  var act_lora = m.params < 5 ? 1.5 : m.params < 15 ? 5 : m.params < 50 ? 10 : 25;
  var total_lora = wMem + loraMem + loraGrad + loraOpt + act_lora;

  var qMem = m.params * 0.5;
  var total_qlora = qMem + loraMem + loraGrad + loraOpt + act_lora * 0.7;

  var pctTrained = ((loraParamsM) / (m.params * 1000) * 100).toFixed(3);

  return (
    <div>
      <SectionTitle title="Memory & Cost" subtitle={"Where LoRA saves memory " + DASH + " and why the savings are so dramatic"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 12, flexWrap: "wrap" }}>
        {models.map(function(mod, i) {
          var on = modelSel === i;
          return (<button key={i} onClick={function() { setModelSel(i); }} style={{
            padding: "8px 16px", borderRadius: 8, border: "1.5px solid " + (on ? mod.color : C.border),
            background: on ? mod.color + "20" : C.card, color: on ? mod.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{mod.name}</button>);
        })}
      </div>

      <Card style={{ maxWidth: 1100, margin: "0 auto 12px" }}>
        <div style={{ fontSize: 10, color: C.accent, fontWeight: 700, marginBottom: 6 }}>{"LoRA Rank r = " + rank + " \u2192 ~" + loraParamsM.toFixed(1) + "M trainable params (" + pctTrained + "% of model)"}</div>
        <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }}
          style={{ width: "100%", accentColor: C.accent }} />
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
          <span style={{ fontSize: 8, color: C.muted }}>r=1 (minimal)</span>
          <span style={{ fontSize: 8, color: C.muted }}>r=64 (aggressive)</span>
        </div>
      </Card>

      {/* Three-way comparison */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"GPU Memory Comparison " + DASH + " " + m.name + " (BF16)"}</div>

        {[
          { label: "Full Fine-Tuning", total: total_full, color: C.red, items: [
            { name: "Weights", val: wMem, desc: "BF16" },
            { name: "Gradients", val: gMem_full, desc: "eliminated by LoRA" },
            { name: "Optimizer (Adam)", val: oMem_full, desc: "eliminated by LoRA" },
            { name: "Activations", val: act_full, desc: "full backprop paths" },
          ]},
          { label: "LoRA (BF16 frozen)", total: total_lora, color: C.accent, items: [
            { name: "Frozen Weights", val: wMem, desc: "forward only" },
            { name: "LoRA Weights", val: loraMem, desc: "~" + (loraParamsM * 2).toFixed(0) + " MB" },
            { name: "LoRA Gradients", val: loraGrad, desc: "~" + (loraParamsM * 2).toFixed(0) + " MB" },
            { name: "LoRA Optimizer", val: loraOpt, desc: "~" + (loraParamsM * 8).toFixed(0) + " MB" },
            { name: "Activations", val: act_lora, desc: "reduced paths" },
          ]},
          { label: "QLoRA (4-bit frozen)", total: total_qlora, color: C.green, items: [
            { name: "4-bit Weights", val: qMem, desc: "NF4 quantized" },
            { name: "LoRA Weights", val: loraMem, desc: "BF16" },
            { name: "LoRA Gradients", val: loraGrad, desc: "BF16" },
            { name: "LoRA Optimizer", val: loraOpt, desc: "FP32" },
            { name: "Activations", val: act_lora * 0.7, desc: "reduced" },
          ]},
        ].map(function(method, mi) {
          var maxVal = total_full;
          return (
            <div key={mi} style={{ marginBottom: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: method.color }}>{method.label}</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: method.color }}>{"~" + method.total.toFixed(0) + " GB"}</div>
                  {mi > 0 && <div style={{ fontSize: 9, color: C.green, fontFamily: "monospace" }}>{Math.round((1 - method.total / total_full) * 100) + "% less"}</div>}
                </div>
              </div>
              <div style={{ display: "flex", height: 28, borderRadius: 6, overflow: "hidden", border: "1px solid " + method.color + "30" }}>
                {method.items.map(function(item, ii) {
                  var w = Math.max(1, (item.val / maxVal) * 100);
                  var colors = [C.blue, C.red, C.orange, C.yellow, C.cyan, C.pink];
                  return (
                    <div key={ii} title={item.name + ": " + item.val.toFixed(1) + " GB"} style={{
                      width: w + "%", height: "100%", background: colors[ii % colors.length] + "40",
                      borderRight: "1px solid " + C.bg, transition: "width 0.5s",
                      display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
                    }}>
                      {w > 5 && <span style={{ fontSize: 7, color: colors[ii % colors.length], fontFamily: "monospace", whiteSpace: "nowrap" }}>{item.name}</span>}
                    </div>
                  );
                })}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 4 }}>
                {method.items.map(function(item, ii) {
                  var colors = [C.blue, C.red, C.orange, C.yellow, C.cyan, C.pink];
                  return (
                    <span key={ii} style={{ fontSize: 8, color: colors[ii % colors.length], fontFamily: "monospace" }}>
                      {item.name + ": " + item.val.toFixed(1) + "GB"}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })}
      </Card>

      {/* Gradient / optimizer insight */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.cyan, marginBottom: 10 }}>{"Why Frozen = No Optimizer States " + DASH + " The Adam Savings"}</div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
          {[
            { label: "Per-param in Adam", items: ["Weight: 2 bytes (BF16)", "Gradient: 2 bytes", "Momentum: 4 bytes (FP32)", "Variance: 4 bytes (FP32)", "Total: 12 bytes/param"], color: C.red },
            { label: "Frozen params (7B)", items: ["Weight: 2 bytes " + CHK, "Gradient: NONE " + CHK, "Momentum: NONE " + CHK, "Variance: NONE " + CHK, "Total: 2 bytes/param"], color: C.green },
          ].map(function(col, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 200, padding: "10px 14px", borderRadius: 8, background: col.color + "08", border: "1px solid " + col.color + "25" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: col.color, marginBottom: 6 }}>{col.label}</div>
                {col.items.map(function(it, j) { return (<div key={j} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8, fontFamily: "monospace" }}>{"  " + it}</div>); })}
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={CHART} title={"Storage Per Task " + DASH + " The Multi-Task Advantage"}>
        Full fine-tuning: 3 tasks = <span style={{color:C.red, fontWeight:700}}>{(m.params * 2 * 3).toFixed(0) + " GB"}</span> of checkpoints. LoRA: 3 tasks = <span style={{color:C.green, fontWeight:700}}>{m.params * 2 + " GB base + 3 \u00D7 ~50MB"}</span>. Scale to 100 tasks: Full FT <span style={{color:C.red}}>{(m.params * 2 * 100).toFixed(0) + " GB"}</span> vs LoRA <span style={{color:C.green}}>{(m.params * 2 + 5).toFixed(0) + " GB"}</span>. One base model, swap adapters in milliseconds.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: DATA PIPELINE
   =============================================================== */
function TabDataPipeline() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _au = useState(false); var autoP = _au[0], setAutoP = _au[1];

  var pSteps = [
    { title: "Raw Data (JSONL)", color: C.blue,
      desc: "Identical to full fine-tuning. LoRA doesn't change what goes INTO the model, only what happens INSIDE it during training.",
      note: "Same as Full FT",
      content: '{"instruction": "Classify sentiment",  "input": "This movie was breathtaking",  "output": "positive"}' },
    { title: "Template Formatting", color: C.cyan,
      desc: "Model-specific chat template applied. Same as full fine-tuning. The model was pre-trained expecting a specific format - wrong template degrades performance.",
      note: "Same as Full FT",
      content: '<s>[INST] Classify sentiment: This movie was breathtaking [/INST] positive</s>' },
    { title: "Tokenization", color: C.purple,
      desc: "Text converted to integer IDs. Tokenizer is fixed - never changes during training or fine-tuning of any kind.",
      note: "Same as Full FT",
      content: '[1, 518, 25580, 29962, 4134, 1598, ..., 6374, 2]  -> 20 tokens' },
    { title: "Labels + Loss Mask", color: C.yellow,
      desc: "-100 means ignore this token. The model is graded ONLY on predicting the response tokens. Instruction tokens are masked out.",
      note: "Same as Full FT",
      content: 'Labels: [-100, -100, ..., -100,  6374,  2 ]  (instruction=ignored, output=graded)' },
    { title: "Batch Tensors to GPU", color: C.orange,
      desc: "Padded to same length, collated into [batch, seq_len] tensors, moved to GPU. Embedding converts IDs to 4096-dim vectors inside the model.",
      note: "Same as Full FT",
      content: 'input_ids [4, 512] -> Embed -> hidden [4, 512, 4096] -> Layer 0 ...' },
    { title: "Forward Pass: W\u2080x + BAx", color: C.accent,
      desc: "HERE is where LoRA diverges. Every target weight gets a parallel bypass. h = W0*x + (alpha/r)*B*A*x. Both paths run simultaneously and are summed.",
      note: "LoRA DIVERGES HERE",
      content: 'h = W0*x  (frozen, no grad stored)  +  (alpha/r) * B*(A*x)  (LoRA path, trainable)' },
    { title: "Loss Computation", color: C.red,
      desc: "CrossEntropyLoss on logits vs labels. Identical to full fine-tuning - loss doesn't know or care about LoRA.",
      note: "Same as Full FT",
      content: 'loss = CrossEntropyLoss(logits[4,512,32000], labels[4,512])  // only where labels != -100' },
    { title: "Backward: LoRA Only", color: C.pink,
      desc: "Gradients flow THROUGH frozen weights (needed for chain rule) but are NOT STORED for them. Only LoRA A and B accumulate gradient tensors.",
      note: "LoRA KEY DIFFERENCE",
      content: 'grad(W0) = computed, NOT stored  |  grad(A), grad(B) = computed AND stored (~33MB total)' },
    { title: "Optimizer: Update A and B Only", color: C.green,
      desc: "Adam updates ONLY the LoRA matrices. 7B frozen weights are completely untouched. ~67MB optimizer states vs ~56GB for full FT.",
      note: "LoRA KEY DIFFERENCE",
      content: 'A -= lr * adam_step(grad_A)  |  B -= lr * adam_step(grad_B)  |  W0 = unchanged forever' },
  ];

  useEffect(function() { if (!autoP) return; var t = setInterval(function() { setStep(function(s) { return (s + 1) % pSteps.length; }); }, 3000); return function() { clearInterval(t); }; }, [autoP]);

  var isLoRaStep = step === 5 || step === 7 || step === 8;

  return (
    <div>
      <SectionTitle title="Data Pipeline" subtitle={"The LoRA pipeline is identical to full FT until step 6 " + DASH + " then it diverges"} />

      <div style={{ display: "flex", gap: 4, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {pSteps.map(function(s, i) {
          var on = step === i;
          var isL = i === 5 || i === 7 || i === 8;
          return (<button key={i} onClick={function() { setStep(i); setAutoP(false); }} style={{
            padding: "6px 9px", borderRadius: 6, border: "1.5px solid " + (on ? s.color : isL ? C.accent + "40" : C.border),
            background: on ? s.color + "20" : isL ? C.accent + "08" : C.card, color: on ? s.color : isL ? C.accent + "70" : C.muted,
            cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
          }}>{i + 1}</button>);
        })}
        <button onClick={function() { setAutoP(!autoP); }} style={{ padding: "6px 9px", borderRadius: 6, border: "1.5px solid " + (autoP ? C.yellow : C.border), background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted, cursor: "pointer", fontSize: 9, fontFamily: "monospace" }}>{autoP ? PAUSE : PLAY}</button>
      </div>

      {/* Pipeline flow */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={80} viewBox="0 0 780 80" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {pSteps.map(function(s, i) {
            var x = 10 + i * 84;
            var on = step === i;
            var done = i < step;
            var isL = i === 5 || i === 7 || i === 8;
            return (<g key={i}>
              <rect x={x} y={12} width={76} height={50} rx={6}
                fill={on ? s.color + "25" : done ? s.color + "08" : isL ? C.accent + "06" : C.card}
                stroke={on ? s.color : done ? s.color + "30" : isL ? C.accent + "30" : C.dim + "40"}
                strokeWidth={on ? 2 : 1} style={{ transition: "all 0.3s" }} />
              <text x={x + 38} y={32} textAnchor="middle" fill={on ? s.color : done ? s.color + "70" : C.dim} fontSize={7} fontWeight={700} fontFamily="monospace">{"Step " + (i + 1)}</text>
              <text x={x + 38} y={46} textAnchor="middle" fill={on ? s.color + "80" : C.dim} fontSize={6} fontFamily="monospace">{s.title.substring(0, 12)}</text>
              {isL && !on && <rect x={x} y={58} width={76} height={2} rx={1} fill={C.accent + "40"} />}
              {on && <rect x={x} y={58} width={76} height={3} rx={1.5} fill={s.color} style={{ animation: "pulse 1.5s infinite" }} />}
              {i < pSteps.length - 1 && <g>
                <line x1={x + 76} y1={37} x2={x + 84} y2={37} stroke={done ? s.color + "40" : C.dim + "20"} strokeWidth={1} />
              </g>}
            </g>);
          })}
        </svg>
      </div>

      {/* Step detail */}
      <Card highlight={isLoRaStep} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: isLoRaStep ? C.accent : pSteps[step].color }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexWrap: "wrap", gap: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: pSteps[step].color }}>{"Step " + (step + 1) + ": " + pSteps[step].title}</div>
          <Badge color={isLoRaStep ? C.accent : C.dim}>{pSteps[step].note}</Badge>
        </div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8, marginBottom: 12 }}>{pSteps[step].desc}</div>
        <div style={{ padding: "12px 16px", background: "#08080d", borderRadius: 8, border: "1px solid " + C.border }}>
          <pre style={{ fontSize: 10, color: pSteps[step].color, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.8, margin: 0, whiteSpace: "pre-wrap", overflowX: "auto" }}>{pSteps[step].content}</pre>
        </div>
      </Card>

      <Insight icon={GEAR} title="The Critical Distinction">
        Steps 1-5 are <span style={{color:C.blue, fontWeight:700}}>byte-for-byte identical</span> to full fine-tuning. The data pipeline doesn't know what training method you're using. LoRA's changes are entirely in the <span style={{color:C.accent, fontWeight:700}}>forward computation</span> (parallel bypass), the <span style={{color:C.pink, fontWeight:700}}>backward pass</span> (no gradient storage for frozen params), and the <span style={{color:C.green, fontWeight:700}}>optimizer</span> (only A and B get updated).
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: RANK & ALPHA
   =============================================================== */
function TabRankAlpha() {
  var _r = useState(8); var rank = _r[0], setRank = _r[1];
  var _al = useState(16); var alpha = _al[0], setAlpha = _al[1];
  var _t = useState(0); var tgtSel = _t[0], setTgtSel = _t[1];

  var DIM = 4096;
  var targets = [
    { name: "Minimal: q,v", count: 2, color: C.blue },
    { name: "Standard: q,k,v,o", count: 4, color: C.accent },
    { name: "Aggressive: all-linear", count: 7, color: C.orange },
  ];

  var tgt = targets[tgtSel];
  var totalMatrices = 32 * tgt.count * 2;
  var totalParams = 32 * tgt.count * 2 * rank * DIM;
  var totalParamsM = totalParams / 1e6;
  var pct = (totalParamsM / 7000 * 100).toFixed(3);
  var sizeMB = (totalParamsM * 2).toFixed(0);
  var scaling = (alpha / rank).toFixed(2);

  var rankPresets = [
    { r: 4, desc: "Very lean. Simple tasks, tiny data." },
    { r: 8, desc: "Default sweet spot. Most instruction-tuning." },
    { r: 16, desc: "Complex tasks, nuanced adaptation." },
    { r: 32, desc: "Very different domains. Diminishing returns." },
    { r: 64, desc: "Rarely needed. Consider full FT." },
  ];

  return (
    <div>
      <SectionTitle title="Rank & Alpha" subtitle={"The two core hyperparameters that control LoRA expressiveness and training stability"} />

      {/* Controls */}
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        <Card style={{ flex: 1, minWidth: 280 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 8 }}>{"Rank r = " + rank + " (expressiveness)"}</div>
          <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.accent }} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, marginBottom: 12 }}>
            <span style={{ fontSize: 8, color: C.muted }}>r=1</span><span style={{ fontSize: 8, color: C.muted }}>r=64</span>
          </div>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.yellow, marginBottom: 8 }}>{"\u03B1 = " + alpha + " \u2192 scaling \u03B1/r = " + scaling}</div>
          <input type="range" min={1} max={128} value={alpha} onChange={function(e) { setAlpha(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.yellow }} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <span style={{ fontSize: 8, color: C.muted }}>\u03B1=1</span><span style={{ fontSize: 8, color: C.muted }}>\u03B1=128</span>
          </div>
        </Card>

        <Card style={{ flex: 1, minWidth: 240 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.text, marginBottom: 10 }}>Target Modules</div>
          {targets.map(function(t, i) {
            var on = tgtSel === i;
            return (<button key={i} onClick={function() { setTgtSel(i); }} style={{
              display: "block", width: "100%", marginBottom: 6, padding: "7px 12px", borderRadius: 6,
              border: "1.5px solid " + (on ? t.color : C.border), background: on ? t.color + "18" : C.card,
              color: on ? t.color : C.muted, cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace", textAlign: "left"
            }}>{t.name}</button>);
          })}
        </Card>

        <Card style={{ flex: 1, minWidth: 220 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <StatBox label="TRAINABLE PARAMS" value={totalParamsM.toFixed(1) + "M"} color={C.accent} />
            <StatBox label="% OF MODEL" value={pct + "%"} color={C.yellow} />
            <StatBox label="ADAPTER SIZE" value={sizeMB + " MB"} color={C.green} />
            <StatBox label="SCALING \u03B1/r" value={scaling} color={tgtSel === 0 ? C.blue : tgtSel === 1 ? C.accent : C.orange} />
          </div>
        </Card>
      </div>

      {/* Rank visualization */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Rank = Bandwidth of Adaptation"}</div>
        <div style={{ position: "relative", height: 40, background: C.border, borderRadius: 6, overflow: "hidden", marginBottom: 8 }}>
          <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: "100%", background: C.dim + "30", borderRadius: 6 }} />
          <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: (rank / 64 * 100) + "%", background: "linear-gradient(90deg," + C.accent + "60," + C.pink + "40)", borderRadius: 6, transition: "width 0.3s", display: "flex", alignItems: "center", paddingLeft: 10 }}>
            <span style={{ fontSize: 9, color: C.text, fontFamily: "monospace", fontWeight: 700 }}>{rank + " channels of adaptation"}</span>
          </div>
          <div style={{ position: "absolute", right: 8, top: 11, fontSize: 8, color: C.dim }}>4096 (full rank)</div>
        </div>
        <div style={{ marginBottom: 12, fontSize: 9, color: C.muted }}>{"Think of rank as the number of independent directions of change. r=8 captures 8 " + LQ + "dimensions" + RQ + " of task adaptation."}</div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {rankPresets.map(function(p, i) {
            var on = rank === p.r;
            return (
              <div key={i} onClick={function() { setRank(p.r); }} style={{
                flex: 1, minWidth: 120, padding: "10px 12px", borderRadius: 8, cursor: "pointer",
                background: on ? C.accent + "15" : C.bg, border: "1.5px solid " + (on ? C.accent : C.border),
                transition: "all 0.25s",
              }}>
                <div style={{ fontSize: 14, fontWeight: 800, color: on ? C.accent : C.muted, fontFamily: "monospace", marginBottom: 4 }}>{"r=" + p.r}</div>
                <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.6 }}>{p.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Alpha / scaling */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"The \u03B1 Scaling Factor " + DASH + " Volume Knob on LoRA Output"}</div>
        <div style={{ fontSize: 11, color: C.muted, marginBottom: 12 }}>{"h = W\u2080x + (\u03B1/r) \u00B7 BAx   \u2192   current scaling: " + alpha + "/" + rank + " = " + scaling}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { label: "\u03B1 = r/2 (gentle)", ratio: 0.5, desc: "Adapters contribute cautiously. Useful if base model behavior should dominate." },
            { label: "\u03B1 = r (neutral)", ratio: 1.0, desc: "Safe default. Balanced contribution from frozen and LoRA paths. Start here." },
            { label: "\u03B1 = 2r (amplified)", ratio: 2.0, desc: "Most common production choice. Slightly stronger adaptation signal." },
            { label: "\u03B1 = 4r (aggressive)", ratio: 4.0, desc: "Use with lower LR. Can destabilize training if too high." },
          ].map(function(v, i) {
            var isCurrent = Math.abs(scaling - v.ratio) < 0.25;
            return (
              <div key={i} style={{ flex: 1, minWidth: 160, padding: "10px 12px", borderRadius: 8, background: isCurrent ? C.yellow + "12" : C.bg, border: "1.5px solid " + (isCurrent ? C.yellow : C.border), transition: "all 0.3s" }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: isCurrent ? C.yellow : C.muted, marginBottom: 6 }}>{v.label}</div>
                <div style={{ height: 12, background: C.border, borderRadius: 3, marginBottom: 6 }}>
                  <div style={{ width: Math.min(100, v.ratio * 40) + "%", height: "100%", borderRadius: 3, background: C.yellow + (isCurrent ? "80" : "30") }} />
                </div>
                <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.6 }}>{v.desc}</div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Target modules table */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Which Layers to Target (32-layer, r=" + rank + ")"}</div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9, fontFamily: "monospace" }}>
            <thead>
              <tr>{["Config", "target_modules", "LoRA pairs", "Params", "% of 7B", "Adapter MB"].map(function(h, i) { return (<th key={i} style={{ padding: "6px 10px", textAlign: "left", color: C.muted, borderBottom: "1px solid " + C.border }}>{h}</th>); })}</tr>
            </thead>
            <tbody>
              {[
                { name: "Minimal", mods: "q, v", count: 2, star: false },
                { name: "Standard", mods: "q, k, v, o", count: 4, star: true },
                { name: "Aggressive", mods: "q, k, v, o, gate, up, down", count: 7, star: false },
              ].map(function(row, i) {
                var pairs = 32 * row.count * 2;
                var params = (32 * row.count * 2 * rank * DIM / 1e6).toFixed(1);
                var pctRow = (32 * row.count * 2 * rank * DIM / 1e9 / 7 * 100).toFixed(3);
                var mb = (32 * row.count * 2 * rank * DIM * 2 / 1e6).toFixed(0);
                var colors = [C.blue, C.accent, C.orange];
                var on = tgtSel === i;
                return (
                  <tr key={i} onClick={function() { setTgtSel(i); }} style={{ cursor: "pointer", background: on ? colors[i] + "10" : "transparent", transition: "background 0.2s" }}>
                    <td style={{ padding: "7px 10px", color: colors[i], fontWeight: 700 }}>{row.name + (row.star ? " *" : "")}</td>
                    <td style={{ padding: "7px 10px", color: C.muted }}>{row.mods}</td>
                    <td style={{ padding: "7px 10px", color: C.dim }}>{pairs}</td>
                    <td style={{ padding: "7px 10px", color: colors[i] }}>{params + "M"}</td>
                    <td style={{ padding: "7px 10px", color: C.dim }}>{pctRow + "%"}</td>
                    <td style={{ padding: "7px 10px", color: colors[i] }}>{mb + " MB"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8, fontSize: 9, color: C.muted }}>{"* recommended starting point. Click a row to update all stats."}</div>
      </Card>

      <Insight icon={TARG} title="Choosing r and alpha in Practice">
        Start with <span style={{color:C.accent, fontWeight:700}}>r=8, \u03B1=16</span> (standard). If the task is complex or very different from pre-training, try <span style={{color:C.yellow}}>r=16 or r=32</span>. Keep <span style={{color:C.yellow, fontWeight:700}}>\u03B1 = 2r</span> as a rule of thumb. If you change r, update \u03B1 proportionally to keep the scaling ratio stable. More targets = more params = better quality but more memory. "<span style={{color:C.orange}}>all-linear</span>" is overkill for most tasks but is the safest "just make it work" option.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 6: MERGE & DEPLOY
   =============================================================== */
function TabMergeDeploy() {
  var _m = useState(false); var merged = _m[0], setMerged = _m[1];
  var _s = useState(0); var swapIdx = _s[0], setSwapIdx = _s[1];
  var _au = useState(false); var autoSwap = _au[0], setAutoSwap = _au[1];

  useEffect(function() {
    if (!autoSwap) return;
    var t = setInterval(function() { setSwapIdx(function(i) { return (i + 1) % 3; }); }, 2000);
    return function() { clearInterval(t); };
  }, [autoSwap]);

  var adapters = [
    { name: "Medical QA", color: C.blue, size: "~33 MB", r: 8, alpha: 16 },
    { name: "Code Gen", color: C.green, size: "~67 MB", r: 16, alpha: 32 },
    { name: "Legal Summ.", color: C.orange, size: "~33 MB", r: 8, alpha: 16 },
  ];

  return (
    <div>
      <SectionTitle title="Merge & Deploy" subtitle={"LoRA's killer advantage: adapters can merge into the base model and completely vanish"} />

      {/* Merge visual */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"The Merge Operation " + DASH + " W_merged = W\u2080 + (\u03B1/r) \u00B7 B \u00D7 A"}</div>

        <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
          <svg width={1050} height={180} viewBox="0 0 780 180" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

            {/* DURING TRAINING label */}
            <text x={190} y={18} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">DURING TRAINING</text>

            {/* W0 box */}
            <rect x={20} y={30} width={130} height={60} rx={8} fill={C.dim + "20"} stroke={!merged ? C.dim + "60" : C.dim + "30"} strokeWidth={1.5} style={{ transition: "all 0.6s" }} />
            <text x={85} y={56} textAnchor="middle" fill={!merged ? C.muted : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">W\u2080 (frozen)</text>
            <text x={85} y={72} textAnchor="middle" fill={!merged ? C.dim : C.dim + "50"} fontSize={8} fontFamily="monospace">14 GB (7B model)</text>
            <text x={85} y={84} textAnchor="middle" fill={!merged ? C.dim : C.dim + "40"} fontSize={7} fontFamily="monospace">[4096 \u00D7 4096]</text>

            {/* Plus */}
            <text x={165} y={65} textAnchor="middle" fill={!merged ? C.accent + "80" : C.dim + "30"} fontSize={22} fontWeight={800} style={{ transition: "all 0.6s" }}>{PLUS}</text>

            {/* B*A box */}
            <rect x={185} y={30} width={130} height={60} rx={8} fill={!merged ? C.accent + "15" : C.dim + "05"} stroke={!merged ? C.accent + "60" : C.dim + "20"} strokeWidth={1.5} style={{ transition: "all 0.6s" }} />
            <text x={250} y={56} textAnchor="middle" fill={!merged ? C.accent : C.dim + "40"} fontSize={10} fontWeight={700} fontFamily="monospace">(\u03B1/r) B\u00D7A</text>
            <text x={250} y={72} textAnchor="middle" fill={!merged ? C.accent + "80" : C.dim + "30"} fontSize={8} fontFamily="monospace">~33 MB adapter</text>
            <text x={250} y={84} textAnchor="middle" fill={!merged ? C.accent + "60" : C.dim + "20"} fontSize={7} fontFamily="monospace">trainable</text>

            {/* Arrow to merge */}
            {!merged && <g>
              <line x1={315} y1={60} x2={360} y2={60} stroke={C.yellow} strokeWidth={2} />
              <polygon points="363,60 357,56 357,64" fill={C.yellow} />
              <text x={338} y={50} textAnchor="middle" fill={C.yellow} fontSize={8} fontWeight={700} fontFamily="monospace">MERGE</text>
              <text x={338} y={78} textAnchor="middle" fill={C.yellow + "60"} fontSize={7} fontFamily="monospace">matrix add</text>
            </g>}

            {/* AFTER MERGE label */}
            <text x={560} y={18} textAnchor="middle" fill={merged ? C.green : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace" style={{ transition: "all 0.6s" }}>AFTER MERGE</text>

            {/* Merged box */}
            <rect x={370} y={30} width={150} height={60} rx={8} fill={merged ? C.green + "15" : C.card} stroke={merged ? C.green + "60" : C.border} strokeWidth={merged ? 2 : 1} style={{ transition: "all 0.6s" }} />
            <text x={445} y={56} textAnchor="middle" fill={merged ? C.green : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">W_merged</text>
            <text x={445} y={72} textAnchor="middle" fill={merged ? C.green + "90" : C.dim} fontSize={8} fontFamily="monospace">{merged ? "14 GB (same size!)" : "W\u2080 + (\u03B1/r)BA"}</text>
            <text x={445} y={84} textAnchor="middle" fill={merged ? C.green + "60" : C.dim + "50"} fontSize={7} fontFamily="monospace">{merged ? "adapters GONE" : "[4096 \u00D7 4096]"}</text>

            {/* Properties comparison */}
            {[
              { label: "Inference overhead", before: "2 paths + sum", after: "Single path", color: C.green },
              { label: "Extra memory", before: "+33 MB adapters", after: "0 bytes extra", color: C.green },
              { label: "PEFT library needed", before: "Yes (runtime)", after: "No (standard model)", color: C.green },
              { label: "Adapter swapping", before: "Yes (keep separate)", after: "No (baked in)", color: C.yellow },
            ].map(function(row, i) {
              return (
                <g key={i}>
                  <text x={15} y={120 + i * 14} fill={C.dim} fontSize={7} fontFamily="monospace">{row.label + ":"}</text>
                  <text x={180} y={120 + i * 14} fill={!merged ? C.muted : C.dim + "40"} fontSize={7} fontFamily="monospace" style={{ transition: "all 0.5s" }}>{row.before}</text>
                  <text x={380} y={120 + i * 14} fill={merged ? row.color : C.dim + "30"} fontSize={7} fontFamily="monospace" fontWeight={merged ? "700" : "400"} style={{ transition: "all 0.5s" }}>{row.after}</text>
                </g>
              );
            })}
          </svg>
        </div>

        <div style={{ display: "flex", justifyContent: "center", gap: 12 }}>
          <button onClick={function() { setMerged(false); }} style={{
            padding: "10px 24px", borderRadius: 8, border: "1.5px solid " + (!merged ? C.accent : C.border),
            background: !merged ? C.accent + "20" : C.card, color: !merged ? C.accent : C.muted,
            cursor: "pointer", fontSize: 11, fontWeight: 700, fontFamily: "monospace"
          }}>{"During Training"}</button>
          <button onClick={function() { setMerged(true); }} style={{
            padding: "10px 24px", borderRadius: 8, border: "1.5px solid " + (merged ? C.green : C.border),
            background: merged ? C.green + "20" : C.card, color: merged ? C.green : C.muted,
            cursor: "pointer", fontSize: 11, fontWeight: 700, fontFamily: "monospace"
          }}>{"After Merge " + MERGE}</button>
        </div>
      </Card>

      {/* Adapter swapping */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 4 }}>{"Adapter Swapping " + DASH + " One Base Model, Many Tasks"}</div>
        <div style={{ fontSize: 10, color: C.muted, marginBottom: 16 }}>{"Load base model once (14 GB). Swap tiny adapters in milliseconds. No model reloading."}</div>

        <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
          <svg width={1050} height={200} viewBox="0 0 780 200" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
            {/* Base model */}
            <rect x={290} y={10} width={200} height={55} rx={10} fill={C.dim + "25"} stroke={C.muted + "50"} strokeWidth={2} />
            <text x={390} y={32} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">BASE MODEL</text>
            <text x={390} y={48} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " 14 GB frozen (loaded once)"}</text>

            {/* Connector lines */}
            {adapters.map(function(ad, i) {
              var x = 65 + i * 220;
              var active = swapIdx === i;
              return (<g key={"line" + i}>
                <line x1={390} y1={65} x2={x + 75} y2={110} stroke={active ? ad.color + "60" : C.dim + "20"} strokeWidth={active ? 2 : 1} strokeDasharray={active ? "0" : "4,4"} style={{ transition: "all 0.4s" }} />
              </g>);
            })}

            {/* Adapters */}
            {adapters.map(function(ad, i) {
              var x = 65 + i * 220;
              var active = swapIdx === i;
              return (<g key={"ad" + i}>
                <rect x={x} y={110} width={150} height={70} rx={10} fill={active ? ad.color + "20" : C.card} stroke={active ? ad.color : C.border} strokeWidth={active ? 2.5 : 1} style={{ transition: "all 0.4s" }} />
                <text x={x + 75} y={133} textAnchor="middle" fill={active ? ad.color : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{ad.name}</text>
                <text x={x + 75} y={148} textAnchor="middle" fill={active ? ad.color + "90" : C.dim + "50"} fontSize={8} fontFamily="monospace">{"r=" + ad.r + ", \u03B1=" + ad.alpha}</text>
                <text x={x + 75} y={163} textAnchor="middle" fill={active ? ad.color + "80" : C.dim + "40"} fontSize={8} fontFamily="monospace">{ad.size}</text>
                {active && <rect x={x} y={177} width={150} height={3} rx={1.5} fill={ad.color} style={{ animation: "pulse 1.5s infinite" }} />}
              </g>);
            })}
          </svg>
        </div>

        <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 10 }}>
          {adapters.map(function(ad, i) {
            var on = swapIdx === i;
            return (<button key={i} onClick={function() { setSwapIdx(i); setAutoSwap(false); }} style={{
              padding: "8px 18px", borderRadius: 8, border: "1.5px solid " + (on ? ad.color : C.border),
              background: on ? ad.color + "20" : C.card, color: on ? ad.color : C.muted,
              cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
            }}>{ad.name}</button>);
          })}
          <button onClick={function() { setAutoSwap(!autoSwap); }} style={{ padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (autoSwap ? C.yellow : C.border), background: autoSwap ? C.yellow + "20" : C.card, color: autoSwap ? C.yellow : C.muted, cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>{autoSwap ? PAUSE + " Stop" : PLAY + " Auto-swap"}</button>
        </div>

        <div style={{ padding: "10px 14px", background: adapters[swapIdx].color + "08", borderRadius: 8, border: "1px solid " + adapters[swapIdx].color + "30" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: adapters[swapIdx].color, marginBottom: 4 }}>{"Active: " + adapters[swapIdx].name + " Adapter"}</div>
          <div style={{ display: "flex", gap: 20 }}>
            <StatBox label="RANK" value={"r=" + adapters[swapIdx].r} color={adapters[swapIdx].color} bigFont={14} />
            <StatBox label="ALPHA" value={"\u03B1=" + adapters[swapIdx].alpha} color={adapters[swapIdx].color} bigFont={14} />
            <StatBox label="SCALING" value={(adapters[swapIdx].alpha / adapters[swapIdx].r).toFixed(1) + "x"} color={adapters[swapIdx].color} bigFont={14} />
            <StatBox label="SIZE" value={adapters[swapIdx].size} color={adapters[swapIdx].color} bigFont={12} minW={80} />
          </div>
        </div>
      </Card>

      {/* Storage comparison */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Storage Comparison " + DASH + " Scale to Many Tasks"}</div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {[1, 3, 10, 100].map(function(n, i) {
            var fullGB = n * 14;
            var loraGB = 14 + n * 0.033;
            return (
              <div key={i} style={{ flex: 1, minWidth: 120, padding: "10px 12px", borderRadius: 8, background: C.bg, border: "1px solid " + C.border }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: C.text, marginBottom: 8, textAlign: "center" }}>{n + " task" + (n > 1 ? "s" : "")}</div>
                <div style={{ marginBottom: 6 }}>
                  <div style={{ fontSize: 8, color: C.red, marginBottom: 2 }}>Full FT: {fullGB + " GB"}</div>
                  <div style={{ height: 10, background: C.border, borderRadius: 2 }}>
                    <div style={{ width: "100%", height: "100%", borderRadius: 2, background: C.red + "40" }} />
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 8, color: C.green, marginBottom: 2 }}>LoRA: {loraGB.toFixed(1) + " GB"}</div>
                  <div style={{ height: 10, background: C.border, borderRadius: 2 }}>
                    <div style={{ width: (loraGB / fullGB * 100) + "%", height: "100%", borderRadius: 2, background: C.green + "50", transition: "width 0.3s" }} />
                  </div>
                </div>
                <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, textAlign: "center", marginTop: 6 }}>{Math.round((1 - loraGB / fullGB) * 100) + "% smaller"}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={MERGE} title="The Mergeability Advantage">
        This is <span style={{color:C.accent, fontWeight:700}}>LoRA's defining property</span> and why it dominates over Adapters. Bottleneck Adapters stay in the forward path forever, adding latency. LoRA's W_merged = W\u2080 + (\u03B1/r)BA is a <span style={{color:C.yellow, fontWeight:700}}>one-time matrix addition</span>. After that, A and B are discarded. The merged model is a <span style={{color:C.green, fontWeight:700}}>standard model</span> - same architecture, same size, zero PEFT overhead. No PEFT library needed at inference. Indistinguishable from full fine-tuning.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Big Picture", "Forward Pass", "Memory & Cost", "Data Pipeline", "Rank & Alpha", "Merge & Deploy"];
  return (
    <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.cyan + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>PEFT / LoRA</div>
        <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " Low-Rank Adaptation from theory to deployment"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab === 0 && <TabBigPicture />}
      {tab === 1 && <TabForwardPass />}
      {tab === 2 && <TabMemory />}
      {tab === 3 && <TabDataPipeline />}
      {tab === 4 && <TabRankAlpha />}
      {tab === 5 && <TabMergeDeploy />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

LORA_VISUAL_HEIGHT = 1600

# """
# Self-contained HTML for the PEFT LoRA interactive walkthrough.
# Covers: Big Picture, Forward Pass, Memory & Cost, Data Pipeline,
# Rank & Alpha, and Merge & Deploy.
# Embed in Streamlit via st.components.v1.html(LORA_VISUAL_HTML, height=LORA_VISUAL_HEIGHT).
# """
#
# LORA_VISUAL_HTML = """
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
#   @keyframes flowDown { 0%{transform:translateY(-6px);opacity:0} 50%{opacity:1} 100%{transform:translateY(6px);opacity:0} }
#   @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
# </style>
# </head>
# <body>
# <div id="root"></div>
# <script type="text/babel">
#
# var useState = React.useState;
# var useEffect = React.useEffect;
# var useMemo = React.useMemo;
#
# var C = {
#   bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
#   accent: "#a78bfa", blue: "#4ecdc4", purple: "#a78bfa",
#   yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
#   dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
#   cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
#   lora: "#a78bfa", frozen: "#3f3f46",
# };
#
# var MUL = "\u00D7";
# var ARR = "\u2192";
# var DASH = "\u2014";
# var CHK = "\u2713";
# var WARN = "\u26A0";
# var LQ = "\u201C";
# var RQ = "\u201D";
# var LARR = "\u2190";
# var PLAY = "\u25B6";
# var PAUSE = "\u23F8";
# var BULB = "\uD83D\uDCA1";
# var TARG = "\uD83C\uDFAF";
# var LOCK = "\uD83D\uDD12";
# var UNLOCK = "\uD83D\uDD13";
# var BRAIN = "\uD83E\uDDE0";
# var FIRE = "\uD83D\uDD25";
# var GEAR = "\u2699";
# var CHART = "\uD83D\uDCC8";
# var DARR = "\u2193";
# var UARR = "\u2191";
# var MERGE = "\u21A0";
# var PLUS = "+";
#
# /* shared components */
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
#           }}>{t}</button>
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
# function Badge(props) {
#   return (
#     <span style={{
#       display: "inline-block", padding: "2px 8px", borderRadius: 4,
#       background: (props.color || C.accent) + "20",
#       border: "1px solid " + (props.color || C.accent) + "50",
#       color: props.color || C.accent, fontSize: 9, fontWeight: 700,
#       fontFamily: "monospace", margin: "0 3px",
#     }}>{props.children}</span>
#   );
# }
#
# /* ===============================================================
#    TAB 1: THE BIG PICTURE
#    =============================================================== */
# function TabBigPicture() {
#   var _a = useState(false); var animated = _a[0], setAnimated = _a[1];
#   var _h = useState(-1); var hov = _h[0], setHov = _h[1];
#
#   useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);
#
#   var methods = [
#     { name: "Additive", ex: "Adapters", desc: "Insert new modules in-series. Stays at inference.", color: C.blue },
#     { name: "Reparameterization", ex: "LoRA", desc: "Decompose weight updates into A\u00D7B. Merges and vanishes.", color: C.accent, star: true },
#     { name: "Selective", ex: "BitFit", desc: "Unfreeze tiny subset of existing params only.", color: C.cyan },
#     { name: "Hybrid", ex: "QLoRA", desc: "LoRA adapters + 4-bit quantized base.", color: C.orange, star: true },
#     { name: "Prompt-based", ex: "Prefix Tuning", desc: "Learn soft vectors prepended to input. Fewest params.", color: C.pink },
#   ];
#
#   return (
#     <div>
#       <SectionTitle title="The Big Picture" subtitle={"PEFT: get 95%+ of full fine-tuning quality at 1% of the memory " + DASH + " by freezing everything and training almost nothing"} />
#
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
#         <svg width={1050} height={290} viewBox="0 0 780 290" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#
#           {/* === LEFT: FROZEN BASE === */}
#           <text x={110} y={22} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">PRE-TRAINED BASE</text>
#           <text x={110} y={36} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">(7B parameters - ALL FROZEN)</text>
#           {[0,1,2,3,4,5,6,7].map(function(i) {
#             return (
#               <g key={"base" + i}>
#                 <rect x={30} y={48 + i * 26} width={160} height={20} rx={4} fill={C.dim + "20"} stroke={C.dim + "50"} strokeWidth={1} />
#                 <rect x={30} y={48 + i * 26} width={animated ? 160 : 0} height={20} rx={4} fill={C.dim + "15"} style={{ transition: "width 1.2s ease-out", transitionDelay: (i * 0.08) + "s" }} />
#                 <text x={110} y={62 + i * 26} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{LOCK + " Layer " + (i + 1)}</text>
#               </g>
#             );
#           })}
#           <text x={110} y={268} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">requires_grad = False</text>
#           <text x={110} y={280} textAnchor="middle" fill={C.dim + "80"} fontSize={8} fontFamily="monospace">No gradients. No optimizer states.</text>
#
#           {/* === PLUS === */}
#           <text x={228} y={152} textAnchor="middle" fill={C.accent} fontSize={28} fontWeight={800}>{PLUS}</text>
#           <text x={228} y={170} textAnchor="middle" fill={C.accent + "60"} fontSize={8} fontFamily="monospace">frozen +</text>
#
#           {/* === CENTER: LORA ADAPTERS === */}
#           <text x={370} y={22} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">LoRA ADAPTERS</text>
#           <text x={370} y={36} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">(~8.4M params - TRAINABLE)</text>
#           {[0,1,2,3].map(function(i) {
#             return (
#               <g key={"lora" + i}>
#                 <rect x={305} y={55 + i * 52} width={130} height={40} rx={6} fill={C.accent + "12"} stroke={C.accent + "50"} strokeWidth={1.5} />
#                 <text x={370} y={72 + i * 52} textAnchor="middle" fill={C.accent} fontSize={8} fontWeight={700} fontFamily="monospace">{"Layer " + (i + 1) + " Adapter"}</text>
#                 <text x={345} y={86 + i * 52} textAnchor="middle" fill={C.blue} fontSize={7} fontFamily="monospace">A [8\u00D74096]</text>
#                 <text x={395} y={86 + i * 52} textAnchor="middle" fill={C.pink} fontSize={7} fontFamily="monospace">B [4096\u00D78]</text>
#               </g>
#             );
#           })}
#           <text x={370} y={268} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">0.12% of model</text>
#           <text x={370} y={280} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">~33 MB saved per task</text>
#
#           {/* === ARROW === */}
#           <line x1={500} y1={145} x2={560} y2={145} stroke={C.green} strokeWidth={2} />
#           <polygon points="565,145 558,140 558,150" fill={C.green} />
#           <text x={530} y={133} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">MERGE</text>
#           <text x={530} y={162} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">W\u2080 + BA</text>
#
#           {/* === RIGHT: RESULT === */}
#           <text x={670} y={22} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">FINE-TUNED MODEL</text>
#           <text x={670} y={36} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">(Zero inference overhead)</text>
#           {[0,1,2,3,4,5,6,7].map(function(i) {
#             var w = animated ? 160 : 0;
#             return (
#               <g key={"ft" + i}>
#                 <rect x={590} y={48 + i * 26} width={160} height={20} rx={4} fill={C.green + "08"} stroke={C.green + "30"} strokeWidth={1} />
#                 <rect x={590} y={48 + i * 26} width={w} height={20} rx={4} fill={C.green + "15"} style={{ transition: "width 1s ease-out", transitionDelay: (0.8 + i * 0.1) + "s" }} />
#                 <text x={670} y={62 + i * 26} textAnchor="middle" fill={C.green + "80"} fontSize={7} fontFamily="monospace">{"W\u2080 + (α/r)BA"}</text>
#               </g>
#             );
#           })}
#           <text x={670} y={268} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">Same size as original</text>
#           <text x={670} y={280} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">Adapters vanished into weights</text>
#         </svg>
#       </div>
#
#       {/* PEFT taxonomy */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"PEFT Methods Taxonomy " + DASH + " Five Families"}</div>
#         <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
#           {methods.map(function(m, i) {
#             var isH = hov === i;
#             return (
#               <div key={i} onMouseEnter={function() { setHov(i); }} onMouseLeave={function() { setHov(-1); }}
#                 style={{ flex: 1, minWidth: 140, padding: "12px 14px", borderRadius: 8, cursor: "pointer",
#                   background: m.color + (isH ? "18" : "08"), border: "1.5px solid " + m.color + (isH ? "70" : "25"),
#                   transition: "all 0.25s" }}>
#                 <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
#                   <div style={{ fontSize: 10, fontWeight: 800, color: m.color }}>{m.name}</div>
#                   {m.star && <div style={{ fontSize: 9, color: C.yellow }}>{"\u2605 Popular"}</div>}
#                 </div>
#                 <div style={{ fontSize: 9, color: m.color + "90", fontFamily: "monospace", marginBottom: 4 }}>{m.ex}</div>
#                 <div style={{ fontSize: 9, color: C.muted, lineHeight: 1.6 }}>{m.desc}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       {/* Memory summary */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Why PEFT? " + DASH + " The Memory Math (7B Model)"}</div>
#         <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
#           {[
#             { label: "Full Fine-Tuning", mem: "~100 GB", breakdown: "14 weights + 14 grad + 56 optimizer + activations", color: C.red },
#             { label: "LoRA (BF16 base)", mem: "~20 GB", breakdown: "14 frozen + 0.1 LoRA weights + 0.1 grad + 0.5 optimizer", color: C.accent },
#             { label: "QLoRA (4-bit base)", mem: "~10 GB", breakdown: "3.5 NF4 frozen + 0.1 LoRA weights + 0.1 grad + 0.5 optimizer", color: C.green },
#           ].map(function(v, i) {
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 220, padding: "12px 16px", borderRadius: 8, background: v.color + "08", border: "1px solid " + v.color + "30" }}>
#                 <div style={{ fontSize: 11, fontWeight: 800, color: v.color, marginBottom: 6 }}>{v.label}</div>
#                 <div style={{ fontSize: 22, fontWeight: 800, color: v.color, marginBottom: 6 }}>{v.mem}</div>
#                 <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.7 }}>{v.breakdown}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={BRAIN} title="The Core Insight">
#         Fine-tuning changes (<span style={{color:C.accent, fontWeight:700}}>\u0394W</span>) have <span style={{color:C.yellow, fontWeight:700}}>low intrinsic rank</span>. You don't need 16.7M parameters to express "now be a medical assistant." The real adaptation lives in a <span style={{color:C.green, fontWeight:700}}>tiny subspace</span>. LoRA finds that subspace directly via two small matrices A and B whose product approximates \u0394W. Base model <span style={{color:C.cyan, fontWeight:700}}>stays frozen</span>. Zero catastrophic forgetting. Zero inference overhead after merging.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 2: FORWARD PASS
#    =============================================================== */
# function TabForwardPass() {
#   var _r = useState(8); var rank = _r[0], setRank = _r[1];
#   var _s = useState(0); var step = _s[0], setStep = _s[1];
#   var _au = useState(false); var autoP = _au[0], setAutoP = _au[1];
#   var _al = useState(16); var alpha = _al[0], setAlpha = _al[1];
#
#   var DIM = 4096;
#   var aParams = rank * DIM;
#   var bParams = DIM * rank;
#   var totalLoRA = aParams + bParams;
#   var fullDW = DIM * DIM;
#   var reduction = Math.round(fullDW / totalLoRA);
#   var scaling = (alpha / rank).toFixed(2);
#
#   var steps = [
#     { title: "Input x arrives", color: C.cyan, desc: "A token's hidden state x (shape [4096]) enters the layer. It will be sent down BOTH paths simultaneously." },
#     { title: "Frozen path: W\u2080x", color: C.dim, desc: "The frozen pre-trained matrix W\u2080 [4096\u00D74096] computes its output. No gradient is stored. This is pure forward pass computation." },
#     { title: "LoRA down: Ax", color: C.blue, desc: "Matrix A [" + rank + "\u00D74096] compresses x from 4096 dimensions down to just " + rank + ". This is the bottleneck - capturing the essential adaptation signal." },
#     { title: "LoRA up: B(Ax)", color: C.pink, desc: "Matrix B [4096\u00D7" + rank + "] expands the compressed signal back up to 4096 dimensions. B starts at zero so initial contribution is exactly zero." },
#     { title: "Scale by \u03B1/r = " + scaling, color: C.yellow, desc: "Multiply the LoRA output by " + alpha + "/" + rank + " = " + scaling + ". This controls how strongly the adapter steers the output. Tune \u03B1 without retraining." },
#     { title: "Sum: h = W\u2080x + (\u03B1/r)BAx", color: C.green, desc: "Element-wise addition. The frozen path provides the pre-trained behavior. The LoRA path provides the task-specific correction. Both are [4096] vectors." },
#   ];
#
#   useEffect(function() { if (!autoP) return; var t = setInterval(function() { setStep(function(s) { return (s + 1) % steps.length; }); }, 2500); return function() { clearInterval(t); }; }, [autoP, rank, alpha]);
#
#   return (
#     <div>
#       <SectionTitle title="The Forward Pass" subtitle={"How x flows through the frozen path AND the LoRA bypass simultaneously"} />
#
#       {/* Rank and Alpha controls */}
#       <div style={{ display: "flex", gap: 16, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
#         <Card style={{ minWidth: 280 }}>
#           <div style={{ fontSize: 10, color: C.accent, fontWeight: 700, marginBottom: 8 }}>{"Rank r = " + rank}</div>
#           <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }}
#             style={{ width: "100%", accentColor: C.accent }} />
#           <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
#             <span style={{ fontSize: 8, color: C.muted }}>r=1 (minimal)</span>
#             <span style={{ fontSize: 8, color: C.muted }}>r=64 (heavy)</span>
#           </div>
#         </Card>
#         <Card style={{ minWidth: 280 }}>
#           <div style={{ fontSize: 10, color: C.yellow, fontWeight: 700, marginBottom: 8 }}>{"\u03B1 (alpha) = " + alpha + "  \u2192  \u03B1/r = " + scaling}</div>
#           <input type="range" min={1} max={64} value={alpha} onChange={function(e) { setAlpha(parseInt(e.target.value)); }}
#             style={{ width: "100%", accentColor: C.yellow }} />
#           <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
#             <span style={{ fontSize: 8, color: C.muted }}>\u03B1=1 (gentle)</span>
#             <span style={{ fontSize: 8, color: C.muted }}>\u03B1=64 (strong)</span>
#           </div>
#         </Card>
#         <Card>
#           <div style={{ display: "flex", gap: 24 }}>
#             <StatBox label={"LORA PARAMS"} value={(totalLoRA / 1000).toFixed(0) + "K"} color={C.accent} />
#             <StatBox label={"FULL \u0394W"} value={(fullDW / 1e6).toFixed(1) + "M"} color={C.red} />
#             <StatBox label={"REDUCTION"} value={reduction + "x"} color={C.green} />
#             <StatBox label={"SCALING"} value={"\u03B1/r=" + scaling} color={C.yellow} />
#           </div>
#         </Card>
#       </div>
#
#       {/* Step selector */}
#       <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
#         {steps.map(function(s, i) {
#           var on = step === i;
#           return (<button key={i} onClick={function() { setStep(i); setAutoP(false); }} style={{
#             padding: "6px 10px", borderRadius: 6, border: "1.5px solid " + (on ? s.color : C.border),
#             background: on ? s.color + "20" : C.card, color: on ? s.color : C.muted,
#             cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
#           }}>{i + 1}</button>);
#         })}
#         <button onClick={function() { setAutoP(!autoP); }} style={{ padding: "6px 10px", borderRadius: 6, border: "1.5px solid " + (autoP ? C.yellow : C.border), background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted, cursor: "pointer", fontSize: 9, fontFamily: "monospace" }}>{autoP ? PAUSE : PLAY}</button>
#       </div>
#
#       {/* Forward pass diagram */}
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#         <svg width={1050} height={300} viewBox="0 0 780 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#
#           {/* Input x */}
#           <rect x={20} y={120} width={80} height={40} rx={8} fill={step === 0 ? C.cyan + "30" : C.card} stroke={step === 0 ? C.cyan : C.border} strokeWidth={step === 0 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={60} y={137} textAnchor="middle" fill={step === 0 ? C.cyan : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">x</text>
#           <text x={60} y={151} textAnchor="middle" fill={step === 0 ? C.cyan + "90" : C.dim} fontSize={8} fontFamily="monospace">[4096]</text>
#
#           {/* Split lines */}
#           <line x1={100} y1={140} x2={130} y2={140} stroke={C.dim + "60"} strokeWidth={1.5} />
#           <line x1={130} y1={140} x2={130} y2={80} stroke={C.dim + "60"} strokeWidth={1.5} />
#           <line x1={130} y1={140} x2={130} y2={210} stroke={C.dim + "60"} strokeWidth={1.5} />
#           <line x1={130} y1={80} x2={165} y2={80} stroke={step >= 1 ? C.dim + "90" : C.dim + "30"} strokeWidth={1.5} />
#           <line x1={130} y1={210} x2={165} y2={210} stroke={step >= 2 ? C.accent + "80" : C.dim + "30"} strokeWidth={1.5} />
#
#           {/* Frozen W0 */}
#           <rect x={165} y={55} width={130} height={50} rx={8} fill={step === 1 ? C.dim + "40" : C.card} stroke={step === 1 ? C.muted : C.border} strokeWidth={step === 1 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={230} y={75} textAnchor="middle" fill={step === 1 ? C.muted : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">W\u2080 (FROZEN)</text>
#           <text x={230} y={90} textAnchor="middle" fill={step === 1 ? C.dim : C.dim + "60"} fontSize={8} fontFamily="monospace">[4096 \u00D7 4096]</text>
#           <text x={230} y={102} textAnchor="middle" fill={C.dim + "60"} fontSize={7} fontFamily="monospace">{LOCK + " no grad stored"}</text>
#           <line x1={295} y1={80} x2={450} y2={80} stroke={step >= 1 ? C.muted + "60" : C.dim + "20"} strokeWidth={1.5} />
#
#           {/* LoRA A */}
#           <rect x={165} y={185} width={110} height={50} rx={8} fill={step === 2 ? C.blue + "30" : C.card} stroke={step === 2 ? C.blue : C.border} strokeWidth={step === 2 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={220} y={205} textAnchor="middle" fill={step === 2 ? C.blue : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">A (DOWN)</text>
#           <text x={220} y={219} textAnchor="middle" fill={step === 2 ? C.blue + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">{"[" + rank + " \u00D7 4096]"}</text>
#           <text x={220} y={231} textAnchor="middle" fill={C.accent + "60"} fontSize={7} fontFamily="monospace">{"random init"}</text>
#
#           {/* Arrow A to B */}
#           <line x1={275} y1={210} x2={315} y2={210} stroke={step >= 2 ? C.blue + "60" : C.dim + "20"} strokeWidth={1.5} />
#           <polygon points={"318,210 313,207 313,213"} fill={step >= 2 ? C.blue + "60" : C.dim + "20"} />
#           <text x={296} y={205} textAnchor="middle" fill={step >= 2 ? C.blue + "60" : C.dim + "30"} fontSize={7} fontFamily="monospace">{"[" + rank + "]"}</text>
#
#           {/* LoRA B */}
#           <rect x={318} y={185} width={110} height={50} rx={8} fill={step === 3 ? C.pink + "30" : C.card} stroke={step === 3 ? C.pink : C.border} strokeWidth={step === 3 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={373} y={205} textAnchor="middle" fill={step === 3 ? C.pink : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">B (UP)</text>
#           <text x={373} y={219} textAnchor="middle" fill={step === 3 ? C.pink + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">{"[4096 \u00D7 " + rank + "]"}</text>
#           <text x={373} y={231} textAnchor="middle" fill={C.accent + "60"} fontSize={7} fontFamily="monospace">{"zeros init"}</text>
#
#           {/* Arrow B to scale */}
#           <line x1={428} y1={210} x2={460} y2={210} stroke={step >= 3 ? C.pink + "60" : C.dim + "20"} strokeWidth={1.5} />
#           <polygon points={"463,210 458,207 458,213"} fill={step >= 3 ? C.pink + "60" : C.dim + "20"} />
#
#           {/* Scale */}
#           <rect x={463} y={187} width={85} height={46} rx={8} fill={step === 4 ? C.yellow + "20" : C.card} stroke={step === 4 ? C.yellow : C.border} strokeWidth={step === 4 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={505} y={207} textAnchor="middle" fill={step === 4 ? C.yellow : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace">{"\u00D7 \u03B1/r"}</text>
#           <text x={505} y={220} textAnchor="middle" fill={step === 4 ? C.yellow + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">{"=" + scaling}</text>
#           <text x={505} y={230} textAnchor="middle" fill={C.yellow + "50"} fontSize={7} fontFamily="monospace">{"volume knob"}</text>
#
#           {/* Arrow scale to sum */}
#           <line x1={548} y1={210} x2={580} y2={210} stroke={step >= 4 ? C.yellow + "60" : C.dim + "20"} strokeWidth={1.5} />
#           <line x1={580} y1={210} x2={580} y2={140} stroke={step >= 4 ? C.yellow + "60" : C.dim + "20"} strokeWidth={1.5} />
#           <line x1={580} y1={140} x2={610} y2={140} stroke={step >= 4 ? C.yellow + "60" : C.dim + "20"} strokeWidth={1.5} />
#
#           {/* Arrow frozen to sum */}
#           <line x1={450} y1={80} x2={580} y2={80} stroke={step >= 1 ? C.muted + "40" : C.dim + "20"} strokeWidth={1.5} />
#           <line x1={580} y1={80} x2={580} y2={140} stroke={step >= 1 ? C.muted + "40" : C.dim + "20"} strokeWidth={1.5} />
#
#           {/* Sum node */}
#           <circle cx={630} cy={140} r={22} fill={step === 5 ? C.green + "25" : C.card} stroke={step === 5 ? C.green : C.border} strokeWidth={step === 5 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={630} y={145} textAnchor="middle" fill={step === 5 ? C.green : C.dim} fontSize={16} fontWeight={800}>{PLUS}</text>
#
#           {/* Output h */}
#           <line x1={652} y1={140} x2={700} y2={140} stroke={step === 5 ? C.green + "80" : C.dim + "30"} strokeWidth={2} />
#           <polygon points={"703,140 697,136 697,144"} fill={step === 5 ? C.green + "80" : C.dim + "30"} />
#           <rect x={703} y={118} width={65} height={44} rx={8} fill={step === 5 ? C.green + "15" : C.card} stroke={step === 5 ? C.green : C.border} strokeWidth={step === 5 ? 2 : 1} style={{ transition: "all 0.3s" }} />
#           <text x={735} y={137} textAnchor="middle" fill={step === 5 ? C.green : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">h</text>
#           <text x={735} y={151} textAnchor="middle" fill={step === 5 ? C.green + "90" : C.dim + "60"} fontSize={8} fontFamily="monospace">[4096]</text>
#
#           {/* Formula at bottom */}
#           <text x={390} y={278} textAnchor="middle" fill={C.accent + "80"} fontSize={11} fontWeight={700} fontFamily="monospace">{"h = W\u2080x + (\u03B1/r) \u00B7 BAx"}</text>
#           <text x={390} y={292} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"   = W\u2080x + (" + alpha + "/" + rank + ") \u00B7 BAx    [scaling = " + scaling + "]"}</text>
#         </svg>
#       </div>
#
#       {/* Active step detail */}
#       <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: steps[step].color }}>
#         <div style={{ fontSize: 13, fontWeight: 700, color: steps[step].color, marginBottom: 6 }}>{"Step " + (step + 1) + ": " + steps[step].title}</div>
#         <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{steps[step].desc}</div>
#       </Card>
#
#       {/* Initialization callout */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"Initialization " + DASH + " Why B Starts at Zero"}</div>
#         <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
#           {[
#             { label: "A init", value: "Random Gaussian", desc: "Small random values provide diverse projections", color: C.blue },
#             { label: "B init", value: "All Zeros", desc: "\u0394W = B\u00D7A = 0\u00D7A = 0 at step 0", color: C.pink },
#             { label: "Effect", value: "W\u2080 + 0 = W\u2080", desc: "Model starts EXACTLY as pre-trained. No disruption.", color: C.green },
#             { label: "As training progresses", value: "B learns", desc: "Gradually steers toward task. Safe, stable start.", color: C.yellow },
#           ].map(function(v, i) {
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 160, padding: "10px 14px", borderRadius: 8, background: v.color + "08", border: "1px solid " + v.color + "25" }}>
#                 <div style={{ fontSize: 9, color: v.color, fontWeight: 700, marginBottom: 3 }}>{v.label}</div>
#                 <div style={{ fontSize: 13, fontWeight: 800, color: v.color, marginBottom: 3, fontFamily: "monospace" }}>{v.value}</div>
#                 <div style={{ fontSize: 9, color: C.muted }}>{v.desc}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={TARG} title="Why the Parallel Path Matters">
#         Bottleneck Adapters insert modules <span style={{color:C.red, fontWeight:700}}>in series</span> between layers - they're always there at inference, adding latency permanently. LoRA runs <span style={{color:C.green, fontWeight:700}}>in parallel</span> to the frozen weights. After training, W_merged = W\u2080 + (\u03B1/r)BA is just <span style={{color:C.yellow, fontWeight:700}}>one matrix addition</span>. The adapters vanish completely. <span style={{color:C.accent, fontWeight:700}}>Zero inference overhead.</span>
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
#   var _m = useState(1); var modelSel = _m[0], setModelSel = _m[1];
#   var _r = useState(8); var rank = _r[0], setRank = _r[1];
#
#   var models = [
#     { name: "LLaMA 1B", params: 1.24, color: C.green },
#     { name: "LLaMA 7B", params: 7, color: C.blue },
#     { name: "LLaMA 13B", params: 13, color: C.purple },
#     { name: "LLaMA 70B", params: 70, color: C.orange },
#   ];
#
#   var m = models[modelSel];
#   var wMem = m.params * 2;
#   var gMem_full = m.params * 2;
#   var oMem_full = m.params * 8;
#   var act_full = m.params < 5 ? 3 : m.params < 15 ? 12 : m.params < 50 ? 25 : 60;
#   var total_full = wMem + gMem_full + oMem_full + act_full;
#
#   var loraParamsM = (32 * 4 * 2 * rank * 4096) / 1e6;
#   var loraMem = loraParamsM * 2 / 1000;
#   var loraGrad = loraMem;
#   var loraOpt = loraParamsM * 8 / 1000;
#   var act_lora = m.params < 5 ? 1.5 : m.params < 15 ? 5 : m.params < 50 ? 10 : 25;
#   var total_lora = wMem + loraMem + loraGrad + loraOpt + act_lora;
#
#   var qMem = m.params * 0.5;
#   var total_qlora = qMem + loraMem + loraGrad + loraOpt + act_lora * 0.7;
#
#   var pctTrained = ((loraParamsM) / (m.params * 1000) * 100).toFixed(3);
#
#   return (
#     <div>
#       <SectionTitle title="Memory & Cost" subtitle={"Where LoRA saves memory " + DASH + " and why the savings are so dramatic"} />
#
#       <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 12, flexWrap: "wrap" }}>
#         {models.map(function(mod, i) {
#           var on = modelSel === i;
#           return (<button key={i} onClick={function() { setModelSel(i); }} style={{
#             padding: "8px 16px", borderRadius: 8, border: "1.5px solid " + (on ? mod.color : C.border),
#             background: on ? mod.color + "20" : C.card, color: on ? mod.color : C.muted,
#             cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
#           }}>{mod.name}</button>);
#         })}
#       </div>
#
#       <Card style={{ maxWidth: 1100, margin: "0 auto 12px" }}>
#         <div style={{ fontSize: 10, color: C.accent, fontWeight: 700, marginBottom: 6 }}>{"LoRA Rank r = " + rank + " \u2192 ~" + loraParamsM.toFixed(1) + "M trainable params (" + pctTrained + "% of model)"}</div>
#         <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }}
#           style={{ width: "100%", accentColor: C.accent }} />
#         <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
#           <span style={{ fontSize: 8, color: C.muted }}>r=1 (minimal)</span>
#           <span style={{ fontSize: 8, color: C.muted }}>r=64 (aggressive)</span>
#         </div>
#       </Card>
#
#       {/* Three-way comparison */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"GPU Memory Comparison " + DASH + " " + m.name + " (BF16)"}</div>
#
#         {[
#           { label: "Full Fine-Tuning", total: total_full, color: C.red, items: [
#             { name: "Weights", val: wMem, desc: "BF16" },
#             { name: "Gradients", val: gMem_full, desc: "eliminated by LoRA" },
#             { name: "Optimizer (Adam)", val: oMem_full, desc: "eliminated by LoRA" },
#             { name: "Activations", val: act_full, desc: "full backprop paths" },
#           ]},
#           { label: "LoRA (BF16 frozen)", total: total_lora, color: C.accent, items: [
#             { name: "Frozen Weights", val: wMem, desc: "forward only" },
#             { name: "LoRA Weights", val: loraMem, desc: "~" + (loraParamsM * 2).toFixed(0) + " MB" },
#             { name: "LoRA Gradients", val: loraGrad, desc: "~" + (loraParamsM * 2).toFixed(0) + " MB" },
#             { name: "LoRA Optimizer", val: loraOpt, desc: "~" + (loraParamsM * 8).toFixed(0) + " MB" },
#             { name: "Activations", val: act_lora, desc: "reduced paths" },
#           ]},
#           { label: "QLoRA (4-bit frozen)", total: total_qlora, color: C.green, items: [
#             { name: "4-bit Weights", val: qMem, desc: "NF4 quantized" },
#             { name: "LoRA Weights", val: loraMem, desc: "BF16" },
#             { name: "LoRA Gradients", val: loraGrad, desc: "BF16" },
#             { name: "LoRA Optimizer", val: loraOpt, desc: "FP32" },
#             { name: "Activations", val: act_lora * 0.7, desc: "reduced" },
#           ]},
#         ].map(function(method, mi) {
#           var maxVal = total_full;
#           return (
#             <div key={mi} style={{ marginBottom: 18 }}>
#               <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
#                 <div style={{ fontSize: 11, fontWeight: 700, color: method.color }}>{method.label}</div>
#                 <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
#                   <div style={{ fontSize: 22, fontWeight: 800, color: method.color }}>{"~" + method.total.toFixed(0) + " GB"}</div>
#                   {mi > 0 && <div style={{ fontSize: 9, color: C.green, fontFamily: "monospace" }}>{Math.round((1 - method.total / total_full) * 100) + "% less"}</div>}
#                 </div>
#               </div>
#               <div style={{ display: "flex", height: 28, borderRadius: 6, overflow: "hidden", border: "1px solid " + method.color + "30" }}>
#                 {method.items.map(function(item, ii) {
#                   var w = Math.max(1, (item.val / maxVal) * 100);
#                   var colors = [C.blue, C.red, C.orange, C.yellow, C.cyan, C.pink];
#                   return (
#                     <div key={ii} title={item.name + ": " + item.val.toFixed(1) + " GB"} style={{
#                       width: w + "%", height: "100%", background: colors[ii % colors.length] + "40",
#                       borderRight: "1px solid " + C.bg, transition: "width 0.5s",
#                       display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
#                     }}>
#                       {w > 5 && <span style={{ fontSize: 7, color: colors[ii % colors.length], fontFamily: "monospace", whiteSpace: "nowrap" }}>{item.name}</span>}
#                     </div>
#                   );
#                 })}
#               </div>
#               <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 4 }}>
#                 {method.items.map(function(item, ii) {
#                   var colors = [C.blue, C.red, C.orange, C.yellow, C.cyan, C.pink];
#                   return (
#                     <span key={ii} style={{ fontSize: 8, color: colors[ii % colors.length], fontFamily: "monospace" }}>
#                       {item.name + ": " + item.val.toFixed(1) + "GB"}
#                     </span>
#                   );
#                 })}
#               </div>
#             </div>
#           );
#         })}
#       </Card>
#
#       {/* Gradient / optimizer insight */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.cyan, marginBottom: 10 }}>{"Why Frozen = No Optimizer States " + DASH + " The Adam Savings"}</div>
#         <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
#           {[
#             { label: "Per-param in Adam", items: ["Weight: 2 bytes (BF16)", "Gradient: 2 bytes", "Momentum: 4 bytes (FP32)", "Variance: 4 bytes (FP32)", "Total: 12 bytes/param"], color: C.red },
#             { label: "Frozen params (7B)", items: ["Weight: 2 bytes " + CHK, "Gradient: NONE " + CHK, "Momentum: NONE " + CHK, "Variance: NONE " + CHK, "Total: 2 bytes/param"], color: C.green },
#           ].map(function(col, i) {
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 200, padding: "10px 14px", borderRadius: 8, background: col.color + "08", border: "1px solid " + col.color + "25" }}>
#                 <div style={{ fontSize: 10, fontWeight: 700, color: col.color, marginBottom: 6 }}>{col.label}</div>
#                 {col.items.map(function(it, j) { return (<div key={j} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8, fontFamily: "monospace" }}>{"  " + it}</div>); })}
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={CHART} title={"Storage Per Task " + DASH + " The Multi-Task Advantage"}>
#         Full fine-tuning: 3 tasks = <span style={{color:C.red, fontWeight:700}}>{(m.params * 2 * 3).toFixed(0) + " GB"}</span> of checkpoints. LoRA: 3 tasks = <span style={{color:C.green, fontWeight:700}}>{m.params * 2 + " GB base + 3 \u00D7 ~50MB"}</span>. Scale to 100 tasks: Full FT <span style={{color:C.red}}>{(m.params * 2 * 100).toFixed(0) + " GB"}</span> vs LoRA <span style={{color:C.green}}>{(m.params * 2 + 5).toFixed(0) + " GB"}</span>. One base model, swap adapters in milliseconds.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 4: DATA PIPELINE
#    =============================================================== */
# function TabDataPipeline() {
#   var _s = useState(0); var step = _s[0], setStep = _s[1];
#   var _au = useState(false); var autoP = _au[0], setAutoP = _au[1];
#
#   var pSteps = [
#     { title: "Raw Data (JSONL)", color: C.blue,
#       desc: "Identical to full fine-tuning. LoRA doesn't change what goes INTO the model, only what happens INSIDE it during training.",
#       note: "Same as Full FT",
#       content: '{"instruction": "Classify sentiment",  "input": "This movie was breathtaking",  "output": "positive"}' },
#     { title: "Template Formatting", color: C.cyan,
#       desc: "Model-specific chat template applied. Same as full fine-tuning. The model was pre-trained expecting a specific format - wrong template degrades performance.",
#       note: "Same as Full FT",
#       content: '<s>[INST] Classify sentiment: This movie was breathtaking [/INST] positive</s>' },
#     { title: "Tokenization", color: C.purple,
#       desc: "Text converted to integer IDs. Tokenizer is fixed - never changes during training or fine-tuning of any kind.",
#       note: "Same as Full FT",
#       content: '[1, 518, 25580, 29962, 4134, 1598, ..., 6374, 2]  -> 20 tokens' },
#     { title: "Labels + Loss Mask", color: C.yellow,
#       desc: "-100 means ignore this token. The model is graded ONLY on predicting the response tokens. Instruction tokens are masked out.",
#       note: "Same as Full FT",
#       content: 'Labels: [-100, -100, ..., -100,  6374,  2 ]  (instruction=ignored, output=graded)' },
#     { title: "Batch Tensors to GPU", color: C.orange,
#       desc: "Padded to same length, collated into [batch, seq_len] tensors, moved to GPU. Embedding converts IDs to 4096-dim vectors inside the model.",
#       note: "Same as Full FT",
#       content: 'input_ids [4, 512] -> Embed -> hidden [4, 512, 4096] -> Layer 0 ...' },
#     { title: "Forward Pass: W\u2080x + BAx", color: C.accent,
#       desc: "HERE is where LoRA diverges. Every target weight gets a parallel bypass. h = W0*x + (alpha/r)*B*A*x. Both paths run simultaneously and are summed.",
#       note: "LoRA DIVERGES HERE",
#       content: 'h = W0*x  (frozen, no grad stored)  +  (alpha/r) * B*(A*x)  (LoRA path, trainable)' },
#     { title: "Loss Computation", color: C.red,
#       desc: "CrossEntropyLoss on logits vs labels. Identical to full fine-tuning - loss doesn't know or care about LoRA.",
#       note: "Same as Full FT",
#       content: 'loss = CrossEntropyLoss(logits[4,512,32000], labels[4,512])  // only where labels != -100' },
#     { title: "Backward: LoRA Only", color: C.pink,
#       desc: "Gradients flow THROUGH frozen weights (needed for chain rule) but are NOT STORED for them. Only LoRA A and B accumulate gradient tensors.",
#       note: "LoRA KEY DIFFERENCE",
#       content: 'grad(W0) = computed, NOT stored  |  grad(A), grad(B) = computed AND stored (~33MB total)' },
#     { title: "Optimizer: Update A and B Only", color: C.green,
#       desc: "Adam updates ONLY the LoRA matrices. 7B frozen weights are completely untouched. ~67MB optimizer states vs ~56GB for full FT.",
#       note: "LoRA KEY DIFFERENCE",
#       content: 'A -= lr * adam_step(grad_A)  |  B -= lr * adam_step(grad_B)  |  W0 = unchanged forever' },
#   ];
#
#   useEffect(function() { if (!autoP) return; var t = setInterval(function() { setStep(function(s) { return (s + 1) % pSteps.length; }); }, 3000); return function() { clearInterval(t); }; }, [autoP]);
#
#   var isLoRaStep = step === 5 || step === 7 || step === 8;
#
#   return (
#     <div>
#       <SectionTitle title="Data Pipeline" subtitle={"The LoRA pipeline is identical to full FT until step 6 " + DASH + " then it diverges"} />
#
#       <div style={{ display: "flex", gap: 4, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
#         {pSteps.map(function(s, i) {
#           var on = step === i;
#           var isL = i === 5 || i === 7 || i === 8;
#           return (<button key={i} onClick={function() { setStep(i); setAutoP(false); }} style={{
#             padding: "6px 9px", borderRadius: 6, border: "1.5px solid " + (on ? s.color : isL ? C.accent + "40" : C.border),
#             background: on ? s.color + "20" : isL ? C.accent + "08" : C.card, color: on ? s.color : isL ? C.accent + "70" : C.muted,
#             cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
#           }}>{i + 1}</button>);
#         })}
#         <button onClick={function() { setAutoP(!autoP); }} style={{ padding: "6px 9px", borderRadius: 6, border: "1.5px solid " + (autoP ? C.yellow : C.border), background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted, cursor: "pointer", fontSize: 9, fontFamily: "monospace" }}>{autoP ? PAUSE : PLAY}</button>
#       </div>
#
#       {/* Pipeline flow */}
#       <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#         <svg width={1050} height={80} viewBox="0 0 780 80" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#           {pSteps.map(function(s, i) {
#             var x = 10 + i * 84;
#             var on = step === i;
#             var done = i < step;
#             var isL = i === 5 || i === 7 || i === 8;
#             return (<g key={i}>
#               <rect x={x} y={12} width={76} height={50} rx={6}
#                 fill={on ? s.color + "25" : done ? s.color + "08" : isL ? C.accent + "06" : C.card}
#                 stroke={on ? s.color : done ? s.color + "30" : isL ? C.accent + "30" : C.dim + "40"}
#                 strokeWidth={on ? 2 : 1} style={{ transition: "all 0.3s" }} />
#               <text x={x + 38} y={32} textAnchor="middle" fill={on ? s.color : done ? s.color + "70" : C.dim} fontSize={7} fontWeight={700} fontFamily="monospace">{"Step " + (i + 1)}</text>
#               <text x={x + 38} y={46} textAnchor="middle" fill={on ? s.color + "80" : C.dim} fontSize={6} fontFamily="monospace">{s.title.substring(0, 12)}</text>
#               {isL && !on && <rect x={x} y={58} width={76} height={2} rx={1} fill={C.accent + "40"} />}
#               {on && <rect x={x} y={58} width={76} height={3} rx={1.5} fill={s.color} style={{ animation: "pulse 1.5s infinite" }} />}
#               {i < pSteps.length - 1 && <g>
#                 <line x1={x + 76} y1={37} x2={x + 84} y2={37} stroke={done ? s.color + "40" : C.dim + "20"} strokeWidth={1} />
#               </g>}
#             </g>);
#           })}
#         </svg>
#       </div>
#
#       {/* Step detail */}
#       <Card highlight={isLoRaStep} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: isLoRaStep ? C.accent : pSteps[step].color }}>
#         <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexWrap: "wrap", gap: 8 }}>
#           <div style={{ fontSize: 13, fontWeight: 700, color: pSteps[step].color }}>{"Step " + (step + 1) + ": " + pSteps[step].title}</div>
#           <Badge color={isLoRaStep ? C.accent : C.dim}>{pSteps[step].note}</Badge>
#         </div>
#         <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8, marginBottom: 12 }}>{pSteps[step].desc}</div>
#         <div style={{ padding: "12px 16px", background: "#08080d", borderRadius: 8, border: "1px solid " + C.border }}>
#           <pre style={{ fontSize: 10, color: pSteps[step].color, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.8, margin: 0, whiteSpace: "pre-wrap", overflowX: "auto" }}>{pSteps[step].content}</pre>
#         </div>
#       </Card>
#
#       <Insight icon={GEAR} title="The Critical Distinction">
#         Steps 1-5 are <span style={{color:C.blue, fontWeight:700}}>byte-for-byte identical</span> to full fine-tuning. The data pipeline doesn't know what training method you're using. LoRA's changes are entirely in the <span style={{color:C.accent, fontWeight:700}}>forward computation</span> (parallel bypass), the <span style={{color:C.pink, fontWeight:700}}>backward pass</span> (no gradient storage for frozen params), and the <span style={{color:C.green, fontWeight:700}}>optimizer</span> (only A and B get updated).
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 5: RANK & ALPHA
#    =============================================================== */
# function TabRankAlpha() {
#   var _r = useState(8); var rank = _r[0], setRank = _r[1];
#   var _al = useState(16); var alpha = _al[0], setAlpha = _al[1];
#   var _t = useState(0); var tgtSel = _t[0], setTgtSel = _t[1];
#
#   var DIM = 4096;
#   var targets = [
#     { name: "Minimal: q,v", count: 2, color: C.blue },
#     { name: "Standard: q,k,v,o", count: 4, color: C.accent },
#     { name: "Aggressive: all-linear", count: 7, color: C.orange },
#   ];
#
#   var tgt = targets[tgtSel];
#   var totalMatrices = 32 * tgt.count * 2;
#   var totalParams = 32 * tgt.count * 2 * rank * DIM;
#   var totalParamsM = totalParams / 1e6;
#   var pct = (totalParamsM / 7000 * 100).toFixed(3);
#   var sizeMB = (totalParamsM * 2).toFixed(0);
#   var scaling = (alpha / rank).toFixed(2);
#
#   var rankPresets = [
#     { r: 4, desc: "Very lean. Simple tasks, tiny data." },
#     { r: 8, desc: "Default sweet spot. Most instruction-tuning." },
#     { r: 16, desc: "Complex tasks, nuanced adaptation." },
#     { r: 32, desc: "Very different domains. Diminishing returns." },
#     { r: 64, desc: "Rarely needed. Consider full FT." },
#   ];
#
#   return (
#     <div>
#       <SectionTitle title="Rank & Alpha" subtitle={"The two core hyperparameters that control LoRA expressiveness and training stability"} />
#
#       {/* Controls */}
#       <div style={{ display: "flex", gap: 16, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
#         <Card style={{ flex: 1, minWidth: 280 }}>
#           <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 8 }}>{"Rank r = " + rank + " (expressiveness)"}</div>
#           <input type="range" min={1} max={64} value={rank} onChange={function(e) { setRank(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.accent }} />
#           <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, marginBottom: 12 }}>
#             <span style={{ fontSize: 8, color: C.muted }}>r=1</span><span style={{ fontSize: 8, color: C.muted }}>r=64</span>
#           </div>
#           <div style={{ fontSize: 10, fontWeight: 700, color: C.yellow, marginBottom: 8 }}>{"\u03B1 = " + alpha + " \u2192 scaling \u03B1/r = " + scaling}</div>
#           <input type="range" min={1} max={128} value={alpha} onChange={function(e) { setAlpha(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.yellow }} />
#           <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
#             <span style={{ fontSize: 8, color: C.muted }}>\u03B1=1</span><span style={{ fontSize: 8, color: C.muted }}>\u03B1=128</span>
#           </div>
#         </Card>
#
#         <Card style={{ flex: 1, minWidth: 240 }}>
#           <div style={{ fontSize: 10, fontWeight: 700, color: C.text, marginBottom: 10 }}>Target Modules</div>
#           {targets.map(function(t, i) {
#             var on = tgtSel === i;
#             return (<button key={i} onClick={function() { setTgtSel(i); }} style={{
#               display: "block", width: "100%", marginBottom: 6, padding: "7px 12px", borderRadius: 6,
#               border: "1.5px solid " + (on ? t.color : C.border), background: on ? t.color + "18" : C.card,
#               color: on ? t.color : C.muted, cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace", textAlign: "left"
#             }}>{t.name}</button>);
#           })}
#         </Card>
#
#         <Card style={{ flex: 1, minWidth: 220 }}>
#           <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
#             <StatBox label="TRAINABLE PARAMS" value={totalParamsM.toFixed(1) + "M"} color={C.accent} />
#             <StatBox label="% OF MODEL" value={pct + "%"} color={C.yellow} />
#             <StatBox label="ADAPTER SIZE" value={sizeMB + " MB"} color={C.green} />
#             <StatBox label="SCALING \u03B1/r" value={scaling} color={tgtSel === 0 ? C.blue : tgtSel === 1 ? C.accent : C.orange} />
#           </div>
#         </Card>
#       </div>
#
#       {/* Rank visualization */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Rank = Bandwidth of Adaptation"}</div>
#         <div style={{ position: "relative", height: 40, background: C.border, borderRadius: 6, overflow: "hidden", marginBottom: 8 }}>
#           <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: "100%", background: C.dim + "30", borderRadius: 6 }} />
#           <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: (rank / 64 * 100) + "%", background: "linear-gradient(90deg," + C.accent + "60," + C.pink + "40)", borderRadius: 6, transition: "width 0.3s", display: "flex", alignItems: "center", paddingLeft: 10 }}>
#             <span style={{ fontSize: 9, color: C.text, fontFamily: "monospace", fontWeight: 700 }}>{rank + " channels of adaptation"}</span>
#           </div>
#           <div style={{ position: "absolute", right: 8, top: 11, fontSize: 8, color: C.dim }}>4096 (full rank)</div>
#         </div>
#         <div style={{ marginBottom: 12, fontSize: 9, color: C.muted }}>{"Think of rank as the number of independent directions of change. r=8 captures 8 " + LQ + "dimensions" + RQ + " of task adaptation."}</div>
#
#         <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
#           {rankPresets.map(function(p, i) {
#             var on = rank === p.r;
#             return (
#               <div key={i} onClick={function() { setRank(p.r); }} style={{
#                 flex: 1, minWidth: 120, padding: "10px 12px", borderRadius: 8, cursor: "pointer",
#                 background: on ? C.accent + "15" : C.bg, border: "1.5px solid " + (on ? C.accent : C.border),
#                 transition: "all 0.25s",
#               }}>
#                 <div style={{ fontSize: 14, fontWeight: 800, color: on ? C.accent : C.muted, fontFamily: "monospace", marginBottom: 4 }}>{"r=" + p.r}</div>
#                 <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.6 }}>{p.desc}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       {/* Alpha / scaling */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"The \u03B1 Scaling Factor " + DASH + " Volume Knob on LoRA Output"}</div>
#         <div style={{ fontSize: 11, color: C.muted, marginBottom: 12 }}>{"h = W\u2080x + (\u03B1/r) \u00B7 BAx   \u2192   current scaling: " + alpha + "/" + rank + " = " + scaling}</div>
#         <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
#           {[
#             { label: "\u03B1 = r/2 (gentle)", ratio: 0.5, desc: "Adapters contribute cautiously. Useful if base model behavior should dominate." },
#             { label: "\u03B1 = r (neutral)", ratio: 1.0, desc: "Safe default. Balanced contribution from frozen and LoRA paths. Start here." },
#             { label: "\u03B1 = 2r (amplified)", ratio: 2.0, desc: "Most common production choice. Slightly stronger adaptation signal." },
#             { label: "\u03B1 = 4r (aggressive)", ratio: 4.0, desc: "Use with lower LR. Can destabilize training if too high." },
#           ].map(function(v, i) {
#             var isCurrent = Math.abs(scaling - v.ratio) < 0.25;
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 160, padding: "10px 12px", borderRadius: 8, background: isCurrent ? C.yellow + "12" : C.bg, border: "1.5px solid " + (isCurrent ? C.yellow : C.border), transition: "all 0.3s" }}>
#                 <div style={{ fontSize: 9, fontWeight: 700, color: isCurrent ? C.yellow : C.muted, marginBottom: 6 }}>{v.label}</div>
#                 <div style={{ height: 12, background: C.border, borderRadius: 3, marginBottom: 6 }}>
#                   <div style={{ width: Math.min(100, v.ratio * 40) + "%", height: "100%", borderRadius: 3, background: C.yellow + (isCurrent ? "80" : "30") }} />
#                 </div>
#                 <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.6 }}>{v.desc}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       {/* Target modules table */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Which Layers to Target (32-layer, r=" + rank + ")"}</div>
#         <div style={{ overflowX: "auto" }}>
#           <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9, fontFamily: "monospace" }}>
#             <thead>
#               <tr>{["Config", "target_modules", "LoRA pairs", "Params", "% of 7B", "Adapter MB"].map(function(h, i) { return (<th key={i} style={{ padding: "6px 10px", textAlign: "left", color: C.muted, borderBottom: "1px solid " + C.border }}>{h}</th>); })}</tr>
#             </thead>
#             <tbody>
#               {[
#                 { name: "Minimal", mods: "q, v", count: 2, star: false },
#                 { name: "Standard", mods: "q, k, v, o", count: 4, star: true },
#                 { name: "Aggressive", mods: "q, k, v, o, gate, up, down", count: 7, star: false },
#               ].map(function(row, i) {
#                 var pairs = 32 * row.count * 2;
#                 var params = (32 * row.count * 2 * rank * DIM / 1e6).toFixed(1);
#                 var pctRow = (32 * row.count * 2 * rank * DIM / 1e9 / 7 * 100).toFixed(3);
#                 var mb = (32 * row.count * 2 * rank * DIM * 2 / 1e6).toFixed(0);
#                 var colors = [C.blue, C.accent, C.orange];
#                 var on = tgtSel === i;
#                 return (
#                   <tr key={i} onClick={function() { setTgtSel(i); }} style={{ cursor: "pointer", background: on ? colors[i] + "10" : "transparent", transition: "background 0.2s" }}>
#                     <td style={{ padding: "7px 10px", color: colors[i], fontWeight: 700 }}>{row.name + (row.star ? " *" : "")}</td>
#                     <td style={{ padding: "7px 10px", color: C.muted }}>{row.mods}</td>
#                     <td style={{ padding: "7px 10px", color: C.dim }}>{pairs}</td>
#                     <td style={{ padding: "7px 10px", color: colors[i] }}>{params + "M"}</td>
#                     <td style={{ padding: "7px 10px", color: C.dim }}>{pctRow + "%"}</td>
#                     <td style={{ padding: "7px 10px", color: colors[i] }}>{mb + " MB"}</td>
#                   </tr>
#                 );
#               })}
#             </tbody>
#           </table>
#         </div>
#         <div style={{ marginTop: 8, fontSize: 9, color: C.muted }}>{"* recommended starting point. Click a row to update all stats."}</div>
#       </Card>
#
#       <Insight icon={TARG} title="Choosing r and alpha in Practice">
#         Start with <span style={{color:C.accent, fontWeight:700}}>r=8, \u03B1=16</span> (standard). If the task is complex or very different from pre-training, try <span style={{color:C.yellow}}>r=16 or r=32</span>. Keep <span style={{color:C.yellow, fontWeight:700}}>\u03B1 = 2r</span> as a rule of thumb. If you change r, update \u03B1 proportionally to keep the scaling ratio stable. More targets = more params = better quality but more memory. "<span style={{color:C.orange}}>all-linear</span>" is overkill for most tasks but is the safest "just make it work" option.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 6: MERGE & DEPLOY
#    =============================================================== */
# function TabMergeDeploy() {
#   var _m = useState(false); var merged = _m[0], setMerged = _m[1];
#   var _s = useState(0); var swapIdx = _s[0], setSwapIdx = _s[1];
#   var _au = useState(false); var autoSwap = _au[0], setAutoSwap = _au[1];
#
#   useEffect(function() {
#     if (!autoSwap) return;
#     var t = setInterval(function() { setSwapIdx(function(i) { return (i + 1) % 3; }); }, 2000);
#     return function() { clearInterval(t); };
#   }, [autoSwap]);
#
#   var adapters = [
#     { name: "Medical QA", color: C.blue, size: "~33 MB", r: 8, alpha: 16 },
#     { name: "Code Gen", color: C.green, size: "~67 MB", r: 16, alpha: 32 },
#     { name: "Legal Summ.", color: C.orange, size: "~33 MB", r: 8, alpha: 16 },
#   ];
#
#   return (
#     <div>
#       <SectionTitle title="Merge & Deploy" subtitle={"LoRA's killer advantage: adapters can merge into the base model and completely vanish"} />
#
#       {/* Merge visual */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"The Merge Operation " + DASH + " W_merged = W\u2080 + (\u03B1/r) \u00B7 B \u00D7 A"}</div>
#
#         <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#           <svg width={1050} height={180} viewBox="0 0 780 180" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#
#             {/* DURING TRAINING label */}
#             <text x={190} y={18} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">DURING TRAINING</text>
#
#             {/* W0 box */}
#             <rect x={20} y={30} width={130} height={60} rx={8} fill={C.dim + "20"} stroke={!merged ? C.dim + "60" : C.dim + "30"} strokeWidth={1.5} style={{ transition: "all 0.6s" }} />
#             <text x={85} y={56} textAnchor="middle" fill={!merged ? C.muted : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">W\u2080 (frozen)</text>
#             <text x={85} y={72} textAnchor="middle" fill={!merged ? C.dim : C.dim + "50"} fontSize={8} fontFamily="monospace">14 GB (7B model)</text>
#             <text x={85} y={84} textAnchor="middle" fill={!merged ? C.dim : C.dim + "40"} fontSize={7} fontFamily="monospace">[4096 \u00D7 4096]</text>
#
#             {/* Plus */}
#             <text x={165} y={65} textAnchor="middle" fill={!merged ? C.accent + "80" : C.dim + "30"} fontSize={22} fontWeight={800} style={{ transition: "all 0.6s" }}>{PLUS}</text>
#
#             {/* B*A box */}
#             <rect x={185} y={30} width={130} height={60} rx={8} fill={!merged ? C.accent + "15" : C.dim + "05"} stroke={!merged ? C.accent + "60" : C.dim + "20"} strokeWidth={1.5} style={{ transition: "all 0.6s" }} />
#             <text x={250} y={56} textAnchor="middle" fill={!merged ? C.accent : C.dim + "40"} fontSize={10} fontWeight={700} fontFamily="monospace">(\u03B1/r) B\u00D7A</text>
#             <text x={250} y={72} textAnchor="middle" fill={!merged ? C.accent + "80" : C.dim + "30"} fontSize={8} fontFamily="monospace">~33 MB adapter</text>
#             <text x={250} y={84} textAnchor="middle" fill={!merged ? C.accent + "60" : C.dim + "20"} fontSize={7} fontFamily="monospace">trainable</text>
#
#             {/* Arrow to merge */}
#             {!merged && <g>
#               <line x1={315} y1={60} x2={360} y2={60} stroke={C.yellow} strokeWidth={2} />
#               <polygon points="363,60 357,56 357,64" fill={C.yellow} />
#               <text x={338} y={50} textAnchor="middle" fill={C.yellow} fontSize={8} fontWeight={700} fontFamily="monospace">MERGE</text>
#               <text x={338} y={78} textAnchor="middle" fill={C.yellow + "60"} fontSize={7} fontFamily="monospace">matrix add</text>
#             </g>}
#
#             {/* AFTER MERGE label */}
#             <text x={560} y={18} textAnchor="middle" fill={merged ? C.green : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace" style={{ transition: "all 0.6s" }}>AFTER MERGE</text>
#
#             {/* Merged box */}
#             <rect x={370} y={30} width={150} height={60} rx={8} fill={merged ? C.green + "15" : C.card} stroke={merged ? C.green + "60" : C.border} strokeWidth={merged ? 2 : 1} style={{ transition: "all 0.6s" }} />
#             <text x={445} y={56} textAnchor="middle" fill={merged ? C.green : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">W_merged</text>
#             <text x={445} y={72} textAnchor="middle" fill={merged ? C.green + "90" : C.dim} fontSize={8} fontFamily="monospace">{merged ? "14 GB (same size!)" : "W\u2080 + (\u03B1/r)BA"}</text>
#             <text x={445} y={84} textAnchor="middle" fill={merged ? C.green + "60" : C.dim + "50"} fontSize={7} fontFamily="monospace">{merged ? "adapters GONE" : "[4096 \u00D7 4096]"}</text>
#
#             {/* Properties comparison */}
#             {[
#               { label: "Inference overhead", before: "2 paths + sum", after: "Single path", color: C.green },
#               { label: "Extra memory", before: "+33 MB adapters", after: "0 bytes extra", color: C.green },
#               { label: "PEFT library needed", before: "Yes (runtime)", after: "No (standard model)", color: C.green },
#               { label: "Adapter swapping", before: "Yes (keep separate)", after: "No (baked in)", color: C.yellow },
#             ].map(function(row, i) {
#               return (
#                 <g key={i}>
#                   <text x={15} y={120 + i * 14} fill={C.dim} fontSize={7} fontFamily="monospace">{row.label + ":"}</text>
#                   <text x={180} y={120 + i * 14} fill={!merged ? C.muted : C.dim + "40"} fontSize={7} fontFamily="monospace" style={{ transition: "all 0.5s" }}>{row.before}</text>
#                   <text x={380} y={120 + i * 14} fill={merged ? row.color : C.dim + "30"} fontSize={7} fontFamily="monospace" fontWeight={merged ? "700" : "400"} style={{ transition: "all 0.5s" }}>{row.after}</text>
#                 </g>
#               );
#             })}
#           </svg>
#         </div>
#
#         <div style={{ display: "flex", justifyContent: "center", gap: 12 }}>
#           <button onClick={function() { setMerged(false); }} style={{
#             padding: "10px 24px", borderRadius: 8, border: "1.5px solid " + (!merged ? C.accent : C.border),
#             background: !merged ? C.accent + "20" : C.card, color: !merged ? C.accent : C.muted,
#             cursor: "pointer", fontSize: 11, fontWeight: 700, fontFamily: "monospace"
#           }}>{"During Training"}</button>
#           <button onClick={function() { setMerged(true); }} style={{
#             padding: "10px 24px", borderRadius: 8, border: "1.5px solid " + (merged ? C.green : C.border),
#             background: merged ? C.green + "20" : C.card, color: merged ? C.green : C.muted,
#             cursor: "pointer", fontSize: 11, fontWeight: 700, fontFamily: "monospace"
#           }}>{"After Merge " + MERGE}</button>
#         </div>
#       </Card>
#
#       {/* Adapter swapping */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 4 }}>{"Adapter Swapping " + DASH + " One Base Model, Many Tasks"}</div>
#         <div style={{ fontSize: 10, color: C.muted, marginBottom: 16 }}>{"Load base model once (14 GB). Swap tiny adapters in milliseconds. No model reloading."}</div>
#
#         <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
#           <svg width={1050} height={200} viewBox="0 0 780 200" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
#             {/* Base model */}
#             <rect x={290} y={10} width={200} height={55} rx={10} fill={C.dim + "25"} stroke={C.muted + "50"} strokeWidth={2} />
#             <text x={390} y={32} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">BASE MODEL</text>
#             <text x={390} y={48} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " 14 GB frozen (loaded once)"}</text>
#
#             {/* Connector lines */}
#             {adapters.map(function(ad, i) {
#               var x = 65 + i * 220;
#               var active = swapIdx === i;
#               return (<g key={"line" + i}>
#                 <line x1={390} y1={65} x2={x + 75} y2={110} stroke={active ? ad.color + "60" : C.dim + "20"} strokeWidth={active ? 2 : 1} strokeDasharray={active ? "0" : "4,4"} style={{ transition: "all 0.4s" }} />
#               </g>);
#             })}
#
#             {/* Adapters */}
#             {adapters.map(function(ad, i) {
#               var x = 65 + i * 220;
#               var active = swapIdx === i;
#               return (<g key={"ad" + i}>
#                 <rect x={x} y={110} width={150} height={70} rx={10} fill={active ? ad.color + "20" : C.card} stroke={active ? ad.color : C.border} strokeWidth={active ? 2.5 : 1} style={{ transition: "all 0.4s" }} />
#                 <text x={x + 75} y={133} textAnchor="middle" fill={active ? ad.color : C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{ad.name}</text>
#                 <text x={x + 75} y={148} textAnchor="middle" fill={active ? ad.color + "90" : C.dim + "50"} fontSize={8} fontFamily="monospace">{"r=" + ad.r + ", \u03B1=" + ad.alpha}</text>
#                 <text x={x + 75} y={163} textAnchor="middle" fill={active ? ad.color + "80" : C.dim + "40"} fontSize={8} fontFamily="monospace">{ad.size}</text>
#                 {active && <rect x={x} y={177} width={150} height={3} rx={1.5} fill={ad.color} style={{ animation: "pulse 1.5s infinite" }} />}
#               </g>);
#             })}
#           </svg>
#         </div>
#
#         <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 10 }}>
#           {adapters.map(function(ad, i) {
#             var on = swapIdx === i;
#             return (<button key={i} onClick={function() { setSwapIdx(i); setAutoSwap(false); }} style={{
#               padding: "8px 18px", borderRadius: 8, border: "1.5px solid " + (on ? ad.color : C.border),
#               background: on ? ad.color + "20" : C.card, color: on ? ad.color : C.muted,
#               cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
#             }}>{ad.name}</button>);
#           })}
#           <button onClick={function() { setAutoSwap(!autoSwap); }} style={{ padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (autoSwap ? C.yellow : C.border), background: autoSwap ? C.yellow + "20" : C.card, color: autoSwap ? C.yellow : C.muted, cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>{autoSwap ? PAUSE + " Stop" : PLAY + " Auto-swap"}</button>
#         </div>
#
#         <div style={{ padding: "10px 14px", background: adapters[swapIdx].color + "08", borderRadius: 8, border: "1px solid " + adapters[swapIdx].color + "30" }}>
#           <div style={{ fontSize: 10, fontWeight: 700, color: adapters[swapIdx].color, marginBottom: 4 }}>{"Active: " + adapters[swapIdx].name + " Adapter"}</div>
#           <div style={{ display: "flex", gap: 20 }}>
#             <StatBox label="RANK" value={"r=" + adapters[swapIdx].r} color={adapters[swapIdx].color} bigFont={14} />
#             <StatBox label="ALPHA" value={"\u03B1=" + adapters[swapIdx].alpha} color={adapters[swapIdx].color} bigFont={14} />
#             <StatBox label="SCALING" value={(adapters[swapIdx].alpha / adapters[swapIdx].r).toFixed(1) + "x"} color={adapters[swapIdx].color} bigFont={14} />
#             <StatBox label="SIZE" value={adapters[swapIdx].size} color={adapters[swapIdx].color} bigFont={12} minW={80} />
#           </div>
#         </div>
#       </Card>
#
#       {/* Storage comparison */}
#       <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
#         <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Storage Comparison " + DASH + " Scale to Many Tasks"}</div>
#         <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
#           {[1, 3, 10, 100].map(function(n, i) {
#             var fullGB = n * 14;
#             var loraGB = 14 + n * 0.033;
#             return (
#               <div key={i} style={{ flex: 1, minWidth: 120, padding: "10px 12px", borderRadius: 8, background: C.bg, border: "1px solid " + C.border }}>
#                 <div style={{ fontSize: 11, fontWeight: 800, color: C.text, marginBottom: 8, textAlign: "center" }}>{n + " task" + (n > 1 ? "s" : "")}</div>
#                 <div style={{ marginBottom: 6 }}>
#                   <div style={{ fontSize: 8, color: C.red, marginBottom: 2 }}>Full FT: {fullGB + " GB"}</div>
#                   <div style={{ height: 10, background: C.border, borderRadius: 2 }}>
#                     <div style={{ width: "100%", height: "100%", borderRadius: 2, background: C.red + "40" }} />
#                   </div>
#                 </div>
#                 <div>
#                   <div style={{ fontSize: 8, color: C.green, marginBottom: 2 }}>LoRA: {loraGB.toFixed(1) + " GB"}</div>
#                   <div style={{ height: 10, background: C.border, borderRadius: 2 }}>
#                     <div style={{ width: (loraGB / fullGB * 100) + "%", height: "100%", borderRadius: 2, background: C.green + "50", transition: "width 0.3s" }} />
#                   </div>
#                 </div>
#                 <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, textAlign: "center", marginTop: 6 }}>{Math.round((1 - loraGB / fullGB) * 100) + "% smaller"}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={MERGE} title="The Mergeability Advantage">
#         This is <span style={{color:C.accent, fontWeight:700}}>LoRA's defining property</span> and why it dominates over Adapters. Bottleneck Adapters stay in the forward path forever, adding latency. LoRA's W_merged = W\u2080 + (\u03B1/r)BA is a <span style={{color:C.yellow, fontWeight:700}}>one-time matrix addition</span>. After that, A and B are discarded. The merged model is a <span style={{color:C.green, fontWeight:700}}>standard model</span> - same architecture, same size, zero PEFT overhead. No PEFT library needed at inference. Indistinguishable from full fine-tuning.
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
#   var tabs = ["Big Picture", "Forward Pass", "Memory & Cost", "Data Pipeline", "Rank & Alpha", "Merge & Deploy"];
#   return (
#     <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
#       <div style={{ textAlign: "center", marginBottom: 16 }}>
#         <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.cyan + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>PEFT / LoRA</div>
#         <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " Low-Rank Adaptation from theory to deployment"}</div>
#       </div>
#       <TabBar tabs={tabs} active={tab} onChange={setTab} />
#       {tab === 0 && <TabBigPicture />}
#       {tab === 1 && <TabForwardPass />}
#       {tab === 2 && <TabMemory />}
#       {tab === 3 && <TabDataPipeline />}
#       {tab === 4 && <TabRankAlpha />}
#       {tab === 5 && <TabMergeDeploy />}
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
# LORA_VISUAL_HEIGHT = 1600