"""
Self-contained HTML for the PEFT IA3 interactive walkthrough.
Covers: Big Picture, IA3 Mechanics, Memory & Cost, Configuration,
IA3 vs LoRA, and When to Use IA3.
Embed in Streamlit via st.components.v1.html(IA3_VISUAL_HTML, height=IA3_VISUAL_HEIGHT).
"""

IA3_VISUAL_HTML = """
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
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #4ecdc4; }
  @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
  @keyframes scaleIn { 0%{transform:scaleX(0)} 100%{transform:scaleX(1)} }
  @keyframes flow { 0%{stroke-dashoffset:20} 100%{stroke-dashoffset:0} }
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
  accent: "#4ecdc4", blue: "#38bdf8", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#4ecdc4", pink: "#f472b6", orange: "#fb923c",
  teal: "#14b8a6",
};

var MUL = "\\u00D7";
var ARR = "\\u2192";
var DASH = "\\u2014";
var CHK = "\\u2713";
var WARN = "\\u26A0";
var LQ = "\\u201C";
var RQ = "\\u201D";
var PLAY = "\\u25B6";
var PAUSE = "\\u23F8";
var BULB = "\\uD83D\\uDCA1";
var TARG = "\\uD83C\\uDFAF";
var LOCK = "\\uD83D\\uDD12";
var BRAIN = "\\uD83E\\uDDE0";
var FIRE = "\\uD83D\\uDD25";
var GEAR = "\\u2699";
var CHART = "\\uD83D\\uDCC8";
var ZAP = "\\u26A1";
var MICRO = "\\uD83D\\uDD2C";
var SCALE = "\\u2696\\uFE0F";
var DNA = "\\uD83E\\uDDEC";
var STAR = "\\u2605";
var DARR = "\\u2193";
var UARR = "\\u2191";

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
      padding: "16px 22px", background: "rgba(78,205,196,0.06)",
      borderRadius: 10, border: "1px solid rgba(78,205,196,0.2)",
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
  useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);

  var vectors = [
    { name: "l_k (Key scaler)", color: C.accent, dim: "d_k", where: "Attention Keys", effect: "Rescales key activations" },
    { name: "l_v (Value scaler)", color: C.blue, dim: "d_v", where: "Attention Values", effect: "Rescales value activations" },
    { name: "l_ff (FFN scaler)", color: C.purple, dim: "d_ff", where: "FFN output", effect: "Rescales feed-forward activations" },
  ];

  return (
    <div>
      <SectionTitle title="The Big Picture" subtitle={"IA\u00B3: Infused Adapter by Inhibiting & Amplifying Inner Activations"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={1050} height={300} viewBox="0 0 800 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Base model frozen */}
          <text x={110} y={18} textAnchor="middle" fill={C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">BASE MODEL</text>
          <text x={110} y={32} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LOCK + " ALL frozen"}</text>
          {[0,1,2,3,4,5,6].map(function(i) {
            return <rect key={i} x={40} y={42+i*32} width={140} height={26} rx={4} fill={C.dim+"18"} stroke={C.dim+"40"} strokeWidth={1} />;
          })}
          <text x={110} y={276} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"7B params"}</text>

          {/* Element-wise multiply icons */}
          <text x={245} y={70} textAnchor="middle" fill={C.accent} fontSize={22} fontWeight={800}>{MUL}</text>
          <text x={245} y={150} textAnchor="middle" fill={C.blue} fontSize={22} fontWeight={800}>{MUL}</text>
          <text x={245} y={230} textAnchor="middle" fill={C.purple} fontSize={22} fontWeight={800}>{MUL}</text>

          {/* IA3 learned vectors */}
          <text x={390} y={18} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">IA\u00B3 LEARNED VECTORS</text>
          <text x={390} y={32} textAnchor="middle" fill={C.accent+"80"} fontSize={8} fontFamily="monospace">{"(3 vectors per layer \u2014 EXTREMELY tiny!)"}</text>

          {[
            { y: 48, color: C.accent, label: "l_k", sub: "Key scaler", dim: "d_k = 128" },
            { y: 128, color: C.blue, label: "l_v", sub: "Value scaler", dim: "d_v = 128" },
            { y: 208, color: C.purple, label: "l_ff", sub: "FFN scaler", dim: "d_ff = 16384" },
          ].map(function(v, i) {
            var w = animated ? 160 : 0;
            return (<g key={i}>
              <rect x={290} y={v.y} width={200} height={58} rx={8}
                fill={v.color+"10"} stroke={v.color+"50"} strokeWidth={1.5}
                style={{ filter: "drop-shadow(0 0 8px " + v.color + "25)" }} />
              <rect x={290} y={v.y} width={w} height={58} rx={8}
                fill={v.color+"15"} stroke="none"
                style={{ transition: "width 1s ease-out", transitionDelay: (i*0.2)+"s" }} />
              <text x={390} y={v.y + 28} textAnchor="middle" fill={v.color} fontSize={14} fontWeight={800} fontFamily="monospace">{v.label}</text>
              <text x={390} y={v.y + 44} textAnchor="middle" fill={v.color+"80"} fontSize={8} fontFamily="monospace">{v.sub + " | " + v.dim}</text>
            </g>);
          })}

          {/* Arrow to result */}
          <line x1={500} y1={148} x2={565} y2={148} stroke={C.green} strokeWidth={2} />
          <polygon points="570,148 563,143 563,153" fill={C.green} />
          <text x={533} y={136} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">element-wise</text>
          <text x={533} y={162} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{MUL + " rescale"}</text>

          {/* Result */}
          <text x={680} y={18} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">ADAPTED OUTPUT</text>
          {[0,1,2,3,4,5,6].map(function(i) {
            return (<rect key={i} x={595} y={42+i*32} width={140} height={26} rx={4}
              fill={C.green+(animated?"12":"06")} stroke={C.green+(animated?"40":"20")} strokeWidth={1}
              style={{ transition: "all 0.8s", transitionDelay: (i*0.06)+"s" }} />);
          })}
          <text x={680} y={268} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">{"learned rescaling"}</text>
          <text x={680} y={282} textAnchor="middle" fill={C.green+"80"} fontSize={8} fontFamily="monospace">{"~0.01% extra params"}</text>
        </svg>
      </div>

      {/* Vector breakdown */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"What IA\u00B3 Actually Learns " + DASH + " Three Learned Rescaling Vectors per Layer"}</div>
        {vectors.map(function(v, i) {
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10, padding: "8px 12px", borderRadius: 8, background: v.color + "06", border: "1px solid " + v.color + "15" }}>
              <div style={{ width: 90, fontSize: 11, fontWeight: 800, color: v.color, fontFamily: "monospace" }}>{v.name}</div>
              <div style={{ width: 80, fontSize: 9, color: C.dim, fontFamily: "monospace" }}>{v.dim}</div>
              <div style={{ width: 130, fontSize: 9, color: C.muted }}>{v.where}</div>
              <div style={{ flex: 1, fontSize: 9, color: C.muted }}>{v.effect}</div>
              <div style={{ padding: "3px 8px", borderRadius: 4, background: v.color + "20", fontSize: 8, color: v.color, fontFamily: "monospace" }}>{"x " + MUL + " " + v.name.split(" ")[0]}</div>
            </div>
          );
        })}
        <div style={{ marginTop: 8, fontSize: 9, color: C.muted, textAlign: "center" }}>
          {"IA\u00B3 learns to softly gate which dimensions to inhibit (" + DARR + ") and which to amplify (" + UARR + ")"}
        </div>
      </Card>

      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"Why IA\u00B3?"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 12, flexWrap: "wrap" }}>
          {[
            { icon: MICRO, title: "Ultra-minimal params", desc: "~10k params vs millions for LoRA", c: C.accent },
            { icon: ZAP, title: "Fast to train", desc: "Fewer params = fast convergence", c: C.blue },
            { icon: SCALE, title: "Soft gating", desc: "Inhibit or amplify any dimension", c: C.purple },
            { icon: CHART, title: "Few-shot friendly", desc: "Designed for few-shot fine-tuning", c: C.green },
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
        IA\u00B3 takes a uniquely minimalist philosophy: instead of injecting new matrices, it learns <span style={{color:C.accent,fontWeight:700}}>three learned vectors per transformer layer</span> that element-wise rescale key, value, and FFN activations. This is <span style={{color:C.blue,fontWeight:700}}>inhibiting</span> (scaling near 0) or <span style={{color:C.green,fontWeight:700}}>amplifying</span> (scaling above 1) individual dimensions {DASH} like a learned per-dimension gate on information flow.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: IA3 MECHANICS
   =============================================================== */
function TabMechanics() {
  var _s = useState(0); var site = _s[0], setSite = _s[1];
  var _a = useState(false); var anim = _a[0], setAnim = _a[1];
  var _lv = useState(Array(16).fill(0).map(function(_, i) { return 0.5 + Math.sin(i * 0.8) * 0.4; }));
  var lVals = _lv[0], setLVals = _lv[1];

  useEffect(function() { var t = setTimeout(function() { setAnim(true); }, 300); return function() { clearTimeout(t); }; }, []);

  var sites = [
    { name: "Attention Keys (K)", color: C.accent, formula: "K' = l_k \u2299 K", detail: "l_k \u2208 \u211D^{d_k} rescales each key dimension before attention score computation", effect: "Controls WHAT each head attends to by amplifying/inhibiting key dimensions" },
    { name: "Attention Values (V)", color: C.blue, formula: "V' = l_v \u2299 V", detail: "l_v \u2208 \u211D^{d_v} rescales each value dimension after attention aggregation", effect: "Controls WHAT information flows forward from attended positions" },
    { name: "FFN Intermediate", color: C.purple, formula: "ff' = l_ff \u2299 ff(x)", detail: "l_ff \u2208 \u211D^{d_ff} rescales each neuron output in the feed-forward network", effect: "Controls WHICH features the FFN activates after nonlinearity" },
  ];

  var s = sites[site];

  return (
    <div>
      <SectionTitle title="IA\u00B3 Mechanics" subtitle={"Where and how the three learned vectors rescale transformer internals"} />

      {/* Attention diagram */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={280} viewBox="0 0 800 280" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Input */}
          <rect x={20} y={115} width={70} height={40} rx={5} fill={C.dim+"20"} stroke={C.dim+"40"} strokeWidth={1} />
          <text x={55} y={140} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">Input x</text>
          <line x1={90} y1={135} x2={130} y2={135} stroke={C.dim} strokeWidth={1.5} />
          <polygon points="132,135 126,131 126,139" fill={C.dim} />

          {/* Q, K, V projections */}
          {[
            { label: "W_Q", y: 60, color: C.dim, lbl: "Q" },
            { label: "W_K", y: 130, color: site===0 ? C.accent : C.dim, lbl: "K" },
            { label: "W_V", y: 200, color: site===1 ? C.blue : C.dim, lbl: "V" },
          ].map(function(proj, i) {
            return (<g key={i}>
              <rect x={132} y={proj.y} width={70} height={32} rx={5} fill={proj.color+"15"} stroke={proj.color+"40"} strokeWidth={1} />
              <text x={167} y={proj.y+20} textAnchor="middle" fill={proj.color} fontSize={9} fontWeight={700} fontFamily="monospace">{proj.label}</text>

              {/* IA3 multiply on K and V */}
              {(i === 1 && site === 0) && (<g>
                <rect x={215} y={proj.y} width={50} height={32} rx={4} fill={C.accent+"30"} stroke={C.accent} strokeWidth={1.5} />
                <text x={240} y={proj.y+20} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">{MUL + " l_k"}</text>
              </g>)}
              {(i === 2 && site === 1) && (<g>
                <rect x={215} y={proj.y} width={50} height={32} rx={4} fill={C.blue+"30"} stroke={C.blue} strokeWidth={1.5} />
                <text x={240} y={proj.y+20} textAnchor="middle" fill={C.blue} fontSize={9} fontWeight={700} fontFamily="monospace">{MUL + " l_v"}</text>
              </g>)}
            </g>);
          })}

          {/* Attention */}
          <rect x={285} y={50} width={100} height={170} rx={8} fill={C.cyan+"08"} stroke={C.cyan+"30"} strokeWidth={1.5} />
          <text x={335} y={130} textAnchor="middle" fill={C.cyan} fontSize={9} fontWeight={700} fontFamily="monospace">Softmax</text>
          <text x={335} y={144} textAnchor="middle" fill={C.cyan+"80"} fontSize={8} fontFamily="monospace">(QK^T)</text>
          <text x={335} y={200} textAnchor="middle" fill={C.cyan+"60"} fontSize={7} fontFamily="monospace">{LOCK + " frozen"}</text>

          {/* FFN block */}
          <rect x={500} y={50} width={110} height={170} rx={8} fill={site===2 ? C.purple+"10" : C.dim+"08"} stroke={site===2 ? C.purple+"40" : C.dim+"30"} strokeWidth={1.5} />
          <text x={555} y={110} textAnchor="middle" fill={site===2 ? C.purple : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace">FFN</text>
          <text x={555} y={126} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">W1 \u2192 ReLU</text>
          <text x={555} y={140} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"\u2192 W2"}</text>
          {site===2 && (<g>
            <rect x={510} y={150} width={90} height={30} rx={4} fill={C.purple+"30"} stroke={C.purple} strokeWidth={1.5} />
            <text x={555} y={169} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">{MUL + " l_ff"}</text>
          </g>)}
          <text x={555} y={200} textAnchor="middle" fill={C.dim+"60"} fontSize={7} fontFamily="monospace">{LOCK + " frozen"}</text>

          {/* Output */}
          <rect x={660} y={115} width={80} height={40} rx={5} fill={C.green+"12"} stroke={C.green+"40"} strokeWidth={1} />
          <text x={700} y={140} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">Output</text>

          {/* Connecting lines */}
          <line x1={385} y1={135} x2={498} y2={135} stroke={C.dim} strokeWidth={1.5} />
          <polygon points="500,135 494,131 494,139" fill={C.dim} />
          <line x1={610} y1={135} x2={658} y2={135} stroke={C.dim} strokeWidth={1.5} />
          <polygon points="660,135 654,131 654,139" fill={C.dim} />

          {/* Site labels */}
          {site === 0 && <text x={240} y={108} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">{UARR + DARR + " keys rescaled"}</text>}
          {site === 1 && <text x={240} y={185} textAnchor="middle" fill={C.blue} fontSize={9} fontWeight={700} fontFamily="monospace">{UARR + DARR + " values rescaled"}</text>}
          {site === 2 && <text x={555} y={195} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">{UARR + DARR}</text>}

          {/* Formula */}
          <rect x={100} y={240} width={600} height={26} rx={6} fill={s.color+"08"} stroke={s.color+"20"} strokeWidth={1} />
          <text x={400} y={257} textAnchor="middle" fill={s.color} fontSize={10} fontFamily="monospace" fontWeight={700}>{s.formula}</text>
        </svg>
      </div>

      {/* Site selector */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16 }}>
        {sites.map(function(si, i) {
          var on = site === i;
          return (<button key={i} onClick={function() { setSite(i); }} style={{
            padding: "8px 16px", borderRadius: 8, border: "1.5px solid " + (on ? si.color : C.border),
            background: on ? si.color + "20" : C.card, color: on ? si.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{si.name}</button>);
        })}
      </div>

      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: s.color }}>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          <div style={{ flex: 2, minWidth: 260 }}>
            <div style={{ fontSize: 16, fontWeight: 800, color: s.color, marginBottom: 4 }}>{s.name}</div>
            <div style={{ fontFamily: "monospace", fontSize: 12, color: s.color, marginBottom: 10, padding: "6px 12px", background: s.color+"10", borderRadius: 6, border: "1px solid " + s.color+"30" }}>{s.formula}</div>
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 8 }}>{s.detail}</div>
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, borderLeft: "3px solid " + s.color + "40", paddingLeft: 10, fontStyle: "italic" }}>{s.effect}</div>
          </div>

          {/* Vector visualizer */}
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 8 }}>{"Example learned vector (16 dims shown):"}</div>
            <div style={{ display: "flex", gap: 3, alignItems: "flex-end", height: 80 }}>
              {lVals.map(function(v, i) {
                return (<div key={i} style={{
                  flex: 1, borderRadius: 2,
                  height: (v * 70) + "%",
                  background: v < 0.5 ? C.red + "80" : v > 0.7 ? C.green + "80" : s.color + "60",
                  border: "1px solid " + (v < 0.5 ? C.red + "40" : v > 0.7 ? C.green + "40" : s.color + "30"),
                  transition: "height 0.3s"
                }} />)
              })}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, color: C.dim, marginTop: 4 }}>
              <span style={{color: C.red}}>inhibit (&lt;1)</span>
              <span style={{color: C.green}}>amplify (&gt;1)</span>
            </div>
            <div style={{ marginTop: 8, fontSize: 8, color: C.dim }}>{"Initialized to all-ones (identity). Learns to deviate."}</div>
          </div>
        </div>
      </Card>

      {/* Math detail */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Full IA\u00B3 Attention Forward Pass"}</div>
        <div style={{ background: "#06060a", borderRadius: 8, padding: "14px 18px", fontFamily: "monospace", fontSize: 10, lineHeight: 2, color: C.muted, border: "1px solid " + C.border }}>
          <div><span style={{color: C.dim}}>{"# Standard attention:"}</span></div>
          <div><span style={{color: C.text}}>{"Attn(Q, K, V) = softmax(QK\u1D40 / \u221Bd_k) \u00B7 V"}</span></div>
          <br />
          <div><span style={{color: C.dim}}>{"# IA\u00B3 attention (l_k, l_v are learned vectors):"}</span></div>
          <div><span style={{color: C.accent}}>{"Attn(Q, l_k\u2299K, l_v\u2299V) = softmax(Q(l_k\u2299K)\u1D40 / \u221Bd_k) \u00B7 (l_v\u2299V)"}</span></div>
          <br />
          <div><span style={{color: C.dim}}>{"# IA\u00B3 FFN:"}</span></div>
          <div><span style={{color: C.purple}}>{"ff'(x) = l_ff \u2299 \u03B3(W_1 x + b_1)"}</span><span style={{color: C.dim}}>{" # \u03B3 = nonlinearity"}</span></div>
        </div>
      </Card>

      <Insight icon={SCALE} title="Inhibiting vs Amplifying">
        IA\u00B3 vectors are initialized to <span style={{color:C.green,fontWeight:700}}>all ones (identity)</span>, so training starts from the pretrained baseline. During training, individual dimensions shift away from 1: values <span style={{color:C.red,fontWeight:700}}>below 1 inhibit</span> (suppress) that dimension, values <span style={{color:C.green,fontWeight:700}}>above 1 amplify</span> it. The model learns which internal features to gate for the target task {DASH} with minimal parameters.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: MEMORY & COST
   =============================================================== */
function TabMemory() {
  var _m = useState(7); var modelB = _m[0], setModelB = _m[1];
  var _L = useState(32); var layers = _L[0], setLayers = _L[1];

  var dModel = 4096;
  var dK = 128; var dFF = 16384;

  var ia3Params = layers * (dK + dK + dFF);
  var loraParams = layers * 4 * dModel * 8 * 2;  // q,k,v,o with rank 8
  var fftParams = modelB * 1e9;

  var ia3GB = modelB * 2 + ia3Params * 4 * 4 / 1e9;  // frozen bf16 + tiny fp32 trainable
  var loraGB = modelB * 2 + loraParams * 4 * 4 / 1e9;
  var fftGB = modelB * 4 * 4;

  var loraH = Math.min(220, loraGB / fftGB * 220);
  var ia3H = Math.min(220, ia3GB / fftGB * 220);

  return (
    <div>
      <SectionTitle title="Memory & Cost" subtitle={"IA\u00B3 is the most parameter-efficient PEFT: 10x fewer params than LoRA"} />

      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"Model: " + modelB + "B"}</div>
            <input type="range" min={1} max={70} value={modelB} onChange={function(e) { setModelB(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.accent }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim }}><span>1B</span><span>7B</span><span>13B</span><span>70B</span></div>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>{"Layers: " + layers}</div>
            <input type="range" min={12} max={80} value={layers} onChange={function(e) { setLayers(parseInt(e.target.value)); }} style={{ width: "100%", accentColor: C.purple }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim }}><span>12</span><span>32</span><span>48</span><span>80</span></div>
          </div>
        </div>
      </Card>

      {/* Param bar chart */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={300} viewBox="0 0 800 300" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Grid */}
          {[0,25,50,75,100].map(function(pct) {
            var y = 255 - pct * 2.2;
            return (<g key={pct}>
              <line x1={60} y1={y} x2={740} y2={y} stroke={C.dim + "25"} strokeWidth={0.5} />
              <text x={52} y={y+3} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{pct + "%"}</text>
            </g>);
          })}

          {/* FFT bar */}
          <rect x={100} y={255 - 220} width={100} height={220} fill={C.red+"40"} stroke={C.red+"60"} strokeWidth={1} />
          <text x={150} y={255 - 220 - 8} textAnchor="middle" fill={C.red} fontSize={9} fontFamily="monospace">{(fftGB).toFixed(0) + " GB"}</text>
          <text x={150} y={265} textAnchor="middle" fill={C.red} fontSize={9} fontFamily="monospace">Full FT</text>
          <text x={150} y={278} textAnchor="middle" fill={C.red} fontSize={8} fontFamily="monospace">{(fftParams/1e9).toFixed(0) + "B params"}</text>

          {/* LoRA bar */}
          <rect x={320} y={255-loraH} width={100} height={loraH} fill={C.purple+"40"} stroke={C.purple+"60"} strokeWidth={1} />
          <text x={370} y={255-loraH-8} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">{loraGB.toFixed(1) + " GB"}</text>
          <text x={370} y={265} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">LoRA (r=8)</text>
          <text x={370} y={278} textAnchor="middle" fill={C.purple} fontSize={8} fontFamily="monospace">{(loraParams/1e6).toFixed(0) + "M params"}</text>

          {/* IA3 bar */}
          <rect x={540} y={255-ia3H} width={100} height={Math.max(ia3H, 2)} fill={C.accent+"60"} stroke={C.accent+"80"} strokeWidth={1.5} />
          <text x={590} y={Math.max(255-ia3H-8, 20)} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">{ia3GB.toFixed(1) + " GB"}</text>
          <text x={590} y={265} textAnchor="middle" fill={C.accent} fontSize={9} fontFamily="monospace">IA\u00B3</text>
          <text x={590} y={278} textAnchor="middle" fill={C.accent} fontSize={8} fontFamily="monospace">{(ia3Params/1e3).toFixed(0) + "K params"}</text>

          {/* Annotations */}
          <line x1={420} y1={80} x2={540} y2={80} stroke={C.green+"60"} strokeWidth={1} strokeDasharray="4,3" />
          <text x={480} y={72} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">{((1 - ia3Params/loraParams)*100).toFixed(1) + "% fewer"}</text>
          <text x={480} y={88} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">than LoRA</text>
        </svg>
      </div>

      {/* Stats */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ display: "flex", justifyContent: "center", gap: 30, flexWrap: "wrap" }}>
          <StatBox label="IA\u00B3 PARAMS" value={(ia3Params/1e3).toFixed(0) + "K"} color={C.accent} sub="learned vectors" />
          <StatBox label="LoRA PARAMS" value={(loraParams/1e6).toFixed(0) + "M"} color={C.purple} sub="r=8, q/k/v/o" />
          <StatBox label="FULL FT PARAMS" value={modelB + "B"} color={C.red} sub="all weights" />
          <StatBox label="IA\u00B3 vs LoRA" value={(loraParams/ia3Params).toFixed(0) + "x"} color={C.green} sub="LoRA is bigger" />
          <StatBox label="IA\u00B3 MEMORY" value={ia3GB.toFixed(1) + " GB"} color={C.accent} sub={modelB + "B model"} />
        </div>
      </Card>

      {/* Why so small? */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Why IA\u00B3 Is So Small"}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { label: "LoRA per layer", formula: "2 \u00D7 r \u00D7 d_model \u00D7 n_targets", example: "2 \u00D7 8 \u00D7 4096 \u00D7 4 = 262K", color: C.purple },
            { label: "IA\u00B3 per layer", formula: "d_k + d_v + d_ff", example: "128 + 128 + 16384 = 16.6K", color: C.accent },
            { label: "Ratio", formula: "LoRA / IA\u00B3 \u2248 15\u00D720\u00D7", example: "per layer, 15-20x more for LoRA", color: C.green },
          ].map(function(row, i) {
            return (<div key={i} style={{ flex: 1, minWidth: 200, padding: "10px 14px", borderRadius: 8, background: row.color+"06", border: "1px solid " + row.color+"20" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: row.color, marginBottom: 4 }}>{row.label}</div>
              <div style={{ fontSize: 9, color: C.muted, fontFamily: "monospace", marginBottom: 4 }}>{row.formula}</div>
              <div style={{ fontSize: 9, color: C.dim }}>{row.example}</div>
            </div>);
          })}
        </div>
      </Card>

      <Insight icon={ZAP} title="When Does Tiny Matter?">
        IA\u00B3's extreme efficiency shines when you need <span style={{color:C.accent,fontWeight:700}}>many tasks on one base model</span>, each with their own adapter. 1000 LoRA adapters (r=8) for a 7B model = ~260GB. 1000 IA\u00B3 adapters = ~16GB. For large-scale multi-tenant serving, IA\u00B3 can be <span style={{color:C.green,fontWeight:700}}>16x more storage-efficient</span>. It also trains faster due to fewer parameters to optimize.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: CONFIGURATION
   =============================================================== */
function TabConfig() {
  var _k = useState(true); var useK = _k[0], setUseK = _k[1];
  var _v = useState(true); var useV = _v[0], setUseV = _v[1];
  var _ff = useState(true); var useFF = _ff[0], setUseFF = _ff[1];
  var _fr = useState(false); var feedfwdRescale = _fr[0], setFeedfwdRescale = _fr[1];

  var dK = 128; var dFF = 16384; var L = 32;
  var params = L * ((useK ? dK : 0) + (useV ? dK : 0) + (useFF ? dFF : 0));

  return (
    <div>
      <SectionTitle title="Configuration" subtitle={"Tune which activations IA\u00B3 rescales \u2014 and understand initialization"} />

      <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
        {/* Toggle panel */}
        <Card style={{ flex: 1, minWidth: 300 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"IA\u00B3 Targets (32-layer 7B model)"}</div>

          {[
            { label: "l_k (Key rescaling)", on: useK, set: setUseK, color: C.accent, params: L * dK, desc: "Rescale attention key vectors" },
            { label: "l_v (Value rescaling)", on: useV, set: setUseV, color: C.blue, params: L * dK, desc: "Rescale attention value vectors" },
            { label: "l_ff (FFN rescaling)", on: useFF, set: setUseFF, color: C.purple, params: L * dFF, desc: "Rescale FFN intermediate activations" },
          ].map(function(tog, i) {
            return (<div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 12px", marginBottom: 8, borderRadius: 8, background: tog.on ? tog.color+"08" : "transparent", border: "1px solid " + (tog.on ? tog.color+"25" : C.border) }}>
              <div onClick={function() { tog.set(!tog.on); }} style={{
                width: 36, height: 20, borderRadius: 10, cursor: "pointer",
                background: tog.on ? tog.color : C.dim + "40",
                position: "relative", transition: "background 0.3s"
              }}>
                <div style={{ position: "absolute", top: 2, left: tog.on ? 18 : 2, width: 16, height: 16, borderRadius: "50%", background: "#fff", transition: "left 0.3s" }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: tog.on ? tog.color : C.dim, fontFamily: "monospace" }}>{tog.label}</div>
                <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{tog.desc}</div>
              </div>
              <div style={{ fontSize: 9, color: tog.on ? tog.color : C.dim, fontFamily: "monospace" }}>{(tog.params/1e3).toFixed(1) + "K"}</div>
            </div>);
          })}

          <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: 8, background: C.yellow+"08", border: "1px solid " + C.yellow+"20" }}>
            <div style={{ fontSize: 9, color: C.yellow, fontWeight: 700, marginBottom: 4 }}>{"Total trainable params:"}</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: C.accent, fontFamily: "monospace" }}>{(params/1e3).toFixed(1) + "K"}</div>
            <div style={{ fontSize: 8, color: C.dim }}>{"of 7B total (0." + (params/7e9*1000).toFixed(4) + "%)"}</div>
          </div>
        </Card>

        {/* Initialization card */}
        <Card style={{ flex: 1, minWidth: 300 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"Initialization Strategy"}</div>
          <div style={{ padding: "12px 14px", borderRadius: 8, background: C.green+"06", border: "1px solid " + C.green+"20", marginBottom: 12 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.green, marginBottom: 6 }}>{"All vectors initialized to " + CHK + " ones"}</div>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: C.green, marginBottom: 6 }}>{"l_k = l_v = l_ff = [1, 1, ..., 1]"}</div>
            <div style={{ fontSize: 9, color: C.muted, lineHeight: 1.6 }}>{"This means at step 0, IA\u00B3 is a perfect identity: the adapted model = the base model. Gradients then push each dimension up or down from 1."}</div>
          </div>

          <div style={{ fontSize: 10, fontWeight: 700, color: C.text, marginBottom: 8 }}>{"Typical learned value distribution:"}</div>
          <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
            {[
              { range: "0.0–0.5", pct: 8, desc: "Strong inhibit", color: C.red },
              { range: "0.5–0.8", pct: 22, desc: "Mild inhibit", color: C.orange },
              { range: "0.8–1.2", pct: 40, desc: "Near identity", color: C.dim },
              { range: "1.2–1.5", pct: 20, desc: "Mild amplify", color: C.blue },
              { range: ">1.5", pct: 10, desc: "Strong amplify", color: C.green },
            ].map(function(b, i) {
              return (<div key={i} style={{ flex: 1, textAlign: "center" }}>
                <div style={{ height: b.pct * 3, background: b.color + "60", border: "1px solid " + b.color + "50", borderRadius: "2px 2px 0 0", marginBottom: 4 }} />
                <div style={{ fontSize: 7, color: b.color }}>{b.range}</div>
              </div>);
            })}
          </div>

          <div style={{ fontSize: 10, fontWeight: 700, color: C.text, marginBottom: 8 }}>{"HuggingFace PEFT Config:"}</div>
          <div style={{ background: "#06060a", borderRadius: 8, padding: "12px 16px", fontFamily: "monospace", fontSize: 9, lineHeight: 1.8, color: C.muted, border: "1px solid " + C.border }}>
            <div><span style={{color: C.purple}}>{"from"}</span><span>{" peft "}</span><span style={{color: C.purple}}>{"import"}</span><span>{" IA3Config"}</span></div>
            <br />
            <div><span style={{color: C.cyan}}>{"config"}</span><span>{" = IA3Config("}</span></div>
            <div><span>{" target_modules=["}</span></div>
            {useK && <div><span style={{color: C.accent}}>{' "k_proj",'}</span></div>}
            {useV && <div><span style={{color: C.blue}}>{' "v_proj",'}</span></div>}
            {useFF && <div><span style={{color: C.purple}}>{' "fc2",'}</span></div>}
            <div><span>{"  ],"}</span></div>
            <div><span>{'  feedforward_modules=["fc2"],'}</span></div>
            <div><span>{")"}</span></div>
          </div>
        </Card>
      </div>

      {/* Best practices */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Configuration Best Practices"}</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[
            { title: "Use all 3 by default", desc: "l_k + l_v + l_ff together gives best results. Ablations show each contributes.", color: C.green, icon: STAR },
            { title: "Higher LR than LoRA", desc: "IA\u00B3 benefits from ~3x higher LR (e.g. 3e-3). Fewer params need bigger steps.", color: C.yellow, icon: ZAP },
            { title: "Few-shot friendly", desc: "Designed for T-Few (few-shot fine-tuning). Works well with 8–100 examples.", color: C.accent, icon: MICRO },
            { title: "Needs prompt template", desc: "Best combined with task-specific prompt templates, unlike LoRA.", color: C.blue, icon: BRAIN },
          ].map(function(tip, i) {
            return (<div key={i} style={{ flex: 1, minWidth: 180, padding: "10px 14px", borderRadius: 8, background: tip.color+"06", border: "1px solid " + tip.color+"20" }}>
              <div style={{ fontSize: 13, marginBottom: 4 }}>{tip.icon}</div>
              <div style={{ fontSize: 10, fontWeight: 700, color: tip.color, marginBottom: 4 }}>{tip.title}</div>
              <div style={{ fontSize: 9, color: C.muted, lineHeight: 1.6 }}>{tip.desc}</div>
            </div>);
          })}
        </div>
      </Card>

      <Insight icon={GEAR} title="Configuration Rule of Thumb">
        Start with <span style={{color:C.accent,fontWeight:700}}>all three targets</span> (k, v, ff). Use a <span style={{color:C.yellow}}>learning rate of 3e-3</span> {DASH} IA\u00B3 needs higher LR than LoRA because there are far fewer parameters for the optimizer to work with. <span style={{color:C.blue}}>Combine with good prompt templates</span> for best results. If compute is extremely limited, removing <span style={{color:C.purple}}>l_ff</span> (which has the most parameters) is the safest cut.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: IA3 vs LoRA
   =============================================================== */
function TabVsLora() {
  var _s = useState(0); var scenario = _s[0], setScenario = _s[1];

  var scenarios = [
    {
      name: "Few-shot (8–100 examples)", color: C.accent,
      ia3: 90, lora: 82, fft: 78,
      note: "IA\u00B3 wins here: fewer params = less overfitting on tiny datasets"
    },
    {
      name: "Medium dataset (1K–10K)", color: C.blue,
      ia3: 84, lora: 91, fft: 88,
      note: "LoRA starts to pull ahead: more capacity helps with more data"
    },
    {
      name: "Large dataset (100K+)", color: C.purple,
      ia3: 79, lora: 93, fft: 97,
      note: "LoRA and FFT clearly better: IA\u00B3 underfits with so much data"
    },
    {
      name: "Multi-tenant (1000 tasks)", color: C.green,
      ia3: 88, lora: 85, fft: 60,
      note: "IA\u00B3 wins on practical serving: 1000 adapters = 16GB vs 260GB"
    },
  ];

  var sc = scenarios[scenario];

  var dimensions = [
    { label: "Trainable Params", ia3: 95, lora: 60, fft: 0, note: "IA\u00B3 ~10K, LoRA ~4M, FFT ~7B" },
    { label: "Training Speed", ia3: 90, lora: 70, fft: 30, note: "Fewer params = faster per step" },
    { label: "Memory (GB)", ia3: 90, lora: 70, fft: 10, note: "Lower is better (shown as savings %)" },
    { label: "Few-shot quality", ia3: 85, lora: 72, fft: 65, note: "IA\u00B3 resists overfitting on tiny data" },
    { label: "High-data quality", ia3: 55, lora: 90, fft: 98, note: "LoRA and FFT have more capacity" },
    { label: "Multi-task serving", ia3: 95, lora: 75, fft: 20, note: "Adapter size dominates" },
  ];

  return (
    <div>
      <SectionTitle title="IA\u00B3 vs LoRA" subtitle={"Head-to-head across scenarios \u2014 both have a domain where they win"} />

      {/* Scenario selector */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {scenarios.map(function(sc, i) {
          var on = scenario === i;
          return (<button key={i} onClick={function() { setScenario(i); }} style={{
            padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? sc.color : C.border),
            background: on ? sc.color+"20" : C.card, color: on ? sc.color : C.muted,
            cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
          }}>{sc.name}</button>);
        })}
      </div>

      {/* Scenario bars */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={200} viewBox="0 0 800 200" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {[0,25,50,75,100].map(function(v) {
            var y = 170 - v * 1.6;
            return (<g key={v}>
              <line x1={60} y1={y} x2={740} y2={y} stroke={C.dim+"25"} strokeWidth={0.5} />
              <text x={52} y={y+3} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{v}</text>
            </g>);
          })}

          {[
            { label: "IA\u00B3", val: sc.ia3, x: 200, color: C.accent },
            { label: "LoRA", val: sc.lora, x: 380, color: C.purple },
            { label: "Full FT", val: sc.fft, x: 560, color: C.red },
          ].map(function(bar) {
            var barH = bar.val * 1.6;
            return (<g key={bar.label}>
              <rect x={bar.x-50} y={170-barH} width={100} height={barH}
                fill={bar.color+"40"} stroke={bar.color+"80"} strokeWidth={1.5}
                style={{ transition: "all 0.4s" }} />
              <text x={bar.x} y={170-barH-8} textAnchor="middle" fill={bar.color} fontSize={14} fontWeight={800} fontFamily="monospace">{bar.val}</text>
              <text x={bar.x} y={185} textAnchor="middle" fill={bar.color} fontSize={9} fontFamily="monospace">{bar.label}</text>
            </g>);
          })}

          <text x={400} y={200} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{sc.note}</text>
        </svg>
      </div>

      {/* Radar / comparison bars */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"Overall Comparison Across Dimensions"}</div>
        {dimensions.map(function(dim, i) {
          return (<div key={i} style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.muted, marginBottom: 4 }}>
              <span style={{fontWeight:700}}>{dim.label}</span>
              <span style={{color:C.dim,fontSize:8}}>{dim.note}</span>
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {[
                { label: "IA\u00B3", val: dim.ia3, color: C.accent },
                { label: "LoRA", val: dim.lora, color: C.purple },
                { label: "FFT", val: dim.fft, color: C.red },
              ].map(function(bar, j) {
                return (<div key={j} style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                    <span style={{ fontSize: 7, color: bar.color, width: 30 }}>{bar.label}</span>
                    <div style={{ flex: 1, height: 10, background: C.border, borderRadius: 2 }}>
                      <div style={{ width: bar.val + "%", height: "100%", borderRadius: 2, background: bar.color + "60", border: "1px solid " + bar.color + "40", transition: "width 0.4s" }} />
                    </div>
                    <span style={{ fontSize: 7, color: bar.color, width: 22 }}>{bar.val}</span>
                  </div>
                </div>);
              })}
            </div>
          </div>);
        })}
      </Card>

      {/* Decision table */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{"Quick Decision Guide"}</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 250, padding: "10px 14px", borderRadius: 8, background: C.accent+"06", border: "1px solid " + C.accent+"20" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 6 }}>{STAR + " Choose IA\u00B3 when:"}</div>
            {["Dataset < 1000 examples", "Serving 100s of tasks concurrently", "Minimal storage budget", "Fast iteration / experimentation", "Few-shot or zero-shot setting"].map(function(t, i) {
              return <div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{CHK + " " + t}</div>;
            })}
          </div>
          <div style={{ flex: 1, minWidth: 250, padding: "10px 14px", borderRadius: 8, background: C.purple+"06", border: "1px solid " + C.purple+"20" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.purple, marginBottom: 6 }}>{DNA + " Choose LoRA when:"}</div>
            {["Dataset > 1000 examples", "Single or few task adapters", "Need max quality on medium data", "Can merge for zero inference cost", "Want broad community support"].map(function(t, i) {
              return <div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{CHK + " " + t}</div>;
            })}
          </div>
        </div>
      </Card>

      <Insight icon={TARG} title="The Core Trade-off">
        IA\u00B3 and LoRA are complementary tools. IA\u00B3 is the <span style={{color:C.accent,fontWeight:700}}>minimalist's choice</span>: when you need extreme param efficiency, few-shot performance, or massive multi-task serving. LoRA is the <span style={{color:C.purple,fontWeight:700}}>practitioner's choice</span>: when you have a reasonable dataset, want zero inference overhead via merging, and need maximum community tooling support. Neither is universally better.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 6: WHEN TO USE IA3
   =============================================================== */
function TabWhenToUse() {
  var _s = useState(0); var usecase = _s[0], setUsecase = _s[1];

  var usecases = [
    {
      name: "Few-shot Fine-tuning", color: C.accent, score: 95,
      desc: "IA\u00B3 was designed for T-Few, a few-shot fine-tuning method. With 8-32 examples, IA\u00B3 reliably outperforms LoRA because fewer params = less overfitting.",
      icon: MICRO, fit: "Best",
    },
    {
      name: "Multi-task Serving", color: C.blue, score: 92,
      desc: "When you need to serve hundreds or thousands of tasks from one base, IA\u00B3's ~10K adapter size enables loading all adapters in memory simultaneously.",
      icon: BRAIN, fit: "Excellent",
    },
    {
      name: "Research / Ablations", color: C.purple, score: 85,
      desc: "Training is fast and cheap. Rapid iteration across many configurations is practical. Good for NLP research comparing many fine-tuning settings.",
      icon: DNA, fit: "Great",
    },
    {
      name: "Production NLP (>10K data)", color: C.yellow, score: 55,
      desc: "IA\u00B3 underfits on larger datasets. LoRA or QLoRA will outperform it here. Use IA\u00B3 only if storage is the primary constraint.",
      icon: WARN, fit: "Limited",
    },
    {
      name: "Code / Math Tasks", color: C.red, score: 40,
      desc: "Complex reasoning tasks benefit from more adapter capacity. LoRA (especially with higher rank) significantly outperforms IA\u00B3 on code generation and math.",
      icon: GEAR, fit: "Poor",
    },
  ];

  var u = usecases[usecase];
  var fitColors = { Best: C.green, Excellent: C.blue, Great: C.purple, Limited: C.yellow, Poor: C.red };

  return (
    <div>
      <SectionTitle title="When to Use IA\u00B3" subtitle={"Matching IA\u00B3 to the right problem \u2014 and knowing when to switch to LoRA"} />

      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {usecases.map(function(uc, i) {
          var on = usecase === i;
          return (<button key={i} onClick={function() { setUsecase(i); }} style={{
            padding: "8px 12px", borderRadius: 8, border: "1.5px solid " + (on ? uc.color : C.border),
            background: on ? uc.color+"20" : C.card, color: on ? uc.color : C.muted,
            cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
          }}>{uc.icon + " " + uc.name}</button>);
        })}
      </div>

      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: u.color }}>
        <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 280 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <div style={{ fontSize: 22 }}>{u.icon}</div>
              <div style={{ fontSize: 16, fontWeight: 800, color: u.color }}>{u.name}</div>
              <div style={{ padding: "3px 10px", borderRadius: 12, background: fitColors[u.fit]+"20", color: fitColors[u.fit], fontSize: 9, fontWeight: 700, fontFamily: "monospace" }}>{"IA\u00B3 Fit: " + u.fit}</div>
            </div>
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{u.desc}</div>
          </div>
          <div style={{ minWidth: 120, textAlign: "center" }}>
            <div style={{ fontSize: 8, color: C.muted, marginBottom: 8 }}>IA\u00B3 SUITABILITY</div>
            <svg width={100} height={100} viewBox="0 0 100 100">
              <circle cx={50} cy={50} r={42} fill="none" stroke={C.border} strokeWidth={8} />
              <circle cx={50} cy={50} r={42} fill="none" stroke={u.color} strokeWidth={8}
                strokeDasharray={"" + (u.score * 2.638) + " 264"}
                strokeLinecap="round"
                transform="rotate(-90 50 50)"
                style={{ transition: "stroke-dasharray 0.6s" }} />
              <text x={50} y={55} textAnchor="middle" fill={u.color} fontSize={22} fontWeight={800} fontFamily="monospace">{u.score}</text>
            </svg>
          </div>
        </div>
      </Card>

      {/* Checklist */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"IA\u00B3 vs LoRA Decision Checklist"}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 240 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 8 }}>{MICRO + " Use IA\u00B3 if:"}</div>
            {[
              ["Training examples", "< 500", true],
              ["Adapter storage budget", "Very tight", true],
              ["Number of tasks", "> 100", true],
              ["Training speed priority", "High", true],
              ["Task type", "Classification/NLU", true],
            ].map(function(row, i) {
              return (<div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                <div style={{ width: 140, fontSize: 8, color: C.muted }}>{row[0]}</div>
                <div style={{ padding: "2px 8px", borderRadius: 4, background: C.accent+"20", color: C.accent, fontSize: 8, fontFamily: "monospace" }}>{row[1]}</div>
              </div>);
            })}
          </div>
          <div style={{ flex: 1, minWidth: 240 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.purple, marginBottom: 8 }}>{DNA + " Use LoRA if:"}</div>
            {[
              ["Training examples", "> 1000", false],
              ["Inference latency", "Critical", false],
              ["Task type", "Generation/Code", false],
              ["Rank tuning", "Needed", false],
              ["Community support", "Important", false],
            ].map(function(row, i) {
              return (<div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                <div style={{ width: 140, fontSize: 8, color: C.muted }}>{row[0]}</div>
                <div style={{ padding: "2px 8px", borderRadius: 4, background: C.purple+"20", color: C.purple, fontSize: 8, fontFamily: "monospace" }}>{row[1]}</div>
              </div>);
            })}
          </div>
        </div>
      </Card>

      {/* T-Few reference */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 8 }}>{"IA\u00B3 in Practice: T-Few Method"}</div>
        <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8 }}>
          {"IA\u00B3 was introduced in the paper "}<span style={{color:C.accent,fontWeight:700}}>{LQ + "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" + RQ}</span>{" (Liu et al., 2022) as part of the T-Few recipe. The T-Few setup: (1) pre-compute IA\u00B3 vectors, (2) use task-specific prompt templates, (3) train on ≤1000 labeled examples. On the T0 benchmark, T-Few matches or beats GPT-3 in-context learning at a fraction of the inference cost."}
        </div>
      </Card>

      <Insight icon={FIRE} title="The Bottom Line">
        IA\u00B3 is the right choice when parameters are the primary bottleneck {DASH} not quality. If you're doing <span style={{color:C.accent,fontWeight:700}}>few-shot learning</span>, serving <span style={{color:C.blue,fontWeight:700}}>many tasks at once</span>, or need the <span style={{color:C.green,fontWeight:700}}>fastest possible training loop</span>, IA\u00B3 delivers competitive results at a fraction of the storage and memory cost. For everything else, <span style={{color:C.purple}}>LoRA remains the gold standard</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Big Picture", "IA\u00B3 Mechanics", "Memory & Cost", "Configuration", "IA\u00B3 vs LoRA", "When to Use"];
  return (
    <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.blue + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>PEFT IA\u00B3</div>
        <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " Infused Adapter by Inhibiting & Amplifying Inner Activations"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab === 0 && <TabBigPicture />}
      {tab === 1 && <TabMechanics />}
      {tab === 2 && <TabMemory />}
      {tab === 3 && <TabConfig />}
      {tab === 4 && <TabVsLora />}
      {tab === 5 && <TabWhenToUse />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

IA3_VISUAL_HEIGHT = 1600