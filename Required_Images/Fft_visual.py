"""
Self-contained HTML for the Full Fine-Tuning interactive walkthrough.
Covers: Big Picture, Training Loop, Memory & Cost, Data Pipeline,
Variants & Spectrum, and Catastrophic Forgetting.
Embed in Streamlit via st.components.v1.html(FFT_VISUAL_HTML, height=FFT_VISUAL_HEIGHT).
"""

FFT_VISUAL_HTML = """
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
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #ff6b35; }
  @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
  @keyframes flowRight { 0%{transform:translateX(-8px);opacity:0} 50%{opacity:1} 100%{transform:translateX(8px);opacity:0} }
  @keyframes gradientShift { 0%{stop-color:#ff6b35} 50%{stop-color:#fbbf24} 100%{stop-color:#ff6b35} }
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
  accent: "#ff6b35", blue: "#4ecdc4", purple: "#a78bfa",
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
var WARN_E = "\\u26A0\\uFE0F";
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
      padding: "16px 22px", background: "rgba(255,107,53,0.06)",
      borderRadius: 10, border: "1px solid rgba(255,107,53,0.2)",
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
  var _h = useState(0); var hoverLayer = _h[0], setHoverLayer = _h[1];

  useEffect(function() { var t = setTimeout(function() { setAnimated(true); }, 300); return function() { clearTimeout(t); }; }, []);

  var layers = [
    { name: "Embedding", params: "131M", pct: 1.9, color: C.cyan },
    { name: "Attention Layers", params: "4.7B", pct: 67.1, color: C.purple },
    { name: "FFN Layers", params: "2.0B", pct: 28.6, color: C.blue },
    { name: "LM Head", params: "131M", pct: 1.9, color: C.yellow },
    { name: "LayerNorms", params: "33M", pct: 0.5, color: C.green },
  ];

  return (
    <div>
      <SectionTitle title="The Big Picture" subtitle={"Full fine-tuning: take a generalist, retrain EVERYTHING to become a specialist"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={1050} height={280} viewBox="0 0 780 280" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {/* Pre-trained model */}
          <text x={120} y={22} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">PRE-TRAINED MODEL</text>
          <text x={120} y={36} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"(General Knowledge)"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            return (<rect key={"p"+i} x={40} y={48 + i * 26} width={160} height={20} rx={4}
              fill={C.dim + "30"} stroke={C.dim + "60"} strokeWidth={1}
              style={{ transition: "all 0.6s", transitionDelay: (i * 0.05) + "s" }} />);
          })}
          {[0,1,2,3,4,5,6,7].map(function(i) {
            var w = animated ? 160 : 0;
            return (<rect key={"pf"+i} x={40} y={48 + i * 26} width={w} height={20} rx={4}
              fill={C.dim + "20"} stroke="none"
              style={{ transition: "width 1.2s ease-out", transitionDelay: (i * 0.08) + "s" }} />);
          })}
          <text x={120} y={270} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"7B parameters"}</text>

          {/* Arrow + unlock */}
          <g>
            <line x1={220} y1={145} x2={290} y2={145} stroke={C.accent} strokeWidth={2} strokeDasharray={animated ? "0" : "6,4"} style={{ transition: "stroke-dasharray 0.5s" }} />
            <polygon points="295,145 288,140 288,150" fill={C.accent} />
            <text x={258} y={133} textAnchor="middle" fill={C.accent} fontSize={16}>{UNLOCK}</text>
            <text x={258} y={166} textAnchor="middle" fill={C.accent} fontSize={8} fontWeight={700} fontFamily="monospace">ALL UNLOCKED</text>
          </g>

          {/* Training */}
          <rect x={310} y={55} width={140} height={180} rx={10} fill={C.accent + "08"} stroke={C.accent + "40"} strokeWidth={1.5} strokeDasharray="6,3" />
          <text x={380} y={22} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">TRAINING LOOP</text>
          <text x={380} y={36} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">{"(Task-Specific Data)"}</text>
          {["Forward", "Loss", "Backprop", "Update"].map(function(s, i) {
            var cols = [C.blue, C.red, C.purple, C.green];
            return (<g key={s}>
              <rect x={330} y={70 + i * 42} width={100} height={28} rx={6} fill={cols[i] + "15"} stroke={cols[i] + "50"} strokeWidth={1} />
              <text x={380} y={88 + i * 42} textAnchor="middle" fill={cols[i]} fontSize={9} fontWeight={700} fontFamily="monospace">{s}</text>
              {i < 3 && <line x1={380} y1={98 + i * 42} x2={380} y2={112 + i * 42} stroke={C.dim} strokeWidth={1} />}
              {i < 3 && <polygon points={(380) + "," + (115 + i * 42) + " " + (377) + "," + (110 + i * 42) + " " + (383) + "," + (110 + i * 42)} fill={C.dim} />}
            </g>);
          })}
          <text x={380} y={250} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{MUL + " N epochs"}</text>

          {/* Arrow to result */}
          <g>
            <line x1={470} y1={145} x2={540} y2={145} stroke={C.green} strokeWidth={2} />
            <polygon points="545,145 538,140 538,150" fill={C.green} />
            <text x={505} y={133} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">EVERY WEIGHT</text>
            <text x={505} y={162} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">NUDGED</text>
          </g>

          {/* Fine-tuned model */}
          <text x={660} y={22} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">FINE-TUNED MODEL</text>
          <text x={660} y={36} textAnchor="middle" fill={C.green + "80"} fontSize={8} fontFamily="monospace">{"(Task Specialist)"}</text>
          {[0,1,2,3,4,5,6,7].map(function(i) {
            var w = animated ? 160 : 0;
            return (<rect key={"ft"+i} x={580} y={48 + i * 26} width={160} height={20} rx={4}
              fill={C.green + "10"} stroke={C.green + "40"} strokeWidth={1}
              style={{ transition: "all 0.8s", transitionDelay: (0.6 + i * 0.08) + "s" }} />);
          })}
          {[0,1,2,3,4,5,6,7].map(function(i) {
            var w = animated ? 160 : 0;
            return (<rect key={"ftf"+i} x={580} y={48 + i * 26} width={w} height={20} rx={4}
              fill={C.green + "20"} stroke="none"
              style={{ transition: "width 1s ease-out", transitionDelay: (0.8 + i * 0.1) + "s" }} />);
          })}
          <text x={660} y={270} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">{"7B parameters (ALL updated)"}</text>
        </svg>
      </div>

      {/* Layer breakdown */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 14 }}>{"What " + LQ + "ALL Parameters" + RQ + " Means " + DASH + " Inside a 7B Model"}</div>
        {layers.map(function(l, i) {
          var isH = hoverLayer === i + 1;
          return (
            <div key={i} onMouseEnter={function() { setHoverLayer(i + 1); }} onMouseLeave={function() { setHoverLayer(0); }}
              style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, cursor: "pointer", padding: "4px 8px", borderRadius: 6, background: isH ? l.color + "10" : "transparent", transition: "background 0.2s" }}>
              <div style={{ width: 110, fontSize: 9, color: isH ? l.color : C.muted, fontFamily: "monospace", fontWeight: isH ? 700 : 400 }}>{l.name}</div>
              <div style={{ flex: 1, position: "relative", height: 16 }}>
                <div style={{ width: Math.max(4, l.pct) + "%", height: "100%", borderRadius: 4, background: l.color + (isH ? "70" : "35"), border: "1px solid " + l.color + (isH ? "90" : "40"), transition: "all 0.3s" }} />
              </div>
              <div style={{ width: 55, fontSize: 9, color: isH ? l.color : C.dim, fontFamily: "monospace", textAlign: "right" }}>{l.params}</div>
              <div style={{ width: 40, fontSize: 9, color: isH ? l.color : C.dim, fontFamily: "monospace" }}>{l.pct + "%"}</div>
            </div>
          );
        })}
        <div style={{ marginTop: 10, fontSize: 9, color: C.muted, textAlign: "center" }}>
          {"In full fine-tuning, " + UNLOCK + " ALL of these components are unlocked and updated via gradient descent"}
        </div>
      </Card>

      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 10 }}>{"Why Fine-Tune?"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 12, flexWrap: "wrap" }}>
          {[
            { icon: BRAIN, title: "General " + ARR + " Specific", desc: "Poetry model can't classify legal contracts", c: C.blue },
            { icon: FIRE, title: "Reshape ALL weights", desc: "Every connection adapts to your task", c: C.accent },
            { icon: CHART, title: "Maximum quality", desc: "Best possible task performance", c: C.green },
            { icon: WARN_E, title: "Highest cost", desc: "GPU memory, compute, forgetting risk", c: C.red },
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
        Think of a pre-trained model as a <span style={{color:C.blue,fontWeight:700}}>brilliant generalist</span> who has read millions of books. Full fine-tuning is <span style={{color:C.accent,fontWeight:700}}>intensive retraining</span> {DASH} reshaping every single connection toward your specific goal. The result: a <span style={{color:C.green,fontWeight:700}}>specialist</span>, but one that may forget some general knowledge.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: THE TRAINING LOOP
   =============================================================== */
function TabTrainingLoop() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _au = useState(false); var autoP = _au[0], setAutoP = _au[1];
  var _lr = useState(3); var lrIdx = _lr[0], setLrIdx = _lr[1];

  var steps = [
    { title: "Forward Pass", subtitle: "The Model Makes a Prediction", color: C.blue,
      desc: "Input flows through ALL layers " + DASH + " embedding, attention, feed-forward " + DASH + " and the model produces an output prediction. Think of water flowing through pipes where each weight controls how much flows through." },
    { title: "Loss Calculation", subtitle: "How Wrong Was It?", color: C.red,
      desc: "Compare prediction to the correct answer. The difference = loss. High loss means very wrong, low loss means close. For classification: predicted " + LQ + "negative" + RQ + " for a " + LQ + "positive" + RQ + " review " + ARR + " high loss." },
    { title: "Backpropagation", subtitle: "Tracing the Blame", color: C.purple,
      desc: "Work backward through the ENTIRE network asking: " + LQ + "which weights contributed to this mistake?" + RQ + " This produces a gradient for every single weight " + DASH + " a direction saying " + LQ + "adjust this much to reduce error." + RQ },
    { title: "Weight Update", subtitle: "Adjusting Everything", color: C.green,
      desc: "Using an optimizer (Adam), EVERY weight is nudged in the direction that reduces error. The learning rate controls nudge size. In full fine-tuning, ALL billions of weights are updated. This is the defining characteristic." },
    { title: "Repeat", subtitle: "Batch After Batch, Epoch After Epoch", color: C.yellow,
      desc: "This cycle repeats for every batch of examples, across multiple passes through the dataset (epochs). Gradient accumulation lets you simulate larger batches: 4 " + MUL + " 8 micro-batches = effective batch size of 32." },
  ];

  useEffect(function() { if (!autoP) return; var t = setInterval(function() { setStep(function(s) { return (s + 1) % 5; }); }, 2800); return function() { clearInterval(t); }; }, [autoP]);

  var lrVals = ["1e-3", "5e-4", "1e-4", "2e-5", "1e-6"];
  var lrDescs = ["Way too high " + DASH + " destroys knowledge!", "Still too aggressive for fine-tuning", "Upper bound " + DASH + " risky but fast", "Sweet spot for most full fine-tuning", "Very safe but slow convergence"];
  var lrColors = [C.red, C.orange, C.yellow, C.green, C.blue];
  var lrBarW = [100, 80, 55, 30, 10];

  return (
    <div>
      <SectionTitle title="The Training Loop" subtitle={"Forward " + ARR + " Loss " + ARR + " Backward " + ARR + " Update " + DASH + " the heartbeat of fine-tuning"} />

      <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          var on = step === i;
          return (<button key={i} onClick={function() { setStep(i); setAutoP(false); }} style={{
            padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? s.color : C.border),
            background: on ? s.color + "20" : C.card, color: on ? s.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{(i + 1) + ". " + s.title}</button>);
        })}
        <button onClick={function() { setAutoP(!autoP); }} style={{ padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (autoP ? C.yellow : C.border), background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted, cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>{autoP ? PAUSE : PLAY}</button>
      </div>

      {/* Main training loop SVG */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={260} viewBox="0 0 750 260" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {/* Cycle visualization */}
          {steps.map(function(s, i) {
            var x = 80 + i * 150;
            var on = step === i;
            var done = i < step;
            return (<g key={i}>
              <rect x={x - 55} y={70} width={110} height={65} rx={10}
                fill={on ? s.color + "20" : done ? s.color + "08" : C.card}
                stroke={on ? s.color : done ? s.color + "40" : C.border}
                strokeWidth={on ? 2.5 : 1}
                style={{ transition: "all 0.3s" }} />
              <text x={x} y={95} textAnchor="middle" fill={on ? s.color : done ? s.color + "80" : C.dim}
                fontSize={10} fontWeight={700} fontFamily="monospace">{s.title}</text>
              <text x={x} y={112} textAnchor="middle" fill={on ? s.color + "90" : C.dim}
                fontSize={7} fontFamily="monospace">{s.subtitle.substring(0, 20)}</text>
              {on && <rect x={x - 55} y={132} width={110} height={3} rx={1.5} fill={s.color}
                style={{ animation: "pulse 1.5s infinite" }} />}
              {i < 4 && <g>
                <line x1={x + 55} y1={102} x2={x + 95} y2={102} stroke={done || on ? steps[Math.min(i + 1, 4)].color + "60" : C.dim + "40"} strokeWidth={1.5} />
                <polygon points={(x + 95) + ",102 " + (x + 90) + ",98 " + (x + 90) + ",106"} fill={done || on ? steps[Math.min(i + 1, 4)].color + "60" : C.dim + "40"} />
              </g>}
            </g>);
          })}

          {/* Repeat arrow (curved from step 5 back to step 1) */}
          <path d={"M 720,102 C 740,102 740,200 375,210 C 10,220 10,102 25,102"} fill="none" stroke={C.yellow + "30"} strokeWidth={1.5} strokeDasharray="6,4" />
          <text x={375} y={235} textAnchor="middle" fill={C.yellow} fontSize={9} fontWeight={700} fontFamily="monospace">{MUL + " repeat for each batch " + MUL + " N epochs"}</text>

          {/* Param count indicator */}
          <rect x={270} y={15} width={210} height={34} rx={8} fill={C.accent + "10"} stroke={C.accent + "30"} />
          <text x={375} y={30} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">{UNLOCK + " ALL parameters updated at Step 4"}</text>
          <text x={375} y={42} textAnchor="middle" fill={C.accent + "80"} fontSize={8} fontFamily="monospace">{"(7B+ weights " + MUL + " every batch)"}</text>
        </svg>
      </div>

      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: steps[step].color }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 14, background: steps[step].color + "25", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 800, color: steps[step].color, fontFamily: "monospace" }}>{step + 1}</div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: steps[step].color }}>{steps[step].title}</div>
            <div style={{ fontSize: 9, color: C.muted }}>{steps[step].subtitle}</div>
          </div>
        </div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{steps[step].desc}</div>
      </Card>

      {/* Learning Rate Interactive */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 12 }}>{"The Learning Rate Dilemma " + DASH + " The Most Critical Hyperparameter"}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 12 }}>
          <div style={{ fontSize: 9, color: C.muted, whiteSpace: "nowrap" }}>LR:</div>
          <input type="range" min={0} max={4} value={lrIdx} onChange={function(e) { setLrIdx(parseInt(e.target.value)); }}
            style={{ flex: 1, accentColor: lrColors[lrIdx] }} />
          <div style={{ fontSize: 14, fontWeight: 800, color: lrColors[lrIdx], fontFamily: "monospace", minWidth: 50 }}>{lrVals[lrIdx]}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <div style={{ fontSize: 9, color: C.muted, width: 80 }}>Change size:</div>
          <div style={{ flex: 1, position: "relative", height: 20 }}>
            <div style={{ width: lrBarW[lrIdx] + "%", height: "100%", borderRadius: 4, background: lrColors[lrIdx] + "50", border: "1px solid " + lrColors[lrIdx], transition: "all 0.3s" }} />
          </div>
        </div>
        <div style={{ padding: "8px 12px", background: lrColors[lrIdx] + "10", borderRadius: 6, border: "1px solid " + lrColors[lrIdx] + "30" }}>
          <div style={{ fontSize: 10, color: lrColors[lrIdx], fontWeight: 700 }}>{lrDescs[lrIdx]}</div>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 4 }}>{"Fine-tuning range: 1e-6 to 5e-5 (10" + MUL + " to 100" + MUL + " smaller than pre-training). Pre-trained weights already encode valuable knowledge " + DASH + " you want to gently reshape, not destroy."}</div>
        </div>
      </Card>

      <Insight icon={TARG} title="The Sculpture Analogy">
        Pre-trained weights are a <span style={{color:C.purple,fontWeight:700}}>beautifully sculpted statue</span>. Fine-tuning is <span style={{color:C.accent,fontWeight:700}}>carefully chiseling new details</span>. Too aggressive (high LR) {ARR} destroy the sculpture. Too gentle (low LR) {ARR} never finish. The <span style={{color:C.green,fontWeight:700}}>sweet spot: 1e-5 to 3e-5</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: MEMORY & COST
   =============================================================== */
function TabMemory() {
  var _m = useState(0); var modelSel = _m[0], setModelSel = _m[1];
  var _p = useState(true); var mixedP = _p[0], setMixedP = _p[1];

  var models = [
    { name: "LLaMA 1B", params: 1.24, color: C.green },
    { name: "LLaMA 7B", params: 7, color: C.blue },
    { name: "LLaMA 13B", params: 13, color: C.purple },
    { name: "LLaMA 70B", params: 70, color: C.accent },
  ];

  var m = models[modelSel];
  var bp = mixedP ? 2 : 4;
  var wMem = m.params * bp;
  var gMem = m.params * bp;
  var oMem = m.params * 8;
  var aMem = m.params < 5 ? 3 : m.params < 15 ? 12 : m.params < 50 ? 25 : 60;
  var total = wMem + gMem + oMem + aMem;

  var components = [
    { name: "Model Weights", mem: wMem, formula: m.params + "B " + MUL + " " + bp + " bytes", color: C.blue, desc: mixedP ? "BF16" : "FP32" },
    { name: "Gradients", mem: gMem, formula: m.params + "B " + MUL + " " + bp + " bytes", color: C.purple, desc: "Same dtype as weights" },
    { name: "Optimizer (Adam)", mem: oMem, formula: m.params + "B " + MUL + " 8 bytes", color: C.red, desc: "Momentum + Variance (FP32)" },
    { name: "Activations", mem: aMem, formula: "~" + aMem + " GB (variable)", color: C.yellow, desc: "Batch size + seq length dependent" },
  ];

  var maxMem = Math.max.apply(null, components.map(function(c) { return c.mem; }));

  return (
    <div>
      <SectionTitle title={"Memory & Cost"} subtitle={"Why full fine-tuning is expensive " + DASH + " and where all the GPU memory goes"} />

      {/* Model selector */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 12, flexWrap: "wrap" }}>
        {models.map(function(mod, i) {
          var on = modelSel === i;
          return (<button key={i} onClick={function() { setModelSel(i); }} style={{
            padding: "8px 16px", borderRadius: 8, border: "1.5px solid " + (on ? mod.color : C.border),
            background: on ? mod.color + "20" : C.card, color: on ? mod.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{mod.name}</button>);
        })}
        <button onClick={function() { setMixedP(!mixedP); }} style={{
          padding: "8px 16px", borderRadius: 8, border: "1.5px solid " + (mixedP ? C.cyan : C.border),
          background: mixedP ? C.cyan + "15" : C.card, color: mixedP ? C.cyan : C.muted,
          cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
        }}>{mixedP ? "BF16 Mixed " + CHK : "FP32 Full"}</button>
      </div>

      {/* Memory breakdown bars */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>{"GPU Memory Breakdown " + DASH + " " + m.name}</div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 22, fontWeight: 800, color: total > 80 ? C.red : total > 40 ? C.orange : C.green }}>{"~" + total.toFixed(0) + " GB"}</div>
            <div style={{ fontSize: 8, color: C.muted }}>{"TOTAL VRAM NEEDED"}</div>
          </div>
        </div>

        {components.map(function(comp, i) {
          var pct = (comp.mem / total) * 100;
          var barW = Math.max(6, (comp.mem / maxMem) * 100);
          return (
            <div key={i} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <div style={{ fontSize: 9, color: comp.color, fontWeight: 700, fontFamily: "monospace" }}>{comp.name + " (" + comp.desc + ")"}</div>
                <div style={{ fontSize: 9, color: comp.color, fontFamily: "monospace" }}>{comp.mem.toFixed(1) + " GB (" + pct.toFixed(0) + "%)"}</div>
              </div>
              <div style={{ position: "relative", height: 22, background: C.border, borderRadius: 4 }}>
                <div style={{ width: barW + "%", height: "100%", borderRadius: 4, background: comp.color + "50", border: "1px solid " + comp.color, transition: "width 0.5s" }} />
                <div style={{ position: "absolute", left: 8, top: 4, fontSize: 8, color: C.muted, fontFamily: "monospace" }}>{comp.formula}</div>
              </div>
            </div>
          );
        })}

        <div style={{ marginTop: 12, padding: "8px 12px", background: C.red + "08", borderRadius: 6, border: "1px solid " + C.red + "25" }}>
          <div style={{ fontSize: 9, color: C.red, fontWeight: 700 }}>{WARN + " Optimizer states dominate memory"}</div>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>{"Adam stores 2 extra values per weight in FP32 for numerical stability. This is why 8-bit Adam (bitsandbytes) exists " + DASH + " to compress these states."}</div>
        </div>
      </Card>

      {/* Comparison: Full FT vs LoRA */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"Full Fine-Tuning vs LoRA " + DASH + " " + m.name}</div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
          {[
            { label: "Full FT", mem: total.toFixed(0) + " GB", params: "100%", tasks: m.params * bp + " GB/task", color: C.accent },
            { label: "LoRA", mem: (wMem + 3).toFixed(0) + " GB", params: "~0.3%", tasks: "~50 MB/task", color: C.green },
          ].map(function(v, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 200, padding: "12px 16px", borderRadius: 8, background: v.color + "08", border: "1px solid " + v.color + "30" }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: v.color, marginBottom: 8 }}>{v.label}</div>
                {[
                  { l: "VRAM", v: v.mem },
                  { l: "Params trained", v: v.params },
                  { l: "Storage/task", v: v.tasks },
                ].map(function(r, j) {
                  return (<div key={j} style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 9, color: C.muted }}>{r.l}</span>
                    <span style={{ fontSize: 9, color: v.color, fontWeight: 700, fontFamily: "monospace" }}>{r.v}</span>
                  </div>);
                })}
              </div>
            );
          })}
        </div>
      </Card>

      {/* Gradient checkpointing */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.cyan, marginBottom: 10 }}>{"Gradient Checkpointing " + DASH + " Trading Compute for Memory"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 30, flexWrap: "wrap" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>WITHOUT</div>
            <div style={{ padding: "8px 16px", borderRadius: 6, background: C.red + "10", border: "1px solid " + C.red + "30" }}>
              <div style={{ fontSize: 10, color: C.red, fontFamily: "monospace" }}>Store ALL activations</div>
              <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>{UARR + " Memory / Normal speed"}</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: 16, color: C.dim }}>vs</div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>WITH</div>
            <div style={{ padding: "8px 16px", borderRadius: 6, background: C.green + "10", border: "1px solid " + C.green + "30" }}>
              <div style={{ fontSize: 10, color: C.green, fontFamily: "monospace" }}>Recompute on-the-fly</div>
              <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>{"30-40% less mem / 20-30% slower"}</div>
            </div>
          </div>
        </div>
      </Card>

      <Insight icon={TARG} title={"Why PEFT Was Invented"}>
        Every pain point above has a PEFT solution: <span style={{color:C.red,fontWeight:700}}>{total.toFixed(0) + " GB"}</span> for {m.name} {ARR} LoRA: <span style={{color:C.green,fontWeight:700}}>{"~" + (wMem + 3).toFixed(0) + " GB"}</span>. Full copy per task {ARR} <span style={{color:C.green,fontWeight:700}}>~50 MB adapters</span>. Catastrophic forgetting {ARR} <span style={{color:C.green,fontWeight:700}}>base model frozen</span>.
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
      desc: "Your fine-tuning data is just text in simple files. No vectors, no embeddings " + DASH + " just text.",
      content: '{"instruction": "Classify sentiment",\\n "input": "This movie was breathtaking",\\n "output": "positive"}' },
    { title: "Template Formatting", color: C.cyan,
      desc: "Apply a model-specific chat template to combine fields into a single string with special tokens.",
      content: '<s>[INST] Classify sentiment:\\nThis movie was breathtaking\\n[/INST] positive</s>' },
    { title: "Tokenization", color: C.purple,
      desc: "Break text into subword tokens and map each to an integer ID. The tokenizer is FIXED " + DASH + " never changes during training.",
      content: '[1, 518, 25580, 29962, 4134,\\n 1598, 278, ..., 6374, 2]\\n     20 tokens' },
    { title: "Labels + Loss Mask", color: C.yellow,
      desc: "Create labels array. -100 = " + LQ + "ignore this token." + RQ + " Only grade the output/response portion, not the instruction.",
      content: 'Labels:\\n[-100, -100, ..., -100, 6374, 2]\\n ^^^^^^ instruction ^^^^^^  ^output^' },
    { title: "Pad + Attention Mask", color: C.orange,
      desc: "All sequences in a batch must be same length. Shorter ones get PAD tokens (0). Attention mask: 1=real, 0=padding.",
      content: 'input_ids: [..., 6374, 2, 0, 0, 0]\\nattn_mask: [ 1,  1, ..., 1, 0, 0, 0]\\nlabels:    [-100,..., 6374, 2,-100,-100]' },
    { title: "Batch Tensors " + ARR + " GPU", color: C.green,
      desc: "Stack into rectangular tensors [batch_size, seq_length] and move to GPU. The embedding layer converts IDs " + ARR + " vectors INSIDE the model.",
      content: 'input_ids     [4, 2048]     int\\nattn_mask      [4, 2048]     int\\nlabels        [4, 2048]     int\\n        ' + DARR + ' GPU ' + DARR + '\\nhidden_states [4, 2048, 4096] float' },
    { title: "Forward + Backward", color: C.accent,
      desc: "Embedding converts IDs to 4096-dim vectors. Forward through 32 transformer layers. Compute loss. Backprop. Update ALL weights. Next batch.",
      content: '[4, 2048] ' + ARR + ' Embed ' + ARR + ' [4, 2048, 4096]\\n  ' + ARR + ' 32 Transformer Layers\\n  ' + ARR + ' Loss ' + ARR + ' Backprop\\n  ' + ARR + ' Update ALL weights\\n  ' + ARR + ' Next batch!' },
  ];

  useEffect(function() { if (!autoP) return; var t = setInterval(function() { setStep(function(s) { return (s + 1) % pSteps.length; }); }, 3500); return function() { clearInterval(t); }; }, [autoP]);

  return (
    <div>
      <SectionTitle title="Data Pipeline" subtitle={"From raw JSONL on disk to gradient updates on GPU " + DASH + " every transformation step"} />

      <div style={{ display: "flex", justifyContent: "center", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
        {pSteps.map(function(s, i) {
          var on = step === i;
          return (<button key={i} onClick={function() { setStep(i); setAutoP(false); }} style={{
            padding: "6px 10px", borderRadius: 6, border: "1.5px solid " + (on ? s.color : C.border),
            background: on ? s.color + "20" : C.card, color: on ? s.color : C.muted,
            cursor: "pointer", fontSize: 9, fontWeight: 700, fontFamily: "monospace"
          }}>{(i + 1)}</button>);
        })}
        <button onClick={function() { setAutoP(!autoP); }} style={{ padding: "6px 10px", borderRadius: 6, border: "1.5px solid " + (autoP ? C.yellow : C.border), background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted, cursor: "pointer", fontSize: 9, fontFamily: "monospace" }}>{autoP ? PAUSE : PLAY}</button>
      </div>

      {/* Pipeline flow visualization */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1050} height={80} viewBox="0 0 780 80" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {pSteps.map(function(s, i) {
            var x = 18 + i * 108;
            var on = step === i;
            var done = i < step;
            return (<g key={i}>
              <rect x={x} y={15} width={95} height={50} rx={8}
                fill={on ? s.color + "20" : done ? s.color + "08" : C.card}
                stroke={on ? s.color : done ? s.color + "30" : C.dim + "40"}
                strokeWidth={on ? 2 : 1} style={{ transition: "all 0.3s" }} />
              <text x={x + 47} y={35} textAnchor="middle" fill={on ? s.color : done ? s.color + "70" : C.dim}
                fontSize={8} fontWeight={700} fontFamily="monospace">{"Step " + (i + 1)}</text>
              <text x={x + 47} y={50} textAnchor="middle" fill={on ? s.color + "80" : C.dim}
                fontSize={7} fontFamily="monospace">{s.title.split(" ")[0]}</text>
              {on && <rect x={x} y={62} width={95} height={3} rx={1.5} fill={s.color} style={{ animation: "pulse 1.5s infinite" }} />}
              {i < pSteps.length - 1 && <g>
                <line x1={x + 95} y1={40} x2={x + 108} y2={40} stroke={done ? s.color + "40" : C.dim + "30"} strokeWidth={1} />
                <polygon points={(x + 108) + ",40 " + (x + 104) + ",37 " + (x + 104) + ",43"} fill={done ? s.color + "40" : C.dim + "30"} />
              </g>}
            </g>);
          })}
        </svg>
      </div>

      {/* Active step detail */}
      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: pSteps[step].color }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: pSteps[step].color, marginBottom: 4 }}>{"Step " + (step + 1) + ": " + pSteps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 12 }}>{pSteps[step].desc}</div>
        <div style={{ padding: "12px 16px", background: "#08080d", borderRadius: 8, border: "1px solid " + C.border }}>
          <pre style={{ fontSize: 10, color: pSteps[step].color, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.8, margin: 0, whiteSpace: "pre-wrap", overflowX: "auto" }}>{pSteps[step].content}</pre>
        </div>
      </Card>

      {/* Where data lives */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"What Lives Where During Training"}</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
          {[
            { title: "DISK (SSD)", items: ["JSONL/Parquet files", "Model checkpoints"], color: C.blue, icon: "\\uD83D\\uDCBE" },
            { title: "CPU RAM", items: ["DataLoader workers", "Prefetched batches", "Memory-mapped index"], color: C.purple, icon: "\\uD83D\\uDDA5" },
            { title: "GPU VRAM", items: ["Current micro-batch", "Model weights", "Gradients + Optimizer", "Activations"], color: C.accent, icon: "\\u26A1" },
          ].map(function(loc, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 180, padding: "12px 14px", borderRadius: 8, background: loc.color + "06", border: "1px solid " + loc.color + "25" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: loc.color, marginBottom: 8 }}>{loc.icon + " " + loc.title}</div>
                {loc.items.map(function(it, j) {
                  return (<div key={j} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8, paddingLeft: 8, borderLeft: "2px solid " + loc.color + "20" }}>{it}</div>);
                })}
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={BULB} title="Common Misconception">
        The input is <span style={{color:C.red,fontWeight:700}}>NOT</span> pre-computed embedding vectors. You feed raw text. The model's own embedding layer converts text {ARR} vectors <span style={{color:C.accent,fontWeight:700}}>on-the-fly inside the model</span> during training. The embedding table itself gets updated during full fine-tuning.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: VARIANTS & SPECTRUM
   =============================================================== */
function TabVariants() {
  var _v = useState(0); var sel = _v[0], setSel = _v[1];

  var variants = [
    { name: "Feature Extraction", color: C.blue, flexibility: 10, safety: 95, cost: 15,
      unlock: "0%", head: "New head only",
      desc: "Freeze the ENTIRE pre-trained model. Only train a new classification head added on top. The model is a fixed feature extractor.",
      analogy: "Don" + "'" + "t retrain the generalist at all. Just add a specialist translator at the end who interprets the generalist" + "'" + "s output.",
      pros: ["Zero forgetting", "Lowest cost", "Fastest training"],
      cons: ["Least flexible", "Can't adapt internal representations", "Limited performance ceiling"] },
    { name: "Layer-Selective", color: C.cyan, flexibility: 35, safety: 75, cost: 35,
      unlock: "10-30%", head: "Last N layers + head",
      desc: "Choose specific layers to train while freezing the rest. Common: last N layers, first+last, or every Nth layer.",
      analogy: "Retrain just part of the generalist" + "'" + "s brain " + DASH + " the parts most relevant to your task.",
      pros: ["Good balance", "Lower cost than full", "Predictable behavior"],
      cons: ["Need to choose layers wisely", "May miss cross-layer adaptations"] },
    { name: "Gradual Unfreezing", color: C.purple, flexibility: 65, safety: 70, cost: 60,
      unlock: "0% " + ARR + " 100%", head: "Top " + ARR + " deeper layers",
      desc: "Start by training only top layers. Epoch by epoch, unfreeze deeper layers. The model progressively adapts from output to input.",
      analogy: "First train the specialist translator (top layers), then gradually let the generalist adapt their thinking (deeper layers).",
      pros: ["Balanced adaptation", "Reduces forgetting", "Elegant approach"],
      cons: ["More complex to configure", "Slower than standard full FT", "Needs careful scheduling"] },
    { name: "Standard Full FT", color: C.accent, flexibility: 100, safety: 30, cost: 100,
      unlock: "100%", head: "All layers from start",
      desc: "ALL layers, ALL weights, updated from the very first step. Maximum flexibility, maximum risk of catastrophic forgetting, maximum compute cost.",
      analogy: "Completely retrain the generalist into a specialist. Every neural pathway reshaped for your task.",
      pros: ["Maximum flexibility", "Highest performance ceiling", "Deepest task adaptation"],
      cons: ["Catastrophic forgetting risk", "Highest GPU cost", "Full model copy per task"] },
  ];

  var v = variants[sel];

  return (
    <div>
      <SectionTitle title="Variants & Spectrum" subtitle={"From safest to most aggressive " + DASH + " choose the right approach for your situation"} />

      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 20, flexWrap: "wrap" }}>
        {variants.map(function(vr, i) {
          var on = sel === i;
          return (<button key={i} onClick={function() { setSel(i); }} style={{
            padding: "8px 16px", borderRadius: 8, border: "1.5px solid " + (on ? vr.color : C.border),
            background: on ? vr.color + "20" : C.card, color: on ? vr.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{vr.name}</button>);
        })}
      </div>

      {/* Spectrum bar */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 9, color: C.muted, textAlign: "center", marginBottom: 8 }}>{"FULL FINE-TUNING SPECTRUM"}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 0, marginBottom: 6 }}>
          {variants.map(function(vr, i) {
            var on = sel === i;
            return (<div key={i} onClick={function() { setSel(i); }} style={{
              flex: 1, height: on ? 36 : 24, background: vr.color + (on ? "40" : "15"),
              borderTop: "3px solid " + vr.color + (on ? "" : "50"),
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer", transition: "all 0.3s",
              borderRadius: i === 0 ? "6px 0 0 6px" : i === 3 ? "0 6px 6px 0" : 0,
            }}>
              <span style={{ fontSize: 8, color: on ? vr.color : C.dim, fontWeight: 700, fontFamily: "monospace" }}>{on ? vr.name : ""}</span>
            </div>);
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ fontSize: 8, color: C.blue }}>{LOCK + " Safest"}</span>
          <span style={{ fontSize: 8, color: C.accent }}>{FIRE + " Most aggressive"}</span>
        </div>
      </Card>

      {/* Detail card */}
      <Card highlight={true} style={{ maxWidth: 1100, margin: "0 auto 16px", borderColor: v.color }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
          <div style={{ flex: 1, minWidth: 250 }}>
            <div style={{ fontSize: 18, fontWeight: 800, color: v.color }}>{v.name}</div>
            <div style={{ fontSize: 11, color: C.muted, marginTop: 8, lineHeight: 1.7 }}>{v.desc}</div>
            <div style={{ marginTop: 10, fontSize: 10, color: C.muted, fontStyle: "italic", borderLeft: "3px solid " + v.color + "40", paddingLeft: 10, lineHeight: 1.7 }}>{v.analogy}</div>
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <StatBox label="UNLOCKED" value={v.unlock} color={v.color} />
            <StatBox label="WHAT TRAINS" value={v.head} color={v.color} bigFont={11} minW={100} />
          </div>
        </div>

        {/* Meter bars */}
        <div style={{ marginTop: 16 }}>
          {[
            { l: "Flexibility", v: v.flexibility, c: C.green },
            { l: "Safety (forgetting)", v: v.safety, c: C.blue },
            { l: "Compute Cost", v: v.cost, c: C.red },
          ].map(function(bar, i) {
            return (<div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div style={{ width: 120, fontSize: 9, color: C.muted, fontFamily: "monospace" }}>{bar.l}</div>
              <div style={{ flex: 1, height: 14, background: C.border, borderRadius: 3 }}>
                <div style={{ width: bar.v + "%", height: "100%", borderRadius: 3, background: bar.c + "50", border: "1px solid " + bar.c, transition: "width 0.5s" }} />
              </div>
              <div style={{ width: 30, fontSize: 9, color: bar.c, fontFamily: "monospace", fontWeight: 700 }}>{bar.v + "%"}</div>
            </div>);
          })}
        </div>

        {/* Pros / Cons */}
        <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.green, fontWeight: 700, marginBottom: 4 }}>{CHK + " Advantages"}</div>
            {v.pros.map(function(p, i) { return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + CHK + " " + p}</div>); })}
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 9, color: C.red, fontWeight: 700, marginBottom: 4 }}>{WARN + " Trade-offs"}</div>
            {v.cons.map(function(p, i) { return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + WARN + " " + p}</div>); })}
          </div>
        </div>
      </Card>

      <Insight icon={TARG} title="Choosing the Right Variant">
        <span style={{color:C.blue}}>Feature Extraction</span> for quick baselines and tiny datasets. <span style={{color:C.purple}}>Gradual Unfreezing</span> for the best balance. <span style={{color:C.accent}}>Standard Full FT</span> when you have massive data + GPUs and need maximum performance. Most practitioners today reach for <span style={{color:C.green,fontWeight:700}}>PEFT/LoRA</span> instead of full FT.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 6: CATASTROPHIC FORGETTING
   =============================================================== */
function TabForgetting() {
  var _e = useState(0); var epoch = _e[0], setEpoch = _e[1];
  var _au = useState(false); var autoP = _au[0], setAutoP = _au[1];
  var _s = useState(0); var strategy = _s[0], setStrategy = _s[1];

  useEffect(function() { if (!autoP) return; var t = setInterval(function() { setEpoch(function(e) { return Math.min(e + 1, 10); }); }, 1000); return function() { clearInterval(t); }; }, [autoP]);
  useEffect(function() { setEpoch(0); setAutoP(false); }, [strategy]);

  var strategies = [
    { name: "No Mitigation", color: C.red,
      taskPerf: function(e) { return Math.min(95, 40 + e * 8); },
      genKnow: function(e) { return Math.max(15, 95 - e * 9); },
      desc: "All layers unlocked, aggressive LR (5e-4), no regularization. Task performance climbs fast but general knowledge collapses." },
    { name: "Small Learning Rate", color: C.yellow,
      taskPerf: function(e) { return Math.min(88, 40 + e * 5.5); },
      genKnow: function(e) { return Math.max(55, 95 - e * 4.5); },
      desc: "LR = 2e-5. Gentle updates preserve more original knowledge. Slower to converge but much safer." },
    { name: "Early Stopping", color: C.blue,
      taskPerf: function(e) { return e <= 5 ? Math.min(82, 40 + e * 9) : 82; },
      genKnow: function(e) { return e <= 5 ? Math.max(65, 95 - e * 6) : 65; },
      desc: "Stop at epoch 5 when validation loss stops improving. You sacrifice a few points of task accuracy to retain general knowledge." },
    { name: "Data Mixing", color: C.green,
      taskPerf: function(e) { return Math.min(85, 40 + e * 5); },
      genKnow: function(e) { return Math.max(72, 95 - e * 2.5); },
      desc: "Include ~10% general-purpose data alongside task data. The model is continuously reminded of general knowledge during training." },
  ];

  var strat = strategies[strategy];
  var taskV = strat.taskPerf(epoch);
  var genV = strat.genKnow(epoch);

  var chartPoints = [];
  for (var i = 0; i <= 10; i++) {
    chartPoints.push({ e: i, task: strat.taskPerf(i), gen: strat.genKnow(i) });
  }

  return (
    <div>
      <SectionTitle title="Catastrophic Forgetting" subtitle={"The biggest danger of full fine-tuning " + DASH + " and how to mitigate it"} />

      {/* Strategy selector */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {strategies.map(function(s, i) {
          var on = strategy === i;
          return (<button key={i} onClick={function() { setStrategy(i); }} style={{
            padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (on ? s.color : C.border),
            background: on ? s.color + "20" : C.card, color: on ? s.color : C.muted,
            cursor: "pointer", fontSize: 10, fontWeight: 700, fontFamily: "monospace"
          }}>{s.name}</button>);
        })}
      </div>

      {/* Chart */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={1000} height={280} viewBox="0 0 700 280" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(function(v) {
            var y = 250 - v * 2.2;
            return (<g key={v}>
              <line x1={60} y1={y} x2={660} y2={y} stroke={C.dim + "30"} strokeWidth={0.5} />
              <text x={52} y={y + 3} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{v + "%"}</text>
            </g>);
          })}

          {/* X axis labels */}
          {[0, 2, 4, 6, 8, 10].map(function(e) {
            var x = 60 + e * 60;
            return (<text key={e} x={x} y={268} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"E" + e}</text>);
          })}
          <text x={380} y={278} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">Epoch</text>

          {/* Task performance line */}
          <polyline
            points={chartPoints.map(function(p) { return (60 + p.e * 60) + "," + (250 - p.task * 2.2); }).join(" ")}
            fill="none" stroke={C.green} strokeWidth={2} />

          {/* General knowledge line */}
          <polyline
            points={chartPoints.map(function(p) { return (60 + p.e * 60) + "," + (250 - p.gen * 2.2); }).join(" ")}
            fill="none" stroke={C.red} strokeWidth={2} strokeDasharray="6,3" />

          {/* Current epoch marker */}
          <line x1={60 + epoch * 60} y1={30} x2={60 + epoch * 60} y2={255} stroke={strat.color + "40"} strokeWidth={1} strokeDasharray="4,4" />
          <circle cx={60 + epoch * 60} cy={250 - taskV * 2.2} r={5} fill={C.green} stroke="#0a0a0f" strokeWidth={2} />
          <circle cx={60 + epoch * 60} cy={250 - genV * 2.2} r={5} fill={C.red} stroke="#0a0a0f" strokeWidth={2} />

          {/* Legend */}
          <line x1={490} y1={20} x2={510} y2={20} stroke={C.green} strokeWidth={2} />
          <text x={515} y={24} fill={C.green} fontSize={9} fontFamily="monospace">Task Performance</text>
          <line x1={490} y1={36} x2={510} y2={36} stroke={C.red} strokeWidth={2} strokeDasharray="6,3" />
          <text x={515} y={40} fill={C.red} fontSize={9} fontFamily="monospace">General Knowledge</text>
        </svg>
      </div>

      {/* Epoch slider */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <div style={{ fontSize: 9, color: C.muted }}>Epoch:</div>
          <input type="range" min={0} max={10} value={epoch}
            onChange={function(e) { setEpoch(parseInt(e.target.value)); setAutoP(false); }}
            style={{ flex: 1, accentColor: strat.color }} />
          <div style={{ fontSize: 14, fontWeight: 800, color: strat.color, fontFamily: "monospace", minWidth: 30 }}>{epoch}</div>
          <button onClick={function() { setEpoch(0); setAutoP(!autoP); }} style={{
            padding: "4px 10px", borderRadius: 6, border: "1px solid " + (autoP ? C.yellow : C.border),
            background: autoP ? C.yellow + "20" : C.card, color: autoP ? C.yellow : C.muted,
            cursor: "pointer", fontSize: 9, fontFamily: "monospace"
          }}>{autoP ? PAUSE : PLAY + " Animate"}</button>
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: 30 }}>
          <StatBox label="TASK ACCURACY" value={taskV.toFixed(0) + "%"} color={C.green} />
          <StatBox label="GENERAL KNOWLEDGE" value={genV.toFixed(0) + "%"} color={genV < 50 ? C.red : genV < 70 ? C.orange : C.blue} />
          <StatBox label="STRATEGY" value={strat.name} color={strat.color} bigFont={11} minW={120} />
        </div>
        <div style={{ marginTop: 10, fontSize: 10, color: C.muted, lineHeight: 1.7, fontStyle: "italic", borderLeft: "3px solid " + strat.color + "40", paddingLeft: 10 }}>{strat.desc}</div>
      </Card>

      {/* When it makes sense */}
      <Card style={{ maxWidth: 1100, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{"When Full Fine-Tuning Makes Sense vs. When It Doesn" + "'" + "t"}</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 250, padding: "10px 14px", borderRadius: 8, background: C.green + "06", border: "1px solid " + C.green + "20" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.green, marginBottom: 6 }}>{CHK + " DO use Full FT when:"}</div>
            {["Enough data (10K+ examples)", "Enough compute (multi-GPU)", "Task is very different from pre-training", "Maximum quality is critical", "Large org with infrastructure"].map(function(t, i) {
              return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + CHK + " " + t}</div>);
            })}
          </div>
          <div style={{ flex: 1, minWidth: 250, padding: "10px 14px", borderRadius: 8, background: C.red + "06", border: "1px solid " + C.red + "20" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.red, marginBottom: 6 }}>{WARN + " DON" + "'" + "T use Full FT when:"}</div>
            {["Limited compute budget " + ARR + " use PEFT", "Small dataset (<1K) " + ARR + " overfitting risk", "Multiple tasks " + ARR + " LoRA adapters are modular", "Quick experimentation " + ARR + " too slow", "Budget constrained " + ARR + " QLoRA on single GPU"].map(function(t, i) {
              return (<div key={i} style={{ fontSize: 9, color: C.muted, lineHeight: 1.8 }}>{"  " + WARN + " " + t}</div>);
            })}
          </div>
        </div>
      </Card>

      <Insight icon={BRAIN} title="The Core Tension">
        Catastrophic forgetting is a <span style={{color:C.red,fontWeight:700}}>fundamental trade-off</span> of full fine-tuning. New task-specific gradients can <span style={{color:C.red}}>overwrite patterns from pre-training</span>. Best mitigations: <span style={{color:C.yellow,fontWeight:700}}>small LR</span> + <span style={{color:C.blue,fontWeight:700}}>early stopping</span> + <span style={{color:C.green,fontWeight:700}}>data mixing</span>. Or skip the problem entirely with <span style={{color:C.purple,fontWeight:700}}>PEFT/LoRA</span> (base model frozen).
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Big Picture", "Training Loop", "Memory & Cost", "Data Pipeline", "Variants & Spectrum", "Catastrophic Forgetting"];
  return (
    <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 1400, margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.yellow + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>Full Fine-Tuning</div>
        <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " from concept to catastrophic forgetting"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab === 0 && <TabBigPicture />}
      {tab === 1 && <TabTrainingLoop />}
      {tab === 2 && <TabMemory />}
      {tab === 3 && <TabDataPipeline />}
      {tab === 4 && <TabVariants />}
      {tab === 5 && <TabForgetting />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

FFT_VISUAL_HEIGHT = 1600