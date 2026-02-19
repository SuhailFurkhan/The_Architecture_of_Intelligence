"""
Self-contained HTML for an interactive "Perceptron → MLP → CNN → RNN → Transformer" progression.
Includes quiz checkpoints between tabs (unlock next tab when quiz is correct).

Embed in Streamlit via:
    import streamlit as st
    import streamlit.components.v1 as components
    from nn_evolution_visual import NN_EVOLUTION_VISUAL_HTML, NN_EVOLUTION_VISUAL_HEIGHT
    components.html(NN_EVOLUTION_VISUAL_HTML, height=NN_EVOLUTION_VISUAL_HEIGHT)
"""

NN_EVOLUTION_VISUAL_HEIGHT = 1250

NN_EVOLUTION_VISUAL_HTML = r"""
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
  @keyframes slideIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes pulse { 0%,100%{opacity:0.65} 50%{opacity:1} }
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
  accent: "#ff6b35", blue: "#4ecdc4", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
};

var MUL = "\u00D7";
var ARR = "\u2192";
var DASH = "\u2014";
var CHK = "\u2713";
var LOCK = "\uD83D\uDD12";
var BULB = "\uD83D\uDCA1";
var TARG = "\uD83C\uDFAF";

/* ─────────────────────────────────────────────────────────────
   Shared UI bits
   ───────────────────────────────────────────────────────────── */

function Card(props) {
  return (
    <div style={{
      background: C.card,
      border: "1.5px solid " + C.border,
      borderRadius: 14,
      padding: 16,
      boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
      animation: "slideIn 0.35s ease-out",
      ...props.style
    }}>
      {props.children}
    </div>
  );
}

function Pill(props) {
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:6,
      padding:"6px 10px", borderRadius:999,
      border:"1px solid " + (props.border || C.border),
      background: (props.bg || (C.border + "30")),
      color: props.color || C.text,
      fontSize: 11, fontWeight: 700,
      fontFamily: "'JetBrains Mono','SF Mono',monospace"
    }}>
      {props.children}
    </span>
  );
}

function SectionTitle(props) {
  return (
    <div style={{marginBottom:12}}>
      <div style={{
        fontSize: 18, fontWeight: 900,
        fontFamily: "'JetBrains Mono','SF Mono',monospace",
        letterSpacing: "0.01em"
      }}>
        {props.title}
      </div>
      {props.subtitle && (
        <div style={{fontSize: 12, color: C.muted, marginTop: 4, lineHeight: 1.35}}>
          {props.subtitle}
        </div>
      )}
    </div>
  );
}

function TabBar(props) {
  var tabs = props.tabs, active = props.active, onChange = props.onChange, unlocked = props.unlocked;
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 18, overflowX: "auto" }}>
      {tabs.map(function(t, i) {
        var ok = unlocked[i];
        var isActive = active === i;
        return (
          <button key={i}
            onClick={function() { if(ok) onChange(i); }}
            title={ok ? "" : "Complete the quiz to unlock"}
            style={{
              padding: "12px 16px",
              background: "none",
              border: "none",
              borderBottom: isActive ? "2px solid " + C.accent : "2px solid transparent",
              color: isActive ? C.accent : (ok ? C.text : C.muted),
              cursor: ok ? "pointer" : "not-allowed",
              fontSize: 12,
              fontWeight: 800,
              fontFamily: "'JetBrains Mono','SF Mono',monospace",
              opacity: ok ? 1 : 0.65,
              whiteSpace: "nowrap"
            }}>
            {ok ? t : (LOCK + " " + t)}
          </button>
        );
      })}
    </div>
  );
}

function InfoRow(props) {
  return (
    <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center", marginBottom: 10}}>
      <Pill border={props.color + "55"} bg={props.color + "18"} color={props.color}>
        {BULB} {props.kicker}
      </Pill>
      <div style={{color:C.muted, fontSize:12, lineHeight:1.35}}>
        {props.text}
      </div>
    </div>
  );
}

function Divider() {
  return <div style={{height:1, background:C.border, margin:"14px 0"}} />;
}

/* ─────────────────────────────────────────────────────────────
   Quiz System (unlock next tab)
   ───────────────────────────────────────────────────────────── */

function QuizCard(props) {
  var q = props.q;
  var solved = props.solved;
  var setSolved = props.setSolved;

  var _a = useState(null); var choice = _a[0]; var setChoice = _a[1];
  var _f = useState(null); var feedback = _f[0]; var setFeedback = _f[1];

  useEffect(function(){
    // reset when question changes
    setChoice(null);
    setFeedback(null);
  }, [q && q.id]);

  function submit() {
    if (choice == null) {
      setFeedback({ ok:false, msg:"Pick an option first." });
      return;
    }
    var ok = choice === q.correct;
    setFeedback({ ok: ok, msg: ok ? ("Correct " + CHK + " — " + q.explain_ok) : ("Not quite — " + q.explain_no) });
    if (ok) setSolved(true);
  }

  return (
    <Card style={{marginTop:16}}>
      <SectionTitle title={"Quiz Checkpoint " + TARG} subtitle="Answer correctly to unlock the next tab." />
      <div style={{fontSize:13, fontWeight:800, marginBottom:10}}>{q.prompt}</div>

      <div style={{display:"grid", gap:8}}>
        {q.options.map(function(opt, idx){
          var selected = choice === idx;
          var disabled = solved;
          return (
            <button key={idx}
              onClick={function(){ if(!disabled) setChoice(idx); }}
              style={{
                textAlign:"left",
                padding:"10px 12px",
                borderRadius:10,
                border:"1.5px solid " + (selected ? C.accent : C.border),
                background: selected ? (C.accent + "18") : "#0b0b12",
                color: selected ? C.accent : C.text,
                cursor: disabled ? "not-allowed" : "pointer",
                fontFamily:"'JetBrains Mono','SF Mono',monospace",
                fontSize: 12,
                fontWeight: 700,
                opacity: disabled ? 0.85 : 1
              }}>
              {String.fromCharCode(65 + idx) + ". "} {opt}
            </button>
          );
        })}
      </div>

      <div style={{display:"flex", gap:10, alignItems:"center", marginTop:12, flexWrap:"wrap"}}>
        <button onClick={submit}
          style={{
            padding:"10px 12px",
            borderRadius:10,
            border:"1.5px solid " + (solved ? (C.green+"80") : C.border),
            background: solved ? (C.green+"18") : C.card,
            color: solved ? C.green : C.text,
            cursor: solved ? "not-allowed" : "pointer",
            fontFamily:"'JetBrains Mono','SF Mono',monospace",
            fontSize: 12,
            fontWeight: 800
          }}>
          {solved ? (CHK + " Unlocked") : "Check Answer"}
        </button>

        {feedback && (
          <div style={{
            flex:1,
            padding:"10px 12px",
            borderRadius:10,
            border:"1.5px solid " + (feedback.ok ? (C.green+"70") : (C.red+"70")),
            background: (feedback.ok ? C.green : C.red) + "10",
            color: feedback.ok ? C.green : C.red,
            fontSize: 12,
            fontWeight: 800,
            lineHeight: 1.35
          }}>
            {feedback.msg}
          </div>
        )}
      </div>
    </Card>
  );
}

/* ─────────────────────────────────────────────────────────────
   Diagrams (SVG)
   ───────────────────────────────────────────────────────────── */

function SvgBox(props) {
  return (
    <svg width="100%" height={props.h || 320} viewBox={props.viewBox || "0 0 860 320"}
      style={{ background:"#08080d", borderRadius: 12, border: "1px solid " + C.border }}>
      {props.children}
    </svg>
  );
}

function Node(props) {
  var r = props.r || 18;
  var fill = (props.fill || "#0b0b12");
  var stroke = props.stroke || C.border;
  var label = props.label || "";
  return (
    <g>
      <circle cx={props.x} cy={props.y} r={r} fill={fill} stroke={stroke} strokeWidth={props.sw || 2}/>
      <text x={props.x} y={props.y + 4} textAnchor="middle"
        fill={props.textColor || C.text}
        fontSize={props.fs || 11}
        fontWeight={props.fw || 900}
        fontFamily="'JetBrains Mono','SF Mono',monospace">
        {label}
      </text>
    </g>
  );
}

function Arrow(props) {
  return (
    <line x1={props.x1} y1={props.y1} x2={props.x2} y2={props.y2}
      stroke={props.color || C.muted}
      strokeWidth={props.w || 2}
      strokeDasharray={props.dash || "0"}
      markerEnd="url(#arrowHead)"
      opacity={props.op == null ? 1 : props.op} />
  );
}

function Defs() {
  return (
    <defs>
      <marker id="arrowHead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill={C.muted} />
      </marker>
    </defs>
  );
}

/* ─────────────────────────────────────────────────────────────
   Tabs content
   ───────────────────────────────────────────────────────────── */

function TabPerceptron() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="1) Single Perceptron"
          subtitle="One neuron = dot product + bias + activation. Great for linear patterns, fails on XOR."
        />
        <InfoRow color={C.cyan} kicker="Core Operation"
          text={"Compute z = w·x + b, then apply an activation (step/sigmoid/ReLU) to get the output."} />
        <SvgBox h={320} viewBox="0 0 860 320">
          <Defs/>
          <text x="24" y="28" fill={C.cyan} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"z = w·x + b  " + ARR + "  a = act(z)"}
          </text>

          <Node x={120} y={90} label="x₁" r={18} fill="#0b0b12" stroke={C.border} textColor={C.text} />
          <Node x={120} y={150} label="x₂" r={18} fill="#0b0b12" stroke={C.border} textColor={C.text} />
          <Node x={120} y={210} label="x₃" r={18} fill="#0b0b12" stroke={C.border} textColor={C.text} />

          <Node x={360} y={150} label="Σ" r={26} fill={C.cyan+"10"} stroke={C.cyan} textColor={C.cyan} fs={16} />
          <Node x={520} y={150} label="act" r={28} fill={C.accent+"10"} stroke={C.accent} textColor={C.accent} fs={12} />
          <Node x={700} y={150} label="y" r={22} fill={C.green+"10"} stroke={C.green} textColor={C.green} fs={14} />

          <Arrow x1={140} y1={90} x2={330} y2={140} color={C.muted} />
          <Arrow x1={140} y1={150} x2={330} y2={150} color={C.muted} />
          <Arrow x1={140} y1={210} x2={330} y2={160} color={C.muted} />

          <Arrow x1={388} y1={150} x2={492} y2={150} color={C.muted} />
          <Arrow x1={548} y1={150} x2={678} y2={150} color={C.muted} />

          <text x="250" y="92" fill={C.muted} fontSize="10" fontFamily="monospace">weights w</text>
          <text x="410" y="120" fill={C.muted} fontSize="10" fontFamily="monospace">+ bias b</text>
          <text x="595" y="120" fill={C.muted} fontSize="10" fontFamily="monospace">activation</text>

          <g>
            <rect x="24" y="250" width="812" height="52" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="273" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"Limitation: only linear decision boundaries " + DASH + " can't solve XOR with a single line."}
            </text>
            <text x="40" y="292" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: add hidden layers (MLP) to compose non-linear features."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabMLP() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="2) Multi-Layer Perceptron (MLP)"
          subtitle="Hidden layers + non-linear activations let you solve XOR and other non-linear problems."
        />
        <InfoRow color={C.purple} kicker="Key Upgrade"
          text={"Add hidden neurons: output becomes a composition of non-linear functions. This creates curved / piecewise boundaries."} />

        <SvgBox h={330} viewBox="0 0 860 330">
          <Defs/>
          <text x="24" y="28" fill={C.purple} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"Non-linear composition: x " + ARR + " hidden features " + ARR + " y"}
          </text>

          {/* Inputs */}
          <Node x={90} y={90} label="x₁" r={18} fill="#0b0b12" stroke={C.border} />
          <Node x={90} y={160} label="x₂" r={18} fill="#0b0b12" stroke={C.border} />
          <Node x={90} y={230} label="x₃" r={18} fill="#0b0b12" stroke={C.border} />

          {/* Hidden */}
          <Node x={320} y={110} label="h₁" r={20} fill={C.purple+"12"} stroke={C.purple} textColor={C.purple} />
          <Node x={320} y={170} label="h₂" r={20} fill={C.purple+"12"} stroke={C.purple} textColor={C.purple} />
          <Node x={320} y={230} label="h₃" r={20} fill={C.purple+"12"} stroke={C.purple} textColor={C.purple} />

          {/* Output */}
          <Node x={560} y={170} label="act" r={28} fill={C.accent+"10"} stroke={C.accent} textColor={C.accent} fs={12} />
          <Node x={730} y={170} label="y" r={22} fill={C.green+"10"} stroke={C.green} textColor={C.green} fs={14} />

          {/* Connections input->hidden */}
          {[
            [90,90,320,110],[90,90,320,170],[90,90,320,230],
            [90,160,320,110],[90,160,320,170],[90,160,320,230],
            [90,230,320,110],[90,230,320,170],[90,230,320,230],
          ].map(function(p,i){
            return <line key={i} x1={p[0]+18} y1={p[1]} x2={p[2]-20} y2={p[3]}
              stroke={C.muted} strokeWidth="1.5" opacity="0.7" />;
          })}

          {/* hidden->output */}
          {[
            [320,110,560,170],[320,170,560,170],[320,230,560,170],
          ].map(function(p,i){
            return <line key={i} x1={p[0]+20} y1={p[1]} x2={p[2]-28} y2={p[3]}
              stroke={C.muted} strokeWidth="2" opacity="0.85" />;
          })}

          <Arrow x1={588} y1={170} x2={708} y2={170} color={C.muted} />

          <g>
            <rect x="24" y="260" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="283" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"MLP solves XOR, but scales poorly for images: too many parameters + no spatial inductive bias."}
            </text>
            <text x="40" y="302" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: use local connectivity + weight sharing (CNN)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabCNN() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="3) CNN (Convolutional Neural Network)"
          subtitle="For images/grids: local receptive fields + weight sharing. Detects edges → textures → shapes."
        />
        <InfoRow color={C.accent} kicker="Inductive Bias"
          text={"A small filter (kernel) slides across space: same weights reused everywhere, capturing local spatial patterns efficiently."} />

        <SvgBox h={340} viewBox="0 0 860 340">
          <Defs/>
          <text x="24" y="28" fill={C.accent} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"(kernel) " + ARR + " feature maps " + ARR + " pooling/downsample " + ARR + " head"}
          </text>

          {/* Image grid */}
          <g transform="translate(36,60)">
            <text x="0" y="-10" fill={C.blue} fontSize="10" fontWeight="800" fontFamily="monospace">INPUT IMAGE</text>
            {Array.from({length: 6}).map(function(_, r){
              return Array.from({length: 6}).map(function(_, c){
                var v = (r===c || r+c===5) ? 1 : 0;
                var col = v ? (C.blue+"aa") : (C.border+"55");
                return <rect key={r+"-"+c} x={c*22} y={r*22} width="20" height="20" rx="4"
                  fill={col} stroke={C.border} />;
              });
            })}
            {/* Kernel overlay */}
            <rect x={22} y={22} width={22*3-2} height={22*3-2} rx="8"
              fill={C.accent+"12"} stroke={C.accent} strokeWidth="2"
              style={{animation:"pulse 1.8s ease-in-out infinite"}} />
            <text x={22} y={22*6+22} fill={C.muted} fontSize="10" fontFamily="monospace">3×3 kernel slides</text>
          </g>

          {/* Arrows */}
          <Arrow x1={220} y1={160} x2={320} y2={160} color={C.muted} />

          {/* Feature map */}
          <g transform="translate(340,76)">
            <text x="0" y="-10" fill={C.cyan} fontSize="10" fontWeight="800" fontFamily="monospace">FEATURE MAP</text>
            {Array.from({length: 4}).map(function(_, r){
              return Array.from({length: 4}).map(function(_, c){
                var hot = (r===1 && c===1) || (r===2 && c===2);
                return <rect key={r+"-"+c} x={c*26} y={r*26} width="24" height="24" rx="5"
                  fill={hot ? (C.cyan+"b0") : (C.border+"55")} stroke={C.border} />;
              });
            })}
            <text x="0" y={4*26+24} fill={C.muted} fontSize="10" fontFamily="monospace">high = pattern found</text>
          </g>

          <Arrow x1={470} y1={160} x2={560} y2={160} color={C.muted} />

          {/* Pooling */}
          <g transform="translate(580,108)">
            <text x="0" y="-10" fill={C.yellow} fontSize="10" fontWeight="800" fontFamily="monospace">POOL/DOWNSAMPLE</text>
            {Array.from({length: 2}).map(function(_, r){
              return Array.from({length: 2}).map(function(_, c){
                return <rect key={r+"-"+c} x={c*34} y={r*34} width="32" height="32" rx="7"
                  fill={(r===0 && c===0) ? (C.yellow+"b0") : (C.border+"55")} stroke={C.border} />;
              });
            })}
            <text x="0" y={2*34+30} fill={C.muted} fontSize="10" fontFamily="monospace">smaller, robust</text>
          </g>

          <Arrow x1={680} y1={160} x2={760} y2={160} color={C.muted} />
          <Node x={805} y={160} label="ŷ" r={22} fill={C.green+"10"} stroke={C.green} textColor={C.green} fs={14} />

          <g>
            <rect x="24" y="274" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="297" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"CNNs are great for space, but not for time/order (language, audio)."}
            </text>
            <text x="40" y="316" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: add recurrence / memory across steps (RNN)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabRNN() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="4) RNN (Recurrent Neural Network)"
          subtitle="For sequences: reuse the same weights at each time step and carry a hidden state (memory)."
        />
        <InfoRow color={C.pink} kicker="Core Idea"
          text={"Hidden state hₜ summarizes the past. Process tokens one-by-one: (xₜ, hₜ₋₁) → hₜ → output."} />

        <SvgBox h={350} viewBox="0 0 860 350">
          <Defs/>
          <text x="24" y="28" fill={C.pink} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"x₁ " + ARR + " h₁ " + ARR + " x₂ " + ARR + " h₂ " + ARR + " x₃ " + ARR + " h₃ ..."}
          </text>

          {/* Time steps */}
          function cell(x, y, label, color) { return (
            <g>
              <rect x={x-34} y={y-20} width="68" height="40" rx="10" fill={color+"14"} stroke={color} strokeWidth="2"/>
              <text x={x} y={y+4} fill={color} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">{label}</text>
            </g>
          ); }

          {/* tokens */}
          <g>
            {/* x boxes */}
            <rect x="70" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="115" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₁</text>

            <rect x="250" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="295" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₂</text>

            <rect x="430" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="475" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₃</text>

            <rect x="610" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="655" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₄</text>
          </g>

          {/* RNN cells */}
          <g>
            <g transform="translate(115,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
            <g transform="translate(295,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
            <g transform="translate(475,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
            <g transform="translate(655,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
          </g>

          {/* x -> cell arrows */}
          <Arrow x1={115} y1={112} x2={115} y2={142} color={C.muted} />
          <Arrow x1={295} y1={112} x2={295} y2={142} color={C.muted} />
          <Arrow x1={475} y1={112} x2={475} y2={142} color={C.muted} />
          <Arrow x1={655} y1={112} x2={655} y2={142} color={C.muted} />

          {/* hidden state arrows (recurrent) */}
          <path d="M141 170 C 185 170, 225 170, 269 170" fill="none" stroke={C.cyan} strokeWidth="3" markerEnd="url(#arrowHead)" opacity="0.95"/>
          <path d="M321 170 C 365 170, 405 170, 449 170" fill="none" stroke={C.cyan} strokeWidth="3" markerEnd="url(#arrowHead)" opacity="0.95"/>
          <path d="M501 170 C 545 170, 585 170, 629 170" fill="none" stroke={C.cyan} strokeWidth="3" markerEnd="url(#arrowHead)" opacity="0.95"/>
          <text x="208" y="156" fill={C.cyan} fontSize="10" fontWeight="900" fontFamily="monospace">h₁</text>
          <text x="388" y="156" fill={C.cyan} fontSize="10" fontWeight="900" fontFamily="monospace">h₂</text>
          <text x="568" y="156" fill={C.cyan} fontSize="10" fontWeight="900" fontFamily="monospace">h₃</text>

          {/* outputs */}
          <g>
            <rect x="70" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="115" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₁</text>
            <rect x="250" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="295" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₂</text>
            <rect x="430" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="475" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₃</text>
            <rect x="610" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="655" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₄</text>
          </g>

          <Arrow x1={115} y1={196} x2={115} y2={230} color={C.muted} />
          <Arrow x1={295} y1={196} x2={295} y2={230} color={C.muted} />
          <Arrow x1={475} y1={196} x2={475} y2={230} color={C.muted} />
          <Arrow x1={655} y1={196} x2={655} y2={230} color={C.muted} />

          <g>
            <rect x="24" y="286" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="309" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"RNNs struggle with very long sequences (vanishing gradients, slow sequential processing)."}
            </text>
            <text x="40" y="328" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: replace recurrence with self-attention (Transformer)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabTransformer() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="5) Transformer"
          subtitle="Processes all tokens in parallel. Self-attention lets every token attend to every other token."
        />
        <InfoRow color={C.yellow} kicker="Key Mechanism"
          text={"Self-attention computes weighted mixes of tokens using Q/K/V. This captures long-range dependencies without recurrence."} />

        <SvgBox h={420} viewBox="0 0 860 420">
          <Defs/>
          <text x="24" y="28" fill={C.yellow} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"Tokens " + ARR + " Q,K,V " + ARR + " Attention matrix " + ARR + " context vectors"}
          </text>

          {/* Token row */}
          <g transform="translate(60,70)">
            {["x₁","x₂","x₃","x₄","x₅","x₆"].map(function(t, i){
              var x = i*120;
              return (
                <g key={i}>
                  <rect x={x} y="0" width="88" height="40" rx="10" fill={C.border+"25"} stroke={C.border}/>
                  <text x={x+44} y="26" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">{t}</text>
                </g>
              );
            })}
            <text x="0" y="-10" fill={C.blue} fontSize="10" fontWeight="800" fontFamily="monospace">INPUT TOKENS</text>
          </g>

          {/* Attention matrix block */}
          <g transform="translate(270,140)">
            <rect x="0" y="0" width="320" height="210" rx="14" fill={C.yellow+"0f"} stroke={C.yellow+"aa"} strokeWidth="2"/>
            <text x="16" y="26" fill={C.yellow} fontSize="11" fontWeight="900" fontFamily="monospace">SELF-ATTENTION</text>
            <text x="16" y="46" fill={C.muted} fontSize="10" fontFamily="monospace">softmax(QKᵀ/√d) · V</text>

            {/* Grid */}
            {Array.from({length: 6}).map(function(_, r){
              return Array.from({length: 6}).map(function(_, c){
                var hot = (r===1 && c===4) || (r===2 && c===0) || (r===4 && c===2) || (r===0 && c===0) || (r===5 && c===5);
                var fill = hot ? (C.yellow+"b0") : (C.border+"55");
                return <rect key={r+"-"+c} x={18 + c*46} y={70 + r*22} width="40" height="18" rx="4"
                  fill={fill} stroke={C.border} />;
              });
            })}
            <text x="16" y="200" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"each row = where token attends"}
            </text>
          </g>

          {/* attention lines from tokens to matrix */}
          {[
            [104,110,270,160],[224,110,270,180],[344,110,270,200],
            [464,110,270,220],[584,110,270,240],[704,110,270,260],
          ].map(function(p,i){
            return <line key={i} x1={p[0]} y1={p[1]} x2={p[2]} y2={p[3]}
              stroke={C.muted} strokeWidth="1.5" opacity="0.6" />;
          })}

          {/* output context vectors */}
          <g transform="translate(640,164)">
            <text x="0" y="-10" fill={C.green} fontSize="10" fontWeight="800" fontFamily="monospace">CONTEXT VECTORS</text>
            {["c₁","c₂","c₃","c₄","c₅","c₆"].map(function(t, i){
              return (
                <g key={i}>
                  <rect x="0" y={i*34} width="110" height="28" rx="9" fill={C.green+"10"} stroke={C.green+"70"}/>
                  <text x="55" y={i*34+19} fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">{t}</text>
                </g>
              );
            })}
          </g>

          <Arrow x1={590} y1={240} x2={640} y2={240} color={C.muted} />

          {/* Bottom note */}
          <g>
            <rect x="24" y="358" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="381" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"Transformer replaces recurrence with attention " + DASH + " better long-range context + parallel compute."}
            </text>
            <text x="40" y="400" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"This is the backbone of modern LLMs (GPT/BERT/T5 families)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabSummary() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="6) Summary: The Progression"
          subtitle="Each model adds an inductive bias that matches the structure of the data."
        />

        <div style={{display:"grid", gridTemplateColumns:"repeat(1, 1fr)", gap:12}}>
          <Card style={{background:"#0b0b12"}}>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.cyan+"55"} bg={C.cyan+"18"} color={C.cyan}>Perceptron</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Linear decision boundary (dot product + activation)
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.purple+"55"} bg={C.purple+"18"} color={C.purple}>MLP</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Non-linear composition via hidden layers (solves XOR)
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.accent+"55"} bg={C.accent+"18"} color={C.accent}>CNN</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Spatial inductive bias: locality + weight sharing
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.pink+"55"} bg={C.pink+"18"} color={C.pink}>RNN</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Temporal memory via hidden state across time
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.yellow+"55"} bg={C.yellow+"18"} color={C.yellow}>Transformer</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Global context via self-attention (parallel, long-range)
              </div>
            </div>
          </Card>

          <Card style={{background:"#0b0b12"}}>
            <SectionTitle title="One-line takeaway" subtitle="" />
            <div style={{fontFamily:"'JetBrains Mono','SF Mono',monospace", fontSize:12, lineHeight:1.5, color:C.text}}>
              {"Perceptron " + ARR + " MLP " + ARR + " CNN " + ARR + " RNN " + ARR + " Transformer"}
              <div style={{marginTop:10, color:C.muted}}>
                {"Linear " + ARR + " Non-linear " + ARR + " Spatial " + ARR + " Sequential " + ARR + " Global Attention"}
              </div>
            </div>
          </Card>
        </div>
      </Card>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   Quizzes (unlock next tab)
   ───────────────────────────────────────────────────────────── */

var QUIZZES = [
  {
    id: "q0",
    prompt: "Why can a single perceptron NOT solve XOR?",
    options: [
      "Because it has too many parameters",
      "Because it only forms a linear decision boundary",
      "Because it cannot multiply inputs",
      "Because it cannot use bias"
    ],
    correct: 1,
    explain_ok: "XOR is not linearly separable — you need hidden layers / non-linear composition.",
    explain_no: "Think geometry: one line can’t separate XOR’s corners."
  },
  {
    id: "q1",
    prompt: "What’s the key feature that lets an MLP solve XOR?",
    options: [
      "Pooling layers",
      "Self-attention",
      "Hidden layer(s) with non-linear activation",
      "Weight sharing across space"
    ],
    correct: 2,
    explain_ok: "Hidden + non-linearity creates feature composition, enabling non-linear boundaries.",
    explain_no: "MLPs win by adding hidden representations + activation."
  },
  {
    id: "q2",
    prompt: "What makes CNNs efficient for images?",
    options: [
      "They use recurrence across time",
      "Local receptive fields + weight sharing (same filter reused across space)",
      "They avoid activations",
      "They require fixed-length sequences"
    ],
    correct: 1,
    explain_ok: "CNN filters scan local patches and reuse the same weights everywhere.",
    explain_no: "Think: small kernel sliding with shared parameters."
  },
  {
    id: "q3",
    prompt: "What is the RNN’s 'memory' mechanism?",
    options: [
      "Pooling window",
      "Hidden state carried across time steps",
      "Convolution kernel",
      "Positional encoding"
    ],
    correct: 1,
    explain_ok: "RNNs carry a hidden state hₜ that summarizes prior steps.",
    explain_no: "RNNs remember via hidden state, not via filters or pooling."
  },
  {
    id: "q4",
    prompt: "What replaces recurrence in Transformers?",
    options: [
      "A larger convolution kernel",
      "A deeper MLP head",
      "Self-attention over all tokens (parallel)",
      "Max pooling over time"
    ],
    correct: 2,
    explain_ok: "Self-attention lets each token use information from all others without sequential loops.",
    explain_no: "Transformers use attention, not recurrent hidden states."
  }
];

/* ─────────────────────────────────────────────────────────────
   Root App
   ───────────────────────────────────────────────────────────── */

function App() {
  var tabs = ["Perceptron", "MLP", "CNN", "RNN", "Transformer", "Summary"];

  // unlocked[i] indicates whether tab i is accessible
  // tab 0 starts unlocked; each quiz unlocks the next tab
  var _u = useState([true, false, false, false, false, false]);
  var unlocked = _u[0];
  var setUnlocked = _u[1];

  var _t = useState(0);
  var tab = _t[0];
  var setTab = _t[1];

  // quizSolved[k] corresponds to quiz between tab k and k+1 (k=0..4)
  var _qs = useState([false, false, false, false, false]);
  var quizSolved = _qs[0];
  var setQuizSolved = _qs[1];

  function markSolved(k) {
    var nextSolved = quizSolved.slice();
    nextSolved[k] = true;
    setQuizSolved(nextSolved);

    var nextUnlocked = unlocked.slice();
    nextUnlocked[k+1] = true; // unlock next tab
    setUnlocked(nextUnlocked);
  }

  // Convenience: find quiz index for current tab (if any)
  var quizIndex = tab; // quiz after tab 0..4
  var showQuiz = tab >= 0 && tab <= 4; // summary has no quiz

  return (
    <div style={{
      background: C.bg,
      minHeight:"100vh",
      padding:"24px 16px",
      fontFamily:"'JetBrains Mono','SF Mono',monospace",
      color:C.text,
      maxWidth: 980,
      margin:"0 auto"
    }}>
      <div style={{textAlign:"center", marginBottom: 14}}>
        <div style={{
          fontSize: 22,
          fontWeight: 900,
          background: "linear-gradient(135deg," + C.accent + "," + C.yellow + ")",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          display:"inline-block"
        }}>
          Neural Network Evolution
        </div>
        <div style={{fontSize: 11, color: C.muted, marginTop: 4}}>
          {"Interactive progression " + DASH + " Perceptron → MLP → CNN → RNN → Transformer (with quizzes)"}
        </div>
      </div>

      <TabBar tabs={tabs} active={tab} onChange={setTab} unlocked={unlocked} />

      {tab===0 && <TabPerceptron />}
      {tab===1 && <TabMLP />}
      {tab===2 && <TabCNN />}
      {tab===3 && <TabRNN />}
      {tab===4 && <TabTransformer />}
      {tab===5 && <TabSummary />}

      {showQuiz && (
        <QuizCard
          q={QUIZZES[quizIndex]}
          solved={quizSolved[quizIndex]}
          setSolved={function(v){ if(v) markSolved(quizIndex); }}
        />
      )}

      <div style={{marginTop: 16, color: C.muted, fontSize: 10, textAlign:"center"}}>
        {"Tip: quizzes unlock the next tab. If you want free navigation, remove the unlock logic in App()."}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
</body>
</html>
"""
