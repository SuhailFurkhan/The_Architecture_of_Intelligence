"""
Self-contained HTML for the Transformer interactive walkthrough.
Covers: Architecture Overview, Q/K/V Computation, Single-Head Attention,
Multi-Head Attention, and Full Transformer Block.
Embed in Streamlit via st.components.v1.html(TRANSFORMER_VISUAL_HTML, height=TRANSFORMER_VISUAL_HEIGHT).
"""

TRANSFORMER_VISUAL_HTML = """
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
  @keyframes flowRight { 0%{stroke-dashoffset:20} 100%{stroke-dashoffset:0} }
  @keyframes fadeIn { 0%{opacity:0;transform:translateY(6px)} 100%{opacity:1;transform:translateY(0)} }
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
  q: "#38bdf8", k: "#f472b6", v: "#4ade80",
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
var SQRT = "\u221A";

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
      maxWidth: 750, margin: "16px auto 0",
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

function fmt(v) { return (Math.round(v * 1000) / 1000).toString(); }
function fmt2(v) { return (Math.round(v * 100) / 100).toString(); }


/* ===============================================================
   TAB 1: ARCHITECTURE OVERVIEW
   =============================================================== */
function TabArchitecture() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];

  var steps = [
    { title: "Input Embedding + Positional Encoding", desc: "Each word is converted to a dense vector (embedding). Since Transformers process ALL words in parallel (no recurrence!), we add positional encoding so the model knows word order. The result is a matrix where each row represents a word with its position baked in.", color: C.blue },
    { title: "Multi-Head Self-Attention", desc: "Every word looks at every other word to figure out what's relevant. The input is projected into Queries, Keys, and Values. Attention scores are computed, and then used to create context-aware representations. Multiple heads let the model attend to different aspects simultaneously.", color: C.purple },
    { title: "Add & Normalize (Residual Connection)", desc: "The attention output is ADDED back to the original input (residual/skip connection). This is then layer-normalized. Residual connections help gradients flow and allow the model to learn incremental refinements rather than full transformations at each layer.", color: C.green },
    { title: "Feed-Forward Network (FFN)", desc: "Each position passes through the same 2-layer MLP independently: FFN(x) = ReLU(xW\u2081 + b\u2081)W\u2082 + b\u2082. This is where the model does its "+LQ+"thinking"+RQ+" "+DASH+" transforming features, combining patterns, and building higher-level representations.", color: C.yellow },
    { title: "Add & Normalize Again", desc: "Another residual connection + layer norm after the FFN. The output now contains rich, context-aware representations for every position. Stack N of these blocks (GPT-3 uses 96!) and you get deep understanding.", color: C.accent },
  ];

  var bw = 740, bh = 520;
  var blockX = 270, blockW = 200;
  /* vertical positions for each component */
  var yEmb = 460, yAttn = 365, yAN1 = 295, yFFN = 220, yAN2 = 150, yOut = 70;

  function boxStyle(idx, y, h, col) {
    var on = step === idx;
    return {
      x: blockX, y: y, width: blockW, height: h, rx: 8,
      fill: on ? col + "20" : C.card,
      stroke: on ? col : C.dim,
      strokeWidth: on ? 2.5 : 1,
      style: on ? { animation: "pulse 1.5s infinite" } : {},
    };
  }

  return (
    <div>
      <SectionTitle title="Transformer Architecture" subtitle={"One encoder/decoder block "+DASH+" click each layer to understand its role"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          var on = step === i;
          return (
            <button key={i} onClick={function() { setStep(i); }} style={{
              padding: "8px 14px", borderRadius: 8,
              border: "1.5px solid " + (on ? s.color : C.border),
              background: on ? s.color + "20" : C.card,
              color: on ? s.color : C.muted, cursor: "pointer",
              fontSize: 10, fontWeight: 700, fontFamily: "monospace",
            }}>
              {(i + 1) + ". " + s.title.split(" (")[0].split(" +")[0]}
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={bw} height={bh} viewBox={"0 0 " + bw + " " + bh} style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Title */}
          <text x={bw / 2} y={25} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">SINGLE TRANSFORMER BLOCK</text>
          <text x={bw / 2} y={42} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"(stack N"+MUL+" for full model: GPT-2=12, GPT-3=96, GPT-4=120+)"}</text>

          {/* Nx bracket */}
          <rect x={blockX - 50} y={yAN2 - 15} width={blockW + 100} height={yEmb - yAN2 + 65} rx={12} fill="none" stroke={C.accent + "30"} strokeWidth={1.5} strokeDasharray="6,4" />
          <text x={blockX - 35} y={yAN2 + 5} fill={C.accent} fontSize={11} fontWeight={800} fontFamily="monospace">{MUL+"N"}</text>

          {/* Output */}
          <rect x={blockX + 30} y={yOut} width={blockW - 60} height={34} rx={6} fill={C.accent + "10"} stroke={C.accent + "50"} strokeWidth={1} />
          <text x={bw / 2} y={yOut + 21} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">Output</text>
          <line x1={bw / 2} y1={yAN2 + 40} x2={bw / 2} y2={yOut + 34} stroke={C.dim} strokeWidth={1.5} />
          <polygon points={(bw/2)+","+(yOut+36)+" "+(bw/2-4)+","+(yOut+42)+" "+(bw/2+4)+","+(yOut+42)} fill={C.dim} />

          {/* Add & Norm 2 */}
          <rect {...boxStyle(4, yAN2, 40, C.accent)} />
          <text x={bw / 2} y={yAN2 + 18} textAnchor="middle" fill={step === 4 ? C.accent : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Add & LayerNorm</text>
          <text x={bw / 2} y={yAN2 + 33} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">residual + normalize</text>

          {/* Arrow AN2 <- FFN */}
          <line x1={bw / 2} y1={yFFN + 50} x2={bw / 2} y2={yAN2} stroke={step >= 4 ? C.yellow : C.dim} strokeWidth={1.5} />

          {/* FFN */}
          <rect {...boxStyle(3, yFFN, 50, C.yellow)} />
          <text x={bw / 2} y={yFFN + 20} textAnchor="middle" fill={step === 3 ? C.yellow : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Feed-Forward Network</text>
          <text x={bw / 2} y={yFFN + 35} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"FFN(x) = ReLU(xW"+"\u2081"+")W"+"\u2082"}</text>
          <text x={bw / 2} y={yFFN + 46} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">applied independently per position</text>

          {/* Arrow FFN <- AN1 */}
          <line x1={bw / 2} y1={yAN1 + 40} x2={bw / 2} y2={yFFN} stroke={step >= 3 ? C.green : C.dim} strokeWidth={1.5} />

          {/* Residual skip around FFN */}
          <path d={"M" + (blockX + blockW + 10) + "," + (yAN1 + 20) + " L" + (blockX + blockW + 30) + "," + (yAN1 + 20) + " L" + (blockX + blockW + 30) + "," + (yAN2 + 20) + " L" + (blockX + blockW) + "," + (yAN2 + 20)} fill="none" stroke={step >= 4 ? C.accent + "60" : C.dim + "40"} strokeWidth={1.5} strokeDasharray="4,3" />
          <text x={blockX + blockW + 35} y={(yAN1 + yAN2) / 2 + 25} fill={step >= 4 ? C.accent : C.dim} fontSize={7} fontFamily="monospace" transform={"rotate(90," + (blockX + blockW + 35) + "," + ((yAN1 + yAN2) / 2 + 25) + ")"}>skip</text>

          {/* Add & Norm 1 */}
          <rect {...boxStyle(2, yAN1, 40, C.green)} />
          <text x={bw / 2} y={yAN1 + 18} textAnchor="middle" fill={step === 2 ? C.green : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Add & LayerNorm</text>
          <text x={bw / 2} y={yAN1 + 33} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">residual + normalize</text>

          {/* Arrow AN1 <- Attention */}
          <line x1={bw / 2} y1={yAttn + 55} x2={bw / 2} y2={yAN1} stroke={step >= 2 ? C.purple : C.dim} strokeWidth={1.5} />

          {/* Residual skip around Attention */}
          <path d={"M" + (blockX + blockW + 10) + "," + (yEmb + 20) + " L" + (blockX + blockW + 50) + "," + (yEmb + 20) + " L" + (blockX + blockW + 50) + "," + (yAN1 + 20) + " L" + (blockX + blockW) + "," + (yAN1 + 20)} fill="none" stroke={step >= 2 ? C.green + "60" : C.dim + "40"} strokeWidth={1.5} strokeDasharray="4,3" />
          <text x={blockX + blockW + 55} y={(yEmb + yAN1) / 2 + 25} fill={step >= 2 ? C.green : C.dim} fontSize={7} fontFamily="monospace" transform={"rotate(90," + (blockX + blockW + 55) + "," + ((yEmb + yAN1) / 2 + 25) + ")"}>skip</text>

          {/* Multi-Head Attention */}
          <rect {...boxStyle(1, yAttn, 55, C.purple)} />
          <text x={bw / 2} y={yAttn + 20} textAnchor="middle" fill={step === 1 ? C.purple : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Multi-Head Attention</text>
          <text x={bw / 2} y={yAttn + 35} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"Q, K, V "+ARR+" Attention "+ARR+" Concat "+ARR+" Linear"}</text>
          {/* QKV labels */}
          <text x={blockX + 30} y={yAttn + 50} fill={C.q} fontSize={8} fontWeight={700} fontFamily="monospace">Q</text>
          <text x={blockX + blockW / 2} y={yAttn + 50} textAnchor="middle" fill={C.k} fontSize={8} fontWeight={700} fontFamily="monospace">K</text>
          <text x={blockX + blockW - 30} y={yAttn + 50} fill={C.v} fontSize={8} fontWeight={700} fontFamily="monospace">V</text>

          {/* Arrow Attn <- Embedding */}
          <line x1={bw / 2} y1={yEmb} x2={bw / 2} y2={yAttn + 55} stroke={step >= 1 ? C.blue : C.dim} strokeWidth={1.5} />

          {/* Input Embedding + Pos Encoding */}
          <rect {...boxStyle(0, yEmb, 45, C.blue)} />
          <text x={bw / 2} y={yEmb + 17} textAnchor="middle" fill={step === 0 ? C.blue : C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Embedding + Position</text>
          <text x={bw / 2} y={yEmb + 32} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"word vectors + sin/cos position"}</text>
          <text x={bw / 2} y={yEmb + 43} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{"parallel: ALL words at once!"}</text>

          {/* Input tokens */}
          {["The", "cat", "sat"].map(function(w, i) {
            var tx = blockX + 30 + i * 70;
            return (
              <g key={i}>
                <rect x={tx} y={yEmb + 52} width={50} height={22} rx={4} fill={C.blue + "10"} stroke={C.blue + "40"} strokeWidth={1} />
                <text x={tx + 25} y={yEmb + 67} textAnchor="middle" fill={C.blue} fontSize={9} fontWeight={700} fontFamily="monospace">{LQ + w + RQ}</text>
              </g>
            );
          })}

        </svg>
      </div>

      <Card highlight={true} style={{ maxWidth: 750, margin: "0 auto 16px", borderColor: steps[step].color }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: steps[step].color, marginBottom: 4 }}>{"Layer " + (step + 1) + ": " + steps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{steps[step].desc}</div>
      </Card>

      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.purple, marginBottom: 10 }}>RNN vs Transformer</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 20, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200, padding: 12, background: "#08080d", borderRadius: 8, border: "1px solid " + C.border }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.cyan, marginBottom: 6 }}>RNN</div>
            <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8, fontFamily: "monospace" }}>
              {"Sequential: word by word"}<br />
              {"Fixed-size hidden state bottleneck"}<br />
              {"Vanishing gradients over long sequences"}<br />
              {"Cannot parallelize training"}
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 200, padding: 12, background: "#08080d", borderRadius: 8, border: "1px solid " + C.accent + "30" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 6 }}>Transformer</div>
            <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8, fontFamily: "monospace" }}>
              {"Parallel: ALL words at once"}<br />
              {"Every word attends to every word"}<br />
              {"No vanishing gradient (direct connections)"}<br />
              {"Massively parallelizable on GPUs"}
            </div>
          </div>
        </div>
      </Card>

      <Insight>
        The Transformer processes the entire sequence <span style={{ color: C.accent, fontWeight: 700 }}>in parallel</span>. No recurrence, no sequential bottleneck. The magic is <span style={{ color: C.purple, fontWeight: 700 }}>self-attention</span>: every word can directly look at every other word, regardless of distance. This is what powers GPT, BERT, and Claude.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: Q, K, V COMPUTATION
   =============================================================== */
function TabQKV() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _a = useState(false); var autoPlay = _a[0], setAutoPlay = _a[1];

  useEffect(function() {
    if (!autoPlay) return;
    var t = setInterval(function() { setStep(function(s) { return (s + 1) % 4; }); }, 3500);
    return function() { clearInterval(t); };
  }, [autoPlay]);

  var steps = [
    { title: "Start: Input Embeddings", desc: "Each word has been converted to a vector (embedding + position). For this example, let's say our model dimension d=4. So each word is a vector of 4 numbers. The sentence "+LQ+"The cat sat"+RQ+" becomes a 3"+MUL+"4 matrix." },
    { title: "Compute Queries (Q)", desc: "Multiply the input by weight matrix W_Q (learned during training). Q = X "+MUL+" W_Q. The Query represents "+LQ+"what am I looking for?"+RQ+" For each word, Q encodes what kind of information this word wants to find in other words." },
    { title: "Compute Keys (K)", desc: "Multiply the input by weight matrix W_K. K = X "+MUL+" W_K. The Key represents "+LQ+"what do I contain?"+RQ+" For each word, K encodes a description of what information this word offers to other words." },
    { title: "Compute Values (V)", desc: "Multiply the input by weight matrix W_V. V = X "+MUL+" W_V. The Value represents "+LQ+"what information do I actually provide?"+RQ+" Once attention scores decide WHICH words to focus on, V provides the CONTENT that gets passed along." },
  ];

  var bw = 740, bh = 340;

  /* Simulated embedding values */
  var embeddings = [
    [0.2, 0.8, -0.1, 0.5],
    [0.9, -0.3, 0.7, 0.1],
    [-0.4, 0.6, 0.3, 0.8],
  ];
  var words = ["The", "cat", "sat"];

  var qColors = [C.q, C.k, C.v];
  var qLabels = ["Q", "K", "V"];
  var qDesc = ['"What am I looking for?"', '"What do I contain?"', '"What info do I provide?"'];

  return (
    <div>
      <SectionTitle title="Query, Key, Value Computation" subtitle={"How inputs are transformed into Q, K, V "+DASH+" the building blocks of attention"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          var on = step === i;
          var col = i === 0 ? C.blue : qColors[i - 1];
          return (
            <button key={i} onClick={function() { setStep(i); setAutoPlay(false); }} style={{
              padding: "8px 14px", borderRadius: 8,
              border: "1.5px solid " + (on ? col : C.border),
              background: on ? col + "20" : C.card,
              color: on ? col : C.muted, cursor: "pointer",
              fontSize: 10, fontWeight: 700, fontFamily: "monospace",
            }}>
              {i === 0 ? "Input" : qLabels[i - 1]}
            </button>
          );
        })}
        <button onClick={function() { setAutoPlay(!autoPlay); }} style={{ padding: "8px 14px", borderRadius: 8, border: "1.5px solid " + (autoPlay ? C.yellow : C.border), background: autoPlay ? C.yellow + "20" : C.card, color: autoPlay ? C.yellow : C.muted, cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>{autoPlay ? PAUSE : PLAY}</button>
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={bw} height={bh} viewBox={"0 0 " + bw + " " + bh} style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Input Matrix X */}
          <text x={80} y={30} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={700} fontFamily="monospace">Input X</text>
          <text x={80} y={45} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">(3 words {MUL} d=4)</text>

          {words.map(function(w, r) {
            var y = 60 + r * 40;
            return (
              <g key={"x" + r}>
                <text x={15} y={y + 18} fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{w}</text>
                {embeddings[r].map(function(v, c) {
                  return (
                    <g key={c}>
                      <rect x={45 + c * 35} y={y} width={32} height={26} rx={4} fill={step === 0 ? C.blue + "15" : C.card} stroke={step === 0 ? C.blue + "60" : C.dim} strokeWidth={1} />
                      <text x={61 + c * 35} y={y + 17} textAnchor="middle" fill={step === 0 ? C.blue : C.muted} fontSize={9} fontFamily="monospace">{fmt2(v)}</text>
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* Multiplication symbol */}
          <text x={200} y={105} textAnchor="middle" fill={C.muted} fontSize={20} fontWeight={700} fontFamily="monospace">{MUL}</text>

          {/* Weight Matrices */}
          {[0, 1, 2].map(function(mi) {
            var on = step === mi + 1;
            var col = qColors[mi];
            var wx = 240 + mi * 170;

            return (
              <g key={"w" + mi}>
                <text x={wx + 35} y={30} textAnchor="middle" fill={on ? col : C.dim} fontSize={11} fontWeight={700} fontFamily="monospace">{"W_" + qLabels[mi]}</text>
                <text x={wx + 35} y={45} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">(d=4 {MUL} d=4)</text>

                <rect x={wx - 32} y={56} width={134} height={120} rx={6} fill={on ? col + "08" : "transparent"} stroke={on ? col + "40" : C.dim + "30"} strokeWidth={on ? 1.5 : 1} />

                {[0, 1, 2, 3].map(function(r) {
                  return [0, 1, 2, 3].map(function(c2) {
                    var val = (Math.sin((mi + 1) * (r + 1) * (c2 + 1) * 0.7) * 0.9);
                    return (
                      <g key={r + "-" + c2}>
                        <rect x={wx - 28 + c2 * 32} y={62 + r * 28} width={28} height={22} rx={3} fill={on ? col + "10" : "transparent"} stroke={on ? col + "30" : C.dim + "20"} strokeWidth={0.5} />
                        <text x={wx - 14 + c2 * 32} y={77 + r * 28} textAnchor="middle" fill={on ? col : C.dim} fontSize={8} fontFamily="monospace">{fmt2(val)}</text>
                      </g>
                    );
                  });
                })}

                <text x={wx + 35} y={195} textAnchor="middle" fill={on ? col : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace">{qLabels[mi]}</text>
                <text x={wx + 35} y={210} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{qDesc[mi]}</text>

                {/* Arrow from weights to result label */}
                {on && <g>
                  <line x1={wx + 35} y1={178} x2={wx + 35} y2={188} stroke={col} strokeWidth={1.5} />
                  <polygon points={(wx + 35) + ",190 " + (wx + 31) + ",185 " + (wx + 39) + ",185"} fill={col} />
                </g>}
              </g>
            );
          })}

          {/* Equation at bottom */}
          <rect x={60} y={235} width={620} height={40} rx={8} fill={C.purple + "08"} stroke={C.purple + "25"} />
          <text x={bw / 2} y={253} textAnchor="middle" fill={C.purple} fontSize={10} fontWeight={700} fontFamily="monospace">
            {"Q = X"+MUL+"W_Q    K = X"+MUL+"W_K    V = X"+MUL+"W_V"}
          </text>
          <text x={bw / 2} y={268} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">
            {"Same input X, THREE different learned projections "+ARR+" THREE different roles"}
          </text>

          {/* Result matrices */}
          <text x={bw / 2} y={298} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">
            {"Each produces a (3 words "+MUL+" 4 dims) matrix. Same shape as input, different information!"}
          </text>

          {/* Active indicator */}
          {step > 0 && <g>
            <rect x={240 + (step - 1) * 170 - 38} y={52} width={146} height={132} rx={8} fill="none" stroke={qColors[step - 1]} strokeWidth={2} style={{ animation: "pulse 1.5s infinite" }} />
          </g>}

        </svg>
      </div>

      <Card highlight={true} style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: step === 0 ? C.blue : qColors[step - 1], marginBottom: 4 }}>{"Step " + (step + 1) + ": " + steps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{steps[step].desc}</div>
      </Card>

      {/* Q K V analogy card */}
      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 12 }}>{TARG + " Library Analogy"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 16, flexWrap: "wrap" }}>
          {[
            { label: "Query (Q)", color: C.q, icon: "🔍", desc: "Your search question", ex: '"I want a book about cats"' },
            { label: "Key (K)", color: C.k, icon: "🏷️", desc: "Each book's label / tag", ex: '"Animals", "Cooking", "Fiction"' },
            { label: "Value (V)", color: C.v, icon: "📖", desc: "The actual book content", ex: "The text inside each book" },
          ].map(function(item, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 180, textAlign: "center", padding: "14px 12px", background: item.color + "08", borderRadius: 8, border: "1px solid " + item.color + "25" }}>
                <div style={{ fontSize: 20, marginBottom: 4 }}>{item.icon}</div>
                <div style={{ fontSize: 11, fontWeight: 700, color: item.color, marginBottom: 4 }}>{item.label}</div>
                <div style={{ fontSize: 10, color: C.text, marginBottom: 4 }}>{item.desc}</div>
                <div style={{ fontSize: 9, color: C.dim, fontStyle: "italic" }}>{item.ex}</div>
              </div>
            );
          })}
        </div>
        <div style={{ textAlign: "center", marginTop: 12, fontSize: 10, color: C.muted }}>
          Q{MUL}K{"\u1D40"} = <span style={{ color: C.q }}>search</span> matches <span style={{ color: C.k }}>labels</span> {ARR} attention scores {ARR} weighted sum of <span style={{ color: C.v }}>contents</span>
        </div>
      </Card>

      <Insight>
        Q, K, V all come from the <span style={{ color: C.blue, fontWeight: 700 }}>same input X</span>, but through <span style={{ color: C.purple, fontWeight: 700 }}>different learned weight matrices</span>. This is what makes it <em>self</em>-attention {DASH} the input is attending to itself. The three projections give each word three "roles": what it's <span style={{ color: C.q }}>searching for</span>, what it <span style={{ color: C.k }}>advertises</span>, and what it <span style={{ color: C.v }}>provides</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: SINGLE-HEAD ATTENTION (Scaled Dot-Product)
   =============================================================== */
function TabSingleHead() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];

  /* Simulated Q, K, V matrices (3 words, d_k = 3 for simplicity) */
  var Q = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
  var K = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]];
  var V = [[0.5, 0.2, 0.8], [0.1, 0.9, 0.3], [0.7, 0.4, 0.6]];
  var dk = 3;
  var sqrtDk = Math.sqrt(dk);

  /* Step 1: Q * K^T */
  function dotProd(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  var QKT = Q.map(function(qr) { return K.map(function(kr) { return dotProd(qr, kr); }); });

  /* Step 2: Scale */
  var scaled = QKT.map(function(row) { return row.map(function(v) { return v / sqrtDk; }); });

  /* Step 3: Softmax */
  function softmax(arr) {
    var maxV = Math.max.apply(null, arr);
    var exps = arr.map(function(v) { return Math.exp(v - maxV); });
    var sum = exps.reduce(function(a, b) { return a + b; }, 0);
    return exps.map(function(e) { return e / sum; });
  }
  var attnWeights = scaled.map(function(row) { return softmax(row); });

  /* Step 4: Weighted sum with V */
  var output = attnWeights.map(function(weights) {
    var out = [0, 0, 0];
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        out[j] += weights[i] * V[i][j];
      }
    }
    return out;
  });

  var words = ["The", "cat", "sat"];

  var steps = [
    { title: "Q "+MUL+" K\u1D40 : Compute Raw Scores", desc: "Each Query is dot-produced with every Key. This measures how much each word should attend to every other word. High dot product = high relevance. Result is a 3"+MUL+"3 attention score matrix.", color: C.yellow },
    { title: "Scale by "+SQRT+"d_k", desc: "Divide every score by "+SQRT+"d_k = "+SQRT+"3 "+"\u2248"+" 1.73. Without scaling, when d_k is large, dot products become huge, pushing softmax into near-zero gradient regions. Scaling keeps gradients healthy.", color: C.orange },
    { title: "Softmax: Normalize to Probabilities", desc: "Apply softmax row-wise so each row sums to 1. Now each row is a probability distribution "+DASH+" it tells each word how much to attend to every other word. These are the attention weights.", color: C.purple },
    { title: "Multiply by V: Get Output", desc: "Each word's output is a weighted combination of ALL Value vectors, weighted by attention scores. Words with high attention weights contribute more to the output. This is the context-aware representation!", color: C.green },
  ];

  var bw = 740, bh = 300;

  function renderMatrix(data, x, y, w, h, label, col, cellW, cellH, highlight, rowLabels) {
    return (
      <g>
        <text x={x + (data[0].length * cellW) / 2} y={y - 8} textAnchor="middle" fill={col} fontSize={10} fontWeight={700} fontFamily="monospace">{label}</text>
        {data.map(function(row, r) {
          return row.map(function(val, c) {
            var isHL = highlight && highlight(r, c);
            var bgOp = isHL ? "30" : "10";
            return (
              <g key={r + "-" + c}>
                <rect x={x + c * cellW} y={y + r * cellH} width={cellW - 2} height={cellH - 2} rx={3}
                  fill={col + bgOp} stroke={isHL ? col : col + "30"} strokeWidth={isHL ? 1.5 : 0.5} />
                <text x={x + c * cellW + (cellW - 2) / 2} y={y + r * cellH + (cellH - 2) / 2 + 4}
                  textAnchor="middle" fill={isHL ? col : C.muted} fontSize={8} fontFamily="monospace">
                  {typeof val === "number" ? fmt2(val) : val}
                </text>
              </g>
            );
          });
        })}
        {rowLabels && rowLabels.map(function(lbl, i) {
          return <text key={i} x={x - 6} y={y + i * cellH + cellH / 2 + 3} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{lbl}</text>;
        })}
      </g>
    );
  }

  return (
    <div>
      <SectionTitle title="Scaled Dot-Product Attention" subtitle={"Step-by-step: how a single attention head works with real numbers"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          var on = step === i;
          return (
            <button key={i} onClick={function() { setStep(i); }} style={{
              padding: "8px 14px", borderRadius: 8,
              border: "1.5px solid " + (on ? s.color : C.border),
              background: on ? s.color + "20" : C.card,
              color: on ? s.color : C.muted, cursor: "pointer",
              fontSize: 10, fontWeight: 700, fontFamily: "monospace",
            }}>
              {(i + 1) + ". " + s.title.split(":")[0]}
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={bw} height={bh} viewBox={"0 0 " + bw + " " + bh} style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {/* Equation */}
          <text x={bw / 2} y={22} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">
            {"Attention(Q,K,V) = softmax( Q"+MUL+"K\u1D40 / "+SQRT+"d_k ) "+MUL+" V"}
          </text>

          {/* Step 1: QK^T */}
          {renderMatrix(Q, 30, 52, 36, 28, "Q", C.q, 36, 28, step === 0 ? function() { return true; } : null, words)}

          <text x={145} y={95} textAnchor="middle" fill={C.muted} fontSize={16} fontFamily="monospace">{MUL}</text>

          {renderMatrix(
            K[0].map(function(_, ci) { return K.map(function(row) { return row[ci]; }); }),
            160, 52, 36, 28, "K\u1D40", C.k, 36, 28, step === 0 ? function() { return true; } : null
          )}

          <text x={280} y={95} textAnchor="middle" fill={C.muted} fontSize={14} fontFamily="monospace">=</text>

          {renderMatrix(QKT, 295, 52, 42, 28, step === 0 ? "Q"+MUL+"K\u1D40 (raw)" : step === 1 ? "Scaled (/"+SQRT+"3)" : step >= 2 ? "Softmax" : "Scores",
            step === 0 ? C.yellow : step === 1 ? C.orange : C.purple,
            42, 28,
            function(r, c) {
              if (step === 0) return true;
              if (step >= 1) return true;
              return false;
            }, words
          )}

          {/* Show actual values based on step */}
          {step >= 1 && <g>
            <text x={340} y={145} textAnchor="middle" fill={C.orange} fontSize={9} fontFamily="monospace">{"\u00F7 "+SQRT+"3 \u2248 1.73"}</text>
          </g>}

          {/* Softmax result (overlay) */}
          {step >= 2 && renderMatrix(attnWeights, 420, 52, 48, 28, "Attention Weights", C.purple, 48, 28, function(r, c) {
            return attnWeights[r][c] > 0.35;
          }, words)}

          {/* Multiply by V */}
          {step >= 3 && <g>
            <text x={580} y={68} textAnchor="middle" fill={C.muted} fontSize={14} fontFamily="monospace">{MUL}</text>

            {renderMatrix(V, 600, 52, 36, 28, "V", C.v, 36, 28, function() { return true; })}

            {/* Output */}
            {renderMatrix(output, 295, 190, 42, 28, "Output (context-aware!)", C.green, 42, 28, function() { return true; }, words)}

            {/* Arrow */}
            <line x1={420} y1={140} x2={370} y2={185} stroke={C.green} strokeWidth={1.5} strokeDasharray="4,3" />
          </g>}

          {/* Heat map visualization */}
          {step >= 2 && <g>
            <text x={185} y={190} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">Attention Heatmap</text>
            {attnWeights.map(function(row, r) {
              return row.map(function(val, c) {
                var opacity = Math.max(0.1, val);
                return (
                  <g key={"heat" + r + c}>
                    <rect x={115 + c * 50} y={200 + r * 30} width={46} height={26} rx={4}
                      fill={C.purple} fillOpacity={opacity} stroke={C.purple + "40"} strokeWidth={0.5} />
                    <text x={138 + c * 50} y={217 + r * 30} textAnchor="middle" fill={val > 0.3 ? "#fff" : C.muted} fontSize={8} fontWeight={700} fontFamily="monospace">
                      {(Math.round(val * 100)) + "%"}
                    </text>
                  </g>
                );
              });
            })}
            {/* Column labels */}
            {words.map(function(w, i) {
              return <text key={i} x={138 + i * 50} y={294} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{w}</text>;
            })}
            {/* Row labels */}
            {words.map(function(w, i) {
              return <text key={i} x={110} y={217 + i * 30} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">{w}</text>;
            })}
          </g>}

        </svg>
      </div>

      <Card highlight={true} style={{ maxWidth: 750, margin: "0 auto 16px", borderColor: steps[step].color }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: steps[step].color, marginBottom: 4 }}>{"Step " + (step + 1) + ": " + steps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{steps[step].desc}</div>
      </Card>

      {step === 1 && <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.orange, marginBottom: 8 }}>{WARN + " Why Scale?"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 24, flexWrap: "wrap" }}>
          <div style={{ textAlign: "center", padding: "12px 16px", background: C.red + "08", borderRadius: 8, border: "1px solid " + C.red + "20" }}>
            <div style={{ fontSize: 9, color: C.muted }}>Without Scaling (d_k=512)</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: C.red, marginTop: 4 }}>Scores ~ 100+</div>
            <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>Softmax {ARR} [0.99, 0.01, 0.00]</div>
            <div style={{ fontSize: 8, color: C.red }}>Near one-hot! No gradient flow</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: 16, color: C.dim }}>vs</div>
          <div style={{ textAlign: "center", padding: "12px 16px", background: C.green + "08", borderRadius: 8, border: "1px solid " + C.green + "20" }}>
            <div style={{ fontSize: 9, color: C.muted }}>With Scaling (/{SQRT}d_k)</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: C.green, marginTop: 4 }}>Scores ~ 1-5</div>
            <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>Softmax {ARR} [0.45, 0.35, 0.20]</div>
            <div style={{ fontSize: 8, color: C.green }}>Smooth distribution! Healthy gradients</div>
          </div>
        </div>
      </Card>}

      <Insight icon={TARG} title="The Attention Flow">
        <span style={{ color: C.q }}>Q</span>{MUL}<span style={{ color: C.k }}>K{"\u1D40"}</span> asks "how relevant is each word to each other word?" {ARR} Scale keeps gradients healthy {ARR} Softmax gives probabilities {ARR} Multiply by <span style={{ color: C.v }}>V</span> to get the <span style={{ color: C.green, fontWeight: 700 }}>weighted mix</span> of information. Each word's output is now <span style={{ color: C.accent, fontWeight: 700 }}>context-aware</span>: it blends in relevant information from all other words.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: MULTI-HEAD ATTENTION
   =============================================================== */
function TabMultiHead() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _h = useState(4); var nHeads = _h[0], setNHeads = _h[1];
  var dModel = 512;
  var dHead = Math.floor(dModel / nHeads);

  var headColors = [C.q, C.k, C.v, C.yellow, C.pink, C.orange, C.cyan, C.accent];

  var steps = [
    { title: "Split into Multiple Heads", desc: "The d_model=" + dModel + " dimensional Q, K, V are split (or projected) into " + nHeads + " separate heads, each with d_k = d_model/" + nHeads + " = " + dHead + " dimensions. Each head gets its OWN learned W_Q, W_K, W_V projections. This lets each head learn to attend to different things!", color: C.blue },
    { title: "Parallel Attention Computations", desc: "Each head independently computes Attention(Q_i, K_i, V_i). Head 1 might focus on syntactic relationships (subject-verb), Head 2 on semantic similarity, Head 3 on positional proximity, Head 4 on coreference. They all run in PARALLEL on the GPU " + DASH + " same compute cost as single-head!", color: C.purple },
    { title: "Concatenate Heads", desc: "All head outputs are concatenated back: Concat(head_1, ..., head_" + nHeads + "). Each head produced d_k=" + dHead + " dims, so concatenated = " + nHeads + MUL + dHead + " = " + dModel + " dims. We're back to the original dimension.", color: C.yellow },
    { title: "Final Linear Projection", desc: "Multiply by W_O (d_model " + MUL + " d_model) to mix information across heads. This learned projection lets the model combine the different perspectives from all heads into a single coherent representation. Output: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O.", color: C.green },
  ];

  var bw = 740, bh = 360;

  return (
    <div>
      <SectionTitle title="Multi-Head Attention" subtitle={"Multiple attention heads "+DASH+" each learns to focus on different relationships"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 12, flexWrap: "wrap" }}>
        {steps.map(function(s, i) {
          var on = step === i;
          return (
            <button key={i} onClick={function() { setStep(i); }} style={{
              padding: "8px 14px", borderRadius: 8,
              border: "1.5px solid " + (on ? s.color : C.border),
              background: on ? s.color + "20" : C.card,
              color: on ? s.color : C.muted, cursor: "pointer",
              fontSize: 10, fontWeight: 700, fontFamily: "monospace",
            }}>
              {(i + 1) + ". " + s.title.split("(")[0].trim()}
            </button>
          );
        })}
      </div>

      {/* Heads slider */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 14, gap: 12, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace" }}>Heads:</span>
        {[2, 4, 8].map(function(h) {
          var on = nHeads === h;
          return (
            <button key={h} onClick={function() { setNHeads(h); }} style={{
              padding: "4px 12px", borderRadius: 6,
              border: "1px solid " + (on ? C.accent : C.border),
              background: on ? C.accent + "20" : C.card,
              color: on ? C.accent : C.muted, cursor: "pointer",
              fontSize: 10, fontWeight: 700, fontFamily: "monospace",
            }}>{h}</button>
          );
        })}
        <span style={{ fontSize: 9, color: C.dim, fontFamily: "monospace" }}>d_head = {dModel}/{nHeads} = {dHead}</span>
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={bw} height={bh} viewBox={"0 0 " + bw + " " + bh} style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>

          {/* Title equation */}
          <text x={bw / 2} y={22} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">
            {"MultiHead(Q,K,V) = Concat(head_1, ..., head_h) "+MUL+" W_O    where head_i = Attention(Q_i, K_i, V_i)"}
          </text>

          {/* Input */}
          <rect x={310} y={38} width={120} height={35} rx={8} fill={C.blue + "15"} stroke={C.blue} strokeWidth={step === 0 ? 2 : 1} />
          <text x={370} y={55} textAnchor="middle" fill={C.blue} fontSize={10} fontWeight={700} fontFamily="monospace">Input X</text>
          <text x={370} y={68} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{dModel + " dims"}</text>

          {/* Split arrows */}
          {Array.from({ length: nHeads }).map(function(_, i) {
            var headW = Math.min(120, (bw - 80) / nHeads - 8);
            var startX = (bw - (nHeads * (headW + 8) - 8)) / 2;
            var hx = startX + i * (headW + 8);
            var col = headColors[i % headColors.length];
            var isActive = step >= 1;

            return (
              <g key={"head" + i}>
                {/* Arrow from input */}
                <line x1={370} y1={73} x2={hx + headW / 2} y2={98} stroke={step >= 0 ? col + "60" : C.dim} strokeWidth={1} />

                {/* Head box */}
                <rect x={hx} y={100} width={headW} height={step >= 1 ? 95 : 55} rx={6}
                  fill={isActive ? col + "10" : C.card}
                  stroke={isActive ? col : C.dim} strokeWidth={isActive ? 1.5 : 1}
                  style={step === 1 ? { animation: "pulse 1.5s infinite" } : {}} />

                <text x={hx + headW / 2} y={116} textAnchor="middle" fill={isActive ? col : C.dim} fontSize={9} fontWeight={700} fontFamily="monospace">
                  {"Head " + (i + 1)}
                </text>

                {/* QKV labels inside head */}
                {step >= 0 && <g>
                  <text x={hx + headW / 2} y={132} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">
                    {"d_k=" + dHead}
                  </text>
                </g>}

                {/* Attention computation */}
                {step >= 1 && <g>
                  <text x={hx + headW / 2} y={148} textAnchor="middle" fill={col} fontSize={7} fontFamily="monospace">
                    {"Q"+MUL+"K\u1D40/"+SQRT+"d"}
                  </text>
                  <text x={hx + headW / 2} y={160} textAnchor="middle" fill={col} fontSize={7} fontFamily="monospace">
                    {"softmax "+MUL+" V"}
                  </text>
                  {nHeads <= 4 && <text x={hx + headW / 2} y={175} textAnchor="middle" fill={C.dim} fontSize={6} fontFamily="monospace">
                    {["syntax", "semantic", "position", "coreference"][i] || "pattern " + (i + 1)}
                  </text>}
                  <text x={hx + headW / 2} y={188} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">
                    {"out: " + dHead + "d"}
                  </text>
                </g>}

                {/* Concat arrows */}
                {step >= 2 && <g>
                  <line x1={hx + headW / 2} y1={step >= 1 ? 195 : 155} x2={hx + headW / 2} y2={225}
                    stroke={C.yellow} strokeWidth={1.5} />
                </g>}
              </g>
            );
          })}

          {/* Concat box */}
          {step >= 2 && <g>
            <rect x={120} y={228} width={500} height={35} rx={8}
              fill={C.yellow + "12"} stroke={C.yellow} strokeWidth={step === 2 ? 2 : 1}
              style={step === 2 ? { animation: "pulse 1.5s infinite" } : {}} />
            <text x={370} y={245} textAnchor="middle" fill={C.yellow} fontSize={10} fontWeight={700} fontFamily="monospace">
              {"Concat: " + nHeads + " heads "+MUL+" " + dHead + "d = " + dModel + "d"}
            </text>
            <text x={370} y={258} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">
              {"[head_1 | head_2 | ... | head_" + nHeads + "]"}
            </text>
          </g>}

          {/* Final linear */}
          {step >= 3 && <g>
            <line x1={370} y1={263} x2={370} y2={285} stroke={C.green} strokeWidth={1.5} />
            <polygon points="370,287 366,282 374,282" fill={C.green} />

            <rect x={260} y={288} width={220} height={35} rx={8}
              fill={C.green + "12"} stroke={C.green} strokeWidth={2}
              style={{ animation: "pulse 1.5s infinite" }} />
            <text x={370} y={305} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">
              {MUL + " W_O (Linear Projection)"}
            </text>
            <text x={370} y={318} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">
              {dModel + MUL + dModel + " "+ARR+" mixes head info"}
            </text>

            {/* Final output */}
            <line x1={370} y1={323} x2={370} y2={340} stroke={C.accent} strokeWidth={2} />
            <text x={370} y={354} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={800} fontFamily="monospace">
              {"Output: " + dModel + " dims (context-enriched!)"}
            </text>
          </g>}

        </svg>
      </div>

      <Card highlight={true} style={{ maxWidth: 750, margin: "0 auto 16px", borderColor: steps[step].color }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: steps[step].color, marginBottom: 4 }}>{"Step " + (step + 1) + ": " + steps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7 }}>{steps[step].desc}</div>
      </Card>

      {/* What each head learns */}
      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>{TARG + " What Different Heads Learn (Discovered via Research)"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
          {[
            { head: "Head A", focus: "Syntax", ex: "Subject " + ARR + " Verb", color: C.q },
            { head: "Head B", focus: "Coreference", ex: '"it" ' + ARR + ' "the cat"', color: C.k },
            { head: "Head C", focus: "Proximity", ex: "nearby words", color: C.v },
            { head: "Head D", focus: "Rare patterns", ex: "idioms, negation", color: C.yellow },
          ].map(function(item, i) {
            return (
              <div key={i} style={{ textAlign: "center", padding: "10px 14px", background: item.color + "08", borderRadius: 8, border: "1px solid " + item.color + "25", minWidth: 130 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: item.color }}>{item.head}</div>
                <div style={{ fontSize: 11, fontWeight: 800, color: C.text, marginTop: 2 }}>{item.focus}</div>
                <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{item.ex}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.purple, marginBottom: 8 }}>Single-Head vs Multi-Head</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 20, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200, padding: 12, background: "#08080d", borderRadius: 8, border: "1px solid " + C.border }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.cyan, marginBottom: 6 }}>Single Head (d_k = {dModel})</div>
            <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8, fontFamily: "monospace" }}>
              {"One attention pattern per layer"}<br />
              {"All " + dModel + " dims in one perspective"}<br />
              {"Must compress all relationships into one pattern"}<br />
              {"Like seeing with one eye"}
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 200, padding: 12, background: "#08080d", borderRadius: 8, border: "1px solid " + C.accent + "30" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, marginBottom: 6 }}>Multi-Head ({nHeads} heads, d_k = {dHead})</div>
            <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8, fontFamily: "monospace" }}>
              {nHeads + " attention patterns per layer"}<br />
              {"Each head: " + dHead + " dims, specialized"}<br />
              {"Different heads " + ARR + " different relationships"}<br />
              {"Like seeing with " + nHeads + " eyes, each looking at something different"}
            </div>
          </div>
        </div>
      </Card>

      <Insight>
        Multi-head attention costs the <span style={{ color: C.accent, fontWeight: 700 }}>same as single-head</span>! {nHeads} heads {MUL} {dHead}d = {dModel}d total compute. But we get {nHeads} <span style={{ color: C.purple, fontWeight: 700 }}>specialized perspectives</span>. The original paper used <span style={{ color: C.yellow, fontWeight: 700 }}>8 heads</span> with d_model=512. GPT-3 uses <span style={{ color: C.green, fontWeight: 700 }}>96 heads</span> with d_model=12288.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: FULL FORWARD PASS
   =============================================================== */
function TabFullPass() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _a = useState(false); var autoPlay = _a[0], setAutoPlay = _a[1];

  var totalSteps = 8;

  useEffect(function() {
    if (!autoPlay) return;
    var t = setInterval(function() { setStep(function(s) { return (s + 1) % totalSteps; }); }, 3000);
    return function() { clearInterval(t); };
  }, [autoPlay]);

  var steps = [
    { title: "Tokenize & Embed", desc: "The sentence is split into tokens (subwords). Each token is mapped to a learned embedding vector of size d_model (e.g., 768 or 512). This is a lookup table, no computation.", color: C.blue },
    { title: "Add Positional Encoding", desc: "Since attention is permutation-invariant (doesn't know word order), we add position information using sin/cos functions: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)). Each position gets a unique signal.", color: C.cyan },
    { title: "Compute Q, K, V", desc: "The positioned embeddings are projected through three learned matrices W_Q, W_K, W_V. For multi-head attention, each head gets its own slice. Q = XW_Q, K = XW_K, V = XW_V.", color: C.purple },
    { title: "Attention Scores", desc: "Compute scaled dot-product: scores = Q"+MUL+"K\u1D40 / "+SQRT+"d_k. This creates an attention matrix where entry (i,j) measures how much word i should attend to word j. Apply softmax row-wise to get probabilities.", color: C.yellow },
    { title: "Weighted Values + Concat", desc: "Multiply attention weights by V to get context-aware output per head. Concatenate all heads' outputs and project through W_O. Each word now has information from all other relevant words.", color: C.v },
    { title: "Residual + LayerNorm (1st)", desc: "Add the attention output to the original input (skip connection), then apply LayerNorm. This helps gradient flow and stabilizes training. The model can learn to pass information through unchanged if needed.", color: C.green },
    { title: "Feed-Forward Network", desc: "Each position goes through the same 2-layer MLP: FFN(x) = max(0, xW\u2081+b\u2081)W\u2082+b\u2082. The inner dimension is typically 4x d_model (e.g., 2048 for d_model=512). This is where pattern transformation and "+LQ+"reasoning"+RQ+" happens.", color: C.orange },
    { title: "Residual + LayerNorm (2nd)", desc: "Another skip connection + LayerNorm. The output is now ready to be fed to the NEXT transformer block, or (if this is the final block) to the output projection for prediction. Stack 12-96+ of these blocks for a full model!", color: C.accent },
  ];

  var bw = 740, bh = 120;
  var nodeW = 75, nodeH = 45;
  var startX = 20;
  var gap = (bw - startX * 2 - nodeW * totalSteps) / (totalSteps - 1);

  return (
    <div>
      <SectionTitle title="Complete Forward Pass" subtitle={"Follow a sentence through every step of a Transformer block"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 14, gap: 6 }}>
        <button onClick={function() { setStep(Math.max(0, step - 1)); setAutoPlay(false); }} style={{ padding: "6px 14px", borderRadius: 6, border: "1px solid " + C.border, background: C.card, color: C.muted, cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>{LARR}</button>
        <span style={{ fontSize: 12, color: C.accent, fontWeight: 700, fontFamily: "monospace", padding: "6px 10px" }}>{(step + 1) + " / " + totalSteps}</span>
        <button onClick={function() { setStep(Math.min(totalSteps - 1, step + 1)); setAutoPlay(false); }} style={{ padding: "6px 14px", borderRadius: 6, border: "1px solid " + C.border, background: C.card, color: C.muted, cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>{ARR}</button>
        <button onClick={function() { setAutoPlay(!autoPlay); }} style={{ padding: "6px 14px", borderRadius: 6, border: "1px solid " + (autoPlay ? C.yellow : C.border), background: autoPlay ? C.yellow + "20" : C.card, color: autoPlay ? C.yellow : C.muted, cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>{autoPlay ? PAUSE : PLAY}</button>
      </div>

      {/* Progress pipeline */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={bw} height={bh} viewBox={"0 0 " + bw + " " + bh} style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          {steps.map(function(s, i) {
            var x = startX + i * (nodeW + gap);
            var y = (bh - nodeH) / 2;
            var on = step === i;
            var past = step > i;

            return (
              <g key={i} onClick={function() { setStep(i); setAutoPlay(false); }} style={{ cursor: "pointer" }}>
                {/* Arrow */}
                {i > 0 && <line x1={x - gap + 2} y1={y + nodeH / 2} x2={x - 2} y2={y + nodeH / 2}
                  stroke={past || on ? s.color + "80" : C.dim} strokeWidth={past || on ? 2 : 1} />}

                {/* Node */}
                <rect x={x} y={y} width={nodeW} height={nodeH} rx={6}
                  fill={on ? s.color + "20" : past ? s.color + "08" : C.card}
                  stroke={on ? s.color : past ? s.color + "50" : C.dim}
                  strokeWidth={on ? 2.5 : 1}
                  style={on ? { animation: "pulse 1.5s infinite" } : {}} />

                <text x={x + nodeW / 2} y={y + nodeH / 2 - 3} textAnchor="middle"
                  fill={on ? s.color : past ? s.color + "90" : C.dim}
                  fontSize={8} fontWeight={700} fontFamily="monospace">
                  {s.title.split(" ")[0]}
                </text>
                <text x={x + nodeW / 2} y={y + nodeH / 2 + 9} textAnchor="middle"
                  fill={C.dim} fontSize={7} fontFamily="monospace">
                  {s.title.split(" ").slice(1, 3).join(" ")}
                </text>

                {/* Step number */}
                <circle cx={x + nodeW / 2} cy={y - 6} r={8}
                  fill={on ? s.color : past ? s.color + "40" : C.dim + "40"}
                  stroke={on ? s.color : "none"} strokeWidth={1} />
                <text x={x + nodeW / 2} y={y - 3} textAnchor="middle"
                  fill={on || past ? "#fff" : C.dim} fontSize={8} fontWeight={700} fontFamily="monospace">
                  {i + 1}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Main card */}
      <Card highlight={true} style={{ maxWidth: 750, margin: "0 auto 16px", borderColor: steps[step].color }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: steps[step].color, marginBottom: 6 }}>{"Step " + (step + 1) + ": " + steps[step].title}</div>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{steps[step].desc}</div>
      </Card>

      {/* Data shape tracker */}
      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10 }}>Data Shape Tracker (d_model=512, seq_len=3, heads=8)</div>
        <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
          {[
            { step: 1, label: "Tokens", shape: "3 ints", note: "token IDs" },
            { step: 2, label: "Embed", shape: "3"+MUL+"512", note: "+position" },
            { step: 3, label: "Q, K, V", shape: "8"+MUL+"3"+MUL+"64", note: "per head" },
            { step: 4, label: "Scores", shape: "8"+MUL+"3"+MUL+"3", note: "attention matrix" },
            { step: 5, label: "Attn Out", shape: "3"+MUL+"512", note: "concat heads" },
            { step: 6, label: "+Residual", shape: "3"+MUL+"512", note: "add & norm" },
            { step: 7, label: "FFN", shape: "3"+MUL+"2048"+ARR+"512", note: "expand & compress" },
            { step: 8, label: "Output", shape: "3"+MUL+"512", note: "to next block" },
          ].map(function(item, i) {
            var on = step === i;
            var past = step > i;
            return (
              <div key={i} style={{
                textAlign: "center", padding: "8px 10px", minWidth: 76,
                background: on ? steps[i].color + "12" : "transparent",
                borderRadius: 6, border: "1px solid " + (on ? steps[i].color : past ? steps[i].color + "30" : C.dim + "30"),
              }}>
                <div style={{ fontSize: 8, color: on ? steps[i].color : C.dim, fontWeight: 700 }}>{item.label}</div>
                <div style={{ fontSize: 10, color: on ? C.text : C.dim, fontWeight: 800, marginTop: 2, fontFamily: "monospace" }}>{item.shape}</div>
                <div style={{ fontSize: 7, color: C.dim, marginTop: 2 }}>{item.note}</div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Positional encoding detail */}
      {step === 1 && <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.cyan, marginBottom: 8 }}>{BULB + " Why Sin/Cos for Positions?"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 16, flexWrap: "wrap" }}>
          {[
            { reason: "Unique per position", detail: "Each position gets a distinct encoding pattern" },
            { reason: "Bounded values", detail: "sin/cos always between -1 and 1, won't blow up" },
            { reason: "Relative distances", detail: "PE(pos+k) can be represented as a linear function of PE(pos)" },
            { reason: "Generalize to unseen lengths", detail: "Works for sequences longer than training data" },
          ].map(function(item, i) {
            return (
              <div key={i} style={{ flex: 1, minWidth: 140, padding: "8px 10px", background: C.cyan + "06", borderRadius: 6, border: "1px solid " + C.cyan + "15" }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: C.cyan }}>{CHK + " " + item.reason}</div>
                <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>{item.detail}</div>
              </div>
            );
          })}
        </div>
      </Card>}

      {/* Residual connection detail */}
      {(step === 5 || step === 7) && <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.green, marginBottom: 8 }}>{BULB + " Why Residual Connections?"}</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 20, flexWrap: "wrap" }}>
          <div style={{ textAlign: "center", padding: "12px 16px", background: C.red + "08", borderRadius: 8, border: "1px solid " + C.red + "20" }}>
            <div style={{ fontSize: 9, color: C.muted }}>Without Skip (96 layers)</div>
            <div style={{ fontSize: 12, fontWeight: 800, color: C.red, marginTop: 4 }}>Gradient vanishes</div>
            <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>f(f(f(...f(x)...)))</div>
            <div style={{ fontSize: 8, color: C.red }}>Signal degrades over depth</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: 16, color: C.dim }}>vs</div>
          <div style={{ textAlign: "center", padding: "12px 16px", background: C.green + "08", borderRadius: 8, border: "1px solid " + C.green + "20" }}>
            <div style={{ fontSize: 9, color: C.muted }}>With Skip (96 layers)</div>
            <div style={{ fontSize: 12, fontWeight: 800, color: C.green, marginTop: 4 }}>Gradient highway!</div>
            <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>x + f(x + f(x + ...))</div>
            <div style={{ fontSize: 8, color: C.green }}>Direct path for gradients at all depths</div>
          </div>
        </div>
      </Card>}

      <Insight icon={TARG} title="The Big Picture">
        A full Transformer is just this block <span style={{ color: C.accent, fontWeight: 700 }}>stacked N times</span>. GPT-2 has 12 blocks, GPT-3 has 96, Claude has even more. Each block <span style={{ color: C.purple, fontWeight: 700 }}>refines the representations</span>: early blocks capture syntax and local patterns, middle blocks build semantic understanding, and deep blocks handle complex reasoning and long-range dependencies.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Architecture", "Q, K, V", "Single-Head Attention", "Multi-Head Attention", "Full Forward Pass"];
  return (
    <div style={{ background: C.bg, minHeight: "100vh", padding: "24px 16px", fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.text, maxWidth: 960, margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg," + C.accent + "," + C.purple + ")", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", display: "inline-block" }}>Transformer Architecture</div>
        <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{"Interactive visual walkthrough " + DASH + " from embeddings to multi-head attention"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab === 0 && <TabArchitecture />}
      {tab === 1 && <TabQKV />}
      {tab === 2 && <TabSingleHead />}
      {tab === 3 && <TabMultiHead />}
      {tab === 4 && <TabFullPass />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
<script>
// Auto-resize: dynamically adjust iframe height to match content
(function() {
  function resizeFrame() {
    var root = document.getElementById("root");
    if (!root) return;
    var h = root.scrollHeight + 60;
    // Method 1: Direct iframe resize (works in Streamlit)
    if (window.frameElement) {
      window.frameElement.style.height = h + "px";
    }
    // Method 2: postMessage to parent (fallback for sandboxed iframes)
    try {
      window.parent.postMessage({ type: "streamlit:setFrameHeight", height: h }, "*");
    } catch(e) {}
  }

  // Observe DOM changes to re-measure on tab switches / step changes
  var observer = new MutationObserver(function() {
    requestAnimationFrame(resizeFrame);
  });

  function init() {
    var root = document.getElementById("root");
    if (root) {
      observer.observe(root, { childList: true, subtree: true, attributes: true });
      resizeFrame();
    }
  }

  // Run on load and periodically for safety
  if (document.readyState === "complete") { setTimeout(init, 500); }
  else { window.addEventListener("load", function() { setTimeout(init, 500); }); }
  setInterval(resizeFrame, 1000);
})();
</script>
</body>
</html>
"""

TRANSFORMER_VISUAL_HEIGHT = 1800
