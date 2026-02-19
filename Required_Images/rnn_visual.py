"""
Self-contained HTML for the RNN (Recurrent Neural Network) interactive walkthrough.
Covers: Basic RNN operation, BPTT, Vanishing Gradient Problem, LSTM gates,
and GRU + Evolution timeline (Simple RNN → LSTM → GRU → Transformer).
Embed in Streamlit via st.components.v1.html(RNN_VISUAL_HTML, height=RNN_VISUAL_HEIGHT).
"""

RNN_VISUAL_HTML = """
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

function tanh(x) { var e = Math.exp(2*x); return (e-1)/(e+1); }
function sigmoid(x) { return 1/(1+Math.exp(-x)); }
function fmt(v) { return (Math.round(v*1000)/1000).toString(); }
function fmt2(v) { return (Math.round(v*100)/100).toString(); }


/* ===============================================================
   TAB 1: BASIC RNN — "not good" walkthrough
   =============================================================== */
function TabBasicRNN() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _a = useState(false); var autoPlay = _a[0], setAutoPlay = _a[1];

  useEffect(function(){if(!autoPlay)return;var t=setInterval(function(){setStep(function(s){return(s+1)%3;});},3000);return function(){clearInterval(t);};}, [autoPlay]);

  var W_in = 0.5, W_h = 0.9;
  var words = [{w:'"not"',x:-1,c:C.red},{w:'"good"',x:0.8,c:C.green}];
  var h0 = 0;
  var h1 = tanh(W_in*(-1)+W_h*0);
  var h2 = tanh(W_in*0.8+W_h*h1);
  var h_good_alone = tanh(W_in*0.8);

  var steps = [
    {title:"Initial State",desc:"No memory yet. h(0) = 0. The network starts with a blank slate."},
    {title:'Processing "not"',desc:'X(1) = -1. The RNN computes h(1) = tanh(0.5'+MUL+'(-1) + 0.9'+MUL+'0) = tanh(-0.5) = '+fmt(h1)+'. The hidden state is now negative '+DASH+' the network has a "negative" memory.'},
    {title:'Processing "good"',desc:'X(2) = 0.8. The RNN computes h(2) = tanh(0.5'+MUL+'0.8 + 0.9'+MUL+'('+fmt(h1)+')) = tanh('+fmt2(0.4+W_h*h1)+') = '+fmt(h2)+'. Even though "good" is positive, the memory of "not" dragged the result negative!'},
  ];

  var hVals = [h0,h1,h2];
  var bw = 740, bh = 260;

  return (
    <div>
      <SectionTitle title="How an RNN Works" subtitle={"Watch a single neuron process "+LQ+"not good"+RQ+" one word at a time, carrying memory forward"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:16}}>
        {steps.map(function(s,i){var on=step===i;return (<button key={i} onClick={function(){setStep(i);setAutoPlay(false);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?C.accent:C.border),background:on?C.accent+"20":C.card,color:on?C.accent:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{i===0?"Start":"t="+i}</button>);})}
        <button onClick={function(){setAutoPlay(!autoPlay);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(autoPlay?C.yellow:C.border),background:autoPlay?C.yellow+"20":C.card,color:autoPlay?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoPlay?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <text x={bw/2} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"h(t) = tanh(W_input "+MUL+" X(t)  +  W_hidden "+MUL+" h(t-1))"}</text>

          {/* h(0) */}
          <rect x={40} y={80} width={80} height={55} rx={8} fill={step===0?C.blue+"15":C.card} stroke={step===0?C.blue:C.dim} strokeWidth={step===0?2:1} />
          <text x={80} y={97} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">h(0)</text>
          <text x={80} y={120} textAnchor="middle" fill={C.blue} fontSize={20} fontWeight={800} fontFamily="monospace">0</text>
          <text x={80} y={155} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">no memory</text>

          {/* Arrow h0 -> RNN1 */}
          <line x1={120} y1={107} x2={190} y2={107} stroke={step>=1?C.blue:C.dim} strokeWidth={step>=1?2:1} />
          <polygon points="194,107 187,103 187,111" fill={step>=1?C.blue:C.dim} />
          <text x={155} y={99} textAnchor="middle" fill={step>=1?C.blue:C.dim} fontSize={8} fontFamily="monospace">h(0)=0</text>

          {/* RNN Neuron 1 */}
          <rect x={200} y={60} width={110} height={95} rx={10} fill={step===1?C.accent+"15":C.card} stroke={step===1?C.accent:C.dim} strokeWidth={step===1?2.5:1} style={step===1?{animation:"pulse 1.5s infinite"}:{}} />
          <text x={255} y={82} textAnchor="middle" fill={step>=1?C.accent:C.dim} fontSize={11} fontWeight={700} fontFamily="monospace">RNN</text>
          <text x={255} y={100} textAnchor="middle" fill={step>=1?C.accent:C.dim} fontSize={9} fontFamily="monospace">t=1</text>
          {step>=1 && <text x={255} y={120} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"tanh(-0.5)"}</text>}
          {step>=1 && <text x={255} y={140} textAnchor="middle" fill={C.accent} fontSize={14} fontWeight={800} fontFamily="monospace">{fmt2(h1)}</text>}

          {/* Input "not" */}
          <rect x={215} y={185} width={80} height={35} rx={6} fill={step>=1?C.red+"15":"transparent"} stroke={step>=1?C.red:C.dim} strokeWidth={step>=1?1.5:1} />
          <text x={255} y={207} textAnchor="middle" fill={step>=1?C.red:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{'"not" = -1'}</text>
          <line x1={255} y1={185} x2={255} y2={155} stroke={step>=1?C.red:C.dim} strokeWidth={step>=1?1.5:1} />
          <polygon points="255,157 251,163 259,163" fill={step>=1?C.red:C.dim} />

          {/* Arrow RNN1 -> RNN2 */}
          <line x1={310} y1={107} x2={410} y2={107} stroke={step>=2?C.accent:C.dim} strokeWidth={step>=2?2:1} />
          <polygon points="414,107 407,103 407,111" fill={step>=2?C.accent:C.dim} />
          <text x={360} y={99} textAnchor="middle" fill={step>=2?C.accent:C.dim} fontSize={8} fontFamily="monospace">{"h(1)="+fmt2(h1)}</text>

          {/* RNN Neuron 2 */}
          <rect x={420} y={60} width={110} height={95} rx={10} fill={step===2?C.accent+"15":C.card} stroke={step===2?C.accent:C.dim} strokeWidth={step===2?2.5:1} style={step===2?{animation:"pulse 1.5s infinite"}:{}} />
          <text x={475} y={82} textAnchor="middle" fill={step>=2?C.accent:C.dim} fontSize={11} fontWeight={700} fontFamily="monospace">RNN</text>
          <text x={475} y={100} textAnchor="middle" fill={step>=2?C.accent:C.dim} fontSize={9} fontFamily="monospace">t=2</text>
          {step>=2 && <text x={475} y={120} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"tanh("+fmt2(0.4+W_h*h1)+")"}</text>}
          {step>=2 && <text x={475} y={140} textAnchor="middle" fill={C.accent} fontSize={14} fontWeight={800} fontFamily="monospace">{fmt2(h2)}</text>}

          {/* Input "good" */}
          <rect x={435} y={185} width={80} height={35} rx={6} fill={step>=2?C.green+"15":"transparent"} stroke={step>=2?C.green:C.dim} strokeWidth={step>=2?1.5:1} />
          <text x={475} y={207} textAnchor="middle" fill={step>=2?C.green:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{'"good" = 0.8'}</text>
          <line x1={475} y1={185} x2={475} y2={155} stroke={step>=2?C.green:C.dim} strokeWidth={step>=2?1.5:1} />
          <polygon points="475,157 471,163 479,163" fill={step>=2?C.green:C.dim} />

          {/* Output */}
          {step>=2 && <g>
            <line x1={530} y1={107} x2={590} y2={107} stroke={C.accent} strokeWidth={2} />
            <polygon points="594,107 587,103 587,111" fill={C.accent} />
            <rect x={600} y={80} width={110} height={55} rx={8} fill={h2<0?C.red+"15":C.green+"15"} stroke={h2<0?C.red:C.green} strokeWidth={2} />
            <text x={655} y={100} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">Output</text>
            <text x={655} y={122} textAnchor="middle" fill={h2<0?C.red:C.green} fontSize={16} fontWeight={800} fontFamily="monospace">{fmt(h2)}</text>
          </g>}

          {/* Shared weights label */}
          <rect x={200} y={235} width={330} height={22} rx={4} fill={C.purple+"10"} stroke={C.purple+"30"} />
          <text x={365} y={250} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">{"Same weights reused: W_input=0.5  W_hidden=0.9  b=0"}</text>
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:4}}>{"Step "+(step+1)+": "+steps[step].title}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{steps[step].desc}</div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:12}}>Why Memory Matters</div>
        <div style={{display:"flex",justifyContent:"center",gap:24,flexWrap:"wrap"}}>
          <div style={{textAlign:"center",padding:"12px 20px",background:C.green+"08",borderRadius:8,border:"1px solid "+C.green+"20"}}>
            <div style={{fontSize:9,color:C.muted,marginBottom:4}}>{LQ+"good"+RQ+" ALONE (no memory)"}</div>
            <div style={{fontSize:22,fontWeight:800,color:C.green}}>{fmt2(h_good_alone)}</div>
            <div style={{fontSize:9,color:C.green}}>Positive!</div>
          </div>
          <div style={{display:"flex",alignItems:"center",fontSize:20,color:C.dim}}>vs</div>
          <div style={{textAlign:"center",padding:"12px 20px",background:C.red+"08",borderRadius:8,border:"1px solid "+C.red+"20"}}>
            <div style={{fontSize:9,color:C.muted,marginBottom:4}}>{LQ+"not good"+RQ+" (with memory)"}</div>
            <div style={{fontSize:22,fontWeight:800,color:C.red}}>{fmt2(h2)}</div>
            <div style={{fontSize:9,color:C.red}}>Negative!</div>
          </div>
        </div>
        <div style={{textAlign:"center",marginTop:10,fontSize:10,color:C.muted}}>
          The memory of <span style={{color:C.red,fontWeight:700}}>"not"</span> changed the outcome. The RNN understood context.
        </div>
      </Card>

      <Insight>
        An RNN is the <span style={{color:C.accent,fontWeight:700}}>same neuron reused</span> at each time step {DASH} this is called "unrolling." It carries a <span style={{color:C.blue,fontWeight:700}}>hidden state h(t)</span> that acts as memory. The same weights (<span style={{color:C.purple}}>W_input, W_hidden</span>) are shared across all steps {DASH} just like a CNN filter is shared across all positions.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: BACKPROPAGATION THROUGH TIME (BPTT)
   =============================================================== */
function TabBPTT() {
  var _s = useState(0); var phase = _s[0], setPhase = _s[1];
  var _a = useState(false); var autoP = _a[0], setAutoP = _a[1];

  var phases = [
    {title:"Forward Pass",desc:"The sequence "+LQ+"The cat sat"+RQ+" is processed left-to-right, one word at a time. Each step computes h(t) = tanh(W_input "+MUL+" X(t) + W_hidden "+MUL+" h(t-1)). The same weights are reused at every step."},
    {title:"Compute Loss",desc:'After the full sequence is processed, the network predicts "ran" but the correct word is "on." The loss measures how wrong the prediction is.'},
    {title:"Backward Through Time",desc:"Gradients flow RIGHT to LEFT through every time step. At each step, the gradient is multiplied by W_hidden "+DASH+" this repeated multiplication is what causes vanishing gradients."},
    {title:"Combine & Update",desc:"Since the SAME weights are used at every step, gradients from ALL time steps are summed together to compute one weight update. gradient_total = grad(t=1) + grad(t=2) + grad(t=3)."},
  ];

  useEffect(function(){if(!autoP)return;var t=setInterval(function(){setPhase(function(p){return(p+1)%4;});},4000);return function(){clearInterval(t);};}, [autoP]);

  var bw = 760, bh = 300;
  var words = ["The","cat","sat"];
  var cx = [100,280,460];
  var grads = [0.25,0.5,1.0];

  return (
    <div>
      <SectionTitle title="Backpropagation Through Time (BPTT)" subtitle={"How RNNs learn "+DASH+" gradients must flow backward through LAYERS and TIME"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
        {phases.map(function(p,i){var on=phase===i;var cl=[C.blue,C.red,C.yellow,C.green][i]; return (<button key={i} onClick={function(){setPhase(i);setAutoP(false);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?cl:C.border),background:on?cl+"20":C.card,color:on?cl:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{(i+1)+". "+p.title}</button>);})}
        <button onClick={function(){setAutoP(!autoP);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(autoP?C.yellow:C.border),background:autoP?C.yellow+"20":C.card,color:autoP?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoP?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>

          {/* Forward pass arrows and nodes */}
          {words.map(function(w,i) {
            var x = cx[i], on = phase===0;
            return (<g key={"fwd"+i}>
              {/* Word input */}
              <rect x={x-30} y={180} width={60} height={28} rx={6} fill={on?C.blue+"15":"transparent"} stroke={on?C.blue:C.dim} />
              <text x={x} y={198} textAnchor="middle" fill={on?C.blue:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{'"'+w+'"'}</text>
              <line x1={x} y1={180} x2={x} y2={150} stroke={on?C.blue:C.dim} strokeWidth={on?1.5:1} />
              <polygon points={x+",147 "+(x-3)+",153 "+(x+3)+",153"} fill={on?C.blue:C.dim} />

              {/* RNN cell */}
              <rect x={x-42} y={95} width={84} height={55} rx={8} fill={on?C.accent+"12":C.card} stroke={on?C.accent:C.dim} strokeWidth={on?2:1} />
              <text x={x} y={115} textAnchor="middle" fill={on?C.accent:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">RNN</text>
              <text x={x} y={132} textAnchor="middle" fill={on?C.muted:C.dim} fontSize={9} fontFamily="monospace">{"t="+(i+1)}</text>

              {/* Forward arrow to next */}
              {i<2 && <g>
                <line x1={x+42} y1={122} x2={cx[i+1]-42} y2={122} stroke={on?C.accent:C.dim} strokeWidth={on?2:1} />
                <polygon points={(cx[i+1]-44)+",122 "+(cx[i+1]-50)+",118 "+(cx[i+1]-50)+",126"} fill={on?C.accent:C.dim} />
                <text x={(x+cx[i+1])/2} y={115} textAnchor="middle" fill={on?C.accent:C.dim} fontSize={8} fontFamily="monospace">{"h("+(i+1)+")"}</text>
              </g>}
            </g>);
          })}

          {/* Forward label */}
          {phase===0 && <g>
            <text x={bw/2} y={25} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={700} fontFamily="monospace">{"FORWARD PASS "+ARR+" (left to right)"}</text>
            <line x1={100} y1={38} x2={550} y2={38} stroke={C.blue} strokeWidth={2} />
            <polygon points="554,38 547,34 547,42" fill={C.blue} />
          </g>}

          {/* Loss */}
          {phase>=1 && <g>
            <line x1={502} y1={122} x2={580} y2={122} stroke={C.red} strokeWidth={2} />
            <polygon points="584,122 577,118 577,126" fill={C.red} />
            <rect x={590} y={95} width={120} height={55} rx={8} fill={C.red+"12"} stroke={C.red} strokeWidth={2} />
            <text x={650} y={115} textAnchor="middle" fill={C.red} fontSize={10} fontWeight={700} fontFamily="monospace">{WARN+" LOSS"}</text>
            <text x={650} y={134} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{'"ran" '+MUL+' "on"'}</text>
          </g>}

          {/* Backward arrows */}
          {phase>=2 && <g>
            <text x={bw/2} y={25} textAnchor="middle" fill={C.yellow} fontSize={11} fontWeight={700} fontFamily="monospace">{LARR+" BACKWARD PASS (right to left)"}</text>
            <line x1={550} y1={38} x2={100} y2={38} stroke={C.yellow} strokeWidth={2} />
            <polygon points="96,38 103,34 103,42" fill={C.yellow} />

            {words.map(function(_,i) {
              var x = cx[2-i], g2 = grads[2-i];
              var bh2 = Math.max(8, g2*40);
              return (<g key={"bk"+i}>
                <rect x={x-20} y={60-bh2} width={40} height={bh2} rx={4} fill={C.yellow+(g2>0.5?"50":"25")} stroke={C.yellow} strokeWidth={1} />
                <text x={x} y={55-bh2} textAnchor="middle" fill={C.yellow} fontSize={9} fontWeight={700} fontFamily="monospace">{g2.toFixed(2)}</text>
              </g>);
            })}
            <text x={cx[0]} y={265} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"grad "+MUL+" W "+MUL+" W"}</text>
            <text x={cx[1]} y={265} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"grad "+MUL+" W"}</text>
            <text x={cx[2]} y={265} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"full grad"}</text>
          </g>}

          {/* Combine phase */}
          {phase===3 && <g>
            <rect x={80} y={230} width={440} height={40} rx={8} fill={C.green+"10"} stroke={C.green+"40"} />
            <text x={300} y={254} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">{"Total gradient = 0.25 + 0.5 + 1.0 = 1.75  "+ARR+"  ONE weight update"}</text>
          </g>}

          {/* Shared weights */}
          <rect x={80} y={275} width={440} height={20} rx={4} fill={C.purple+"08"} stroke={C.purple+"25"} />
          <text x={300} y={289} textAnchor="middle" fill={C.purple} fontSize={8} fontWeight={700} fontFamily="monospace">{"Same W_input, W_hidden reused at EVERY time step"}</text>
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:[C.blue,C.red,C.yellow,C.green][phase],marginBottom:4}}>{"Phase "+(phase+1)+": "+phases[phase].title}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{phases[phase].desc}</div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.purple,marginBottom:10}}>Regular NN vs RNN Backprop</div>
        <div style={{display:"flex",justifyContent:"center",gap:20,flexWrap:"wrap"}}>
          <div style={{flex:1,minWidth:200,padding:12,background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
            <div style={{fontSize:10,fontWeight:700,color:C.cyan,marginBottom:6}}>Regular NN</div>
            <div style={{fontSize:10,color:C.muted,lineHeight:1.8,fontFamily:"monospace"}}>
              {"Backward through LAYERS only"}<br/>
              {"3 layers = 3 steps back"}<br/>
              {"Each layer has OWN weights"}
            </div>
          </div>
          <div style={{flex:1,minWidth:200,padding:12,background:"#08080d",borderRadius:8,border:"1px solid "+C.accent+"30"}}>
            <div style={{fontSize:10,fontWeight:700,color:C.accent,marginBottom:6}}>RNN (BPTT)</div>
            <div style={{fontSize:10,color:C.muted,lineHeight:1.8,fontFamily:"monospace"}}>
              {"Backward through LAYERS + TIME"}<br/>
              {"3 layers "+MUL+" 4 steps = 12 back!"}<br/>
              {"SAME weights "+ARR+" gradients COMBINED"}
            </div>
          </div>
        </div>
      </Card>

      <Insight icon={TARG} title="The Core Difference">
        RNN backprop has an <span style={{color:C.accent,fontWeight:700}}>extra dimension {DASH} TIME</span>. More multiplications means gradients <span style={{color:C.yellow,fontWeight:700}}>vanish faster</span>, which means <span style={{color:C.red,fontWeight:700}}>early words get forgotten</span>. This is the fundamental problem LSTM and GRU were invented to solve.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: VANISHING GRADIENT PROBLEM
   =============================================================== */
function TabVanishing() {
  var _w = useState(0.5); var wh = _w[0], setWh = _w[1];
  var _n = useState(10); var nSteps = _n[0], setNSteps = _n[1];

  var gradients = [];
  for (var i=0; i<nSteps; i++) { gradients.push(Math.pow(wh, nSteps-1-i)); }
  var maxG = Math.max.apply(null, gradients.concat([1]));
  var isVanish = wh < 1, isExplode = wh > 1;
  var bw = 740, bh = 240, barW = Math.max(8, Math.min(40, (bw-100)/nSteps - 4));

  return (
    <div>
      <SectionTitle title="The Vanishing Gradient Problem" subtitle={"Adjust W_hidden and time steps "+DASH+" watch gradients vanish or explode"} />

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{display:"flex",gap:30,justifyContent:"center",flexWrap:"wrap",alignItems:"flex-start"}}>
          <div style={{minWidth:200}}>
            <div style={{fontSize:10,color:C.muted,marginBottom:6}}>{"W_hidden = "+wh.toFixed(2)}</div>
            <input type="range" min={10} max={200} value={Math.round(wh*100)} onChange={function(e){setWh(parseInt(e.target.value)/100);}} style={{width:"100%"}} />
            <div style={{display:"flex",justifyContent:"space-between",fontSize:8,color:C.dim,marginTop:2}}><span>0.10</span><span style={{color:C.accent}}>1.00</span><span>2.00</span></div>
          </div>
          <div style={{minWidth:200}}>
            <div style={{fontSize:10,color:C.muted,marginBottom:6}}>{"Time steps = "+nSteps}</div>
            <input type="range" min={3} max={20} value={nSteps} onChange={function(e){setNSteps(parseInt(e.target.value));}} style={{width:"100%"}} />
            <div style={{display:"flex",justifyContent:"space-between",fontSize:8,color:C.dim,marginTop:2}}><span>3</span><span>20</span></div>
          </div>
          <div style={{textAlign:"center",minWidth:120}}>
            <div style={{fontSize:9,color:C.muted}}>Earliest gradient</div>
            <div style={{fontSize:20,fontWeight:800,color:isExplode?C.red:isVanish?C.yellow:C.green}}>{gradients[0]>1000?"overflow!":gradients[0]<0.0001?"~0":gradients[0].toFixed(6)}</div>
            <div style={{fontSize:9,color:isExplode?C.red:isVanish?C.yellow:C.green,fontWeight:700}}>{isExplode?"EXPLODING!":isVanish?"VANISHING":"Stable"}</div>
          </div>
        </div>
      </Card>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <text x={bw/2} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"Gradient magnitude at each time step (W_hidden="+wh.toFixed(2)+")"}</text>

          {/* Axis */}
          <line x1={60} y1={bh-35} x2={bw-20} y2={bh-35} stroke={C.dim} strokeWidth={1} />
          <text x={bw/2} y={bh-8} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"Time step (t=1 earliest "+LARR+"   "+ARR+" t="+nSteps+" latest)"}</text>

          {gradients.map(function(g,i) {
            var cappedG = Math.min(g, maxG);
            var barH = Math.max(2, (cappedG/maxG) * (bh-80));
            var x = 70 + i * ((bw-110)/nSteps);
            var cl = g > 0.5 ? C.green : g > 0.1 ? C.yellow : g > 0.01 ? C.orange : C.red;
            if (isExplode) cl = g > 10 ? C.red : g > 2 ? C.orange : C.yellow;
            return (<g key={i}>
              <rect x={x} y={bh-35-barH} width={barW} height={barH} rx={2} fill={cl+"40"} stroke={cl} strokeWidth={1} />
              {barW > 14 && <text x={x+barW/2} y={bh-40-barH} textAnchor="middle" fill={cl} fontSize={7} fontWeight={700} fontFamily="monospace">{g>999?"Inf":g<0.001?"~0":g.toFixed(3)}</text>}
              {barW > 14 && <text x={x+barW/2} y={bh-22} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{"t="+(i+1)}</text>}
            </g>);
          })}

          {/* Reference line at 1.0 if visible */}
          {maxG >= 1 && <g>
            <line x1={60} y1={bh-35-((1/maxG)*(bh-80))} x2={bw-20} y2={bh-35-((1/maxG)*(bh-80))} stroke={C.accent} strokeWidth={1} strokeDasharray="4,4" />
            <text x={55} y={bh-32-((1/maxG)*(bh-80))} textAnchor="end" fill={C.accent} fontSize={7} fontFamily="monospace">1.0</text>
          </g>}
        </svg>
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:10}}>The Whisper Analogy</div>
        <div style={{display:"flex",justifyContent:"center",gap:4,flexWrap:"wrap",marginBottom:10}}>
          {Array.from({length:Math.min(nSteps,12)},function(_,i) {
            var g = gradients[i];
            var opacity = Math.max(0.1,Math.min(1,g));
            if(isExplode) opacity = Math.min(1, g/maxG + 0.3);
            return (<div key={i} style={{textAlign:"center",fontSize:16,opacity:opacity,transition:"opacity 0.3s"}}>
              {isExplode&&g>5?"\\uD83D\\uDCA5":"\\uD83D\\uDDE3\\uFE0F"}
            </div>);
          })}
        </div>
        <div style={{textAlign:"center",fontSize:10,color:C.muted}}>
          {isVanish ? <span><span style={{color:C.yellow,fontWeight:700}}>Vanishing</span>: each person forgets a little {ARR} the last one hears nothing</span> :
           isExplode ? <span><span style={{color:C.red,fontWeight:700}}>Exploding</span>: each person exaggerates {ARR} the last one hears screaming</span> :
           <span><span style={{color:C.green,fontWeight:700}}>Stable</span>: the message passes through perfectly (W=1.0)</span>}
        </div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Solutions</div>
        <div style={{display:"flex",gap:8,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {p:"Exploding",s:"Gradient Clipping",d:"Cap gradient norm to max value",c:C.red},
            {p:"Vanishing",s:"LSTM / GRU",d:"Gates create gradient highways",c:C.green},
            {p:"Vanishing",s:"Skip Connections",d:"Shortcuts bypass time steps",c:C.blue},
            {p:"Exploding",s:"Orthogonal Init",d:"Keep eigenvalues near 1",c:C.purple},
          ].map(function(item,i) {
            return (<div key={i} style={{padding:"10px 14px",background:item.c+"08",borderRadius:8,border:"1px solid "+item.c+"20",minWidth:130,textAlign:"center"}}>
              <div style={{fontSize:8,color:item.c,marginBottom:2}}>{item.p}</div>
              <div style={{fontSize:10,fontWeight:700,color:C.text}}>{item.s}</div>
              <div style={{fontSize:8,color:C.muted,marginTop:2}}>{item.d}</div>
            </div>);
          })}
        </div>
      </Card>

      <Insight>
        When W_hidden {"<"} 1, gradients decay as <span style={{color:C.yellow,fontWeight:700}}>W^n</span> {DASH} after 20 steps, 0.5^20 = 0.00000095 (practically zero!). Early time steps get <span style={{color:C.red,fontWeight:700}}>no learning signal</span>. This is why the LSTM's <span style={{color:C.green,fontWeight:700}}>cell state highway</span> was a breakthrough.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: LSTM — GATES & NUMERIC WALKTHROUGH
   =============================================================== */
function TabLSTM() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _a = useState(false); var autoL = _a[0], setAutoL = _a[1];

  useEffect(function(){if(!autoL)return;var t=setInterval(function(){setStep(function(s){return(s+1)%6;});},4000);return function(){clearInterval(t);};}, [autoL]);

  var gates = [
    {name:"Overview",desc:"An LSTM adds a cell state (long-term memory) alongside the hidden state. Three gates control what to forget, what to store, and what to output.",icon:"\\uD83C\\uDFD7\\uFE0F"},
    {name:"Forget Gate",desc:'f = sigmoid(W_f '+MUL+' [h(t-1), X(t)] + b_f). Outputs 0-1 for each cell value. 0 = completely forget, 1 = completely keep. Example: new sentence started '+ARR+' forget old subject.',icon:"\\uD83D\\uDDD1\\uFE0F"},
    {name:"Input Gate",desc:'i = sigmoid(W_i '+MUL+' [h(t-1), X(t)]) controls HOW MUCH to write (0-1). C\\u0303 = tanh(W_c '+MUL+' [h(t-1), X(t)]) is WHAT to write (-1 to 1). Two parts working together.',icon:"\\u270D\\uFE0F"},
    {name:"Cell Update",desc:'C(t) = f '+MUL+' C(t-1) + i '+MUL+' C\\u0303. First term: old memories, selectively forgotten. Second term: new info, selectively written. If f=1 and i=0, cell state flows UNCHANGED '+DASH+' this is the magic!',icon:"\\uD83D\\uDD04"},
    {name:"Output Gate",desc:'o = sigmoid(W_o '+MUL+' [h(t-1), X(t)]). h(t) = o '+MUL+' tanh(C(t)). The cell holds full memory, but only a filtered version becomes the output. Not everything in memory is relevant right now.',icon:"\\uD83D\\uDCE4"},
    {name:"Gradient Highway",desc:'In vanilla RNN: gradient '+MUL+' W '+MUL+' tanh\\u2032 at every step '+ARR+' vanishes. In LSTM: \\u2202C(t)/\\u2202C(t-1) = f(t). If forget gate '+DASH+' 1, gradient flows UNDIMINISHED. After 10 steps: 0.9\\u00B9\\u2070 = 0.35 (meaningful!) vs RNN 0.5\\u00B9\\u2070 = 0.001 (dead!).',icon:"\\uD83D\\uDEE3\\uFE0F"},
  ];

  var bw = 740, bh = 280;
  var gateColors = [C.text, C.red, C.green, C.yellow, C.cyan, C.accent];

  return (
    <div>
      <SectionTitle title="LSTM: Long Short-Term Memory" subtitle={"Gates control what to remember, forget, and output "+DASH+" solving the vanishing gradient"} />

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
        {gates.map(function(g,i){var on=step===i;return (<button key={i} onClick={function(){setStep(i);setAutoL(false);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(on?gateColors[i]:C.border),background:on?gateColors[i]+"20":C.card,color:on?gateColors[i]:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{g.icon+" "+g.name}</button>);})}
        <button onClick={function(){setAutoL(!autoL);}} style={{padding:"8px 12px",borderRadius:8,border:"1.5px solid "+(autoL?C.yellow:C.border),background:autoL?C.yellow+"20":C.card,color:autoL?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoL?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>

          {/* Cell state highway */}
          <line x1={40} y1={50} x2={700} y2={50} stroke={step===3||step===5?C.yellow:C.dim} strokeWidth={step===3||step===5?3:2} />
          <text x={370} y={35} textAnchor="middle" fill={step===3||step===5?C.yellow:C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"Cell State C(t) "+DASH+" the memory highway"}</text>
          <text x={30} y={54} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">C(t-1)</text>
          <text x={710} y={54} fill={C.muted} fontSize={8} fontFamily="monospace">C(t)</text>
          <polygon points="704,50 697,46 697,54" fill={step===3||step===5?C.yellow:C.dim} />

          {/* Forget gate */}
          <circle cx={180} cy={50} r={20} fill={step===1?C.red+"25":"transparent"} stroke={step===1?C.red:C.dim} strokeWidth={step===1?2.5:1} />
          <text x={180} y={55} textAnchor="middle" fill={step===1?C.red:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">{MUL}</text>
          <text x={180} y={88} textAnchor="middle" fill={step===1?C.red:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">FORGET</text>
          <text x={180} y={100} textAnchor="middle" fill={step===1?C.red:C.dim} fontSize={8} fontFamily="monospace">f(t)</text>
          <line x1={180} y1={108} x2={180} y2={160} stroke={step===1?C.red:C.dim} strokeWidth={1} strokeDasharray="3,3" />

          {/* Input gate + candidate */}
          <circle cx={330} cy={50} r={20} fill={step===2||step===3?C.green+"25":"transparent"} stroke={step===2||step===3?C.green:C.dim} strokeWidth={step===2||step===3?2.5:1} />
          <text x={330} y={55} textAnchor="middle" fill={step===2||step===3?C.green:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">+</text>
          <text x={330} y={88} textAnchor="middle" fill={step===2?C.green:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">INPUT</text>
          <text x={330} y={100} textAnchor="middle" fill={step===2?C.green:C.dim} fontSize={8} fontFamily="monospace">{"i(t) "+MUL+" C\\u0303(t)"}</text>
          <line x1={330} y1={108} x2={330} y2={160} stroke={step===2?C.green:C.dim} strokeWidth={1} strokeDasharray="3,3" />

          {/* Output gate */}
          <circle cx={530} cy={170} r={20} fill={step===4?C.cyan+"25":"transparent"} stroke={step===4?C.cyan:C.dim} strokeWidth={step===4?2.5:1} />
          <text x={530} y={175} textAnchor="middle" fill={step===4?C.cyan:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">{MUL}</text>
          <text x={530} y={210} textAnchor="middle" fill={step===4?C.cyan:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">OUTPUT</text>
          <text x={530} y={222} textAnchor="middle" fill={step===4?C.cyan:C.dim} fontSize={8} fontFamily="monospace">o(t)</text>

          {/* tanh after cell state */}
          <line x1={530} y1={50} x2={530} y2={110} stroke={step===4?C.cyan:C.dim} strokeWidth={1} />
          <rect x={510} y={112} width={40} height={20} rx={4} fill={step===4?C.purple+"15":"transparent"} stroke={step===4?C.purple:C.dim} />
          <text x={530} y={126} textAnchor="middle" fill={step===4?C.purple:C.dim} fontSize={8} fontWeight={700} fontFamily="monospace">tanh</text>
          <line x1={530} y1={132} x2={530} y2={150} stroke={step===4?C.cyan:C.dim} strokeWidth={1} />

          {/* h(t) output */}
          <line x1={550} y1={170} x2={700} y2={170} stroke={step===4?C.cyan:C.dim} strokeWidth={2} />
          <polygon points="704,170 697,166 697,174" fill={step===4?C.cyan:C.dim} />
          <text x={710} y={174} fill={step===4?C.cyan:C.muted} fontSize={8} fontFamily="monospace">h(t)</text>

          {/* Concat input box */}
          <rect x={200} y={165} width={200} height={40} rx={8} fill={C.card} stroke={C.border} />
          <text x={300} y={180} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">h(t-1) + X(t) concatenated</text>
          <text x={300} y={196} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"4 sigmoids/tanh "+ARR+" f, i, C\\u0303, o"}</text>

          {/* Lines from concat to gates */}
          <line x1={180} y1={160} x2={250} y2={165} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3" />
          <line x1={330} y1={160} x2={330} y2={165} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3" />
          <line x1={530} y1={222} x2={400} y2={205} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3" />

          {/* h(t-1) input */}
          <text x={30} y={174} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">h(t-1)</text>
          <line x1={35} y1={170} x2={200} y2={180} stroke={C.dim} strokeWidth={1} />

          {/* x(t) input */}
          <text x={250} y={250} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">X(t)</text>
          <line x1={250} y1={242} x2={280} y2={205} stroke={C.dim} strokeWidth={1} />
          <polygon points="280,205 275,212 282,212" fill={C.dim} />

          {/* Gradient highway highlight */}
          {step===5 && <g>
            <rect x={50} y={40} width={650} height={20} rx={10} fill={C.accent+"10"} stroke={C.accent} strokeWidth={1.5} strokeDasharray="6,3" />
            <text x={370} y={18} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={800} fontFamily="monospace">{"GRADIENT HIGHWAY "+DASH+" gradients flow directly, no W multiplication!"}</text>
          </g>}

          {/* 4x params note */}
          <rect x={560} y={240} width={160} height={22} rx={4} fill={C.purple+"08"} stroke={C.purple+"25"} />
          <text x={640} y={255} textAnchor="middle" fill={C.purple} fontSize={8} fontWeight={700} fontFamily="monospace">{"4 weight matrices "+ARR+" 4"+MUL+" params"}</text>
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:gateColors[step]}}>
        <div style={{fontSize:13,fontWeight:700,color:gateColors[step],marginBottom:4}}>{gates[step].icon+" "+gates[step].name}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{gates[step].desc}</div>
      </Card>

      {/* Numeric walkthrough */}
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:12}}>{"Numeric Walkthrough: "+LQ+"not"+RQ+" "+ARR+" "+LQ+"good"+RQ}</div>
        <div style={{display:"flex",gap:12,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {t:'t=1 "not"',f:"0.52",i:"0.71",cand:"-0.66",c:"-0.47",o:"0.65",h:"-0.29",cl:C.red,
             detail:'f=0.52 (forget half), i=0.71 (store 71%), c\\u0303=-0.66 (negative content). Cell = 0'+ARR+'-0.47. Memory: NEGATIVE.'},
            {t:'t=2 "good"',f:"0.57",i:"0.67",cand:"0.54",c:"0.09",o:"0.62",h:"0.056",cl:C.green,
             detail:'f=0.57 (keep 57% of "not" memory). i=0.67, c\\u0303=+0.54. Cell = -0.27+0.36 = 0.09. "not" partially survived!'},
          ].map(function(ts,idx) {
            return (<div key={idx} style={{flex:1,minWidth:280,padding:14,background:"#08080d",borderRadius:8,border:"1px solid "+ts.cl+"30"}}>
              <div style={{fontSize:11,fontWeight:700,color:ts.cl,marginBottom:8}}>{ts.t}</div>
              <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:8}}>
                {[{l:"f",v:ts.f,c:C.red},{l:"i",v:ts.i,c:C.green},{l:"c\\u0303",v:ts.cand,c:C.purple},{l:"c",v:ts.c,c:C.yellow},{l:"o",v:ts.o,c:C.cyan},{l:"h",v:ts.h,c:C.accent}].map(function(g,j){
                  return (<div key={j} style={{textAlign:"center",minWidth:36}}>
                    <div style={{fontSize:7,color:C.dim}}>{g.l}</div>
                    <div style={{fontSize:12,fontWeight:800,color:g.c}}>{g.v}</div>
                  </div>);
                })}
              </div>
              <div style={{fontSize:9,color:C.muted,lineHeight:1.6}}>{ts.detail}</div>
            </div>);
          })}
        </div>
      </Card>

      <Insight icon={TARG} title="Why LSTM Solves Vanishing Gradients">
        The cell state update is C(t) = f{MUL}C(t-1) + i{MUL}C\u0303. The gradient of C(t) w.r.t. C(t-1) is simply <span style={{color:C.green,fontWeight:700}}>f(t)</span> {DASH} no W multiplication, no tanh squashing. If f {"\u2248"} 1, gradients flow through <span style={{color:C.accent,fontWeight:700}}>undiminished</span>. After 10 steps: <span style={{color:C.green}}>0.9\u00B9\u2070 = 0.35</span> (meaningful!) vs RNN <span style={{color:C.red}}>0.5\u00B9\u2070 = 0.001</span> (dead!).
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: GRU & EVOLUTION
   =============================================================== */
function TabGRUEvolution() {
  var _s = useState(0); var sel = _s[0], setSel = _s[1];

  var models = [
    {name:"Simple RNN",year:1986,color:C.blue,gates:0,states:1,params:"1x",strength:"Has memory",weakness:"Forgets quickly (vanishing gradient)",
     eq:"h(t) = tanh(W "+MUL+" [h(t-1), x(t)] + b)", insight:"The foundation. Introduced the idea of recurrence and hidden states, but the vanishing gradient problem limits it to short sequences."},
    {name:"LSTM",year:1997,color:C.green,gates:3,states:2,params:"4x",strength:"Long-range memory via cell state highway",weakness:"Complex, expensive (4 weight matrices)",
     eq:"f,i,o = sigmoid(...)   C(t) = f"+MUL+"C(t-1) + i"+MUL+"C\\u0303   h(t) = o"+MUL+"tanh(C(t))", insight:"Forget + Input + Output gates. Two states (cell + hidden). The cell state acts as a gradient highway. Can remember across 100+ time steps."},
    {name:"GRU",year:2014,color:C.yellow,gates:2,states:1,params:"3x",strength:"Nearly as good as LSTM, 25% fewer params",weakness:"Less expressive than LSTM on very long sequences",
     eq:"r = sigmoid(...)   z = sigmoid(...)   h(t) = (1-z)"+MUL+"h(t-1) + z"+MUL+"h\\u0303", insight:"Reset + Update gates. One state only. The update gate does double duty (forget + input linked by z and 1-z). Faster to train, often comparable results."},
    {name:"Transformer",year:2017,color:C.accent,gates:0,states:0,params:"varies",strength:"Every word sees every other word directly (attention)",weakness:"Quadratic memory in sequence length",
     eq:"Attention(Q,K,V) = softmax(QK\\u1D40 / "+"\u221A"+"d_k) "+MUL+" V", insight:"No recurrence at all. Self-attention lets every position attend to every other position directly. This is what GPT, BERT, and Claude use. Solved the long-range problem entirely."},
  ];

  var m = models[sel];

  return (
    <div>
      <SectionTitle title="GRU & The Evolution of Sequence Models" subtitle={"From simple memory to attention "+DASH+" each solved a specific problem"} />

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
        {models.map(function(mod,i){var on=sel===i;return (<button key={i} onClick={function(){setSel(i);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?mod.color:C.border),background:on?mod.color+"20":C.card,color:on?mod.color:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{mod.name+" ("+mod.year+")"}</button>);})}
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:m.color}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:16}}>
          <div>
            <div style={{fontSize:20,fontWeight:800,color:m.color}}>{m.name}</div>
            <div style={{fontSize:11,color:C.muted,marginTop:2}}>{m.year}</div>
            <div style={{fontSize:11,color:C.text,marginTop:10,lineHeight:1.6}}>{m.strength}</div>
          </div>
          <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
            {[{l:"GATES",v:m.gates},{l:"STATES",v:m.states},{l:"PARAMS",v:m.params}].map(function(d,i){
              return (<div key={i} style={{textAlign:"center"}}><div style={{fontSize:8,color:C.muted}}>{d.l}</div><div style={{fontSize:22,fontWeight:800,color:m.color}}>{d.v}</div></div>);
            })}
          </div>
        </div>
        <div style={{marginTop:14,padding:"10px 14px",background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          <div style={{fontSize:9,color:C.muted,marginBottom:4}}>CORE EQUATION</div>
          <div style={{fontSize:10,color:m.color,fontFamily:"monospace",lineHeight:1.8}}>{m.eq}</div>
        </div>
        <div style={{marginTop:12,fontSize:11,color:C.muted,lineHeight:1.7,fontStyle:"italic",borderLeft:"3px solid "+m.color+"40",paddingLeft:12}}>{m.insight}</div>
        {m.weakness && <div style={{marginTop:8,fontSize:10,color:C.red,fontFamily:"monospace"}}>{WARN+" "+m.weakness}</div>}
      </Card>

      {/* LSTM vs GRU comparison */}
      {(sel===1||sel===2) && <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>LSTM vs GRU Side by Side</div>
        <div style={{display:"flex",gap:12,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {aspect:"Gates",lstm:"3 (forget, input, output)",gru:"2 (reset, update)"},
            {aspect:"States",lstm:"Cell + Hidden (2 paths)",gru:"Hidden only (1 path)"},
            {aspect:"Key Difference",lstm:"f and i are INDEPENDENT",gru:"z and (1-z) are LINKED"},
            {aspect:"Parameters",lstm:"4 weight matrices",gru:"3 weight matrices (~25% less)"},
            {aspect:"Best For",lstm:"Very long sequences, large data",gru:"Speed, smaller datasets"},
          ].map(function(row,i){
            return (<div key={i} style={{width:"100%",display:"flex",gap:8}}>
              <div style={{width:110,fontSize:9,color:C.muted,textAlign:"right",paddingTop:4,fontWeight:700}}>{row.aspect}</div>
              <div style={{flex:1,padding:"6px 10px",background:C.green+"08",borderRadius:6,border:"1px solid "+C.green+"20",fontSize:9,color:C.green,fontFamily:"monospace"}}>{row.lstm}</div>
              <div style={{flex:1,padding:"6px 10px",background:C.yellow+"08",borderRadius:6,border:"1px solid "+C.yellow+"20",fontSize:9,color:C.yellow,fontFamily:"monospace"}}>{row.gru}</div>
            </div>);
          })}
        </div>
      </Card>}

      {/* GRU detail */}
      {sel===2 && <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:10}}>GRU: The Update Gate Does Double Duty</div>
        <div style={{display:"flex",justifyContent:"center",gap:24,flexWrap:"wrap"}}>
          <div style={{textAlign:"center",padding:"12px 16px",background:C.yellow+"08",borderRadius:8,border:"1px solid "+C.yellow+"20"}}>
            <div style={{fontSize:9,color:C.muted}}>z {"\u2248"} 0</div>
            <div style={{fontSize:12,fontWeight:800,color:C.yellow,marginTop:4}}>Keep Old</div>
            <div style={{fontSize:9,color:C.muted,marginTop:2}}>{"h(t) "+"\u2248"+" h(t-1)"}</div>
            <div style={{fontSize:8,color:C.dim,marginTop:2}}>Like LSTM forget=1</div>
          </div>
          <div style={{display:"flex",alignItems:"center",fontSize:16,color:C.dim}}>{"\u2194"}</div>
          <div style={{textAlign:"center",padding:"12px 16px",background:C.yellow+"08",borderRadius:8,border:"1px solid "+C.yellow+"20"}}>
            <div style={{fontSize:9,color:C.muted}}>z {"\u2248"} 1</div>
            <div style={{fontSize:12,fontWeight:800,color:C.yellow,marginTop:4}}>Replace New</div>
            <div style={{fontSize:9,color:C.muted,marginTop:2}}>{"h(t) "+"\u2248"+" h\\u0303(t)"}</div>
            <div style={{fontSize:8,color:C.dim,marginTop:2}}>Like LSTM input=1</div>
          </div>
        </div>
        <div style={{textAlign:"center",marginTop:10,fontSize:9,color:C.muted}}>
          LSTM can keep everything AND add everything (independent gates). GRU forces a <span style={{color:C.yellow,fontWeight:700}}>tradeoff</span>: more old = less new.
        </div>
      </Card>}

      {/* Evolution timeline */}
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:16}}>The Full Evolution</div>
        <div style={{display:"flex",flexDirection:"column",gap:2,paddingLeft:20}}>
          {models.map(function(mod,i) {
            var on = sel===i;
            return (<div key={i}>
              <div style={{display:"flex",alignItems:"center",gap:12,padding:"8px 12px",background:on?mod.color+"12":"transparent",borderRadius:8,border:on?"1px solid "+mod.color+"30":"1px solid transparent",cursor:"pointer",transition:"all 0.2s"}} onClick={function(){setSel(i);}}>
                <div style={{width:8,height:8,borderRadius:4,background:mod.color,flexShrink:0}} />
                <div style={{fontSize:12,fontWeight:800,color:on?mod.color:C.muted,minWidth:100}}>{mod.name}</div>
                <div style={{fontSize:10,color:C.dim}}>{mod.year}</div>
                <div style={{fontSize:9,color:on?C.text:C.dim,flex:1}}>{mod.strength}</div>
              </div>
              {i<models.length-1 && <div style={{marginLeft:3,width:2,height:16,background:C.dim}} />}
            </div>);
          })}
        </div>
      </Card>

      <Insight icon={TARG} title="The Trend">
        Each model solved a specific problem: <span style={{color:C.blue}}>RNN</span> added memory, <span style={{color:C.green}}>LSTM</span> solved vanishing gradients with gates, <span style={{color:C.yellow}}>GRU</span> simplified it, and <span style={{color:C.accent,fontWeight:700}}>Transformers</span> abandoned recurrence entirely {DASH} letting every word attend to every other word directly.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Basic RNN", "BPTT", "Vanishing Gradient", "LSTM", "GRU & Evolution"];
  return (
    <div style={{ background:C.bg, minHeight:"100vh", padding:"24px 16px", fontFamily:"'JetBrains Mono','SF Mono',monospace", color:C.text, maxWidth:960, margin:"0 auto" }}>
      <div style={{textAlign:"center",marginBottom:16}}>
        <div style={{ fontSize:22, fontWeight:800, background:"linear-gradient(135deg,"+C.accent+","+C.blue+")", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", display:"inline-block" }}>Recurrent Neural Networks</div>
        <div style={{fontSize:11,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough "+DASH+" from basic RNN to Transformer"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab===0 && <TabBasicRNN />}
      {tab===1 && <TabBPTT />}
      {tab===2 && <TabVanishing />}
      {tab===3 && <TabLSTM />}
      {tab===4 && <TabGRUEvolution />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
<script>
(function() {
  function resizeFrame() {
    var root = document.getElementById("root");
    if (!root) return;
    var h = root.scrollHeight + 60;
    if (window.frameElement) {
      window.frameElement.style.height = h + "px";
    }
    try {
      window.parent.postMessage({ type: "streamlit:setFrameHeight", height: h }, "*");
    } catch(e) {}
  }
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
  if (document.readyState === "complete") { setTimeout(init, 500); }
  else { window.addEventListener("load", function() { setTimeout(init, 500); }); }
  setInterval(resizeFrame, 1000);
})();
</script>
</body>
</html>
"""

RNN_VISUAL_HEIGHT = 1800



# """
# Self-contained HTML for the RNN (Recurrent Neural Network) interactive walkthrough.
# Covers: Basic RNN operation, BPTT, Vanishing Gradient Problem, LSTM gates,
# and GRU + Evolution timeline (Simple RNN → LSTM → GRU → Transformer).
# Embed in Streamlit via st.components.v1.html(RNN_VISUAL_HTML, height=RNN_VISUAL_HEIGHT).
# """
#
# RNN_VISUAL_HTML = """
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
#   input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #ff6b35; }
#   @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
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
#   accent: "#ff6b35", blue: "#4ecdc4", purple: "#a78bfa",
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
#       maxWidth: 750, margin: "16px auto 0",
#       padding: "16px 22px", background: "rgba(255,107,53,0.06)",
#       borderRadius: 10, border: "1px solid rgba(255,107,53,0.2)",
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
# function tanh(x) { var e = Math.exp(2*x); return (e-1)/(e+1); }
# function sigmoid(x) { return 1/(1+Math.exp(-x)); }
# function fmt(v) { return (Math.round(v*1000)/1000).toString(); }
# function fmt2(v) { return (Math.round(v*100)/100).toString(); }
#
#
# /* ===============================================================
#    TAB 1: BASIC RNN — "not good" walkthrough
#    =============================================================== */
# function TabBasicRNN() {
#   var _s = useState(0); var step = _s[0], setStep = _s[1];
#   var _a = useState(false); var autoPlay = _a[0], setAutoPlay = _a[1];
#
#   useEffect(function(){if(!autoPlay)return;var t=setInterval(function(){setStep(function(s){return(s+1)%3;});},3000);return function(){clearInterval(t);};}, [autoPlay]);
#
#   var W_in = 0.5, W_h = 0.9;
#   var words = [{w:'"not"',x:-1,c:C.red},{w:'"good"',x:0.8,c:C.green}];
#   var h0 = 0;
#   var h1 = tanh(W_in*(-1)+W_h*0);
#   var h2 = tanh(W_in*0.8+W_h*h1);
#   var h_good_alone = tanh(W_in*0.8);
#
#   var steps = [
#     {title:"Initial State",desc:"No memory yet. h(0) = 0. The network starts with a blank slate."},
#     {title:'Processing "not"',desc:'X(1) = -1. The RNN computes h(1) = tanh(0.5'+MUL+'(-1) + 0.9'+MUL+'0) = tanh(-0.5) = '+fmt(h1)+'. The hidden state is now negative '+DASH+' the network has a "negative" memory.'},
#     {title:'Processing "good"',desc:'X(2) = 0.8. The RNN computes h(2) = tanh(0.5'+MUL+'0.8 + 0.9'+MUL+'('+fmt(h1)+')) = tanh('+fmt2(0.4+W_h*h1)+') = '+fmt(h2)+'. Even though "good" is positive, the memory of "not" dragged the result negative!'},
#   ];
#
#   var hVals = [h0,h1,h2];
#   var bw = 740, bh = 260;
#
#   return (
#     <div>
#       <SectionTitle title="How an RNN Works" subtitle={"Watch a single neuron process "+LQ+"not good"+RQ+" one word at a time, carrying memory forward"} />
#
#       <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:16}}>
#         {steps.map(function(s,i){var on=step===i;return (<button key={i} onClick={function(){setStep(i);setAutoPlay(false);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?C.accent:C.border),background:on?C.accent+"20":C.card,color:on?C.accent:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{i===0?"Start":"t="+i}</button>);})}
#         <button onClick={function(){setAutoPlay(!autoPlay);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(autoPlay?C.yellow:C.border),background:autoPlay?C.yellow+"20":C.card,color:autoPlay?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoPlay?PAUSE:PLAY}</button>
#       </div>
#
#       <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
#         <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
#           <text x={bw/2} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"h(t) = tanh(W_input "+MUL+" X(t)  +  W_hidden "+MUL+" h(t-1))"}</text>
#
#           {/* h(0) */}
#           <rect x={40} y={80} width={80} height={55} rx={8} fill={step===0?C.blue+"15":C.card} stroke={step===0?C.blue:C.dim} strokeWidth={step===0?2:1} />
#           <text x={80} y={97} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">h(0)</text>
#           <text x={80} y={120} textAnchor="middle" fill={C.blue} fontSize={20} fontWeight={800} fontFamily="monospace">0</text>
#           <text x={80} y={155} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">no memory</text>
#
#           {/* Arrow h0 -> RNN1 */}
#           <line x1={120} y1={107} x2={190} y2={107} stroke={step>=1?C.blue:C.dim} strokeWidth={step>=1?2:1} />
#           <polygon points="194,107 187,103 187,111" fill={step>=1?C.blue:C.dim} />
#           <text x={155} y={99} textAnchor="middle" fill={step>=1?C.blue:C.dim} fontSize={8} fontFamily="monospace">h(0)=0</text>
#
#           {/* RNN Neuron 1 */}
#           <rect x={200} y={60} width={110} height={95} rx={10} fill={step===1?C.accent+"15":C.card} stroke={step===1?C.accent:C.dim} strokeWidth={step===1?2.5:1} style={step===1?{animation:"pulse 1.5s infinite"}:{}} />
#           <text x={255} y={82} textAnchor="middle" fill={step>=1?C.accent:C.dim} fontSize={11} fontWeight={700} fontFamily="monospace">RNN</text>
#           <text x={255} y={100} textAnchor="middle" fill={step>=1?C.accent:C.dim} fontSize={9} fontFamily="monospace">t=1</text>
#           {step>=1 && <text x={255} y={120} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"tanh(-0.5)"}</text>}
#           {step>=1 && <text x={255} y={140} textAnchor="middle" fill={C.accent} fontSize={14} fontWeight={800} fontFamily="monospace">{fmt2(h1)}</text>}
#
#           {/* Input "not" */}
#           <rect x={215} y={185} width={80} height={35} rx={6} fill={step>=1?C.red+"15":"transparent"} stroke={step>=1?C.red:C.dim} strokeWidth={step>=1?1.5:1} />
#           <text x={255} y={207} textAnchor="middle" fill={step>=1?C.red:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{'"not" = -1'}</text>
#           <line x1={255} y1={185} x2={255} y2={155} stroke={step>=1?C.red:C.dim} strokeWidth={step>=1?1.5:1} />
#           <polygon points="255,157 251,163 259,163" fill={step>=1?C.red:C.dim} />
#
#           {/* Arrow RNN1 -> RNN2 */}
#           <line x1={310} y1={107} x2={410} y2={107} stroke={step>=2?C.accent:C.dim} strokeWidth={step>=2?2:1} />
#           <polygon points="414,107 407,103 407,111" fill={step>=2?C.accent:C.dim} />
#           <text x={360} y={99} textAnchor="middle" fill={step>=2?C.accent:C.dim} fontSize={8} fontFamily="monospace">{"h(1)="+fmt2(h1)}</text>
#
#           {/* RNN Neuron 2 */}
#           <rect x={420} y={60} width={110} height={95} rx={10} fill={step===2?C.accent+"15":C.card} stroke={step===2?C.accent:C.dim} strokeWidth={step===2?2.5:1} style={step===2?{animation:"pulse 1.5s infinite"}:{}} />
#           <text x={475} y={82} textAnchor="middle" fill={step>=2?C.accent:C.dim} fontSize={11} fontWeight={700} fontFamily="monospace">RNN</text>
#           <text x={475} y={100} textAnchor="middle" fill={step>=2?C.accent:C.dim} fontSize={9} fontFamily="monospace">t=2</text>
#           {step>=2 && <text x={475} y={120} textAnchor="middle" fill={C.yellow} fontSize={9} fontFamily="monospace">{"tanh("+fmt2(0.4+W_h*h1)+")"}</text>}
#           {step>=2 && <text x={475} y={140} textAnchor="middle" fill={C.accent} fontSize={14} fontWeight={800} fontFamily="monospace">{fmt2(h2)}</text>}
#
#           {/* Input "good" */}
#           <rect x={435} y={185} width={80} height={35} rx={6} fill={step>=2?C.green+"15":"transparent"} stroke={step>=2?C.green:C.dim} strokeWidth={step>=2?1.5:1} />
#           <text x={475} y={207} textAnchor="middle" fill={step>=2?C.green:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{'"good" = 0.8'}</text>
#           <line x1={475} y1={185} x2={475} y2={155} stroke={step>=2?C.green:C.dim} strokeWidth={step>=2?1.5:1} />
#           <polygon points="475,157 471,163 479,163" fill={step>=2?C.green:C.dim} />
#
#           {/* Output */}
#           {step>=2 && <g>
#             <line x1={530} y1={107} x2={590} y2={107} stroke={C.accent} strokeWidth={2} />
#             <polygon points="594,107 587,103 587,111" fill={C.accent} />
#             <rect x={600} y={80} width={110} height={55} rx={8} fill={h2<0?C.red+"15":C.green+"15"} stroke={h2<0?C.red:C.green} strokeWidth={2} />
#             <text x={655} y={100} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">Output</text>
#             <text x={655} y={122} textAnchor="middle" fill={h2<0?C.red:C.green} fontSize={16} fontWeight={800} fontFamily="monospace">{fmt(h2)}</text>
#           </g>}
#
#           {/* Shared weights label */}
#           <rect x={200} y={235} width={330} height={22} rx={4} fill={C.purple+"10"} stroke={C.purple+"30"} />
#           <text x={365} y={250} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">{"Same weights reused: W_input=0.5  W_hidden=0.9  b=0"}</text>
#         </svg>
#       </div>
#
#       <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:4}}>{"Step "+(step+1)+": "+steps[step].title}</div>
#         <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{steps[step].desc}</div>
#       </Card>
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:12}}>Why Memory Matters</div>
#         <div style={{display:"flex",justifyContent:"center",gap:24,flexWrap:"wrap"}}>
#           <div style={{textAlign:"center",padding:"12px 20px",background:C.green+"08",borderRadius:8,border:"1px solid "+C.green+"20"}}>
#             <div style={{fontSize:9,color:C.muted,marginBottom:4}}>{"\"good\" ALONE (no memory)"}</div>
#             <div style={{fontSize:22,fontWeight:800,color:C.green}}>{fmt2(h_good_alone)}</div>
#             <div style={{fontSize:9,color:C.green}}>Positive!</div>
#           </div>
#           <div style={{display:"flex",alignItems:"center",fontSize:20,color:C.dim}}>vs</div>
#           <div style={{textAlign:"center",padding:"12px 20px",background:C.red+"08",borderRadius:8,border:"1px solid "+C.red+"20"}}>
#             <div style={{fontSize:9,color:C.muted,marginBottom:4}}>{"\"not good\" (with memory)"}</div>
#             <div style={{fontSize:22,fontWeight:800,color:C.red}}>{fmt2(h2)}</div>
#             <div style={{fontSize:9,color:C.red}}>Negative!</div>
#           </div>
#         </div>
#         <div style={{textAlign:"center",marginTop:10,fontSize:10,color:C.muted}}>
#           The memory of <span style={{color:C.red,fontWeight:700}}>"not"</span> changed the outcome. The RNN understood context.
#         </div>
#       </Card>
#
#       <Insight>
#         An RNN is the <span style={{color:C.accent,fontWeight:700}}>same neuron reused</span> at each time step {DASH} this is called "unrolling." It carries a <span style={{color:C.blue,fontWeight:700}}>hidden state h(t)</span> that acts as memory. The same weights (<span style={{color:C.purple}}>W_input, W_hidden</span>) are shared across all steps {DASH} just like a CNN filter is shared across all positions.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 2: BACKPROPAGATION THROUGH TIME (BPTT)
#    =============================================================== */
# function TabBPTT() {
#   var _s = useState(0); var phase = _s[0], setPhase = _s[1];
#   var _a = useState(false); var autoP = _a[0], setAutoP = _a[1];
#
#   var phases = [
#     {title:"Forward Pass",desc:"The sequence "+LQ+"The cat sat"+RQ+" is processed left-to-right, one word at a time. Each step computes h(t) = tanh(W_input "+MUL+" X(t) + W_hidden "+MUL+" h(t-1)). The same weights are reused at every step."},
#     {title:"Compute Loss",desc:'After the full sequence is processed, the network predicts "ran" but the correct word is "on." The loss measures how wrong the prediction is.'},
#     {title:"Backward Through Time",desc:"Gradients flow RIGHT to LEFT through every time step. At each step, the gradient is multiplied by W_hidden "+DASH+" this repeated multiplication is what causes vanishing gradients."},
#     {title:"Combine & Update",desc:"Since the SAME weights are used at every step, gradients from ALL time steps are summed together to compute one weight update. gradient_total = grad(t=1) + grad(t=2) + grad(t=3)."},
#   ];
#
#   useEffect(function(){if(!autoP)return;var t=setInterval(function(){setPhase(function(p){return(p+1)%4;});},4000);return function(){clearInterval(t);};}, [autoP]);
#
#   var bw = 760, bh = 300;
#   var words = ["The","cat","sat"];
#   var cx = [100,280,460];
#   var grads = [0.25,0.5,1.0];
#
#   return (
#     <div>
#       <SectionTitle title="Backpropagation Through Time (BPTT)" subtitle={"How RNNs learn "+DASH+" gradients must flow backward through LAYERS and TIME"} />
#
#       <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
#         {phases.map(function(p,i){var on=phase===i;var cl=[C.blue,C.red,C.yellow,C.green][i]; return (<button key={i} onClick={function(){setPhase(i);setAutoP(false);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?cl:C.border),background:on?cl+"20":C.card,color:on?cl:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{(i+1)+". "+p.title}</button>);})}
#         <button onClick={function(){setAutoP(!autoP);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(autoP?C.yellow:C.border),background:autoP?C.yellow+"20":C.card,color:autoP?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoP?PAUSE:PLAY}</button>
#       </div>
#
#       <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
#         <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
#
#           {/* Forward pass arrows and nodes */}
#           {words.map(function(w,i) {
#             var x = cx[i], on = phase===0;
#             return (<g key={"fwd"+i}>
#               {/* Word input */}
#               <rect x={x-30} y={180} width={60} height={28} rx={6} fill={on?C.blue+"15":"transparent"} stroke={on?C.blue:C.dim} />
#               <text x={x} y={198} textAnchor="middle" fill={on?C.blue:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">{'"'+w+'"'}</text>
#               <line x1={x} y1={180} x2={x} y2={150} stroke={on?C.blue:C.dim} strokeWidth={on?1.5:1} />
#               <polygon points={x+",147 "+(x-3)+",153 "+(x+3)+",153"} fill={on?C.blue:C.dim} />
#
#               {/* RNN cell */}
#               <rect x={x-42} y={95} width={84} height={55} rx={8} fill={on?C.accent+"12":C.card} stroke={on?C.accent:C.dim} strokeWidth={on?2:1} />
#               <text x={x} y={115} textAnchor="middle" fill={on?C.accent:C.dim} fontSize={10} fontWeight={700} fontFamily="monospace">RNN</text>
#               <text x={x} y={132} textAnchor="middle" fill={on?C.muted:C.dim} fontSize={9} fontFamily="monospace">{"t="+(i+1)}</text>
#
#               {/* Forward arrow to next */}
#               {i<2 && <g>
#                 <line x1={x+42} y1={122} x2={cx[i+1]-42} y2={122} stroke={on?C.accent:C.dim} strokeWidth={on?2:1} />
#                 <polygon points={(cx[i+1]-44)+",122 "+(cx[i+1]-50)+",118 "+(cx[i+1]-50)+",126"} fill={on?C.accent:C.dim} />
#                 <text x={(x+cx[i+1])/2} y={115} textAnchor="middle" fill={on?C.accent:C.dim} fontSize={8} fontFamily="monospace">{"h("+(i+1)+")"}</text>
#               </g>}
#             </g>);
#           })}
#
#           {/* Forward label */}
#           {phase===0 && <g>
#             <text x={bw/2} y={25} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={700} fontFamily="monospace">{"FORWARD PASS "+ARR+" (left to right)"}</text>
#             <line x1={100} y1={38} x2={550} y2={38} stroke={C.blue} strokeWidth={2} />
#             <polygon points="554,38 547,34 547,42" fill={C.blue} />
#           </g>}
#
#           {/* Loss */}
#           {phase>=1 && <g>
#             <line x1={502} y1={122} x2={580} y2={122} stroke={C.red} strokeWidth={2} />
#             <polygon points="584,122 577,118 577,126" fill={C.red} />
#             <rect x={590} y={95} width={120} height={55} rx={8} fill={C.red+"12"} stroke={C.red} strokeWidth={2} />
#             <text x={650} y={115} textAnchor="middle" fill={C.red} fontSize={10} fontWeight={700} fontFamily="monospace">{WARN+" LOSS"}</text>
#             <text x={650} y={134} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{'"ran" '+MUL+' "on"'}</text>
#           </g>}
#
#           {/* Backward arrows */}
#           {phase>=2 && <g>
#             <text x={bw/2} y={25} textAnchor="middle" fill={C.yellow} fontSize={11} fontWeight={700} fontFamily="monospace">{LARR+" BACKWARD PASS (right to left)"}</text>
#             <line x1={550} y1={38} x2={100} y2={38} stroke={C.yellow} strokeWidth={2} />
#             <polygon points="96,38 103,34 103,42" fill={C.yellow} />
#
#             {words.map(function(_,i) {
#               var x = cx[2-i], g2 = grads[2-i];
#               var bh2 = Math.max(8, g2*40);
#               return (<g key={"bk"+i}>
#                 <rect x={x-20} y={60-bh2} width={40} height={bh2} rx={4} fill={C.yellow+(g2>0.5?"50":"25")} stroke={C.yellow} strokeWidth={1} />
#                 <text x={x} y={55-bh2} textAnchor="middle" fill={C.yellow} fontSize={9} fontWeight={700} fontFamily="monospace">{g2.toFixed(2)}</text>
#               </g>);
#             })}
#             <text x={cx[0]} y={265} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"grad "+MUL+" W "+MUL+" W"}</text>
#             <text x={cx[1]} y={265} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"grad "+MUL+" W"}</text>
#             <text x={cx[2]} y={265} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"full grad"}</text>
#           </g>}
#
#           {/* Combine phase */}
#           {phase===3 && <g>
#             <rect x={80} y={230} width={440} height={40} rx={8} fill={C.green+"10"} stroke={C.green+"40"} />
#             <text x={300} y={254} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">{"Total gradient = 0.25 + 0.5 + 1.0 = 1.75  "+ARR+"  ONE weight update"}</text>
#           </g>}
#
#           {/* Shared weights */}
#           <rect x={80} y={275} width={440} height={20} rx={4} fill={C.purple+"08"} stroke={C.purple+"25"} />
#           <text x={300} y={289} textAnchor="middle" fill={C.purple} fontSize={8} fontWeight={700} fontFamily="monospace">{"Same W_input, W_hidden reused at EVERY time step"}</text>
#         </svg>
#       </div>
#
#       <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:13,fontWeight:700,color:[C.blue,C.red,C.yellow,C.green][phase],marginBottom:4}}>{"Phase "+(phase+1)+": "+phases[phase].title}</div>
#         <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{phases[phase].desc}</div>
#       </Card>
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.purple,marginBottom:10}}>Regular NN vs RNN Backprop</div>
#         <div style={{display:"flex",justifyContent:"center",gap:20,flexWrap:"wrap"}}>
#           <div style={{flex:1,minWidth:200,padding:12,background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
#             <div style={{fontSize:10,fontWeight:700,color:C.cyan,marginBottom:6}}>Regular NN</div>
#             <div style={{fontSize:10,color:C.muted,lineHeight:1.8,fontFamily:"monospace"}}>
#               {"Backward through LAYERS only"}<br/>
#               {"3 layers = 3 steps back"}<br/>
#               {"Each layer has OWN weights"}
#             </div>
#           </div>
#           <div style={{flex:1,minWidth:200,padding:12,background:"#08080d",borderRadius:8,border:"1px solid "+C.accent+"30"}}>
#             <div style={{fontSize:10,fontWeight:700,color:C.accent,marginBottom:6}}>RNN (BPTT)</div>
#             <div style={{fontSize:10,color:C.muted,lineHeight:1.8,fontFamily:"monospace"}}>
#               {"Backward through LAYERS + TIME"}<br/>
#               {"3 layers "+MUL+" 4 steps = 12 back!"}<br/>
#               {"SAME weights "+ARR+" gradients COMBINED"}
#             </div>
#           </div>
#         </div>
#       </Card>
#
#       <Insight icon={TARG} title="The Core Difference">
#         RNN backprop has an <span style={{color:C.accent,fontWeight:700}}>extra dimension {DASH} TIME</span>. More multiplications means gradients <span style={{color:C.yellow,fontWeight:700}}>vanish faster</span>, which means <span style={{color:C.red,fontWeight:700}}>early words get forgotten</span>. This is the fundamental problem LSTM and GRU were invented to solve.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 3: VANISHING GRADIENT PROBLEM
#    =============================================================== */
# function TabVanishing() {
#   var _w = useState(0.5); var wh = _w[0], setWh = _w[1];
#   var _n = useState(10); var nSteps = _n[0], setNSteps = _n[1];
#
#   var gradients = [];
#   for (var i=0; i<nSteps; i++) { gradients.push(Math.pow(wh, nSteps-1-i)); }
#   var maxG = Math.max.apply(null, gradients.concat([1]));
#   var isVanish = wh < 1, isExplode = wh > 1;
#   var bw = 740, bh = 240, barW = Math.max(8, Math.min(40, (bw-100)/nSteps - 4));
#
#   return (
#     <div>
#       <SectionTitle title="The Vanishing Gradient Problem" subtitle={"Adjust W_hidden and time steps "+DASH+" watch gradients vanish or explode"} />
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{display:"flex",gap:30,justifyContent:"center",flexWrap:"wrap",alignItems:"flex-start"}}>
#           <div style={{minWidth:200}}>
#             <div style={{fontSize:10,color:C.muted,marginBottom:6}}>{"W_hidden = "+wh.toFixed(2)}</div>
#             <input type="range" min={10} max={200} value={Math.round(wh*100)} onChange={function(e){setWh(parseInt(e.target.value)/100);}} style={{width:"100%"}} />
#             <div style={{display:"flex",justifyContent:"space-between",fontSize:8,color:C.dim,marginTop:2}}><span>0.10</span><span style={{color:C.accent}}>1.00</span><span>2.00</span></div>
#           </div>
#           <div style={{minWidth:200}}>
#             <div style={{fontSize:10,color:C.muted,marginBottom:6}}>{"Time steps = "+nSteps}</div>
#             <input type="range" min={3} max={20} value={nSteps} onChange={function(e){setNSteps(parseInt(e.target.value));}} style={{width:"100%"}} />
#             <div style={{display:"flex",justifyContent:"space-between",fontSize:8,color:C.dim,marginTop:2}}><span>3</span><span>20</span></div>
#           </div>
#           <div style={{textAlign:"center",minWidth:120}}>
#             <div style={{fontSize:9,color:C.muted}}>Earliest gradient</div>
#             <div style={{fontSize:20,fontWeight:800,color:isExplode?C.red:isVanish?C.yellow:C.green}}>{gradients[0]>1000?"overflow!":gradients[0]<0.0001?"~0":gradients[0].toFixed(6)}</div>
#             <div style={{fontSize:9,color:isExplode?C.red:isVanish?C.yellow:C.green,fontWeight:700}}>{isExplode?"EXPLODING!":isVanish?"VANISHING":"Stable"}</div>
#           </div>
#         </div>
#       </Card>
#
#       <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
#         <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
#           <text x={bw/2} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"Gradient magnitude at each time step (W_hidden="+wh.toFixed(2)+")"}</text>
#
#           {/* Axis */}
#           <line x1={60} y1={bh-35} x2={bw-20} y2={bh-35} stroke={C.dim} strokeWidth={1} />
#           <text x={bw/2} y={bh-8} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"Time step (t=1 earliest "+LARR+"   "+ARR+" t="+nSteps+" latest)"}</text>
#
#           {gradients.map(function(g,i) {
#             var cappedG = Math.min(g, maxG);
#             var barH = Math.max(2, (cappedG/maxG) * (bh-80));
#             var x = 70 + i * ((bw-110)/nSteps);
#             var cl = g > 0.5 ? C.green : g > 0.1 ? C.yellow : g > 0.01 ? C.orange : C.red;
#             if (isExplode) cl = g > 10 ? C.red : g > 2 ? C.orange : C.yellow;
#             return (<g key={i}>
#               <rect x={x} y={bh-35-barH} width={barW} height={barH} rx={2} fill={cl+"40"} stroke={cl} strokeWidth={1} />
#               {barW > 14 && <text x={x+barW/2} y={bh-40-barH} textAnchor="middle" fill={cl} fontSize={7} fontWeight={700} fontFamily="monospace">{g>999?"Inf":g<0.001?"~0":g.toFixed(3)}</text>}
#               {barW > 14 && <text x={x+barW/2} y={bh-22} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{"t="+(i+1)}</text>}
#             </g>);
#           })}
#
#           {/* Reference line at 1.0 if visible */}
#           {maxG >= 1 && <g>
#             <line x1={60} y1={bh-35-((1/maxG)*(bh-80))} x2={bw-20} y2={bh-35-((1/maxG)*(bh-80))} stroke={C.accent} strokeWidth={1} strokeDasharray="4,4" />
#             <text x={55} y={bh-32-((1/maxG)*(bh-80))} textAnchor="end" fill={C.accent} fontSize={7} fontFamily="monospace">1.0</text>
#           </g>}
#         </svg>
#       </div>
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:10}}>The Whisper Analogy</div>
#         <div style={{display:"flex",justifyContent:"center",gap:4,flexWrap:"wrap",marginBottom:10}}>
#           {Array.from({length:Math.min(nSteps,12)},function(_,i) {
#             var g = gradients[i];
#             var opacity = Math.max(0.1,Math.min(1,g));
#             if(isExplode) opacity = Math.min(1, g/maxG + 0.3);
#             return (<div key={i} style={{textAlign:"center",fontSize:16,opacity:opacity,transition:"opacity 0.3s"}}>
#               {isExplode&&g>5?"\\uD83D\\uDCA5":"\\uD83D\\uDDE3\\uFE0F"}
#             </div>);
#           })}
#         </div>
#         <div style={{textAlign:"center",fontSize:10,color:C.muted}}>
#           {isVanish ? <span><span style={{color:C.yellow,fontWeight:700}}>Vanishing</span>: each person forgets a little {ARR} the last one hears nothing</span> :
#            isExplode ? <span><span style={{color:C.red,fontWeight:700}}>Exploding</span>: each person exaggerates {ARR} the last one hears screaming</span> :
#            <span><span style={{color:C.green,fontWeight:700}}>Stable</span>: the message passes through perfectly (W=1.0)</span>}
#         </div>
#       </Card>
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Solutions</div>
#         <div style={{display:"flex",gap:8,flexWrap:"wrap",justifyContent:"center"}}>
#           {[
#             {p:"Exploding",s:"Gradient Clipping",d:"Cap gradient norm to max value",c:C.red},
#             {p:"Vanishing",s:"LSTM / GRU",d:"Gates create gradient highways",c:C.green},
#             {p:"Vanishing",s:"Skip Connections",d:"Shortcuts bypass time steps",c:C.blue},
#             {p:"Exploding",s:"Orthogonal Init",d:"Keep eigenvalues near 1",c:C.purple},
#           ].map(function(item,i) {
#             return (<div key={i} style={{padding:"10px 14px",background:item.c+"08",borderRadius:8,border:"1px solid "+item.c+"20",minWidth:130,textAlign:"center"}}>
#               <div style={{fontSize:8,color:item.c,marginBottom:2}}>{item.p}</div>
#               <div style={{fontSize:10,fontWeight:700,color:C.text}}>{item.s}</div>
#               <div style={{fontSize:8,color:C.muted,marginTop:2}}>{item.d}</div>
#             </div>);
#           })}
#         </div>
#       </Card>
#
#       <Insight>
#         When W_hidden {"<"} 1, gradients decay as <span style={{color:C.yellow,fontWeight:700}}>W^n</span> {DASH} after 20 steps, 0.5^20 = 0.00000095 (practically zero!). Early time steps get <span style={{color:C.red,fontWeight:700}}>no learning signal</span>. This is why the LSTM's <span style={{color:C.green,fontWeight:700}}>cell state highway</span> was a breakthrough.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 4: LSTM — GATES & NUMERIC WALKTHROUGH
#    =============================================================== */
# function TabLSTM() {
#   var _s = useState(0); var step = _s[0], setStep = _s[1];
#   var _a = useState(false); var autoL = _a[0], setAutoL = _a[1];
#
#   useEffect(function(){if(!autoL)return;var t=setInterval(function(){setStep(function(s){return(s+1)%6;});},4000);return function(){clearInterval(t);};}, [autoL]);
#
#   var gates = [
#     {name:"Overview",desc:"An LSTM adds a cell state (long-term memory) alongside the hidden state. Three gates control what to forget, what to store, and what to output.",icon:"\\uD83C\\uDFD7\\uFE0F"},
#     {name:"Forget Gate",desc:'f = sigmoid(W_f '+MUL+' [h(t-1), X(t)] + b_f). Outputs 0-1 for each cell value. 0 = completely forget, 1 = completely keep. Example: new sentence started '+ARR+' forget old subject.',icon:"\\uD83D\\uDDD1\\uFE0F"},
#     {name:"Input Gate",desc:'i = sigmoid(W_i '+MUL+' [h(t-1), X(t)]) controls HOW MUCH to write (0-1). C\\u0303 = tanh(W_c '+MUL+' [h(t-1), X(t)]) is WHAT to write (-1 to 1). Two parts working together.',icon:"\\u270D\\uFE0F"},
#     {name:"Cell Update",desc:'C(t) = f '+MUL+' C(t-1) + i '+MUL+' C\\u0303. First term: old memories, selectively forgotten. Second term: new info, selectively written. If f=1 and i=0, cell state flows UNCHANGED '+DASH+' this is the magic!',icon:"\\uD83D\\uDD04"},
#     {name:"Output Gate",desc:'o = sigmoid(W_o '+MUL+' [h(t-1), X(t)]). h(t) = o '+MUL+' tanh(C(t)). The cell holds full memory, but only a filtered version becomes the output. Not everything in memory is relevant right now.',icon:"\\uD83D\\uDCE4"},
#     {name:"Gradient Highway",desc:'In vanilla RNN: gradient '+MUL+' W '+MUL+' tanh\\u2032 at every step '+ARR+' vanishes. In LSTM: \\u2202C(t)/\\u2202C(t-1) = f(t). If forget gate '+DASH+' 1, gradient flows UNDIMINISHED. After 10 steps: 0.9\\u00B9\\u2070 = 0.35 (meaningful!) vs RNN 0.5\\u00B9\\u2070 = 0.001 (dead!).',icon:"\\uD83D\\uDEE3\\uFE0F"},
#   ];
#
#   var bw = 740, bh = 280;
#   var gateColors = [C.text, C.red, C.green, C.yellow, C.cyan, C.accent];
#
#   return (
#     <div>
#       <SectionTitle title="LSTM: Long Short-Term Memory" subtitle={"Gates control what to remember, forget, and output "+DASH+" solving the vanishing gradient"} />
#
#       <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
#         {gates.map(function(g,i){var on=step===i;return (<button key={i} onClick={function(){setStep(i);setAutoL(false);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(on?gateColors[i]:C.border),background:on?gateColors[i]+"20":C.card,color:on?gateColors[i]:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{g.icon+" "+g.name}</button>);})}
#         <button onClick={function(){setAutoL(!autoL);}} style={{padding:"8px 12px",borderRadius:8,border:"1.5px solid "+(autoL?C.yellow:C.border),background:autoL?C.yellow+"20":C.card,color:autoL?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoL?PAUSE:PLAY}</button>
#       </div>
#
#       <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
#         <svg width={bw} height={bh} viewBox={"0 0 "+bw+" "+bh} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
#
#           {/* Cell state highway */}
#           <line x1={40} y1={50} x2={700} y2={50} stroke={step===3||step===5?C.yellow:C.dim} strokeWidth={step===3||step===5?3:2} />
#           <text x={370} y={35} textAnchor="middle" fill={step===3||step===5?C.yellow:C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"Cell State C(t) "+DASH+" the memory highway"}</text>
#           <text x={30} y={54} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">C(t-1)</text>
#           <text x={710} y={54} fill={C.muted} fontSize={8} fontFamily="monospace">C(t)</text>
#           <polygon points="704,50 697,46 697,54" fill={step===3||step===5?C.yellow:C.dim} />
#
#           {/* Forget gate */}
#           <circle cx={180} cy={50} r={20} fill={step===1?C.red+"25":"transparent"} stroke={step===1?C.red:C.dim} strokeWidth={step===1?2.5:1} />
#           <text x={180} y={55} textAnchor="middle" fill={step===1?C.red:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">{MUL}</text>
#           <text x={180} y={88} textAnchor="middle" fill={step===1?C.red:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">FORGET</text>
#           <text x={180} y={100} textAnchor="middle" fill={step===1?C.red:C.dim} fontSize={8} fontFamily="monospace">f(t)</text>
#           <line x1={180} y1={108} x2={180} y2={160} stroke={step===1?C.red:C.dim} strokeWidth={1} strokeDasharray="3,3" />
#
#           {/* Input gate + candidate */}
#           <circle cx={330} cy={50} r={20} fill={step===2||step===3?C.green+"25":"transparent"} stroke={step===2||step===3?C.green:C.dim} strokeWidth={step===2||step===3?2.5:1} />
#           <text x={330} y={55} textAnchor="middle" fill={step===2||step===3?C.green:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">+</text>
#           <text x={330} y={88} textAnchor="middle" fill={step===2?C.green:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">INPUT</text>
#           <text x={330} y={100} textAnchor="middle" fill={step===2?C.green:C.dim} fontSize={8} fontFamily="monospace">{"i(t) "+MUL+" C\\u0303(t)"}</text>
#           <line x1={330} y1={108} x2={330} y2={160} stroke={step===2?C.green:C.dim} strokeWidth={1} strokeDasharray="3,3" />
#
#           {/* Output gate */}
#           <circle cx={530} cy={170} r={20} fill={step===4?C.cyan+"25":"transparent"} stroke={step===4?C.cyan:C.dim} strokeWidth={step===4?2.5:1} />
#           <text x={530} y={175} textAnchor="middle" fill={step===4?C.cyan:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">{MUL}</text>
#           <text x={530} y={210} textAnchor="middle" fill={step===4?C.cyan:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">OUTPUT</text>
#           <text x={530} y={222} textAnchor="middle" fill={step===4?C.cyan:C.dim} fontSize={8} fontFamily="monospace">o(t)</text>
#
#           {/* tanh after cell state */}
#           <line x1={530} y1={50} x2={530} y2={110} stroke={step===4?C.cyan:C.dim} strokeWidth={1} />
#           <rect x={510} y={112} width={40} height={20} rx={4} fill={step===4?C.purple+"15":"transparent"} stroke={step===4?C.purple:C.dim} />
#           <text x={530} y={126} textAnchor="middle" fill={step===4?C.purple:C.dim} fontSize={8} fontWeight={700} fontFamily="monospace">tanh</text>
#           <line x1={530} y1={132} x2={530} y2={150} stroke={step===4?C.cyan:C.dim} strokeWidth={1} />
#
#           {/* h(t) output */}
#           <line x1={550} y1={170} x2={700} y2={170} stroke={step===4?C.cyan:C.dim} strokeWidth={2} />
#           <polygon points="704,170 697,166 697,174" fill={step===4?C.cyan:C.dim} />
#           <text x={710} y={174} fill={step===4?C.cyan:C.muted} fontSize={8} fontFamily="monospace">h(t)</text>
#
#           {/* Concat input box */}
#           <rect x={200} y={165} width={200} height={40} rx={8} fill={C.card} stroke={C.border} />
#           <text x={300} y={180} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">h(t-1) + X(t) concatenated</text>
#           <text x={300} y={196} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"4 sigmoids/tanh "+ARR+" f, i, C\\u0303, o"}</text>
#
#           {/* Lines from concat to gates */}
#           <line x1={180} y1={160} x2={250} y2={165} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3" />
#           <line x1={330} y1={160} x2={330} y2={165} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3" />
#           <line x1={530} y1={222} x2={400} y2={205} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3" />
#
#           {/* h(t-1) input */}
#           <text x={30} y={174} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">h(t-1)</text>
#           <line x1={35} y1={170} x2={200} y2={180} stroke={C.dim} strokeWidth={1} />
#
#           {/* x(t) input */}
#           <text x={250} y={250} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">X(t)</text>
#           <line x1={250} y1={242} x2={280} y2={205} stroke={C.dim} strokeWidth={1} />
#           <polygon points="280,205 275,212 282,212" fill={C.dim} />
#
#           {/* Gradient highway highlight */}
#           {step===5 && <g>
#             <rect x={50} y={40} width={650} height={20} rx={10} fill={C.accent+"10"} stroke={C.accent} strokeWidth={1.5} strokeDasharray="6,3" />
#             <text x={370} y={18} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={800} fontFamily="monospace">{"GRADIENT HIGHWAY "+DASH+" gradients flow directly, no W multiplication!"}</text>
#           </g>}
#
#           {/* 4x params note */}
#           <rect x={560} y={240} width={160} height={22} rx={4} fill={C.purple+"08"} stroke={C.purple+"25"} />
#           <text x={640} y={255} textAnchor="middle" fill={C.purple} fontSize={8} fontWeight={700} fontFamily="monospace">{"4 weight matrices "+ARR+" 4"+MUL+" params"}</text>
#         </svg>
#       </div>
#
#       <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:gateColors[step]}}>
#         <div style={{fontSize:13,fontWeight:700,color:gateColors[step],marginBottom:4}}>{gates[step].icon+" "+gates[step].name}</div>
#         <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{gates[step].desc}</div>
#       </Card>
#
#       {/* Numeric walkthrough */}
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:12}}>{"Numeric Walkthrough: "+LQ+"not"+RQ+" "+ARR+" "+LQ+"good"+RQ}</div>
#         <div style={{display:"flex",gap:12,flexWrap:"wrap",justifyContent:"center"}}>
#           {[
#             {t:'t=1 "not"',f:"0.52",i:"0.71",cand:"-0.66",c:"-0.47",o:"0.65",h:"-0.29",cl:C.red,
#              detail:'f=0.52 (forget half), i=0.71 (store 71%), c\\u0303=-0.66 (negative content). Cell = 0'+ARR+'-0.47. Memory: NEGATIVE.'},
#             {t:'t=2 "good"',f:"0.57",i:"0.67",cand:"0.54",c:"0.09",o:"0.62",h:"0.056",cl:C.green,
#              detail:'f=0.57 (keep 57% of "not" memory). i=0.67, c\\u0303=+0.54. Cell = -0.27+0.36 = 0.09. "not" partially survived!'},
#           ].map(function(ts,idx) {
#             return (<div key={idx} style={{flex:1,minWidth:280,padding:14,background:"#08080d",borderRadius:8,border:"1px solid "+ts.cl+"30"}}>
#               <div style={{fontSize:11,fontWeight:700,color:ts.cl,marginBottom:8}}>{ts.t}</div>
#               <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:8}}>
#                 {[{l:"f",v:ts.f,c:C.red},{l:"i",v:ts.i,c:C.green},{l:"c\\u0303",v:ts.cand,c:C.purple},{l:"c",v:ts.c,c:C.yellow},{l:"o",v:ts.o,c:C.cyan},{l:"h",v:ts.h,c:C.accent}].map(function(g,j){
#                   return (<div key={j} style={{textAlign:"center",minWidth:36}}>
#                     <div style={{fontSize:7,color:C.dim}}>{g.l}</div>
#                     <div style={{fontSize:12,fontWeight:800,color:g.c}}>{g.v}</div>
#                   </div>);
#                 })}
#               </div>
#               <div style={{fontSize:9,color:C.muted,lineHeight:1.6}}>{ts.detail}</div>
#             </div>);
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={TARG} title="Why LSTM Solves Vanishing Gradients">
#         The cell state update is C(t) = f{MUL}C(t-1) + i{MUL}C\u0303. The gradient of C(t) w.r.t. C(t-1) is simply <span style={{color:C.green,fontWeight:700}}>f(t)</span> {DASH} no W multiplication, no tanh squashing. If f {"\u2248"} 1, gradients flow through <span style={{color:C.accent,fontWeight:700}}>undiminished</span>. After 10 steps: <span style={{color:C.green}}>0.9\u00B9\u2070 = 0.35</span> (meaningful!) vs RNN <span style={{color:C.red}}>0.5\u00B9\u2070 = 0.001</span> (dead!).
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 5: GRU & EVOLUTION
#    =============================================================== */
# function TabGRUEvolution() {
#   var _s = useState(0); var sel = _s[0], setSel = _s[1];
#
#   var models = [
#     {name:"Simple RNN",year:1986,color:C.blue,gates:0,states:1,params:"1x",strength:"Has memory",weakness:"Forgets quickly (vanishing gradient)",
#      eq:"h(t) = tanh(W "+MUL+" [h(t-1), x(t)] + b)", insight:"The foundation. Introduced the idea of recurrence and hidden states, but the vanishing gradient problem limits it to short sequences."},
#     {name:"LSTM",year:1997,color:C.green,gates:3,states:2,params:"4x",strength:"Long-range memory via cell state highway",weakness:"Complex, expensive (4 weight matrices)",
#      eq:"f,i,o = sigmoid(...)   C(t) = f"+MUL+"C(t-1) + i"+MUL+"C\\u0303   h(t) = o"+MUL+"tanh(C(t))", insight:"Forget + Input + Output gates. Two states (cell + hidden). The cell state acts as a gradient highway. Can remember across 100+ time steps."},
#     {name:"GRU",year:2014,color:C.yellow,gates:2,states:1,params:"3x",strength:"Nearly as good as LSTM, 25% fewer params",weakness:"Less expressive than LSTM on very long sequences",
#      eq:"r = sigmoid(...)   z = sigmoid(...)   h(t) = (1-z)"+MUL+"h(t-1) + z"+MUL+"h\\u0303", insight:"Reset + Update gates. One state only. The update gate does double duty (forget + input linked by z and 1-z). Faster to train, often comparable results."},
#     {name:"Transformer",year:2017,color:C.accent,gates:0,states:0,params:"varies",strength:"Every word sees every other word directly (attention)",weakness:"Quadratic memory in sequence length",
#      eq:"Attention(Q,K,V) = softmax(QK\\u1D40 / "+"\u221A"+"d_k) "+MUL+" V", insight:"No recurrence at all. Self-attention lets every position attend to every other position directly. This is what GPT, BERT, and Claude use. Solved the long-range problem entirely."},
#   ];
#
#   var m = models[sel];
#
#   return (
#     <div>
#       <SectionTitle title="GRU & The Evolution of Sequence Models" subtitle={"From simple memory to attention "+DASH+" each solved a specific problem"} />
#
#       <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
#         {models.map(function(mod,i){var on=sel===i;return (<button key={i} onClick={function(){setSel(i);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?mod.color:C.border),background:on?mod.color+"20":C.card,color:on?mod.color:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{mod.name+" ("+mod.year+")"}</button>);})}
#       </div>
#
#       <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:m.color}}>
#         <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:16}}>
#           <div>
#             <div style={{fontSize:20,fontWeight:800,color:m.color}}>{m.name}</div>
#             <div style={{fontSize:11,color:C.muted,marginTop:2}}>{m.year}</div>
#             <div style={{fontSize:11,color:C.text,marginTop:10,lineHeight:1.6}}>{m.strength}</div>
#           </div>
#           <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
#             {[{l:"GATES",v:m.gates},{l:"STATES",v:m.states},{l:"PARAMS",v:m.params}].map(function(d,i){
#               return (<div key={i} style={{textAlign:"center"}}><div style={{fontSize:8,color:C.muted}}>{d.l}</div><div style={{fontSize:22,fontWeight:800,color:m.color}}>{d.v}</div></div>);
#             })}
#           </div>
#         </div>
#         <div style={{marginTop:14,padding:"10px 14px",background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
#           <div style={{fontSize:9,color:C.muted,marginBottom:4}}>CORE EQUATION</div>
#           <div style={{fontSize:10,color:m.color,fontFamily:"monospace",lineHeight:1.8}}>{m.eq}</div>
#         </div>
#         <div style={{marginTop:12,fontSize:11,color:C.muted,lineHeight:1.7,fontStyle:"italic",borderLeft:"3px solid "+m.color+"40",paddingLeft:12}}>{m.insight}</div>
#         {m.weakness && <div style={{marginTop:8,fontSize:10,color:C.red,fontFamily:"monospace"}}>{WARN+" "+m.weakness}</div>}
#       </Card>
#
#       {/* LSTM vs GRU comparison */}
#       {(sel===1||sel===2) && <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>LSTM vs GRU Side by Side</div>
#         <div style={{display:"flex",gap:12,flexWrap:"wrap",justifyContent:"center"}}>
#           {[
#             {aspect:"Gates",lstm:"3 (forget, input, output)",gru:"2 (reset, update)"},
#             {aspect:"States",lstm:"Cell + Hidden (2 paths)",gru:"Hidden only (1 path)"},
#             {aspect:"Key Difference",lstm:"f and i are INDEPENDENT",gru:"z and (1-z) are LINKED"},
#             {aspect:"Parameters",lstm:"4 weight matrices",gru:"3 weight matrices (~25% less)"},
#             {aspect:"Best For",lstm:"Very long sequences, large data",gru:"Speed, smaller datasets"},
#           ].map(function(row,i){
#             return (<div key={i} style={{width:"100%",display:"flex",gap:8}}>
#               <div style={{width:110,fontSize:9,color:C.muted,textAlign:"right",paddingTop:4,fontWeight:700}}>{row.aspect}</div>
#               <div style={{flex:1,padding:"6px 10px",background:C.green+"08",borderRadius:6,border:"1px solid "+C.green+"20",fontSize:9,color:C.green,fontFamily:"monospace"}}>{row.lstm}</div>
#               <div style={{flex:1,padding:"6px 10px",background:C.yellow+"08",borderRadius:6,border:"1px solid "+C.yellow+"20",fontSize:9,color:C.yellow,fontFamily:"monospace"}}>{row.gru}</div>
#             </div>);
#           })}
#         </div>
#       </Card>}
#
#       {/* GRU detail */}
#       {sel===2 && <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:10}}>GRU: The Update Gate Does Double Duty</div>
#         <div style={{display:"flex",justifyContent:"center",gap:24,flexWrap:"wrap"}}>
#           <div style={{textAlign:"center",padding:"12px 16px",background:C.yellow+"08",borderRadius:8,border:"1px solid "+C.yellow+"20"}}>
#             <div style={{fontSize:9,color:C.muted}}>z {"\u2248"} 0</div>
#             <div style={{fontSize:12,fontWeight:800,color:C.yellow,marginTop:4}}>Keep Old</div>
#             <div style={{fontSize:9,color:C.muted,marginTop:2}}>{"h(t) "+"\u2248"+" h(t-1)"}</div>
#             <div style={{fontSize:8,color:C.dim,marginTop:2}}>Like LSTM forget=1</div>
#           </div>
#           <div style={{display:"flex",alignItems:"center",fontSize:16,color:C.dim}}>{"\u2194"}</div>
#           <div style={{textAlign:"center",padding:"12px 16px",background:C.yellow+"08",borderRadius:8,border:"1px solid "+C.yellow+"20"}}>
#             <div style={{fontSize:9,color:C.muted}}>z {"\u2248"} 1</div>
#             <div style={{fontSize:12,fontWeight:800,color:C.yellow,marginTop:4}}>Replace New</div>
#             <div style={{fontSize:9,color:C.muted,marginTop:2}}>{"h(t) "+"\u2248"+" h\\u0303(t)"}</div>
#             <div style={{fontSize:8,color:C.dim,marginTop:2}}>Like LSTM input=1</div>
#           </div>
#         </div>
#         <div style={{textAlign:"center",marginTop:10,fontSize:9,color:C.muted}}>
#           LSTM can keep everything AND add everything (independent gates). GRU forces a <span style={{color:C.yellow,fontWeight:700}}>tradeoff</span>: more old = less new.
#         </div>
#       </Card>}
#
#       {/* Evolution timeline */}
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:16}}>The Full Evolution</div>
#         <div style={{display:"flex",flexDirection:"column",gap:2,paddingLeft:20}}>
#           {models.map(function(mod,i) {
#             var on = sel===i;
#             return (<div key={i}>
#               <div style={{display:"flex",alignItems:"center",gap:12,padding:"8px 12px",background:on?mod.color+"12":"transparent",borderRadius:8,border:on?"1px solid "+mod.color+"30":"1px solid transparent",cursor:"pointer",transition:"all 0.2s"}} onClick={function(){setSel(i);}}>
#                 <div style={{width:8,height:8,borderRadius:4,background:mod.color,flexShrink:0}} />
#                 <div style={{fontSize:12,fontWeight:800,color:on?mod.color:C.muted,minWidth:100}}>{mod.name}</div>
#                 <div style={{fontSize:10,color:C.dim}}>{mod.year}</div>
#                 <div style={{fontSize:9,color:on?C.text:C.dim,flex:1}}>{mod.strength}</div>
#               </div>
#               {i<models.length-1 && <div style={{marginLeft:3,width:2,height:16,background:C.dim}} />}
#             </div>);
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={TARG} title="The Trend">
#         Each model solved a specific problem: <span style={{color:C.blue}}>RNN</span> added memory, <span style={{color:C.green}}>LSTM</span> solved vanishing gradients with gates, <span style={{color:C.yellow}}>GRU</span> simplified it, and <span style={{color:C.accent,fontWeight:700}}>Transformers</span> abandoned recurrence entirely {DASH} letting every word attend to every other word directly.
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
#   var tabs = ["Basic RNN", "BPTT", "Vanishing Gradient", "LSTM", "GRU & Evolution"];
#   return (
#     <div style={{ background:C.bg, minHeight:"100vh", padding:"24px 16px", fontFamily:"'JetBrains Mono','SF Mono',monospace", color:C.text, maxWidth:960, margin:"0 auto" }}>
#       <div style={{textAlign:"center",marginBottom:16}}>
#         <div style={{ fontSize:22, fontWeight:800, background:"linear-gradient(135deg,"+C.accent+","+C.blue+")", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", display:"inline-block" }}>Recurrent Neural Networks</div>
#         <div style={{fontSize:11,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough "+DASH+" from basic RNN to Transformer"}</div>
#       </div>
#       <TabBar tabs={tabs} active={tab} onChange={setTab} />
#       {tab===0 && <TabBasicRNN />}
#       {tab===1 && <TabBPTT />}
#       {tab===2 && <TabVanishing />}
#       {tab===3 && <TabLSTM />}
#       {tab===4 && <TabGRUEvolution />}
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
# RNN_VISUAL_HEIGHT = 1200