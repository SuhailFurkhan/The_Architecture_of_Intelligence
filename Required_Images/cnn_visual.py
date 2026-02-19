"""
Self-contained HTML for the CNN (Convolutional Neural Network) interactive walkthrough.
Covers: Convolution operation, filters & feature maps, pooling, full CNN pipeline,
and architecture evolution (LeNet to AlexNet to VGG to GoogLeNet to ResNet).
Embed in Streamlit via st.components.v1.html(CNN_VISUAL_HTML, height=CNN_VISUAL_HEIGHT).
"""

CNN_VISUAL_HTML = """
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
var TRI = "\\u25E2";

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

function GridCell(props) {
  var x = props.x, y = props.y, s = props.size || 36;
  return (
    <g>
      <rect x={x} y={y} width={s} height={s} fill={props.bg || "transparent"} stroke={props.border || C.border} strokeWidth={1} rx={2} />
      <text x={x + s / 2} y={y + s / 2 + 1} textAnchor="middle" dominantBaseline="middle"
        fill={props.color || C.text} fontSize={props.fontSize || 11} fontWeight={props.bold ? 800 : 500}
        fontFamily="'JetBrains Mono', monospace">
        {props.value}
      </text>
    </g>
  );
}


/* ===============================================================
   TAB 1: THE CONVOLUTION OPERATION
   =============================================================== */
function TabConvolution() {
  var input = [[1,0,1,0,1],[0,1,0,1,0],[1,1,1,0,0],[0,0,1,1,0],[1,0,0,1,1]];
  var filter = [[1,0,-1],[1,0,-1],[1,0,-1]];

  var outputMap = useMemo(function() {
    var out = [];
    for (var r = 0; r < 3; r++) { var row = []; for (var c = 0; c < 3; c++) { var sum = 0; for (var fr = 0; fr < 3; fr++) for (var fc = 0; fc < 3; fc++) sum += input[r+fr][c+fc] * filter[fr][fc]; row.push(sum); } out.push(row); }
    return out;
  }, []);

  var _p = useState(0); var pos = _p[0], setPos = _p[1];
  var _a = useState(false); var autoPlay = _a[0], setAutoPlay = _a[1];

  useEffect(function() { if (!autoPlay) return; var t = setInterval(function() { setPos(function(p) { return (p+1)%9; }); }, 1200); return function() { clearInterval(t); }; }, [autoPlay]);

  var row = Math.floor(pos/3), col = pos%3;
  var products = [], dotSum = 0;
  for (var fr = 0; fr < 3; fr++) for (var fc = 0; fc < 3; fc++) { var p = input[row+fr][col+fc] * filter[fr][fc]; products.push(p); dotSum += p; }
  var cs = 36;

  return (
    <div>
      <SectionTitle title="The Convolution Operation" subtitle={"Watch a 3"+MUL+"3 filter slide across a 5"+MUL+"5 image, computing dot products at each position"} />

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16, gap: 8 }}>
        {Array.from({length:9}, function(_,i) {
          return (<button key={i} onClick={function(){setPos(i);setAutoPlay(false);}} style={{ width:32,height:32,borderRadius:6, border:"1.5px solid "+(pos===i?C.accent:C.border), background:pos===i?C.accent+"25":C.card, color:pos===i?C.accent:C.muted, cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace" }}>{i+1}</button>);
        })}
        <button onClick={function(){setAutoPlay(!autoPlay);}} style={{ padding:"0 12px",borderRadius:6, border:"1.5px solid "+(autoPlay?C.yellow:C.border), background:autoPlay?C.yellow+"20":C.card, color:autoPlay?C.yellow:C.muted, cursor:"pointer",fontSize:11,fontFamily:"monospace" }}>{autoPlay?PAUSE:PLAY}</button>
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <svg width={780} height={360} viewBox="0 0 780 360" style={{ background: "#08080d", borderRadius: 10, border: "1px solid " + C.border }}>
          <text x={105} y={20} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={700} fontFamily="monospace">{"Input (5"+MUL+"5)"}</text>
          {input.map(function(ra,ri) { return ra.map(function(val,ci) { var inP = ri>=row && ri<row+3 && ci>=col && ci<col+3; return <GridCell key={ri+"-"+ci} x={15+ci*cs} y={28+ri*cs} size={cs} value={val} bg={inP?C.accent+"20":"transparent"} border={inP?C.accent:C.dim} color={inP?C.accent:C.muted} bold={inP} />; }); })}
          <rect x={15+col*cs-1} y={28+row*cs-1} width={cs*3+2} height={cs*3+2} fill="none" stroke={C.accent} strokeWidth={2.5} rx={4} style={{animation:"pulse 1.5s infinite"}} />

          <text x={215} y={120} fill={C.yellow} fontSize={22} fontWeight={800} fontFamily="monospace">{MUL}</text>

          <text x={310} y={50} textAnchor="middle" fill={C.purple} fontSize={11} fontWeight={700} fontFamily="monospace">{"Filter (3"+MUL+"3)"}</text>
          {filter.map(function(ra,ri) { return ra.map(function(val,ci) { return <GridCell key={"f"+ri+ci} x={238+ci*cs} y={58+ri*cs} size={cs} value={val} bg={C.purple+"15"} border={C.purple+"60"} color={C.purple} bold={true} />; }); })}

          <text x={370} y={120} fill={C.yellow} fontSize={22} fontWeight={800} fontFamily="monospace">=</text>

          <text x={480} y={50} textAnchor="middle" fill={C.yellow} fontSize={11} fontWeight={700} fontFamily="monospace">Multiply</text>
          {products.map(function(val,idx) { var ri=Math.floor(idx/3), ci=idx%3; return <GridCell key={"p"+idx} x={408+ci*cs+ci*12} y={58+ri*cs} size={cs} value={val>=0?"+"+val:val} bg={val>0?C.green+"15":val<0?C.red+"15":"transparent"} border={val>0?C.green+"50":val<0?C.red+"50":C.dim} color={val>0?C.green:val<0?C.red:C.dim} bold={true} fontSize={10} />; })}

          <line x1={570} y1={115} x2={610} y2={115} stroke={C.yellow} strokeWidth={2} />
          <polygon points="615,115 608,110 608,120" fill={C.yellow} />

          <text x={670} y={80} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Sum</text>
          <rect x={635} y={90} width={70} height={50} rx={8} fill={C.accent+"15"} stroke={C.accent} strokeWidth={2} />
          <text x={670} y={120} textAnchor="middle" dominantBaseline="middle" fill={C.accent} fontSize={22} fontWeight={800} fontFamily="monospace">{dotSum}</text>
          <text x={670} y={165} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{"Position ("+row+","+col+")"}</text>

          <text x={670} y={190} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">Output Feature Map</text>
          {outputMap.map(function(ra,ri) { return ra.map(function(val,ci) { var cur = ri===row&&ci===col; var done = ri<row||(ri===row&&ci<col); return <GridCell key={"o"+ri+ci} x={622+ci*32} y={198+ri*32} size={32} value={cur||done?val:"?"} bg={cur?C.accent+"30":done?C.green+"10":"transparent"} border={cur?C.accent:done?C.green+"40":C.dim} color={cur?C.accent:done?C.green:C.dim} bold={cur} fontSize={10} />; }); })}
        </svg>
      </div>

      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.yellow, marginBottom: 8 }}>{"Position ("+row+","+col+") "+DASH+" Dot Product Breakdown"}</div>
        <div style={{ fontSize: 11, color: C.muted, fontFamily: "monospace", lineHeight: 2.0, overflowX: "auto" }}>
          {(function() { var parts = []; for (var f2=0;f2<3;f2++) for (var c2=0;c2<3;c2++) { var iv=input[row+f2][col+c2], fv=filter[f2][c2], pv=iv*fv; if (parts.length>0) parts.push(<span key={"o"+f2+c2} style={{color:C.dim}}> + </span>); parts.push(<span key={"m"+f2+c2} style={{color:pv>0?C.green:pv<0?C.red:C.dim}}>{"("+iv+MUL+(fv<0?"("+fv+")":fv)+")"}</span>); } parts.push(<span key="eq" style={{color:C.accent,fontWeight:800}}>{" = "+dotSum}</span>); return parts; })()}
        </div>
      </Card>

      <Insight>
        Each position of the convolution is a <span style={{color:C.accent,fontWeight:700}}>dot product</span> {DASH} the same operation a perceptron does. The filter is a <span style={{color:C.purple,fontWeight:700}}>template</span>, and the dot product measures how well each patch matches. This filter (1s left, -1s right) detects <span style={{color:C.green,fontWeight:700}}>vertical edges</span>.
      </Insight>

      <Card style={{ maxWidth: 750, margin: "16px auto 0" }}>
        <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
          <div style={{ textAlign: "center", minWidth: 100 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>FILTER PARAMS</div>
            <div style={{ fontSize: 28, fontWeight: 800, color: C.purple }}>10</div>
            <div style={{ fontSize: 9, color: C.dim }}>{"3"+MUL+"3 + 1 bias"}</div>
          </div>
          <div style={{ fontSize: 20, color: C.dim }}>{ARR}</div>
          <div style={{ textAlign: "center", minWidth: 100 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>POSITIONS SCANNED</div>
            <div style={{ fontSize: 28, fontWeight: 800, color: C.accent }}>9</div>
            <div style={{ fontSize: 9, color: C.dim }}>{"3"+MUL+"3 output"}</div>
          </div>
          <div style={{ fontSize: 20, color: C.dim }}>{ARR}</div>
          <div style={{ textAlign: "center", minWidth: 140 }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 4 }}>FULLY CONNECTED EQUIV.</div>
            <div style={{ fontSize: 28, fontWeight: 800, color: C.red }}>78,500</div>
            <div style={{ fontSize: 9, color: C.dim }}>{"784"+MUL+"100 weights"}</div>
          </div>
        </div>
        <div style={{ textAlign: "center", marginTop: 12, fontSize: 10, color: C.muted }}>
          <span style={{color:C.purple,fontWeight:700}}>Weight sharing</span>: the same 10 parameters scan every position. A FC layer needs <span style={{color:C.red}}>thousands of times more</span>.
        </div>
      </Card>
    </div>
  );
}


/* ===============================================================
   TAB 2: FILTERS & FEATURE MAPS
   =============================================================== */
function TabFilters() {
  var filters = {
    "Vertical Edge": { k:[[1,0,-1],[1,0,-1],[1,0,-1]], color:C.accent, desc:"Detects left-bright / right-dark" },
    "Horizontal Edge": { k:[[1,1,1],[0,0,0],[-1,-1,-1]], color:C.blue, desc:"Detects top-bright / bottom-dark" },
    "Corner": { k:[[0,1,1],[0,0,1],[0,0,0]], color:C.purple, desc:"Detects top-right corners" },
    "Sharpen": { k:[[0,-1,0],[-1,5,-1],[0,-1,0]], color:C.yellow, desc:"Enhances local contrast" },
    "Blur": { k:[[1,1,1],[1,1,1],[1,1,1]], color:C.cyan, desc:"Smooths out noise (avg)" },
    "Sobel X": { k:[[1,0,-1],[2,0,-2],[1,0,-1]], color:C.green, desc:"Strong vertical gradient" },
  };
  var fNames = Object.keys(filters);
  var _s = useState(0); var sel = _s[0], setSel = _s[1];
  var name = fNames[sel], filt = filters[name];

  var testImg = [[0,0,0,1,1,1,0],[0,0,1,1,1,0,0],[0,1,1,1,0,0,0],[0,1,1,1,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,1,0,0],[0,0,0,1,1,1,0]];

  var fMap = useMemo(function() {
    var out = [];
    for (var r=0;r<5;r++) { var row=[]; for (var c=0;c<5;c++) { var sum=0; for (var fr=0;fr<3;fr++) for (var fc=0;fc<3;fc++) sum+=testImg[r+fr][c+fc]*filt.k[fr][fc]; row.push(sum); } out.push(row); }
    return out;
  }, [sel]);

  var maxA = 1; fMap.forEach(function(r){r.forEach(function(v){if(Math.abs(v)>maxA)maxA=Math.abs(v);});});
  var cs = 34;

  return (
    <div>
      <SectionTitle title="Filters & Feature Maps" subtitle={"Different filters detect different patterns "+DASH+" select one to see what it finds"} />

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 20, flexWrap: "wrap" }}>
        {fNames.map(function(n,i) { var on=sel===i; return (<button key={i} onClick={function(){setSel(i);}} style={{ padding:"6px 14px",borderRadius:20, border:"1.5px solid "+(on?filters[n].color:C.border), background:on?filters[n].color+"20":C.card, color:on?filters[n].color:C.muted, cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace",transition:"all 0.2s" }}>{n}</button>); })}
      </div>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <svg width={760} height={270} viewBox="0 0 760 270" style={{ background:"#08080d",borderRadius:10,border:"1px solid "+C.border }}>
          <text x={130} y={20} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={700} fontFamily="monospace">{"Input (7"+MUL+"7)"}</text>
          {testImg.map(function(ra,ri){return ra.map(function(v,ci){return <GridCell key={"i"+ri+ci} x={12+ci*cs} y={28+ri*cs} size={cs} value={v} bg={v===1?C.blue+"25":"transparent"} border={v===1?C.blue+"50":C.dim+"40"} color={v===1?C.blue:C.dim} />;});})}
          <text x={130} y={272} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{LQ+"D"+RQ+" shape pattern"}</text>

          <text x={282} y={100} fill={C.yellow} fontSize={18} fontWeight={800} fontFamily="monospace">{MUL}</text>

          <text x={380} y={55} textAnchor="middle" fill={filt.color} fontSize={11} fontWeight={700} fontFamily="monospace">{name}</text>
          {filt.k.map(function(ra,ri){return ra.map(function(v,ci){return <GridCell key={"f"+ri+ci} x={310+ci*(cs+4)} y={63+ri*(cs+4)} size={cs} value={v} bg={v>0?filt.color+"15":v<0?C.red+"10":"transparent"} border={filt.color+"40"} color={v>0?filt.color:v<0?C.red:C.dim} bold={true} />;});})}
          <text x={380} y={195} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{filt.desc}</text>

          <text x={470} y={100} fill={C.yellow} fontSize={18} fontWeight={800} fontFamily="monospace">{ARR}</text>

          <text x={620} y={20} textAnchor="middle" fill={C.green} fontSize={11} fontWeight={700} fontFamily="monospace">{"Feature Map (5"+MUL+"5)"}</text>
          {fMap.map(function(ra,ri){return ra.map(function(v,ci){ var int=v/maxA; var bg=v>0?"rgba(74,222,128,"+(int*0.5).toFixed(2)+")":v<0?"rgba(239,68,68,"+(Math.abs(int)*0.4).toFixed(2)+")":"transparent"; return <GridCell key={"o"+ri+ci} x={500+ci*(cs+4)} y={28+ri*(cs+4)} size={cs} value={v} bg={bg} border={v!==0?(v>0?C.green:C.red)+"40":C.dim+"30"} color={v>0?C.green:v<0?C.red:C.dim} bold={Math.abs(v)>=maxA*0.6} fontSize={Math.abs(v)>9?9:11} />;});})}
          <text x={620} y={226} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">
            <tspan fill={C.green}>Positive</tspan>{" = match | "}<tspan fill={C.red}>Negative</tspan>{" = inverse"}
          </text>
        </svg>
      </div>

      <Card style={{ maxWidth: 750, margin: "0 auto 16px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.accent, marginBottom: 10 }}>How Hierarchy Emerges</div>
        <div style={{ display: "flex", justifyContent: "center", gap: 6, flexWrap: "wrap" }}>
          {[ {l:"Layer 1",w:"Edges, Gradients",c:C.accent,e:"/"}, {l:"Layer 2",w:"Corners, Textures",c:C.blue,e:TRI}, {l:"Layer 3",w:"Parts (eyes, ears)",c:C.purple,e:"\\u{1F441}"}, {l:"Layer 4",w:"Objects (faces)",c:C.green,e:"\\u{1F642}"} ].map(function(it,i,a) {
            return (<React.Fragment key={i}>
              <div style={{ background:it.c+"10",border:"1px solid "+it.c+"30",borderRadius:8,padding:"10px 14px",textAlign:"center",minWidth:110 }}>
                <div style={{fontSize:18,marginBottom:4}}>{it.e}</div>
                <div style={{fontSize:10,color:it.c,fontWeight:700}}>{it.l}</div>
                <div style={{fontSize:9,color:C.muted,marginTop:2}}>{it.w}</div>
              </div>
              {i<a.length-1 && <div style={{display:"flex",alignItems:"center",fontSize:16,color:C.dim}}>{ARR}</div>}
            </React.Fragment>);
          })}
        </div>
      </Card>

      <Insight>
        A real CNN uses <span style={{color:C.purple,fontWeight:700}}>multiple filters</span> per layer. 32 filters {ARR} 32 feature maps stacked together. The next layer combines these into more complex patterns {DASH} <span style={{color:C.accent,fontWeight:700}}>edges become textures become objects</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: POOLING
   =============================================================== */
function TabPooling() {
  var pIn = [[1,3,2,0],[5,7,1,4],[2,0,6,1],[3,8,9,2]];
  var _pt = useState("max"); var poolType = _pt[0], setPoolType = _pt[1];
  var _hw = useState(0); var hw = _hw[0], setHw = _hw[1];
  var _ap = useState(false); var autoP = _ap[0], setAutoP = _ap[1];

  useEffect(function(){if(!autoP)return;var t=setInterval(function(){setHw(function(w){return(w+1)%4;});},1500);return function(){clearInterval(t);};}, [autoP]);

  var wins = [{r:0,c:0,cells:[[0,0],[0,1],[1,0],[1,1]]},{r:0,c:1,cells:[[0,2],[0,3],[1,2],[1,3]]},{r:1,c:0,cells:[[2,0],[2,1],[3,0],[3,1]]},{r:1,c:1,cells:[[2,2],[2,3],[3,2],[3,3]]}];
  var pooled = wins.map(function(w){var vs=w.cells.map(function(c){return pIn[c[0]][c[1]];}); return poolType==="max"?Math.max.apply(null,vs):+(vs.reduce(function(a,b){return a+b;},0)/4).toFixed(1);});
  var cw = wins[hw]; var cv = cw.cells.map(function(c){return pIn[c[0]][c[1]];}); var mi=0; cv.forEach(function(v,i){if(v>cv[mi])mi=i;});
  var cs = 50;

  return (
    <div>
      <SectionTitle title="Pooling: Downsampling Feature Maps" subtitle={"Shrinks the feature map "+DASH+" keeps important info, discards the rest"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:20}}>
        {["max","avg"].map(function(t){var on=poolType===t;var cl=t==="max"?C.accent:C.cyan; return (<button key={t} onClick={function(){setPoolType(t);}} style={{padding:"8px 20px",borderRadius:20,border:"1.5px solid "+(on?cl:C.border),background:on?cl+"20":C.card,color:on?cl:C.muted,cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"monospace"}}>{t==="max"?"Max Pooling":"Average Pooling"}</button>);})}
        <button onClick={function(){setAutoP(!autoP);}} style={{padding:"8px 14px",borderRadius:20,border:"1.5px solid "+(autoP?C.yellow:C.border),background:autoP?C.yellow+"20":C.card,color:autoP?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{autoP?PAUSE+" Pause":PLAY+" Auto"}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={700} height={310} viewBox="0 0 700 310" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <text x={120} y={24} textAnchor="middle" fill={C.blue} fontSize={12} fontWeight={700} fontFamily="monospace">{"Feature Map (4"+MUL+"4)"}</text>
          {pIn.map(function(ra,ri){return ra.map(function(v,ci){ var inW=cw.cells.some(function(c){return c[0]===ri&&c[1]===ci;}); var isM=poolType==="max"&&inW&&cw.cells[mi][0]===ri&&cw.cells[mi][1]===ci; return <GridCell key={ri+"-"+ci} x={20+ci*cs} y={34+ri*cs} size={cs} value={v} bg={isM?C.accent+"35":inW?C.yellow+"15":"transparent"} border={isM?C.accent:inW?C.yellow:C.dim} color={isM?C.accent:inW?C.yellow:C.muted} bold={inW} fontSize={14} />;});})}
          <rect x={20+cw.c*cs*2-1} y={34+cw.r*cs*2-1} width={cs*2+2} height={cs*2+2} fill="none" stroke={C.yellow} strokeWidth={2.5} rx={4} strokeDasharray="6,3" />

          <g><line x1={240} y1={135} x2={300} y2={135} stroke={C.yellow} strokeWidth={2} /><polygon points="305,135 298,130 298,140" fill={C.yellow} /><text x={272} y={125} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{poolType==="max"?"max()":"avg()"}</text></g>

          <text x={440} y={50} textAnchor="middle" fill={C.yellow} fontSize={11} fontWeight={700} fontFamily="monospace">{"Window "+(hw+1)+" Calculation"}</text>
          <rect x={320} y={58} width={240} height={60} rx={8} fill={C.card} stroke={C.border} strokeWidth={1} />
          <text x={440} y={82} textAnchor="middle" fill={C.muted} fontSize={11} fontFamily="monospace">{poolType==="max"?"max("+cv.join(", ")+")":"("+cv.join(" + ")+") / 4"}</text>
          <text x={440} y={105} textAnchor="middle" fill={poolType==="max"?C.accent:C.cyan} fontSize={18} fontWeight={800} fontFamily="monospace">{"= "+pooled[hw]}</text>

          <text x={440} y={150} textAnchor="middle" fill={C.green} fontSize={12} fontWeight={700} fontFamily="monospace">{(poolType==="max"?"Max":"Avg")+" Pooled (2"+MUL+"2)"}</text>
          {[0,1].map(function(ri){return [0,1].map(function(ci){ var idx=ri*2+ci; var cur=idx===hw; var cl=poolType==="max"?C.accent:C.cyan; return (<g key={"o"+ri+ci} onClick={function(){setHw(idx);setAutoP(false);}} style={{cursor:"pointer"}}><GridCell x={370+ci*(cs+10)} y={160+ri*(cs+10)} size={cs} value={pooled[idx]} bg={cur?cl+"30":cl+"10"} border={cur?cl:cl+"40"} color={cur?cl:C.muted} bold={cur} fontSize={14} /></g>);});})}

          <rect x={320} y={270} width={240} height={30} rx={6} fill={C.red+"08"} stroke={C.red+"30"} />
          <text x={440} y={289} textAnchor="middle" fill={C.red} fontSize={10} fontWeight={700} fontFamily="monospace">{WARN+" 0 learnable parameters"}</text>
        </svg>
      </div>

      <div style={{display:"flex",justifyContent:"center",gap:8,marginBottom:16}}>
        {["Top-Left","Top-Right","Bot-Left","Bot-Right"].map(function(l,i){ return (<button key={i} onClick={function(){setHw(i);setAutoP(false);}} style={{padding:"6px 12px",borderRadius:6,border:"1.5px solid "+(hw===i?C.yellow:C.border),background:hw===i?C.yellow+"20":C.card,color:hw===i?C.yellow:C.muted,cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{l}</button>);})}
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.purple,marginBottom:10}}>Backpropagation Through Pooling</div>
        <div style={{display:"flex",justifyContent:"center",gap:40,flexWrap:"wrap"}}>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:10,fontWeight:700,color:C.accent,marginBottom:6}}>Max Pool Backward</div>
            <div style={{fontSize:10,color:C.muted,fontFamily:"monospace",lineHeight:1.8}}>
              <div>{"gradient \\u03B4 at output"}</div>
              <div style={{marginTop:4}}><span style={{color:C.dim}}>[0, 0]</span></div>
              <div><span style={{color:C.accent,fontWeight:700}}>{"[\\u03B4, 0]"}</span> <span style={{color:C.muted}}>{LARR+" only max gets \\u03B4"}</span></div>
            </div>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:10,fontWeight:700,color:C.cyan,marginBottom:6}}>Avg Pool Backward</div>
            <div style={{fontSize:10,color:C.muted,fontFamily:"monospace",lineHeight:1.8}}>
              <div>{"gradient \\u03B4 at output"}</div>
              <div style={{marginTop:4}}><span style={{color:C.cyan}}>{"[\\u03B4/4, \\u03B4/4]"}</span></div>
              <div><span style={{color:C.cyan}}>{"[\\u03B4/4, \\u03B4/4]"}</span> <span style={{color:C.muted}}>{LARR+" split equally"}</span></div>
            </div>
          </div>
        </div>
      </Card>

      <Insight>
        <span style={{color:C.accent,fontWeight:700}}>Max pooling</span> preserves the strongest signal. It provides <span style={{color:C.yellow,fontWeight:700}}>translation invariance</span>: a cat is still a cat shifted 2px. Pooling has <span style={{color:C.red,fontWeight:700}}>zero learnable parameters</span> {DASH} just a fixed math operation.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: FULL CNN PIPELINE
   =============================================================== */
function TabPipeline() {
  var _st = useState(0); var step = _st[0], setStep = _st[1];
  var _au = useState(false); var auto2 = _au[0], setAuto2 = _au[1];

  var steps = [
    {title:"Input Image", desc:"A 5"+MUL+"5 "+LQ+"X"+RQ+" pattern enters the network. Each pixel is a number (1=bright, 0=dark)."},
    {title:"Conv Layer (2 Filters)", desc:"Two 3"+MUL+"3 filters slide across. Filter A detects "+LQ+"\\\\"+RQ+" diagonals, Filter B detects "+LQ+"/"+RQ+" diagonals. Each produces a 3"+MUL+"3 feature map."},
    {title:"ReLU Activation", desc:"max(0, z) applied to every value. Negatives become 0. Here all values are already positive, so nothing changes."},
    {title:"Max Pooling (2"+MUL+"2)", desc:"Each 3"+MUL+"3 map downsampled to 2"+MUL+"2 by keeping the max in each window. Reduces computation + position invariance."},
    {title:"Flatten + FC", desc:"Two 2"+MUL+"2 maps "+ARR+" 8 values "+ARR+" 2 output neurons (X or O). Weighted sum + softmax."},
    {title:"Prediction!", desc:"Softmax outputs: 92% "+LQ+"X"+RQ+", 8% "+LQ+"O"+RQ+" "+DASH+" correct!"},
  ];

  useEffect(function(){if(!auto2)return;var t=setInterval(function(){setStep(function(s){return(s+1)%steps.length;});},3000);return function(){clearInterval(t);};}, [auto2]);

  var xP=[[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]];
  var fmA=[[3,1,1],[1,3,1],[1,1,3]], fmB=[[1,1,3],[1,3,1],[3,1,1]];
  var pA=[[3,1],[1,3]], pB=[[3,3],[3,1]];
  var cs2=28;

  return (
    <div>
      <SectionTitle title={"Building a CNN: "+LQ+"X"+RQ+" vs "+LQ+"O"+RQ+" Classifier"} subtitle="Trace data through every layer of a complete (tiny) CNN" />

      <div style={{display:"flex",justifyContent:"center",gap:6,marginBottom:16,flexWrap:"wrap"}}>
        {steps.map(function(s,i){return (<button key={i} onClick={function(){setStep(i);setAuto2(false);}} style={{padding:"6px 12px",borderRadius:6,border:"1.5px solid "+(step===i?C.accent:C.border),background:step===i?C.accent+"20":C.card,color:step===i?C.accent:C.muted,cursor:"pointer",fontSize:10,fontWeight:600,fontFamily:"monospace"}}>{i+1}</button>);})}
        <button onClick={function(){setAuto2(!auto2);}} style={{padding:"6px 12px",borderRadius:6,border:"1.5px solid "+(auto2?C.yellow:C.border),background:auto2?C.yellow+"20":C.card,color:auto2?C.yellow:C.muted,cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{auto2?PAUSE:PLAY+" Auto"}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={800} height={320} viewBox="0 0 800 320" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          {/* Input */}
          <g opacity={step>=0?1:0.2} style={{transition:"opacity 0.5s"}}>
            <text x={80} y={18} textAnchor="middle" fill={C.blue} fontSize={9} fontWeight={700} fontFamily="monospace">{"INPUT 5"+MUL+"5"}</text>
            {xP.map(function(ra,ri){return ra.map(function(v,ci){return <GridCell key={"x"+ri+ci} x={10+ci*cs2} y={24+ri*cs2} size={cs2} value={v} bg={v===1?C.blue+"20":"transparent"} border={v===1?C.blue+"50":C.dim+"30"} color={v===1?C.blue:C.dim} fontSize={9} />;});})}
            {step===0 && <rect x={8} y={22} width={cs2*5+4} height={cs2*5+4} fill="none" stroke={C.blue} strokeWidth={2} rx={4} strokeDasharray="5,3" />}
          </g>

          {step>=1 && <g><line x1={152} y1={95} x2={170} y2={95} stroke={C.dim} strokeWidth={1.5}/><polygon points="174,95 168,91 168,99" fill={C.dim}/></g>}

          {/* Feature maps */}
          <g opacity={step>=1?1:0.15} style={{transition:"opacity 0.5s"}}>
            <text x={240} y={18} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">CONV (2 filters)</text>
            <text x={210} y={35} fill={C.accent} fontSize={8} fontFamily="monospace">{"Map A ("+LQ+"\\\\"+RQ+")"}</text>
            {fmA.map(function(ra,ri){return ra.map(function(v,ci){var h=v===3;return <GridCell key={"a"+ri+ci} x={178+ci*cs2} y={40+ri*cs2} size={cs2} value={v} bg={h?C.accent+"30":"transparent"} border={h?C.accent:C.dim+"40"} color={h?C.accent:C.muted} bold={h} fontSize={9} />;});})}
            <text x={210} y={140} fill={C.purple} fontSize={8} fontFamily="monospace">{"Map B ("+LQ+"/"+RQ+")"}</text>
            {fmB.map(function(ra,ri){return ra.map(function(v,ci){var h=v===3;return <GridCell key={"b"+ri+ci} x={178+ci*cs2} y={148+ri*cs2} size={cs2} value={v} bg={h?C.purple+"30":"transparent"} border={h?C.purple:C.dim+"40"} color={h?C.purple:C.muted} bold={h} fontSize={9} />;});})}
          </g>

          {step>=2 && <g><line x1={268} y1={130} x2={286} y2={130} stroke={C.dim} strokeWidth={1.5}/><polygon points="290,130 284,126 284,134" fill={C.dim}/><text x={278} y={120} textAnchor="middle" fill={C.green} fontSize={7} fontFamily="monospace">ReLU</text></g>}

          {step>=3 && <g><line x1={290} y1={130} x2={330} y2={130} stroke={C.dim} strokeWidth={1.5}/><polygon points="334,130 328,126 328,134" fill={C.dim}/></g>}

          {/* Pooling */}
          <g opacity={step>=3?1:0.15} style={{transition:"opacity 0.5s"}}>
            <text x={400} y={18} textAnchor="middle" fill={C.yellow} fontSize={9} fontWeight={700} fontFamily="monospace">{"MAX POOL 2"+MUL+"2"}</text>
            <text x={370} y={42} fill={C.accent} fontSize={8} fontFamily="monospace">Pooled A</text>
            {pA.map(function(ra,ri){return ra.map(function(v,ci){return <GridCell key={"pa"+ri+ci} x={345+ci*34} y={48+ri*34} size={34} value={v} bg={C.accent+"15"} border={C.accent+"40"} color={C.accent} bold={true} fontSize={11} />;});})}
            <text x={370} y={135} fill={C.purple} fontSize={8} fontFamily="monospace">Pooled B</text>
            {pB.map(function(ra,ri){return ra.map(function(v,ci){return <GridCell key={"pb"+ri+ci} x={345+ci*34} y={140+ri*34} size={34} value={v} bg={C.purple+"15"} border={C.purple+"40"} color={C.purple} bold={true} fontSize={11} />;});})}
          </g>

          {step>=4 && <g><line x1={418} y1={130} x2={460} y2={130} stroke={C.dim} strokeWidth={1.5}/><polygon points="464,130 458,126 458,134" fill={C.dim}/><text x={440} y={120} textAnchor="middle" fill={C.muted} fontSize={7} fontFamily="monospace">flatten</text></g>}

          {/* FC */}
          <g opacity={step>=4?1:0.15} style={{transition:"opacity 0.5s"}}>
            <text x={530} y={18} textAnchor="middle" fill={C.cyan} fontSize={9} fontWeight={700} fontFamily="monospace">FLATTEN + FC</text>
            {[3,1,1,3,3,3,3,1].map(function(v,i){return <GridCell key={"fl"+i} x={470} y={28+i*26} size={24} value={v} bg={i<4?C.accent+"10":C.purple+"10"} border={i<4?C.accent+"30":C.purple+"30"} color={i<4?C.accent:C.purple} fontSize={9} />;})}
            <text x={482} y={245} textAnchor="middle" fill={C.muted} fontSize={7} fontFamily="monospace">8 values</text>
            {step>=4 && [0,1,2,3,4,5,6,7].map(function(i){return [0,1].map(function(j){return <line key={"fc"+i+j} x1={494} y1={40+i*26} x2={560} y2={100+j*80} stroke={j===0?C.green:C.pink} strokeWidth={0.5} opacity={0.25} />;});})}
            <circle cx={580} cy={100} r={22} fill={C.card} stroke={step>=5?C.green:C.dim} strokeWidth={step>=5?2.5:1} />
            <text x={580} y={103} textAnchor="middle" fill={step>=5?C.green:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{LQ+"X"+RQ}</text>
            <circle cx={580} cy={180} r={22} fill={C.card} stroke={step>=5?C.pink:C.dim} strokeWidth={step>=5?2.5:1} />
            <text x={580} y={183} textAnchor="middle" fill={step>=5?C.pink:C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{LQ+"O"+RQ}</text>
          </g>

          {step>=5 && <g>
            <line x1={602} y1={100} x2={640} y2={100} stroke={C.green} strokeWidth={2} />
            <line x1={602} y1={180} x2={640} y2={180} stroke={C.pink} strokeWidth={1} opacity={0.4} />
            <rect x={645} y={78} width={90} height={44} rx={8} fill={C.green+"15"} stroke={C.green} strokeWidth={2} />
            <text x={690} y={96} textAnchor="middle" fill={C.green} fontSize={16} fontWeight={800} fontFamily="monospace">92%</text>
            <text x={690} y={114} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{LQ+"X"+RQ+" "+CHK}</text>
            <rect x={645} y={158} width={90} height={44} rx={8} fill={C.pink+"08"} stroke={C.pink+"40"} strokeWidth={1} />
            <text x={690} y={176} textAnchor="middle" fill={C.pink} fontSize={16} fontWeight={800} fontFamily="monospace" opacity={0.6}>8%</text>
            <text x={690} y={194} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{LQ+"O"+RQ}</text>
          </g>}

          <rect x={10} y={270} width={780} height={40} rx={6} fill={C.card} stroke={C.border} />
          <text x={400} y={287} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">
            <tspan fill={C.accent}>{"Conv: 20 params"}</tspan>
            <tspan fill={C.dim}>{" (2"+MUL+"(3"+MUL+"3+1))  |  "}</tspan>
            <tspan fill={C.yellow}>{"Pool: 0"}</tspan>
            <tspan fill={C.dim}>{"  |  "}</tspan>
            <tspan fill={C.cyan}>{"FC: 18 params"}</tspan>
            <tspan fill={C.dim}>{" (2"+MUL+"(8+1))  |  "}</tspan>
            <tspan fill={C.green} fontWeight="700">{"Total: 38"}</tspan>
          </text>
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:4}}>{"Step "+(step+1)+": "+steps[step].title}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{steps[step].desc}</div>
      </Card>

      <Insight icon={TARG} title="The Complete Flow">
        <span style={{color:C.blue}}>Input</span> {ARR} <span style={{color:C.accent}}>Conv</span> {ARR} <span style={{color:C.green}}>ReLU</span> {ARR} <span style={{color:C.yellow}}>Pool</span> {ARR} <span style={{color:C.cyan}}>FC</span> {ARR} <span style={{color:C.green,fontWeight:700}}>Softmax</span>. This pattern repeats with more filters and layers in real CNNs.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: ARCHITECTURE EVOLUTION
   =============================================================== */
function TabArchitectures() {
  var _s2 = useState(0); var sel2 = _s2[0], setSel2 = _s2[1];

  var archs = [
    { name:"LeNet-5",year:1998,color:C.blue,depth:7,params:"60K",error:"--",inventor:"Yann LeCun",
      innovation:"First working CNN deployed commercially (bank check reading)",
      layers:"Conv(6,5"+MUL+"5) "+ARR+" Pool "+ARR+" Conv(16,5"+MUL+"5) "+ARR+" Pool "+ARR+" FC(120) "+ARR+" FC(84) "+ARR+" 10",
      insight:"Proved local filters + weight sharing + stacking works. Computers were too slow for it to take off." },
    { name:"AlexNet",year:2012,color:C.accent,depth:8,params:"60M",error:"16.4%",inventor:"Alex Krizhevsky",
      innovation:"Won ImageNet by a huge margin "+DASH+" reignited deep learning",
      layers:"Conv(96,11"+MUL+"11,s4) "+ARR+" Pool "+ARR+" Conv(256,5"+MUL+"5) "+ARR+" Pool "+ARR+" Conv"+MUL+"3 "+ARR+" Pool "+ARR+" FC(4096)"+MUL+"2 "+ARR+" 1000",
      insight:"Just a bigger LeNet. Breakthrough was GPU training + ReLU + Dropout + massive data." },
    { name:"VGGNet",year:2014,color:C.purple,depth:19,params:"138M",error:"7.3%",inventor:"Simonyan & Zisserman",
      innovation:"Only 3"+MUL+"3 filters, much deeper "+DASH+" depth beats large filters",
      layers:"[Conv 3"+MUL+"3, 64]"+MUL+"2 "+ARR+" Pool "+ARR+" [3"+MUL+"3, 128]"+MUL+"2 "+ARR+" Pool "+ARR+" [256]"+MUL+"3 "+ARR+" [512]"+MUL+"6 "+ARR+" FC"+MUL+"2 "+ARR+" 1000",
      insight:"Two 3"+MUL+"3 convs = one 5"+MUL+"5 receptive field, 28% fewer params, extra ReLU. Simple but 138M params." },
    { name:"GoogLeNet",year:2014,color:C.yellow,depth:22,params:"6.8M",error:"6.7%",inventor:"Google",
      innovation:"Inception module "+DASH+" parallel 1"+MUL+"1, 3"+MUL+"3, 5"+MUL+"5 filters",
      layers:"Conv "+ARR+" [Inception]"+MUL+"9 "+ARR+" Global Avg Pool "+ARR+" 1000",
      insight:"23"+MUL+" fewer params than VGG! 1"+MUL+"1 bottlenecks + global avg pooling were key." },
    { name:"ResNet",year:2015,color:C.green,depth:152,params:"25.6M",error:"3.6%",inventor:"Kaiming He et al.",
      innovation:"Skip connections "+DASH+" enabled 100+ layer networks",
      layers:"Conv 7"+MUL+"7 "+ARR+" [Bottleneck + skip]"+MUL+"~50 "+ARR+" Global Avg Pool "+ARR+" 1000",
      insight:"Residual connection F(x)+x lets gradients flow directly back. Learn identity by setting F(x)=0." },
  ];

  var arch = archs[sel2];

  return (
    <div>
      <SectionTitle title="The Evolution of CNN Architectures" subtitle={"From 60K params to 152 layers "+DASH+" each solved a specific problem"} />

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
        {archs.map(function(a,i){var on=sel2===i;return (<button key={i} onClick={function(){setSel2(i);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?a.color:C.border),background:on?a.color+"20":C.card,color:on?a.color:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{a.name+" ("+a.year+")"}</button>);})}
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:arch.color}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:16}}>
          <div>
            <div style={{fontSize:20,fontWeight:800,color:arch.color}}>{arch.name}</div>
            <div style={{fontSize:11,color:C.muted,marginTop:2}}>{arch.year+" "+DASH+" "+arch.inventor}</div>
            <div style={{fontSize:11,color:C.text,marginTop:10,lineHeight:1.6}}>{arch.innovation}</div>
          </div>
          <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
            {[{l:"DEPTH",v:arch.depth,s:"layers"},{l:"PARAMS",v:arch.params,s:""},{l:"TOP-5 ERR",v:arch.error,s:""}].map(function(d,i){
              return (<div key={i} style={{textAlign:"center"}}><div style={{fontSize:8,color:C.muted}}>{d.l}</div><div style={{fontSize:22,fontWeight:800,color:d.v==="--"?C.dim:arch.color}}>{d.v}</div>{d.s&&<div style={{fontSize:8,color:C.dim}}>{d.s}</div>}</div>);
            })}
          </div>
        </div>
        <div style={{marginTop:14,padding:"10px 14px",background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          <div style={{fontSize:9,color:C.muted,marginBottom:4}}>ARCHITECTURE</div>
          <div style={{fontSize:10,color:arch.color,fontFamily:"monospace",lineHeight:1.8,overflowX:"auto"}}>{arch.layers}</div>
        </div>
        <div style={{marginTop:12,fontSize:11,color:C.muted,lineHeight:1.7,fontStyle:"italic",borderLeft:"3px solid "+arch.color+"40",paddingLeft:12}}>{arch.insight}</div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Comparison</div>
        <div style={{marginBottom:16}}>
          <div style={{fontSize:9,color:C.muted,marginBottom:6}}>NETWORK DEPTH (layers)</div>
          {archs.map(function(a,i){var w=Math.max(8,(a.depth/152)*100); return (<div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}><div style={{width:70,fontSize:9,color:sel2===i?a.color:C.muted,textAlign:"right",fontFamily:"monospace",fontWeight:sel2===i?700:400}}>{a.name}</div><div style={{flex:1,position:"relative",height:18}}><div style={{width:w+"%",height:"100%",borderRadius:4,background:a.color+(sel2===i?"60":"25"),border:"1px solid "+a.color+(sel2===i?"80":"30"),transition:"all 0.3s"}} /></div><div style={{width:35,fontSize:9,color:sel2===i?a.color:C.dim,fontFamily:"monospace"}}>{a.depth}</div></div>);})}
        </div>
        <div>
          <div style={{fontSize:9,color:C.muted,marginBottom:6}}>TOP-5 ERROR RATE (%)</div>
          {archs.map(function(a,i){var err=parseFloat(a.error)||0;var w=err>0?Math.max(8,(err/20)*100):0; return (<div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}><div style={{width:70,fontSize:9,color:sel2===i?a.color:C.muted,textAlign:"right",fontFamily:"monospace",fontWeight:sel2===i?700:400}}>{a.name}</div><div style={{flex:1,position:"relative",height:18}}>{w>0&&<div style={{width:w+"%",height:"100%",borderRadius:4,background:a.color+(sel2===i?"60":"25"),border:"1px solid "+a.color+(sel2===i?"80":"30"),transition:"all 0.3s"}} />}</div><div style={{width:45,fontSize:9,color:sel2===i?a.color:C.dim,fontFamily:"monospace"}}>{a.error}</div></div>);})}
        </div>
      </Card>

      {sel2===4 && <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.green,marginBottom:10}}>The Skip Connection (ResNet)</div>
        <div style={{display:"flex",justifyContent:"center"}}>
          <svg width={500} height={150} viewBox="0 0 500 150" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
            <text x={120} y={18} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">Standard Block</text>
            <rect x={70} y={25} width={100} height={28} rx={4} fill={C.purple+"15"} stroke={C.purple+"40"} /><text x={120} y={43} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">Conv + BN</text>
            <line x1={120} y1={53} x2={120} y2={68} stroke={C.dim} strokeWidth={1} />
            <rect x={70} y={68} width={100} height={28} rx={4} fill={C.purple+"15"} stroke={C.purple+"40"} /><text x={120} y={86} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">Conv + BN</text>
            <line x1={120} y1={96} x2={120} y2={115} stroke={C.dim} strokeWidth={1} /><polygon points="120,118 116,112 124,112" fill={C.dim} />
            <text x={120} y={135} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Output</text>

            <text x={370} y={18} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">Residual Block</text>
            <text x={290} y={55} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">Input x</text>
            <rect x={320} y={25} width={100} height={28} rx={4} fill={C.purple+"15"} stroke={C.purple+"40"} /><text x={370} y={43} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">Conv + BN</text>
            <line x1={370} y1={53} x2={370} y2={68} stroke={C.dim} strokeWidth={1} />
            <rect x={320} y={68} width={100} height={28} rx={4} fill={C.purple+"15"} stroke={C.purple+"40"} /><text x={370} y={86} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">Conv + BN</text>
            <line x1={310} y1={40} x2={310} y2={107} stroke={C.green} strokeWidth={2} strokeDasharray="4,3" />
            <line x1={310} y1={107} x2={360} y2={107} stroke={C.green} strokeWidth={2} strokeDasharray="4,3" />
            <text x={298} y={75} textAnchor="end" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">+x</text>
            <circle cx={370} cy={107} r={10} fill={C.green+"20"} stroke={C.green} strokeWidth={1.5} />
            <text x={370} y={110} textAnchor="middle" fill={C.green} fontSize={11} fontWeight={800} fontFamily="monospace">+</text>
            <line x1={370} y1={117} x2={370} y2={130} stroke={C.dim} strokeWidth={1} /><polygon points="370,133 366,127 374,127" fill={C.dim} />
            <text x={370} y={146} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">F(x) + x</text>
            <text x={445} y={110} fill={C.muted} fontSize={8} fontFamily="monospace">{LARR+" identity shortcut"}</text>
          </svg>
        </div>
      </Card>}

      <Insight icon={TARG} title="The Trend">
        Deeper networks, <span style={{color:C.green,fontWeight:700}}>fewer parameters</span> (after VGG), lower error. ResNet-152 has fewer params than VGG-19, thanks to <span style={{color:C.purple,fontWeight:700}}>bottleneck blocks</span> and <span style={{color:C.yellow,fontWeight:700}}>global average pooling</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Convolution", "Filters & Features", "Pooling", "Full CNN Pipeline", "Architecture Evolution"];
  return (
    <div style={{ background:C.bg, minHeight:"100vh", padding:"24px 16px", fontFamily:"'JetBrains Mono','SF Mono',monospace", color:C.text, maxWidth:960, margin:"0 auto" }}>
      <div style={{textAlign:"center",marginBottom:16}}>
        <div style={{ fontSize:22, fontWeight:800, background:"linear-gradient(135deg,"+C.accent+","+C.yellow+")", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", display:"inline-block" }}>Convolutional Neural Networks</div>
        <div style={{fontSize:11,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough "+DASH+" from convolution to ResNet"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab===0 && <TabConvolution />}
      {tab===1 && <TabFilters />}
      {tab===2 && <TabPooling />}
      {tab===3 && <TabPipeline />}
      {tab===4 && <TabArchitectures />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

CNN_VISUAL_HEIGHT = 1200