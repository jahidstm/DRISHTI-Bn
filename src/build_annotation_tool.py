"""
DRISHTI-XAI -- Task 3.2: Annotation Tool Builder
Generates self-contained HTML annotation tools with embedded images.
Usage:
    python src/build_annotation_tool.py
    python src/build_annotation_tool.py --gen_captions   # also generate model captions
"""

import os, sys, re, json, base64, random, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT    = "/content/DRISHTI-Bn" if os.path.exists("/content/DRISHTI-Bn") else PROJECT_ROOT
SAMPLES_CSV  = os.path.join(REPO_ROOT, "human_eval_outputs", "human_eval_samples.csv")
MASTER_CSV   = os.path.join(REPO_ROOT, "data", "processed", "master_dataset_translated.csv")
IMG_DIR      = os.path.join(REPO_ROOT, "data", "processed")
MODEL_PATH   = os.path.join(REPO_ROOT, "models", "disasternet_multitask_v2.pth")
OUT_DIR      = os.path.join(REPO_ROOT, "human_eval_outputs")


def clean_caption(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def img_to_b64(img_path, max_side=400):
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  [!] Could not encode {img_path}: {e}")
        return ""


def find_model_path(custom_path=None):
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    candidates.extend([
        os.path.join(REPO_ROOT, "models", "disasternet_multitask_v2.pth"),
        os.path.join(PROJECT_ROOT, "models", "disasternet_multitask_v2.pth"),
        "/content/drive/MyDrive/DisasterNet/models/disasternet_multitask_v2.pth",
        "/content/drive/MyDrive/models/disasternet_multitask_v2.pth",
        "/content/disasternet_multitask_v2.pth",
        os.path.join(REPO_ROOT, "src", "disasternet_multitask_v2.pth"),
    ])
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def generate_model_captions(samples_df, model_path=None):
    try:
        import torch
        sys.path.insert(0, SCRIPT_DIR)
        from evaluate import DisasterNetPredictor
    except ImportError as e:
        print(f"  [!] Cannot import model: {e}. Using placeholders.")
        return ["[Model caption placeholder]"] * len(samples_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [+] Device: {device}")

    resolved_path = find_model_path(model_path)
    if not resolved_path:
        print(f"  [!] Model checkpoint not found in standard paths.")
        print(f"      Checked: {model_path or os.path.join(REPO_ROOT, 'models', 'disasternet_multitask_v2.pth')}")
        return ["[Checkpoint not found]"] * len(samples_df)

    print(f"  [+] Loading model from {resolved_path} ...")
    predictor = DisasterNetPredictor(model_path=resolved_path, device=device)

    captions = []
    total = len(samples_df)
    for i, (_, row) in enumerate(samples_df.iterrows()):
        img_path = os.path.join(IMG_DIR, row["image_path"])
        print(f"  [{i+1:3d}/{total}] {row['sample_id']} ...", end=" ", flush=True)
        try:
            img = Image.open(img_path).convert("RGB")
            pv  = predictor.transform(img).unsqueeze(0).to(device)
            cap = predictor.generate_caption(pv, beam_width=1)
            cap = clean_caption(cap)
            captions.append(cap if cap else "[ক্যাপশন পাওয়া যায়নি]")
            print(cap[:70])
        except Exception as e:
            print(f"ERROR: {e}")
            captions.append("[ক্যাপশন পাওয়া যায়নি]")
    return captions


# ── Build HTML ─────────────────────────────────────────────────────────────────
def build_html(annotator_id, tool_samples):
    samples_json = json.dumps(tool_samples, ensure_ascii=False)
    total = len(tool_samples)
    return f"""<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DRISHTI-Bn মানবিক মূল্যায়ন &mdash; {annotator_id}</title>
<link href="https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0f1117;--card:#1a1d27;--card2:#22263a;--acc:#7c6af7;--acc2:#e879f9;--grn:#4ade80;--yel:#fbbf24;--txt:#e2e8f0;--mut:#94a3b8;--bdr:#2d3248;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--txt);font-family:'Hind Siliguri','Inter',sans-serif;min-height:100vh;}}
.hdr{{background:linear-gradient(135deg,#1a1d27,#22263a);border-bottom:1px solid var(--bdr);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}}
.hdr h1{{font-size:1rem;font-weight:700;background:linear-gradient(135deg,var(--acc),var(--acc2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.hdr p{{font-size:0.75rem;color:var(--mut);margin-top:2px;}}
.pb-out{{width:180px;height:7px;background:var(--bdr);border-radius:99px;overflow:hidden;}}
.pb-in{{height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2));border-radius:99px;transition:width .4s;}}
.plbl{{font-size:0.78rem;color:var(--mut);margin-left:10px;white-space:nowrap;}}
.pw{{display:flex;align-items:center;}}
.wrap{{max-width:920px;margin:0 auto;padding:24px 16px 80px;}}
.card{{background:var(--card);border:1px solid var(--bdr);border-radius:16px;overflow:hidden;display:none;animation:fi .3s ease;}}
.card.on{{display:block;}}
@keyframes fi{{from{{opacity:0;transform:translateY(8px);}}to{{opacity:1;transform:translateY(0);}}}}
.ct{{display:grid;grid-template-columns:1fr 1fr;}}
.ipane{{padding:18px;border-right:1px solid var(--bdr);display:flex;flex-direction:column;align-items:center;gap:10px;}}
.si{{width:100%;max-height:270px;object-fit:cover;border-radius:10px;border:1px solid var(--bdr);}}
.no-img{{width:100%;height:200px;background:#1a1d27;border-radius:10px;border:1px solid var(--bdr);display:flex;align-items:center;justify-content:center;color:#475569;font-size:0.8rem;}}
.badge{{display:inline-block;padding:3px 12px;border-radius:99px;font-size:0.7rem;font-weight:700;letter-spacing:.03em;}}
.bs{{background:rgba(248,113,113,.12);color:#f87171;border:1px solid rgba(248,113,113,.3);}}
.bh{{background:rgba(124,106,247,.12);color:#a78bfa;border:1px solid rgba(124,106,247,.3);}}
.ba{{background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.3);}}
.cpane{{padding:18px;display:flex;flex-direction:column;gap:12px;}}
.sid{{font-size:0.75rem;color:var(--mut);display:flex;justify-content:space-between;}}
.cb{{background:var(--card2);border:1px solid var(--bdr);border-radius:10px;padding:13px;}}
.cl{{font-size:0.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px;}}
.cla{{color:var(--grn);}}.clb{{color:var(--acc2);}}
.ct2{{font-size:0.88rem;line-height:1.7;}}
.rs{{padding:18px;border-top:1px solid var(--bdr);display:flex;flex-direction:column;gap:16px;}}
.rs h3{{font-size:0.8rem;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;}}
.rg{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.rr{{display:flex;flex-direction:column;gap:6px;}}
.rq{{font-size:0.8rem;font-weight:600;}}
.rsub{{font-size:0.68rem;color:var(--mut);}}
.stars{{display:flex;gap:5px;}}
.sb{{width:35px;height:35px;border-radius:8px;border:1.5px solid var(--bdr);background:transparent;color:var(--mut);font-size:0.85rem;font-weight:700;cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center;}}
.sb:hover{{border-color:var(--acc);color:var(--acc);background:rgba(124,106,247,.1);}}
.sb.sel{{background:var(--acc);border-color:var(--acc);color:#fff;}}
.pr{{display:flex;flex-direction:column;gap:7px;}}
.pbs{{display:flex;gap:8px;}}
.pb{{flex:1;padding:9px;border-radius:10px;border:1.5px solid var(--bdr);background:transparent;color:var(--mut);font-size:0.83rem;font-weight:600;cursor:pointer;transition:all .15s;font-family:inherit;}}
.pb:hover{{border-color:var(--acc);color:var(--acc);background:rgba(124,106,247,.08);}}
.sA{{background:rgba(74,222,128,.12);border-color:var(--grn);color:var(--grn);}}
.sB{{background:rgba(232,121,249,.12);border-color:var(--acc2);color:var(--acc2);}}
.sE{{background:rgba(251,191,36,.12);border-color:var(--yel);color:var(--yel);}}
.cmt{{width:100%;background:var(--card2);border:1.5px solid var(--bdr);border-radius:10px;color:var(--txt);font-family:'Hind Siliguri',sans-serif;font-size:0.83rem;padding:10px;resize:vertical;min-height:54px;}}
.cmt:focus{{outline:none;border-color:var(--acc);}}
.nav{{display:flex;gap:10px;padding:18px;border-top:1px solid var(--bdr);justify-content:space-between;align-items:center;}}
.nb{{padding:10px 26px;border-radius:10px;border:none;font-family:'Hind Siliguri',sans-serif;font-size:0.88rem;font-weight:600;cursor:pointer;transition:all .15s;}}
.nprev{{background:var(--card2);color:var(--mut);border:1.5px solid var(--bdr);}}
.nprev:hover{{color:var(--txt);border-color:var(--txt);}}
.nnext{{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#fff;}}
.nnext:hover{{opacity:.9;transform:translateY(-1px);}}
.warn{{display:none;font-size:0.75rem;color:#f87171;}}
#fin{{display:none;text-align:center;padding:56px 24px;background:var(--card);border:1px solid var(--bdr);border-radius:16px;}}
#fin .ico{{font-size:3.5rem;margin-bottom:14px;}}
#fin h2{{font-size:1.5rem;margin-bottom:7px;background:linear-gradient(135deg,var(--acc),var(--grn));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
#fin p{{color:var(--mut);margin-bottom:22px;}}
.expo{{display:inline-block;padding:13px 34px;border-radius:12px;background:linear-gradient(135deg,var(--grn),#22d3ee);color:#0f1117;font-weight:700;font-size:0.95rem;cursor:pointer;border:none;font-family:inherit;}}
.expo:hover{{opacity:.9;transform:translateY(-2px);}}
@media(max-width:640px){{.ct{{grid-template-columns:1fr;}}.ipane{{border-right:none;border-bottom:1px solid var(--bdr);}}.rg{{grid-template-columns:1fr;}}.pb-out{{display:none;}}}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>🌊 DRISHTI-Bn মানবিক মূল্যায়ন</h1>
    <p>{annotator_id} &bull; মোট {total}টি দুর্যোগ চিত্র মূল্যায়ন করুন</p>
  </div>
  <div class="pw">
    <div class="pb-out"><div class="pb-in" id="pbi" style="width:0%"></div></div>
    <span class="plbl" id="pbl">0 / {total}</span>
  </div>
</div>
<div class="wrap" id="mc"></div>
<div class="wrap" id="fc" style="display:none">
  <div id="fin">
    <div class="ico">🎉</div>
    <h2>মূল্যায়ন সম্পন্ন!</h2>
    <p>আপনি সবগুলো চিত্র মূল্যায়ন করেছেন।<br>নিচের বোতামে ক্লিক করলেই CSV ডাউনলোড হবে।</p>
    <button class="expo" onclick="xport()">📥 CSV ডাউনলোড করুন</button>
    <p style="margin-top:14px;font-size:0.78rem;">ডাউনলোড হলে গবেষককে ফাইলটি পাঠিয়ে দিন। ধন্যবাদ!</p>
  </div>
</div>
<script>
const AID="{annotator_id}";
const S={samples_json};
let cur=0;
const R=S.map(()=>({{fa:null,aa:null,xa:null,fb:null,ab:null,xb:null,pr:null,cm:""}}));
function bc(lbl){{if(lbl.includes("Severe"))return"bs";if(lbl.includes("Human"))return"bh";return"ba";}}
function bt(lbl){{if(lbl.includes("Severe"))return"মারাত্মক ক্ষতি";if(lbl.includes("Human"))return"মানবিক উদ্ধার";return"ক্ষতিগ্রস্ত মানুষ";}}
function rb(i,k,rk,lbl,sub,col){{
  const btns=[1,2,3,4,5].map(v=>`<button class="sb" id="s${{i}}_${{k}}_${{v}}" onclick="sr(${{i}},'${{k}}',${{v}})">${{v}}</button>`).join("");
  return `<div class="rr"><div class="rq" style="color:${{col}}">ক্যাপশন ${{lbl}}: ${{rk}}</div><div class="rsub">${{sub}}</div><div class="stars">${{btns}}</div></div>`;
}}
function init(){{
  const mc=document.getElementById("mc");
  S.forEach((s,i)=>{{
    const d=document.createElement("div");
    d.className="card"+(i===0?" on":"");d.id="c"+i;
    const imgHtml=s.b64?`<img class="si" src="data:image/jpeg;base64,${{s.b64}}" alt="চিত্র">`:`<div class="no-img">ছবি পাওয়া যায়নি</div>`;
    d.innerHTML=`
<div class="ct">
  <div class="ipane">
    ${{imgHtml}}
    <span class="badge ${{bc(s.cl)}}">${{bt(s.cl)}}</span>
  </div>
  <div class="cpane">
    <div class="sid"><span>চিত্র ${{i+1}} / ${{S.length}}</span><span>${{s.sid}}</span></div>
    <div class="cb"><div class="cl cla">ক্যাপশন A</div><div class="ct2">${{s.cA||"(নেই)"}}</div></div>
    <div class="cb"><div class="cl clb">ক্যাপশন B</div><div class="ct2">${{s.cB||"(নেই)"}}</div></div>
  </div>
</div>
<div class="rs">
  <h3>📊 রেটিং দিন (১ = খুব খারাপ, ৫ = চমৎকার)</h3>
  <div class="rg">
    ${{rb(i,"fa","fa","A","সাবলীলতা","বাক্যটি কতটা স্বাভাবিক বাংলায়?","#4ade80")}}
    ${{rb(i,"fb","fb","B","সাবলীলতা","বাক্যটি কতটা স্বাভাবিক বাংলায়?","#e879f9")}}
    ${{rb(i,"aa","aa","A","পর্যাপ্ততা","ছবির দুর্যোগ তথ্য কতটুকু আছে?","#4ade80")}}
    ${{rb(i,"ab","ab","B","পর্যাপ্ততা","ছবির দুর্যোগ তথ্য কতটুকু আছে?","#e879f9")}}
    ${{rb(i,"xa","xa","A","বিশ্বাসযোগ্যতা","ছবিতে নেই এমন কিছু আছে? ৫=নেই","#4ade80")}}
    ${{rb(i,"xb","xb","B","বিশ্বাসযোগ্যতা","ছবিতে নেই এমন কিছু আছে? ৫=নেই","#e879f9")}}
  </div>
  <div class="pr">
    <div class="rq">কোন ক্যাপশনটি আপনার বেশি ভালো লেগেছে?</div>
    <div class="pbs">
      <button class="pb" id="p${{i}}A" onclick="sp(${{i}},'A')">A ভালো</button>
      <button class="pb" id="p${{i}}E" onclick="sp(${{i}},'E')">সমান</button>
      <button class="pb" id="p${{i}}B" onclick="sp(${{i}},'B')">B ভালো</button>
    </div>
  </div>
  <div>
    <div class="rq" style="margin-bottom:6px">মন্তব্য (ঐচ্ছিক)</div>
    <textarea class="cmt" id="cm${{i}}" placeholder="যেকোনো মন্তব্য লিখুন..." oninput="R[${{i}}].cm=this.value"></textarea>
  </div>
  <div class="warn" id="w${{i}}">⚠️ অনুগ্রহ করে সব রেটিং এবং পছন্দ দিয়ে পরবর্তী চিত্রে যান।</div>
</div>
<div class="nav">
  <button class="nb nprev" onclick="go(-1)" ${{i===0?'style="visibility:hidden"':''}}>&larr; আগে</button>
  <span style="font-size:.75rem;color:var(--mut)">${{i+1}} / ${{S.length}}</span>
  ${{i<S.length-1
    ?`<button class="nb nnext" onclick="go(1)">পরবর্তী &rarr;</button>`
    :`<button class="nb nnext" onclick="done()" style="background:linear-gradient(135deg,#4ade80,#22d3ee);color:#0f1117">&#10003; সম্পন্ন করুন</button>`
  }}
</div>`;
    mc.appendChild(d);
  }});
}}
function sr(i,k,v){{
  R[i][k]=v;
  [1,2,3,4,5].forEach(n=>{{const b=document.getElementById(`s${{i}}_${{k}}_${{n}}`);if(b)b.className="sb"+(n<=v?" sel":"");}});
}}
function sp(i,v){{
  R[i].pr=v;
  ["A","E","B"].forEach(p=>{{const b=document.getElementById(`p${{i}}${{p}}`);if(b)b.className="pb"+(v===p?` s${{p}}`:"");}});
}}
function ok(i){{const r=R[i];return r.fa&&r.aa&&r.xa&&r.fb&&r.ab&&r.xb&&r.pr;}}
function go(d){{
  if(!ok(cur)){{document.getElementById("w"+cur).style.display="block";return;}}
  document.getElementById("w"+cur).style.display="none";
  document.getElementById("c"+cur).classList.remove("on");
  cur=Math.max(0,Math.min(S.length-1,cur+d));
  document.getElementById("c"+cur).classList.add("on");
  upd();window.scrollTo(0,0);
}}
function done(){{
  if(!ok(cur)){{document.getElementById("w"+cur).style.display="block";return;}}
  document.getElementById("mc").style.display="none";
  document.getElementById("fc").style.display="block";
}}
function upd(){{
  const n=R.filter((_,i)=>i<cur||ok(i)).length;
  const p=Math.round(n/S.length*100);
  document.getElementById("pbi").style.width=p+"%";
  document.getElementById("pbl").textContent=n+" / "+S.length;
}}
function xport(){{
  const hdr=["annotator_id","sample_id","class_label","image_path","caption_A","caption_B","fluency_A","adequacy_A","faithfulness_A","fluency_B","adequacy_B","faithfulness_B","preference","comments"];
  const rows=S.map((s,i)=>{{
    const r=R[i];
    return[AID,s.sid,s.cl,s.ip,
      '"'+(s.cA||"").replace(/"/g,'""')+'"','"'+(s.cB||"").replace(/"/g,'""')+'"',
      r.fa||"",r.aa||"",r.xa||"",r.fb||"",r.ab||"",r.xb||"",
      r.pr||"",'"'+(r.cm||"").replace(/"/g,'""')+'"'].join(",");
  }});
  const csv=[hdr.join(","),...rows].join("\\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv;charset=utf-8;"}}));
  a.download=AID.toLowerCase()+"_responses.csv";a.click();
}}
init();upd();
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_captions", action="store_true",
                    help="Generate model captions using trained multitask model")
    ap.add_argument("--model_path", type=str, default=None,
                    help="Path to trained model checkpoint (.pth)")
    ap.add_argument("--max_side", type=int, default=420,
                    help="Max pixel side for embedded images (default 420)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 68)
    print("  DRISHTI-XAI -- Annotation Tool Builder")
    print("=" * 68)

    # 1. Load samples
    print(f"\n[1/4] Loading {SAMPLES_CSV} ...")
    samples = pd.read_csv(SAMPLES_CSV)
    print(f"      {len(samples)} samples loaded.")

    # 2. Merge cleaned Bengali GT captions
    print("[2/4] Merging cleaned Bengali captions ...")
    master = pd.read_csv(MASTER_CSV)
    master["cap_clean"] = master["bengali_caption"].apply(clean_caption)
    samples = samples.merge(
        master[["image_path", "cap_clean"]].drop_duplicates("image_path"),
        on="image_path", how="left"
    )

    # 3. Model captions
    if args.gen_captions:
        print("[3/4] Generating model captions ...")
        model_caps = generate_model_captions(samples, model_path=args.model_path)
    else:
        model_caps = ["[মডেল ক্যাপশন — Colab-এ --gen_captions দিয়ে রান করুন]"] * len(samples)
        print("[3/4] Skipping model caption generation.")

    samples["model_cap"] = model_caps

    # 4. Build per-annotator HTML
    print("\n[4/4] Building HTML annotation tools ...")
    annotators = ["Annotator_1", "Annotator_2", "Annotator_3"]
    generated_files = []

    for ann_id in annotators:
        rng = random.Random(hash(ann_id))
        tool_samples = []

        for _, row in samples.iterrows():
            img_path = os.path.join(IMG_DIR, str(row["image_path"]))
            b64 = img_to_b64(img_path, args.max_side)

            cap_gt    = str(row.get("cap_clean", "") or "")
            cap_model = str(row.get("model_cap", "") or "")

            # Blind A/B assignment: randomly swap GT and model
            if rng.random() > 0.5:
                cA, cB, hA, hB = cap_gt, cap_model, "gt", "model"
            else:
                cA, cB, hA, hB = cap_model, cap_gt, "model", "gt"

            tool_samples.append({
                "sid": row["sample_id"],
                "cl":  row["class_label"],
                "ip":  row["image_path"],
                "cA":  cA,
                "cB":  cB,
                "hA":  hA,
                "hB":  hB,
                "b64": b64,
            })

        html = build_html(ann_id, tool_samples)
        out  = os.path.join(OUT_DIR, f"annotation_tool_{ann_id}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        size_mb = os.path.getsize(out) / (1024 * 1024)
        print(f"  [{ann_id}] {out}  ({size_mb:.1f} MB)")
        generated_files.append((out, f"annotation_tool_{ann_id}.html"))

    # Create zip bundle for convenient 1-click download
    import zipfile
    zip_path = os.path.join(OUT_DIR, "annotation_tools_all.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for full_p, arc_name in generated_files:
            zf.write(full_p, arcname=arc_name)
    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n[+] Created zip bundle: {zip_path} ({zip_mb:.1f} MB)")

    print(f"""
======================================================================
Done! Files saved to: {OUT_DIR}/
  - annotation_tool_Annotator_1.html  -> বন্ধু ১ কে দিন
  - annotation_tool_Annotator_2.html  -> বন্ধু ২ কে দিন
  - annotation_tool_Annotator_3.html  -> বন্ধু ৩ কে দিন
  - annotation_tools_all.zip          -> এক ক্লিকে ৩টি ফাইল ডাউনলোড
======================================================================
""")


if __name__ == "__main__":
    main()
