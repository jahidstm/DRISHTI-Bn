"""
BD external test set — make the labelling package for the two labellers
=======================================================================
Creates ONE self-contained file to send to each labeller (same file for both):
  data/bd_external/BD_labelling_tool.html   (~25 MB; all images embedded, nothing to unzip)
Open it in Chrome/Edge, click buttons, download the result.

The labeller clicks SD / HR / AP / EX (or keys 1-4) for every image, then presses
"ফলাফল ডাউনলোড" and sends back the file  labels_<name>.csv .
Progress is saved in the browser automatically (they can close and continue later).

Usage: python src/bd_make_label_package.py
"""
import os, sys, json, io, base64
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "bd_external", "bd_labels.csv")
IMG = os.path.join(ROOT, "data", "bd_external", "images")
OUT = os.path.join(ROOT, "data", "bd_external", "BD_labelling_tool.html")

df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
def b64(fname):
    im = Image.open(os.path.join(IMG, fname)).convert("RGB"); im.thumbnail((480, 480))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

items = [{"id": r.image_id, "f": b64(r.file)} for r in df.itertuples()]

HTML = r"""<!doctype html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ছবি লেবেলিং — বাংলাদেশের দুর্যোগ</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1a1a1a;--mute:#666;--line:#ddd;--sd:#c0392b;--hr:#1f6fd1;--ap:#d97a00;--ex:#777}
*{box-sizing:border-box}body{margin:0;font-family:"Noto Sans Bengali","Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
header{background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:18px}.bar{flex:1;min-width:160px;height:10px;background:#eee;border-radius:6px;overflow:hidden}
.bar i{display:block;height:100%;background:#2e9d5b;width:0}
main{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px;max-width:1300px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.img{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px;text-align:center}
.img img{max-width:100%;max-height:70vh;object-fit:contain}
.id{font-size:20px;font-weight:700;margin:6px 0}
.side{display:flex;flex-direction:column;gap:10px}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--line);background:#fff;padding:10px}
.lab{display:flex;flex-direction:column;align-items:flex-start;text-align:left;border-width:2px;padding:10px 12px}
.lab span{font-size:13px;color:var(--mute)}.lab b{font-size:17px}
.lab.sd{border-color:var(--sd)}.lab.hr{border-color:var(--hr)}.lab.ap{border-color:var(--ap)}.lab.ex{border-color:var(--ex)}
.lab.on.sd{background:#fbe9e7}.lab.on.hr{background:#e7f0fc}.lab.on.ap{background:#fdf1e0}.lab.on.ex{background:#eee}
.nav{display:flex;gap:8px}.nav button{flex:1}
.dl{background:#2e9d5b;color:#fff;border:none;font-weight:700;padding:12px}
details{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px}
details li{margin:4px 0}.hint{font-size:13px;color:var(--mute)}
</style></head><body>
<header><b>ছবি লেবেলিং</b><span id="who"></span><div class="bar"><i id="prog"></i></div><span id="cnt"></span></header>
<main>
 <div class="img"><div class="id" id="iid"></div><img id="pic" alt=""><div id="last" class="hint" style="color:#2e9d5b;font-weight:600;min-height:18px"></div><div class="hint">একটা বোতাম বেছে নিন (রঙিন হবে), তারপর "পরের →" চাপুন। মত বদলালে অন্য বোতাম চাপুন। কীবোর্ড: 1 = SD · 2 = HR · 3 = AP · 4 = EX · → বা Enter = পরের · ← = আগের</div></div>
 <div class="side">
  <button class="lab sd" data-l="SD"><b>1 · SD — ক্ষয়ক্ষতি</b><span>ভাঙা/ডুবে যাওয়া ঘরবাড়ি, রাস্তা, সেতু, গাছ পড়া, নষ্ট গাড়ি/নৌকা</span></button>
  <button class="lab hr" data-l="HR"><b>2 · HR — উদ্ধার/ত্রাণ</b><span>উদ্ধারকাজ, ত্রাণ বিতরণ, ত্রাণের প্যাকেট, স্বেচ্ছাসেবক, মেডিকেল ক্যাম্প</span></button>
  <button class="lab ap" data-l="AP"><b>3 · AP — ক্ষতিগ্রস্ত মানুষ</b><span>দুর্গত মানুষই মূল বিষয়: পানিতে মানুষ, আশ্রয়কেন্দ্রে, বাস্তুচ্যুত পরিবার</span></button>
  <button class="lab ex" data-l="EX"><b>4 · EX — বাদ</b><span>মানচিত্র, satellite, চার্ট/লেখা, মিটিং, অস্পষ্ট, বাংলাদেশের নয়, দুর্যোগের সম্পর্কহীন</span></button>
  <div class="nav"><button id="prev">← আগের</button><button id="next">পরের →</button><button id="skip">প্রথম বাকি ছবি</button></div>
  <details open><summary><b>দ্বিধা হলে — এই ক্রমে ভাবুন</b></summary><ol>
   <li>উদ্ধার বা ত্রাণের কাজ চলছে দেখা যাচ্ছে? → <b>HR</b></li>
   <li>না হলে: মানুষই ছবির মূল বিষয়? → <b>AP</b></li>
   <li>না হলে: ক্ষতিগ্রস্ত/ডুবে থাকা ঘরবাড়ি-রাস্তা-পরিবেশ মূল বিষয়? → <b>SD</b></li>
   <li>কোনোটাই না / ছবি নয় (মানচিত্র ইত্যাদি) → <b>EX</b></li></ol>
   <p class="hint">শুধু ছবিতে যা দেখছেন তা দেখে বেছে নিন। অন্য কারও উত্তর দেখবেন না, AI-কে জিজ্ঞেস করবেন না। একটা ছবিতে ৫–১০ সেকেন্ডের বেশি ভাবার দরকার নেই।</p></details>
  <button class="dl" id="dl">⬇ ফলাফল ডাউনলোড (শেষে চাপুন)</button>
  <div class="hint">কাজ ব্রাউজারে নিজে থেকে সেভ হয় — বন্ধ করে পরে আবার এই ফাইল খুললে যেখানে ছিলেন সেখান থেকে শুরু হবে।</div>
 </div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "bd_labels_v2";
let st = JSON.parse(localStorage.getItem(KEY) || "{}");
st.labels = st.labels || {}; st.i = st.i || 0;
while (!st.name) { st.name = (prompt("আপনার নাম লিখুন (ইংরেজিতে, যেমন Rafi):") || "").trim(); }
const save = () => localStorage.setItem(KEY, JSON.stringify(st));
const $ = s => document.querySelector(s);
function show() {
  const it = ITEMS[st.i]; $("#pic").src = it.f; $("#iid").textContent = `${it.id}  (${st.i + 1} / ${ITEMS.length})`;
  document.querySelectorAll(".lab").forEach(b => b.classList.toggle("on", st.labels[it.id] === b.dataset.l));
  const n = Object.keys(st.labels).length;
  $("#cnt").textContent = `${n} / ${ITEMS.length} সম্পন্ন`; $("#prog").style.width = (100 * n / ITEMS.length) + "%";
  $("#who").textContent = "লেবেলার: " + st.name; save();
}
function go(d) { st.i = Math.max(0, Math.min(ITEMS.length - 1, st.i + d)); $("#last").textContent = ""; show(); }
function label(l) {
  const id = ITEMS[st.i].id; st.labels[id] = l;
  $("#last").textContent = `✔ ${id} → ${l} সেভ হয়েছে — পরের ছবিতে যেতে "পরের →" চাপুন`;
  show();
}
document.querySelectorAll(".lab").forEach(b => b.onclick = () => label(b.dataset.l));
$("#prev").onclick = () => go(-1); $("#next").onclick = () => go(1);
$("#skip").onclick = () => { const k = ITEMS.findIndex(x => !st.labels[x.id]); if (k >= 0) { st.i = k; show(); } };
document.addEventListener("keydown", e => {
  const m = {"1": "SD", "2": "HR", "3": "AP", "4": "EX"};
  if (m[e.key]) label(m[e.key]); else if (e.key === "ArrowLeft") go(-1); else if (e.key === "ArrowRight" || e.key === "Enter") go(1);
});
$("#dl").onclick = () => {
  const left = ITEMS.filter(x => !st.labels[x.id]).length;
  if (left && !confirm(`${left}টা ছবি এখনো বাকি। তবুও ডাউনলোড করবেন?`)) return;
  const csv = "image_id,label,labeller\n" + ITEMS.map(x => `${x.id},${st.labels[x.id] || ""},${st.name}`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + csv], {type: "text/csv"}));
  a.download = `labels_${st.name.replace(/\W+/g, "_")}.csv`; a.click();
};
show();
</script></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML.replace("__ITEMS__", json.dumps(items)))
print(f"[+] {len(items)} images embedded -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
