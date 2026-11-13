"""
BD external test set — step 1: download candidate images from Wikimedia Commons
==============================================================================
Collects CC-licensed photos of Bangladeshi floods and cyclones, with their source
URL, author, licence and date, and builds a review page for labelling.

Output
  data/bd_external/images/<id>.jpg      images (git-ignored; never redistribute)
  bd_external/bd_labels.csv             metadata + empty label columns (tracked)
  bd_external/review.html               open in a browser: every image with its id

Re-running keeps existing rows and labels and only adds new files.
Usage: python src/bd_fetch_wikimedia.py            (~5-10 min, internet needed)
"""
import os, re, sys, time, json, html
import urllib.parse, urllib.request
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "bd_external", "images")
OUT_DIR = os.path.join(ROOT, "bd_external")
CSV     = os.path.join(OUT_DIR, "bd_labels.csv")
os.makedirs(IMG_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["Floods in Bangladesh", "Cyclone Sidr", "Cyclone Amphan", "Cyclone Aila",
              "Cyclone Remal", "2017 South Asian floods"]
SUBCAT_DEPTH = 1          # also read direct sub-categories
THUMB_WIDTH  = 800        # pixels; the model resizes to 224 anyway
API = "https://commons.wikimedia.org/w/api.php"
UA  = {"User-Agent": "thesis-dataset-script/1.0 (academic, non-commercial)"}
SKIP_EXT = (".svg", ".pdf", ".gif", ".tif", ".tiff", ".webm", ".ogv", ".ogg", ".djvu")
COLS = ["image_id", "file", "title", "source_url", "author", "license", "date", "category",
        "label_a", "label_b", "final_label", "description_bn", "notes"]


def fetch(url, timeout=60):
    """GET with polite pacing; on HTTP 429/5xx waits (Retry-After if given) and retries."""
    for attempt in range(6):
        time.sleep(1.0)                                     # ~1 request per second
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5: raise
            wait = int(e.headers.get("Retry-After", 0) or 0) or 10 * (attempt + 1)
            print(f"  [..] HTTP {e.code}, waiting {wait}s"); time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 5: raise
            print(f"  [..] network error, retrying"); time.sleep(5 * (attempt + 1))


def api(**params):
    params.update(format="json")
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(params), timeout=30))


def members(cat, ns):
    out, cont = [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}",
                cmtype="file" if ns == 6 else "subcat", cmlimit=500, **cont)
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        if "continue" not in d: return out
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


# ---- 1. collect file titles ------------------------------------------------
titles = {}
for cat in CATEGORIES:
    cats = [cat]
    if SUBCAT_DEPTH:
        cats += [t.split(":", 1)[1] for t in members(cat, 14)]
    for c in cats:
        for t in members(c, 6):
            if not t.lower().endswith(SKIP_EXT):
                titles.setdefault(t, cat)
    print(f"[+] {cat}: {sum(v == cat for v in titles.values())} candidate files")

# ---- 2. metadata + download -------------------------------------------------
old = pd.read_csv(CSV, dtype=str, keep_default_na=False) if os.path.exists(CSV) else pd.DataFrame(columns=COLS)
known = set(old.title)
rows, next_id = [], len(old) + 1
todo = [t for t in titles if t not in known]
print(f"[+] {len(todo)} new files to fetch ({len(known)} already in {os.path.basename(CSV)})")

for i in range(0, len(todo), 15):
    batch = todo[i:i + 15]
    d = api(action="query", titles="|".join(batch), prop="imageinfo",
            iiprop="url|extmetadata|mime", iiurlwidth=THUMB_WIDTH)
    for page in d.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        if not info.get("mime", "").startswith("image/"):
            continue
        meta = info.get("extmetadata", {})
        get = lambda k: strip_html(meta.get(k, {}).get("value", ""))
        lic = get("LicenseShortName")
        if not lic or "fair use" in lic.lower():
            continue                                        # keep only clearly licensed files
        image_id = f"BD{next_id:03d}"; fname = f"{image_id}.jpg"
        try:
            data = fetch(info.get("thumburl") or info["url"])
            with open(os.path.join(IMG_DIR, fname), "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"  [!] download failed: {page['title']} ({e})"); continue
        rows.append(dict(image_id=image_id, file=fname, title=page["title"],
                         source_url=info.get("descriptionurl", ""), author=get("Artist")[:120],
                         license=lic, date=(get("DateTimeOriginal") or get("DateTime"))[:10],
                         category=titles[page["title"]], label_a="", label_b="", final_label="",
                         description_bn="", notes=""))
        next_id += 1
    # save after every batch, so an interrupted run resumes where it stopped
    df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"  fetched {min(i + 15, len(todo))}/{len(todo)}")

df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
df.to_csv(CSV, index=False, encoding="utf-8-sig")

# ---- 3. review page ----------------------------------------------------------
rel = os.path.relpath(IMG_DIR, OUT_DIR).replace("\\", "/")
cards = "\n".join(
    f'<div class="c"><img src="{rel}/{r.file}" loading="lazy"><b>{r.image_id}</b>'
    f'<span>{html.escape(r.category)} · {html.escape(r.license)}</span>'
    f'<a href="{html.escape(r.source_url)}" target="_blank">source</a></div>'
    for r in df.itertuples())
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>BD external — review</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.c{{background:#fff;border:1px solid #ddd;padding:6px;display:flex;flex-direction:column;gap:4px}}
.c img{{width:100%;height:220px;object-fit:contain;background:#eee}} span{{font-size:12px;color:#555}}</style>
</head><body><h2>BD external test set — {len(df)} candidates</h2>
<p>Label in <code>bd_labels.csv</code> using the guideline (docs/bd_external_labelling_guide.md).
Label A and label B must be filled independently.</p><div class="g">{cards}</div></body></html>"""
open(os.path.join(OUT_DIR, "review.html"), "w", encoding="utf-8").write(page)
print(f"\n[+] {len(df)} candidates -> {CSV}\n[+] Open {os.path.join(OUT_DIR, 'review.html')} to see them")
