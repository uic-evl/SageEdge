"""
Build a self-contained, offline labeling page for the blinded human validation.

Run on the Thor from the AI_Judge folder (needs val2017/ and Pillow):
    python make_labeling_page.py --pairs human_v2_pairs.csv --images val2017 --out label_tool.html

Then open label_tool.html in a browser (double-click). Each annotator labels ALL
objects independently and clicks "Download CSV" at the end. The page shows only the
image and the object names: no captions, no model names, no judge verdicts (blinded).
Progress is autosaved in the browser, so the page can be closed and reopened.
"""
from __future__ import annotations

import argparse, base64, csv, io, json
from collections import OrderedDict
from pathlib import Path

from PIL import Image

HINTS = {
    "dining table": "any table", "tv": "TV or computer monitor", "sports ball": "any ball",
    "potted plant": "plant in a pot", "cell phone": "any phone", "person": "any human, even partly visible",
    "remote": "remote control / game controller", "couch": "sofa", "hair drier": "hair dryer",
    "skis": "skis (not ski poles)", "car": "car / taxi / SUV (not train cars)",
}

GUIDE = """
<b>Present</b>: at least one such object is visible in the photo (partly visible or small is fine; reflections in mirrors count).<br>
<b>Absent</b>: you cannot find it anywhere in the photo.<br>
<b>Unsure</b>: too small, blurry or ambiguous to decide. Use sparingly.<br>
Objects that only appear <i>inside</i> a picture, poster or TV screen do <b>not</b> count.
Scene words (sky, grass, street, room, wall...) are present if that part of the scene is visible.
Judge only from the image; do not guess what is outside the frame.
"""


def img_b64(path: Path, max_side: int = 900) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="human_v2_pairs.csv")
    ap.add_argument("--images", default="val2017")
    ap.add_argument("--out", default="label_tool.html")
    a = ap.parse_args()

    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in csv.DictReader(open(a.pairs)):
        groups.setdefault(r["image"], []).append(
            {"id": int(r["pair_id"]), "obj": r["object"], "hint": HINTS.get(r["object"], "")})
    data = [{"image": im, "b64": img_b64(Path(a.images) / im), "items": items} for im, items in groups.items()]
    n = sum(len(g["items"]) for g in data)

    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__GUIDE__", GUIDE).replace("__N__", str(n))
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"wrote {a.out}: {len(data)} images, {n} objects, {Path(a.out).stat().st_size/1e6:.1f} MB")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Judge - blinded object labeling</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f6f6f4;color:#111}
 header{background:#001E62;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 header input{padding:4px 6px}
 .wrap{display:flex;gap:18px;padding:16px;flex-wrap:wrap}
 .img{flex:1 1 520px;max-width:900px}.img img{width:100%;border:1px solid #ccc;background:#fff}
 .panel{flex:1 1 340px;max-width:460px}
 .obj{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 10px;margin-bottom:8px}
 .obj.done{border-color:#2a78d6}.name{font-weight:600}.hint{color:#666;font-size:12px}
 .obj label{margin-right:12px;cursor:pointer}
 .nav button,.dl{padding:8px 14px;margin:4px 6px 4px 0;cursor:pointer}
 .guide{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;font-size:13px;margin-bottom:10px}
 .bar{height:6px;background:#ddd}.bar div{height:6px;background:#D50032}
</style></head><body>
<header><b>Blinded object labeling</b>
 <span>Annotator: <input id="who" placeholder="your name" size="12"></span>
 <span id="prog"></span><button class="dl" onclick="dl()">Download CSV</button></header>
<div class="bar"><div id="barf" style="width:0"></div></div>
<div class="wrap"><div class="img"><div id="cap"></div><img id="im"></div>
<div class="panel"><div class="guide">__GUIDE__</div><div id="objs"></div>
<div class="nav"><button onclick="go(-1)">&larr; Previous image</button><button onclick="go(1)">Next image &rarr;</button></div>
<div style="font-size:12px;color:#666">Keys: 1 = present, 2 = absent, 3 = unsure (for the first unlabeled object); &larr;/&rarr; change image.</div></div></div>
<script>
const DATA=__DATA__, N=__N__, BASE='aijudge_labels_v2';
let lab={}, idx=0;
// progress is stored per annotator name, so two people can share one computer
function who(){return (document.getElementById('who').value||'').trim().toLowerCase()}
function key(){return BASE+'__'+who()}
function load(){lab={};try{lab=JSON.parse(localStorage.getItem(key())||'{}')}catch(e){}}
function save(){if(!who()){alert('Please type your name first.');return}
 try{localStorage.setItem(key(),JSON.stringify(lab))}catch(e){}}
try{document.getElementById('who').value=localStorage.getItem(BASE+'_lastwho')||''}catch(e){}
load();
document.getElementById('who').oninput=()=>{try{localStorage.setItem(BASE+'_lastwho',document.getElementById('who').value)}catch(_){}
 load();render()};
function render(){
 const g=DATA[idx]; document.getElementById('im').src='data:image/jpeg;base64,'+g.b64;
 document.getElementById('cap').textContent=`Image ${idx+1} / ${DATA.length}`;
 const box=document.getElementById('objs'); box.innerHTML='';
 g.items.forEach(it=>{const d=document.createElement('div');d.className='obj'+(lab[it.id]?' done':'');
  d.innerHTML=`<div class="name">${it.obj}</div>`+(it.hint?`<div class="hint">(${it.hint})</div>`:'')+
  ['present','absent','unsure'].map(v=>`<label><input type="radio" name="p${it.id}" value="${v}" ${lab[it.id]===v?'checked':''}> ${v}</label>`).join('');
  d.querySelectorAll('input').forEach(r=>r.onchange=()=>{if(!who()){alert('Please type your name first.');r.checked=false;return}lab[it.id]=r.value;save();d.className='obj done';prog()});
  box.appendChild(d)});
 prog()}
function prog(){const k=Object.keys(lab).length;document.getElementById('prog').textContent=`${k} / ${N} labeled`;
 document.getElementById('barf').style.width=(100*k/N)+'%'}
function go(s){idx=Math.max(0,Math.min(DATA.length-1,idx+s));render();window.scrollTo(0,0)}
document.onkeydown=e=>{if(e.target.tagName==='INPUT'&&e.target.type==='text')return;
 if(e.key==='ArrowRight')go(1); if(e.key==='ArrowLeft')go(-1);
 const m={'1':'present','2':'absent','3':'unsure'}[e.key]; if(m){const it=DATA[idx].items.find(i=>!lab[i.id]);
 if(it&&who()){lab[it.id]=m;save();render()}}};
function dl(){const who=(document.getElementById('who').value||'annotator').replace(/\W+/g,'_');
 const miss=N-Object.keys(lab).length; if(miss&&!confirm(miss+' objects are still unlabeled. Download anyway?'))return;
 let s='pair_id,image,object,label,annotator\n';
 DATA.forEach(g=>g.items.forEach(it=>{s+=`${it.id},${g.image},"${it.obj}",${lab[it.id]||''},${who}\n`}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));
 a.download=`labels_${who}.csv`;a.click()}
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
