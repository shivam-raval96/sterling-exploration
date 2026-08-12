from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>English–French token concepts above 5%</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#172033;--muted:#667085;--line:#d7dce5;--en:#008080;--fr:#fa8072}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:10;background:#fffffff2;border-bottom:1px solid var(--line);padding:14px 20px}header>div,main{max-width:1440px;margin:auto}h1{margin:0;font-size:24px}.sub{margin:4px 0 10px;color:var(--muted)}.nav{display:flex;align-items:center;gap:8px}button,select{padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;font:inherit}.counter{margin-left:auto;color:var(--muted)}main{padding:20px}.pair{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px}.content{display:grid;grid-template-columns:minmax(0,1fr) minmax(420px,.9fr);gap:18px}.texts{display:grid;gap:22px}.language{border-top:4px solid;padding-top:8px}.english{border-color:var(--en)}.french{border-color:var(--fr)}h2,h3{margin:0}.raw{color:var(--muted);margin:.35rem 0 .8rem}.tokens{display:flex;flex-wrap:wrap;gap:4px}.token{padding:3px 5px;border:1px solid var(--line);border-radius:5px;background:#fff;font:13px ui-monospace,monospace;cursor:pointer}.token.english{box-shadow:inset 0 -2px var(--en)}.token.french{box-shadow:inset 0 -2px var(--fr)}.token.selected{color:#fff;border-color:#111;box-shadow:none}.token.english.selected{background:var(--en)}.token.french.selected{background:#c45146}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:16px;min-height:340px;max-height:600px;padding:16px;overflow-y:auto;background:#fcfaff;border:1px solid #eadff7;border-radius:10px;font-size:12px}.viewer>header{position:sticky;top:-16px;grid-column:1/-1;margin:-16px -16px 0;padding:12px 16px;background:#f7f2ff;border-bottom:1px solid #eadff7;font:600 13px ui-monospace,monospace}.empty{grid-column:1/-1;align-self:center;text-align:center;color:#746789}.concepts h4{margin:0;color:#57466f}.concepts ol{padding-left:20px}.concepts li{margin-bottom:16px}.concepts p{margin:5px 0;color:#514760}.meta,.score{color:#746789}.activation{margin-top:5px;color:#5f5271;font-size:11px}.bar{width:240px;max-width:100%;height:6px;margin:3px 0 4px;overflow:hidden;background:#e8e1ef;border:1px solid #d8cce5;border-radius:999px}.bar span{display:block;height:100%;background:#8064b3}.flag{display:inline-block;background:#eee6fa;color:#503979;border-radius:10px;padding:1px 6px;margin-right:3px}code{color:#6546a3}@media(max-width:1000px){.content{grid-template-columns:1fr}.viewer{max-height:440px}}@media(max-width:650px){.viewer{grid-template-columns:1fr}.viewer>header{grid-column:1}.counter{margin-left:0}}
</style></head><body><header><div><h1>English–French token concepts</h1><p class="sub">Every native sparse concept above 5% activation · one pair at a time</p><div class="nav"><button id="prev">← Previous</button><select id="pair"></select><button id="next">Next →</button><span class="counter" id="counter"></span></div></div></header><main><article class="pair"><h2 id="heading"></h2><div class="content"><div class="texts" id="texts"></div><section class="viewer" id="viewer"><p class="empty">Hover over or focus a token to inspect its concepts.</p></section></div></article></main><script>
const TOTAL=24,CATALOG_URL='concept_catalog.json',pairSelect=document.querySelector('#pair'),texts=document.querySelector('#texts'),viewer=document.querySelector('#viewer'),heading=document.querySelector('#heading'),counter=document.querySelector('#counter'),prev=document.querySelector('#prev'),next=document.querySelector('#next');let current=0,catalog;
for(let i=0;i<TOTAL;i++)pairSelect.add(new Option(`Pair ${i+1}`,i));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function conceptHTML(c){const m=catalog[`${c.h}:${c.i}`]||{n:'Unlabeled concept',d:'No provider catalog row was found.'},flags=(m.f||[]).map(x=>`<span class="flag">${esc(x)}</span>`).join('');return `<li><div><strong>${esc(m.n)}</strong> <code>${c.h}:${c.i}</code></div><div class="activation">Activation ${c.a.toFixed(3)} (${(c.a*100).toFixed(1)}%)</div><div class="bar" role="meter" aria-valuemin="0" aria-valuemax="1" aria-valuenow="${c.a.toFixed(3)}"><span style="width:${(c.a*100).toFixed(1)}%"></span></div><div class="score">logit ${c.l.toFixed(3)}</div>${m.g?`<div class="meta">${esc(m.g)}</div>`:''}<p>${esc(m.d)}</p>${flags}</li>`}
function choose(token,button){document.querySelectorAll('.token.selected').forEach(x=>x.classList.remove('selected'));button.classList.add('selected');const known=token.k.map(conceptHTML).join(''),unknown=token.u.map(conceptHTML).join('');viewer.innerHTML=`<header>Concept viewer</header><section class="concepts"><h4>Known concepts (${token.k.length})</h4><ol>${known}</ol></section><section class="concepts"><h4>Unknown concepts (${token.u.length})</h4><ol>${unknown}</ol></section>`;viewer.scrollTop=0}
function languageHTML(lang,data){const section=document.createElement('section');section.className=`language ${lang}`;section.innerHTML=`<h3>${lang==='english'?'English':'French'}</h3><p class="raw">${esc(data.text)}</p><div class="tokens"></div>`;const box=section.querySelector('.tokens');data.tokens.forEach(t=>{const b=document.createElement('button');b.className=`token ${lang}`;b.textContent=t.s;b.setAttribute('aria-label',`Token ${t.p}: ${t.s}`);b.addEventListener('mouseenter',()=>choose(t,b));b.addEventListener('focus',()=>choose(t,b));box.appendChild(b)});return section}
async function show(i){current=Math.max(0,Math.min(TOTAL-1,i));const data=await fetch(`pair_data/pair_${String(current).padStart(4,'0')}.json`).then(r=>r.json());heading.textContent=`Pair ${current+1}`;texts.replaceChildren(languageHTML('english',data.english),languageHTML('french',data.french));viewer.innerHTML='<p class="empty">Hover over or focus a token to inspect its concepts.</p>';pairSelect.value=current;counter.textContent=`${current+1} of ${TOTAL}`;prev.disabled=current===0;next.disabled=current===TOTAL-1}
pairSelect.onchange=()=>show(Number(pairSelect.value));prev.onclick=()=>show(current-1);next.onclick=()=>show(current+1);document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')show(current-1);if(e.key==='ArrowRight')show(current+1)});fetch(CATALOG_URL).then(r=>r.json()).then(x=>{catalog=x;show(0)});
</script></body></html>'''


def compact_concept(item: dict[str, Any]) -> dict[str, Any]:
    return {"h": item["head"], "i": item["concept_id"], "l": item["logit"], "a": item["activation"]}


def build(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pair_dir = output / "pair_data"
    pair_dir.mkdir(exist_ok=True)
    catalog: dict[str, Any] = {}
    for index, line in enumerate(source.read_text().splitlines()):
        record = json.loads(line)
        compact = {"pair_id": record["pair_id"]}
        for language in ("english", "french"):
            compact[language] = {"text": record[language]["text"], "tokens": []}
            for token in record[language]["tokens"]:
                entry = {"p": token["position"], "id": token["token_id"], "s": token["token"], "k": [], "u": []}
                for key, short in (("known_concepts", "k"), ("unknown_concepts", "u")):
                    for concept in token[key]:
                        entry[short].append(compact_concept(concept))
                        catalog[f"{concept['head']}:{concept['concept_id']}"] = {
                            "n": concept["concept_name"], "d": concept["concept_description"],
                            "g": concept.get("group_name"),
                            "f": [name.removeprefix("is_") for name in ("is_steerable", "is_tone", "is_alignment", "is_demographic") if concept.get(name)],
                        }
                compact[language]["tokens"].append(entry)
        (pair_dir / f"pair_{index:04d}.json").write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
    (output / "concept_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")))
    (output / "token_concepts.html").write_text(HTML)
    print(json.dumps({"pairs": index + 1, "unique_concepts": len(catalog)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
