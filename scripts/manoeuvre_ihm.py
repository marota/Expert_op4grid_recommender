#!/usr/bin/env python3
"""
scripts/manoeuvre_ihm.py
--------------------------
Petite IHM web (sans dépendance externe — ``http.server`` de la stdlib) pour
tester le module ``manoeuvre`` sur les postes de test.

Fonctionnalités
---------------
1. Choisir un poste parmi ceux disponibles dans le réseau.
2. Visualiser sa **topologie détaillée** (SLD pypowsybl, couleurs natives).
3. Modifier **interactivement** l'état des disjoncteurs / sectionneurs
   (clic sur l'organe dans le schéma, ou via le panneau latéral) pour définir,
   à partir de l'état de **départ**, la topologie détaillée **cible**.
4. Demander la **séquence de manœuvres** (module ``manoeuvre``) pour passer de
   la topologie de départ à la cible.
5. Afficher la séquence **textuellement** et l'**animer** sur le SLD, manœuvre
   par manœuvre, l'organe manipulé étant mis en évidence.

Usage
-----
    python scripts/manoeuvre_ihm.py \
        --grid /chemin/vers/grid.xiidm [--port 8000]

puis ouvrir http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Rendre le package importable quand lancé depuis la racine du dépôt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pypowsybl as pp
import pypowsybl.network as ppn

from expert_op4grid_recommender.manoeuvre.graph import build_vl_graph
from expert_op4grid_recommender.manoeuvre.topologie import (
    PosteTopologique,
    TopologieNodale,
)
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_topo_complete_cible,
)

# Postes de test retenus (intersectés avec les VL réellement présents)
POSTES_TEST = [
    "CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5P3", "BXTO5P6",
    "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6",
    "CORNIP3", "GUARBP6", "MORBRP6",
]

SLD_PAR = None  # défini après import (SldParameters)


class Session:
    """État serveur (mono-utilisateur)."""
    def __init__(self, network):
        self.net = network
        self.vls = set(network.get_voltage_levels().index)
        self.postes = [p for p in POSTES_TEST if p in self.vls]
        self.vl = None
        self.initial: dict[str, bool] = {}   # état de départ (A)
        self.current: dict[str, bool] = {}    # état cible édité (B)

    # --- gestion d'état ---------------------------------------------------
    def switches_df(self, vl):
        df = self.net.get_switches(all_attributes=True)
        return df[df["voltage_level_id"] == vl]

    def load(self, vl):
        self.vl = vl
        df = self.switches_df(vl)
        self.initial = {sid: bool(r["open"]) for sid, r in df.iterrows()}
        self.current = dict(self.initial)

    def reset(self):
        self.current = dict(self.initial)

    def toggle(self, sid):
        if sid in self.current:
            self.current[sid] = not self.current[sid]

    def apply(self, state: dict[str, bool]):
        if state:
            ids = list(state.keys())
            self.net.update_switches(id=ids, open=[state[i] for i in ids])

    # --- rendu ------------------------------------------------------------
    def render(self, state: dict[str, bool]):
        """Retourne (svg_str, switches[list], nb_noeuds) pour un état donné."""
        self.apply(state)
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
        meta = json.loads(svg.metadata)
        switches = []
        for nd in meta.get("nodes", []):
            if nd.get("componentType") in ("BREAKER", "DISCONNECTOR"):
                eq = nd.get("equipmentId")
                if eq is None:
                    continue
                name = eq
                if name.startswith(self.vl + "_"):
                    name = name[len(self.vl) + 1:]
                if name.endswith("_OC"):
                    name = name[:-3]
                switches.append({
                    "id": eq,
                    "name": name,
                    "svgId": nd["id"],
                    "kind": nd["componentType"],
                    "open": bool(state.get(eq, nd.get("open", False))),
                })
        switches.sort(key=lambda s: (s["kind"], s["id"]))
        G = build_vl_graph(self.net, self.vl)
        nb = TopologieNodale.from_graph(G, self.vl).nb_noeuds
        return svg.svg, switches, nb

    def svgid_par_switch(self):
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
        meta = json.loads(svg.metadata)
        return {nd["equipmentId"]: nd["id"] for nd in meta.get("nodes", [])
                if nd.get("componentType") in ("BREAKER", "DISCONNECTOR")
                and nd.get("equipmentId")}

    # --- calcul de séquence ----------------------------------------------
    def sequence(self):
        # Poste à l'état de départ (A)
        self.apply(self.initial)
        poste = PosteTopologique.from_graph(build_vl_graph(self.net, self.vl), self.vl)
        # Topologie nodale cible (B)
        self.apply(self.current)
        topo_cible = TopologieNodale.from_graph(build_vl_graph(self.net, self.vl), self.vl)

        res = determiner_topo_complete_cible(poste, topo_cible)

        svgid = self.svgid_par_switch()

        # Animation : état de départ puis application successive
        steps = []
        running = dict(self.initial)
        svg0, _, nb0 = self.render(running)
        steps.append({"svg": _highlight(svg0, None),
                      "label": f"État de départ — {nb0} nœud(s)",
                      "highlight": None})
        for i, m in enumerate(res.manoeuvres, 1):
            running[m.switch_id] = (m.action == "OPEN")
            svg, _, nb = self.render(running)
            hid = svgid.get(m.switch_id)
            steps.append({
                "svg": _highlight(svg, hid),
                "label": f"{i}. {m.action} {m.switch_id} — {m.raison}",
                "highlight": hid,
            })
        # Restaurer l'affichage sur la cible éditée
        self.apply(self.current)

        return {
            "verified": res.is_verified,
            "message": res.message,
            "nb_manoeuvres": res.nb_manoeuvres,
            "manoeuvres": [{
                "switch_id": m.switch_id, "action": m.action,
                "raison": m.raison, "boucle": m.type_boucle,
            } for m in res.manoeuvres],
            "steps": steps,
        }


def _highlight(svg: str, svg_id: str | None) -> str:
    """Injecte une mise en évidence de l'organe manœuvré dans le SVG."""
    if not svg_id:
        return svg
    rule = (f"#{svg_id} *{{stroke:#e60000 !important;stroke-width:4 !important}}"
            f"#{svg_id}{{stroke:#e60000 !important}}")
    return svg.replace("</style>", rule + "\n</style>", 1)


# ---------------------------------------------------------------------------
# Serveur HTTP
# ---------------------------------------------------------------------------

SESSION: Session = None  # type: ignore


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def log_message(self, *a):  # silencieux
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._html(PAGE)
        if self.path == "/api/postes":
            return self._json({"postes": SESSION.postes})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self._read_json()
            if self.path == "/api/load":
                SESSION.load(data["vl"])
                svg, sw, nb = SESSION.render(SESSION.current)
                return self._json({"svg": svg, "switches": sw, "nb_noeuds": nb})
            if self.path == "/api/toggle":
                SESSION.toggle(data["id"])
                svg, sw, nb = SESSION.render(SESSION.current)
                return self._json({"svg": svg, "switches": sw, "nb_noeuds": nb})
            if self.path == "/api/reset":
                SESSION.reset()
                svg, sw, nb = SESSION.render(SESSION.current)
                return self._json({"svg": svg, "switches": sw, "nb_noeuds": nb})
            if self.path == "/api/sequence":
                return self._json(SESSION.sequence())
            return self._json({"error": "not found"}, 404)
        except Exception as exc:  # pragma: no cover
            import traceback
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)


PAGE = r"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>IHM Manœuvre</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #side{width:330px;background:#f4f5f7;padding:12px;overflow:auto;border-right:1px solid #ccc}
 #main{flex:1;display:flex;flex-direction:column;overflow:hidden}
 #diagram{flex:1;overflow:auto;padding:8px;background:#fff}
 #diagram svg{max-width:100%;height:auto}
 h2{font-size:15px;margin:10px 0 6px}
 button{cursor:pointer;padding:6px 10px;margin:2px 0;border:1px solid #888;border-radius:5px;background:#fff}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 select{width:100%;padding:6px}
 .sw{display:flex;justify-content:space-between;align-items:center;padding:2px 4px;font-size:12px;border-bottom:1px solid #e3e3e3}
 .sw .name{font-family:monospace;font-size:11px}
 .pill{font-size:10px;padding:1px 6px;border-radius:8px;color:#fff}
 .open{background:#c0392b}.closed{background:#27ae60}
 .kind{color:#777;font-size:10px;width:18px}
 #seq{font-family:monospace;font-size:12px;white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:8px;max-height:30vh;overflow:auto}
 #anim{display:flex;align-items:center;gap:8px;padding:6px;background:#eef;border-top:1px solid #ccd}
 .step-cur{background:#fde68a}
 .badge{font-size:11px;padding:2px 6px;border-radius:6px;background:#ddd}
 .ok{background:#27ae60;color:#fff}.ko{background:#c0392b;color:#fff}
</style></head><body>
<div id="side">
  <h2>Poste</h2>
  <select id="poste"></select>
  <div style="margin-top:6px">
    <button onclick="reset()">↺ État de départ</button>
    <button class="primary" onclick="sequence()">⚙ Calculer la séquence</button>
  </div>
  <h2>Nœuds électriques : <span id="nbn" class="badge">–</span></h2>
  <div style="font-size:11px;color:#555">Clic sur un organe du schéma ou ci-dessous pour basculer son état (départ ➜ cible).</div>
  <h2>Disjoncteurs (DJ)</h2><div id="djs"></div>
  <h2>Sectionneurs (SA)</h2><div id="sas"></div>
</div>
<div id="main">
  <div id="diagram">Choisissez un poste…</div>
  <div id="anim" style="display:none">
    <button onclick="step(-1)">◀</button>
    <button onclick="play()">▶ Lecture</button>
    <button onclick="step(1)">▶|</button>
    <span id="stepinfo" class="badge"></span>
    <span id="steplabel" style="font-family:monospace;font-size:12px"></span>
  </div>
  <div id="seqwrap" style="padding:8px;display:none">
    <b>Séquence <span id="seqstatus"></span></b>
    <div id="seq"></div>
  </div>
</div>
<script>
let STATE={steps:[],idx:0,timer:null,switches:[]};
async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json();}
async function init(){const r=await (await fetch('/api/postes')).json();const sel=document.getElementById('poste');
  r.postes.forEach(p=>{const o=document.createElement('option');o.value=p;o.text=p;sel.add(o);});
  sel.onchange=()=>load(sel.value); if(r.postes.length){load(r.postes[0]);}}
async function load(vl){stopAnim();const d=await api('/api/load',{vl});showDiagram(d);hideSeq();}
async function reset(){stopAnim();const d=await api('/api/reset',{});showDiagram(d);hideSeq();}
async function toggle(id){stopAnim();const d=await api('/api/toggle',{id});showDiagram(d);hideSeq();}
function hideSeq(){document.getElementById('seqwrap').style.display='none';document.getElementById('anim').style.display='none';}
function showDiagram(d){
  document.getElementById('diagram').innerHTML=d.svg;
  document.getElementById('nbn').textContent=d.nb_noeuds;
  STATE.switches=d.switches;
  bindSwitches(d.switches);
  renderSwitchPanel(d.switches);
}
function bindSwitches(switches){
  switches.forEach(s=>{const el=document.getElementById(s.svgId);
    if(el){el.style.cursor='pointer';el.onclick=()=>toggle(s.id);
      el.querySelectorAll('*').forEach(c=>c.style.cursor='pointer');}});
}
function renderSwitchPanel(switches){
  const djs=document.getElementById('djs'),sas=document.getElementById('sas');djs.innerHTML='';sas.innerHTML='';
  switches.forEach(s=>{const row=document.createElement('div');row.className='sw';
    row.innerHTML=`<span class="name">${(s.name||s.id).slice(0,30)}</span>`+
      `<span><span class="pill ${s.open?'open':'closed'}">${s.open?'OUVERT':'FERMÉ'}</span></span>`;
    row.title=s.id; row.onclick=()=>toggle(s.id);row.style.cursor='pointer';
    (s.kind==='BREAKER'?djs:sas).appendChild(row);});
}
async function sequence(){stopAnim();const d=await api('/api/sequence',{});
  const sw=document.getElementById('seqwrap');sw.style.display='block';
  const st=document.getElementById('seqstatus');
  st.innerHTML=d.verified?'<span class="badge ok">VÉRIFIÉE</span>':'<span class="badge ko">NON VÉRIFIÉE</span>';
  let txt=d.message+'\n\n';
  d.manoeuvres.forEach((m,i)=>{txt+=`${String(i+1).padStart(2)}. ${m.action.padEnd(5)} ${m.switch_id}  (${m.raison})${m.boucle?' ['+m.boucle+']':''}\n`;});
  if(!d.manoeuvres.length) txt+='(aucune manœuvre)';
  document.getElementById('seq').textContent=txt;
  STATE.steps=d.steps;STATE.idx=0;
  if(d.steps.length){document.getElementById('anim').style.display='flex';showStep(0);}
}
function showStep(i){STATE.idx=Math.max(0,Math.min(i,STATE.steps.length-1));
  const s=STATE.steps[STATE.idx];
  document.getElementById('diagram').innerHTML=s.svg;
  document.getElementById('stepinfo').textContent=`${STATE.idx}/${STATE.steps.length-1}`;
  document.getElementById('steplabel').textContent=s.label;
}
function step(d){stopAnim();showStep(STATE.idx+d);}
function play(){stopAnim();STATE.timer=setInterval(()=>{if(STATE.idx>=STATE.steps.length-1){stopAnim();return;}showStep(STATE.idx+1);},900);}
function stopAnim(){if(STATE.timer){clearInterval(STATE.timer);STATE.timer=null;}}
init();
</script>
</body></html>"""


def main():
    global SESSION, SLD_PAR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, help="Chemin du réseau .xiidm")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    SLD_PAR = ppn.SldParameters(topological_coloring=True)
    print(f"Chargement du réseau {args.grid} …")
    net = pp.network.load(args.grid)
    SESSION = Session(net)
    print(f"Postes de test disponibles : {SESSION.postes}")

    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"IHM Manœuvre : http://localhost:{args.port}  (Ctrl-C pour arrêter)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")


if __name__ == "__main__":
    main()
