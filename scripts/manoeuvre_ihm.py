#!/usr/bin/env python3
"""
scripts/manoeuvre_ihm.py
--------------------------
Petite IHM web (Flask) pour tester le module ``manoeuvre`` sur les postes de
test.

Installation (optionnelle) :
    pip install -e ".[ihm]"      # guillemets requis sous zsh ; ou : pip install flask

Fonctionnalités
---------------
1. Choisir un poste parmi ceux disponibles dans le réseau.
2. Visualiser sa **topologie détaillée** (SLD pypowsybl, couleurs natives
   rendues par le navigateur).
3. Modifier **interactivement** l'état des disjoncteurs / sectionneurs
   (clic sur l'organe dans le schéma, ou via le panneau latéral) pour définir,
   à partir de l'état de **départ**, la topologie détaillée **cible**.
4. **Valider & sauvegarder** la cible dans un fichier JSON réutilisable en test
   (départ + cible détaillés + partitions nodales). La validation est requise
   avant de pouvoir calculer la séquence.
5. Demander la **séquence de manœuvres** (module ``manoeuvre``) pour passer de
   la topologie de départ à la cible.
6. Afficher la séquence **textuellement** et l'**animer** sur le schéma cible,
   manœuvre par manœuvre, l'organe manipulé étant mis en évidence.
7. Recharger un **scénario sauvegardé** : « Rejouer » (départ + cible
   sauvegardés) ou « Comme départ » (la cible sauvegardée devient le nouvel
   état de départ, permettant de chaîner les scénarios depuis une topologie
   validée plutôt que depuis l'état de base du réseau).

Les scénarios sont écrits dans ``--scenarios-dir`` (défaut
``tests/manoeuvre/scenarios``) au format :
``{voltage_level_id, name, depart{sw:open}, cible{sw:open},
   depart_nodale[[..]], cible_nodale[[..]]}``.

Usage
-----
    python scripts/manoeuvre_ihm.py --grid /chemin/vers/grid.xiidm \
        [--port 8000] [--scenarios-dir tests/manoeuvre/scenarios]
    # puis ouvrir http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Rendre le package importable quand lancé depuis la racine du dépôt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from flask import Flask, jsonify, request, Response
except ImportError:  # pragma: no cover
    sys.exit("Flask est requis pour l'IHM : pip install -e .[ihm]  (ou pip install flask)")

import pypowsybl as pp
import pypowsybl.network as ppn

from expert_op4grid_recommender.manoeuvre.graph import build_vl_graph
from expert_op4grid_recommender.manoeuvre.topologie import (
    PosteTopologique,
    TopologieNodale,
)
from expert_op4grid_recommender.manoeuvre.algo import determiner_topo_complete_cible

# Postes de test retenus (intersectés avec les VL réellement présents)
POSTES_TEST = [
    "CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5P3", "BXTO5P6",
    "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6",
    "CORNIP3", "GUARBP6", "MORBRP6",
]

SLD_PAR = ppn.SldParameters(topological_coloring=True)
SCEN_DIR = pathlib.Path("tests/manoeuvre/scenarios")  # redéfini dans main()


class Session:
    """État serveur (mono-utilisateur)."""

    def __init__(self, network):
        self.net = network
        self.vls = set(network.get_voltage_levels().index)
        self.postes = [p for p in POSTES_TEST if p in self.vls]
        # État pristine des organes (référence stable pour « état de départ »,
        # indépendant des modifications appliquées en cours de session).
        df = network.get_switches(all_attributes=True)
        self.pristine = {sid: bool(r["open"]) for sid, r in df.iterrows()}
        self.vl = None
        self.initial: dict[str, bool] = {}   # état de départ (A)
        self.current: dict[str, bool] = {}    # état cible édité (B)
        # Cache de la dernière séquence calculée (pour l'animation lazy)
        self.seq_states: list[dict[str, bool]] = []
        self.seq_highlights: list[str | None] = []

    # --- gestion d'état ---------------------------------------------------
    def switches_df(self, vl):
        df = self.net.get_switches(all_attributes=True)
        return df[df["voltage_level_id"] == vl]

    def load(self, vl):
        self.vl = vl
        df = self.switches_df(vl)
        # Départ = état pristine du poste (et non un état résiduel de session)
        self.initial = {sid: self.pristine[sid] for sid in df.index}
        self.current = dict(self.initial)
        self.seq_states, self.seq_highlights = [], []

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
    def _switches_meta(self, meta, state):
        switches = []
        for nd in meta.get("nodes", []):
            if nd.get("componentType") not in ("BREAKER", "DISCONNECTOR"):
                continue
            eq = nd.get("equipmentId")
            if eq is None:
                continue
            name = eq
            if name.startswith(self.vl + "_"):
                name = name[len(self.vl) + 1:]
            if name.endswith("_OC"):
                name = name[:-3]
            switches.append({
                "id": eq, "name": name, "svgId": nd["id"],
                "kind": nd["componentType"],
                "open": bool(state.get(eq, nd.get("open", False))),
            })
        switches.sort(key=lambda s: (s["kind"], s["id"]))
        return switches

    def view(self, state: dict[str, bool]):
        """(svg, switches, nb_noeuds) pour un état détaillé donné."""
        self.apply(state)
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
        meta = json.loads(svg.metadata)
        switches = self._switches_meta(meta, state)
        nb = TopologieNodale.from_graph(build_vl_graph(self.net, self.vl), self.vl).nb_noeuds
        return svg.svg, switches, nb

    def svgid_par_switch(self):
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
        meta = json.loads(svg.metadata)
        return {nd["equipmentId"]: nd["id"] for nd in meta.get("nodes", [])
                if nd.get("componentType") in ("BREAKER", "DISCONNECTOR")
                and nd.get("equipmentId")}

    def render_step_svg(self, i: int) -> str:
        """SVG de l'étape d'animation i (avec mise en évidence)."""
        if not self.seq_states:
            return ""
        i = max(0, min(i, len(self.seq_states) - 1))
        self.apply(self.seq_states[i])
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR).svg
        return _highlight(svg, self.seq_highlights[i])

    # --- scénarios (sauvegarde / rechargement) ----------------------------
    def groups_of(self, state):
        """Partition nodale (liste de groupes de départs) pour un état donné."""
        self.apply(state)
        topo = TopologieNodale.from_graph(build_vl_graph(self.net, self.vl), self.vl)
        return [sorted(n.equipment_ids) for n in topo.noeuds.values()]

    def save_scenario(self, name: str) -> str:
        import re
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or self.vl
        data = {
            "voltage_level_id": self.vl,
            "name": name,
            "depart": self.initial,
            "cible": self.current,
            "depart_nodale": self.groups_of(self.initial),
            "cible_nodale": self.groups_of(self.current),
        }
        self.apply(self.current)  # restaurer l'affichage courant
        SCEN_DIR.mkdir(parents=True, exist_ok=True)
        path = SCEN_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return str(path)

    def list_scenarios(self):
        if not SCEN_DIR.exists():
            return []
        return sorted(p.stem for p in SCEN_DIR.glob("*.json"))

    def load_scenario(self, name: str, mode: str = "both") -> str:
        """
        Recharge un scénario sauvegardé.

        - ``mode="both"`` (rejouer) : départ = ``depart`` sauvegardé,
          cible = ``cible`` sauvegardée.
        - ``mode="as_depart"`` : la topologie cible sauvegardée devient le
          **nouvel état de départ** (et la cible éditable en repart) ; permet de
          chaîner les scénarios (partir d'une topologie validée, non pristine).
        """
        data = json.loads((SCEN_DIR / f"{name}.json").read_text())
        self.load(data["voltage_level_id"])
        base = {k: bool(self.initial[k]) for k in self.initial}
        if mode == "as_depart":
            self.initial = {k: bool(data["cible"].get(k, base[k])) for k in base}
            self.current = dict(self.initial)
        else:
            self.initial = {k: bool(data["depart"].get(k, base[k])) for k in base}
            self.current = {k: bool(data["cible"].get(k, base[k])) for k in base}
        return data["voltage_level_id"]

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

        # Pré-calcul des états successifs (pour l'animation lazy via /api/step)
        states = [dict(self.initial)]
        highlights: list[str | None] = [None]
        labels = ["État de départ"]
        running = dict(self.initial)
        for i, m in enumerate(res.manoeuvres, 1):
            running = dict(running)
            running[m.switch_id] = (m.action == "OPEN")
            states.append(running)
            highlights.append(svgid.get(m.switch_id))
            labels.append(f"{i}. {m.action} {m.switch_id} — {m.raison}")
        self.seq_states, self.seq_highlights = states, highlights

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
            "n_steps": len(states),
            "labels": labels,
        }


def _highlight(svg: str, svg_id: str | None) -> str:
    if not svg_id:
        return svg
    rule = (f"#{svg_id} *{{stroke:#e60000 !important;stroke-width:4 !important}}"
            f"#{svg_id}{{stroke:#e60000 !important}}")
    return svg.replace("</style>", rule + "\n</style>", 1)


def _prefix_svg_ids(svg: str, pfx: str) -> str:
    """
    Préfixe tous les ids (et leurs références internes) d'un SVG, afin que deux
    SVG du même poste puissent coexister dans le DOM sans collision d'ids.
    Appliqué au schéma « départ » (non interactif) ; le schéma « cible » garde
    ses ids d'origine (cohérents avec le mapping switch → svgId).
    """
    import re
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{pfx}{m.group(1)}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{pfx}{m.group(1)})', svg)
    svg = re.sub(r'(xlink:href|href)="#([^"]+)"',
                 lambda m: f'{m.group(1)}="#{pfx}{m.group(2)}"', svg)
    return svg


# ---------------------------------------------------------------------------
# Application Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)
SESSION: Session = None  # type: ignore


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/postes")
def api_postes():
    return jsonify(postes=SESSION.postes)


@app.post("/api/load")
def api_load():
    SESSION.load(request.json["vl"])
    svg_i, _, nb_i = SESSION.view(SESSION.initial)
    svg_c, sw, nb_c = SESSION.view(SESSION.current)
    return jsonify(initial_svg=_prefix_svg_ids(svg_i, "A_"), nb_initial=nb_i,
                   svg=svg_c, switches=sw, nb_noeuds=nb_c)


@app.post("/api/toggle")
def api_toggle():
    SESSION.toggle(request.json["id"])
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb)


@app.post("/api/reset")
def api_reset():
    SESSION.reset()
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb)


@app.get("/api/scenarios")
def api_scenarios():
    return jsonify(scenarios=SESSION.list_scenarios())


@app.post("/api/save")
def api_save():
    path = SESSION.save_scenario(request.json.get("name", ""))
    return jsonify(path=path, scenarios=SESSION.list_scenarios())


@app.post("/api/load_scenario")
def api_load_scenario():
    SESSION.load_scenario(request.json["name"],
                          request.json.get("mode", "both"))
    svg_i, _, nb_i = SESSION.view(SESSION.initial)
    svg_c, sw, nb_c = SESSION.view(SESSION.current)
    return jsonify(initial_svg=_prefix_svg_ids(svg_i, "A_"), nb_initial=nb_i,
                   svg=svg_c, switches=sw, nb_noeuds=nb_c, vl=SESSION.vl)


@app.post("/api/sequence")
def api_sequence():
    return jsonify(SESSION.sequence())


@app.get("/api/step")
def api_step():
    i = int(request.args.get("i", 0))
    return jsonify(svg=SESSION.render_step_svg(i))


PAGE = r"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>IHM Manœuvre</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #side{width:340px;background:#f4f5f7;padding:12px;overflow:auto;border-right:1px solid #ccc}
 #main{flex:1;display:flex;flex-direction:column;overflow:hidden}
 .pane{flex:1;display:flex;flex-direction:column;min-height:0}
 .pane .ttl{font-size:12px;font-weight:bold;padding:4px 10px;border-bottom:1px solid #ccc}
 .pane .diag{flex:1;overflow:auto;background:#fff}
 .pane .diag svg{max-width:100%;height:auto}
 #paneTop .ttl{background:#eef2ff}
 #paneBot .ttl{background:#fff7ed}
 #paneTop{border-bottom:3px solid #c7d2fe}
 h2{font-size:15px;margin:10px 0 6px}
 button{cursor:pointer;padding:6px 10px;margin:2px 0;border:1px solid #888;border-radius:5px;background:#fff}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 button:disabled{opacity:.5;cursor:default}
 select{width:100%;padding:6px}
 .sw{display:flex;justify-content:space-between;align-items:center;padding:2px 4px;font-size:12px;border-bottom:1px solid #e3e3e3}
 .sw:hover{background:#e8eefc}
 .sw .name{font-family:monospace;font-size:11px}
 .pill{font-size:10px;padding:1px 6px;border-radius:8px;color:#fff}
 .open{background:#c0392b}.closed{background:#27ae60}
 #seq{font-family:monospace;font-size:12px;white-space:pre;background:#fff;border:1px solid #ddd;padding:8px;max-height:30vh;overflow:auto}
 #seq .line{padding:0 2px}
 #seq .cur{background:#fde68a}
 #anim{display:flex;align-items:center;gap:8px;padding:6px;background:#eef;border-top:1px solid #ccd}
 .badge{font-size:11px;padding:2px 6px;border-radius:6px;background:#ddd}
 .ok{background:#27ae60;color:#fff}.ko{background:#c0392b;color:#fff}
</style></head><body>
<div id="side">
  <h2>Poste</h2>
  <select id="poste"></select>
  <div style="margin-top:6px"><button onclick="reset()">↺ État de départ</button></div>

  <h2>1 · Valider la cible</h2>
  <input id="scenName" placeholder="nom du scénario" style="width:62%;padding:5px">
  <button onclick="save()">✓ Valider &amp; sauvegarder</button>
  <div id="savemsg" style="font-size:11px;color:#1a7f37;margin-top:3px"></div>

  <h2>2 · Séquence de manœuvres</h2>
  <button id="bcalc" class="primary" onclick="sequence()" disabled>⚙ Calculer la séquence</button>
  <div id="calchint" style="font-size:11px;color:#b45309">Validez d'abord la cible pour activer le calcul.</div>

  <h2>Scénarios sauvegardés</h2>
  <select id="scenSel" style="width:100%"><option value="">—</option></select>
  <div style="display:flex;gap:4px;margin-top:4px">
    <button onclick="loadScen('both')" title="départ + cible sauvegardés">▷ Rejouer</button>
    <button onclick="loadScen('as_depart')" title="la cible sauvegardée devient le nouvel état de départ">⇧ Comme départ</button>
  </div>
  <h2>Nœuds électriques : <span id="nbn" class="badge">–</span></h2>
  <div style="font-size:11px;color:#555">Clic sur un organe du schéma ou dans la liste pour basculer son état (départ ➜ cible).</div>
  <h2>Disjoncteurs (DJ)</h2><div id="djs"></div>
  <h2>Sectionneurs (SA)</h2><div id="sas"></div>
</div>
<div id="main">
  <div class="pane" id="paneTop">
    <div class="ttl">Topologie de départ — <span id="nbA" class="badge">–</span> nœud(s)</div>
    <div class="diag" id="diagTop">Choisissez un poste…</div>
  </div>
  <div class="pane" id="paneBot">
    <div class="ttl">Topologie cible (éditable, clic sur un organe) — <span id="nbB" class="badge">–</span> nœud(s)
        · animation de la séquence ici</div>
    <div class="diag" id="diagBottom"></div>
  </div>
  <div id="anim" style="display:none">
    <button id="bprev" onclick="step(-1)">◀</button>
    <button id="bplay" onclick="play()">▶ Lecture</button>
    <button id="bnext" onclick="step(1)">▶|</button>
    <span id="stepinfo" class="badge"></span>
    <span id="steplabel" style="font-family:monospace;font-size:12px"></span>
  </div>
  <div id="seqwrap" style="padding:8px;display:none">
    <b>Séquence <span id="seqstatus"></span></b>
    <div id="seq"></div>
  </div>
</div>
<script>
let S={n:0,idx:0,timer:null,labels:[]};
const api=(p,b)=>fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json());
function setValidated(v){document.getElementById('bcalc').disabled=!v;
  document.getElementById('calchint').style.display=v?'none':'block';}
async function init(){const r=await (await fetch('/api/postes')).json();const sel=document.getElementById('poste');
  r.postes.forEach(p=>{const o=document.createElement('option');o.value=p;o.text=p;sel.add(o);});
  sel.onchange=()=>load(sel.value); await refreshScenarios(); if(r.postes.length) load(r.postes[0]);}
async function load(vl){stopAnim();show(await api('/api/load',{vl}));hideSeq();setValidated(false);
  document.getElementById('scenName').value=vl+'_cible';}
async function reset(){stopAnim();show(await api('/api/reset',{}));hideSeq();setValidated(false);}
async function toggle(id){stopAnim();show(await api('/api/toggle',{id}));hideSeq();setValidated(false);}
async function refreshScenarios(){const r=await (await fetch('/api/scenarios')).json();
  const sel=document.getElementById('scenSel');sel.innerHTML='<option value="">—</option>';
  r.scenarios.forEach(s=>{const o=document.createElement('option');o.value=s;o.text=s;sel.add(o);});}
async function save(){const name=document.getElementById('scenName').value;
  const r=await api('/api/save',{name});
  document.getElementById('savemsg').textContent='✓ Sauvegardé : '+r.path;
  setValidated(true);await refreshScenarios();}
async function loadScen(mode){const name=document.getElementById('scenSel').value;if(!name)return;
  stopAnim();const d=await api('/api/load_scenario',{name,mode});
  document.getElementById('poste').value=d.vl;show(d);hideSeq();
  if(mode==='as_depart'){
    document.getElementById('scenName').value=name+'_suite';
    document.getElementById('savemsg').textContent='« '+name+' » chargé comme état de départ — éditez puis validez une nouvelle cible.';
    setValidated(false);
  }else{
    document.getElementById('scenName').value=name;
    document.getElementById('savemsg').textContent='Scénario « '+name+' » rechargé (départ + cible).';
    setValidated(true);
  }}
function hideSeq(){document.getElementById('seqwrap').style.display='none';document.getElementById('anim').style.display='none';}
function show(d){
  if(d.initial_svg!==undefined){document.getElementById('diagTop').innerHTML=d.initial_svg;
    document.getElementById('nbA').textContent=d.nb_initial;}
  document.getElementById('diagBottom').innerHTML=d.svg;
  document.getElementById('nbB').textContent=d.nb_noeuds;
  document.getElementById('nbn').textContent=d.nb_noeuds;
  bind(d.switches);panel(d.switches);}
function bind(switches){const root=document.getElementById('diagBottom');
  switches.forEach(s=>{const el=root.querySelector('[id="'+s.svgId+'"]');
    if(el){el.style.cursor='pointer';el.onclick=()=>toggle(s.id);
      el.querySelectorAll('*').forEach(c=>c.style.cursor='pointer');}});}
function panel(switches){const dj=document.getElementById('djs'),sa=document.getElementById('sas');dj.innerHTML='';sa.innerHTML='';
  switches.forEach(s=>{const row=document.createElement('div');row.className='sw';row.title=s.id;row.style.cursor='pointer';
    row.innerHTML=`<span class="name">${(s.name||s.id).slice(0,30)}</span><span class="pill ${s.open?'open':'closed'}">${s.open?'OUVERT':'FERMÉ'}</span>`;
    row.onclick=()=>toggle(s.id);(s.kind==='BREAKER'?dj:sa).appendChild(row);});}
async function sequence(){stopAnim();const d=await api('/api/sequence',{});
  const w=document.getElementById('seqwrap');w.style.display='block';
  document.getElementById('seqstatus').innerHTML=d.verified?'<span class="badge ok">VÉRIFIÉE</span>':'<span class="badge ko">NON VÉRIFIÉE</span>';
  const seq=document.getElementById('seq');seq.innerHTML='';
  const head=document.createElement('div');head.textContent=d.message+(d.manoeuvres.length?'':'\n(aucune manœuvre)');seq.appendChild(head);
  d.manoeuvres.forEach((m,i)=>{const ln=document.createElement('div');ln.className='line';ln.id='ln'+(i+1);
    ln.textContent=`${String(i+1).padStart(2)}. ${m.action.padEnd(5)} ${m.switch_id}  (${m.raison})${m.boucle?' ['+m.boucle+']':''}`;seq.appendChild(ln);});
  S.n=d.n_steps;S.labels=d.labels;S.idx=0;
  if(S.n>0){document.getElementById('anim').style.display='flex';await showStep(0);}}
async function showStep(i){S.idx=Math.max(0,Math.min(i,S.n-1));
  const r=await (await fetch('/api/step?i='+S.idx)).json();
  document.getElementById('diagBottom').innerHTML=r.svg;
  document.getElementById('stepinfo').textContent=S.idx+'/'+(S.n-1);
  document.getElementById('steplabel').textContent=S.labels[S.idx]||'';
  document.querySelectorAll('#seq .cur').forEach(e=>e.classList.remove('cur'));
  const ln=document.getElementById('ln'+S.idx);if(ln)ln.classList.add('cur');
  document.getElementById('bprev').disabled=(S.idx<=0);document.getElementById('bnext').disabled=(S.idx>=S.n-1);}
function step(d){stopAnim();showStep(S.idx+d);}
function play(){stopAnim();S.timer=setInterval(async()=>{if(S.idx>=S.n-1){stopAnim();return;}await showStep(S.idx+1);},1000);}
function stopAnim(){if(S.timer){clearInterval(S.timer);S.timer=null;}}
init();
</script>
</body></html>"""


def main():
    global SESSION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, help="Chemin du réseau .xiidm")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scenarios-dir", default="tests/manoeuvre/scenarios",
                    help="Dossier de sauvegarde des scénarios cible")
    args = ap.parse_args()

    global SCEN_DIR
    SCEN_DIR = pathlib.Path(args.scenarios_dir)

    print(f"Chargement du réseau {args.grid} …")
    SESSION = Session(pp.network.load(args.grid))
    print(f"Postes de test disponibles : {SESSION.postes}")
    print(f"IHM Manœuvre : http://localhost:{args.port}  (Ctrl-C pour arrêter)")
    # threaded=False : sérialise les requêtes (l'état réseau pypowsybl est partagé).
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
