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
6bis. **Sauvegarder la séquence générée** (``--sequences-dir``, défaut
   ``tests/manoeuvre/sequences``) : JSON autonome avec topologies détaillées et
   nodales de départ/cible, lien vers le scénario, et manœuvres ordonnées —
   réutilisable pour l'analyse et la création de tests.
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
from expert_op4grid_recommender.manoeuvre.algo import (
    Manoeuvre,
    determiner_topo_complete_cible,
    determiner_manoeuvres_cible_detaillee,
    _sectionneurs_sous_charge_par_manoeuvre,
)

# Postes de test retenus (intersectés avec les VL réellement présents)
POSTES_TEST = [
    "CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5P3", "BXTO5P6",
    "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6",
    "CORNIP3", "GUARBP6", "MORBRP6",
]

SLD_PAR = ppn.SldParameters(topological_coloring=True)
SCEN_DIR = pathlib.Path("tests/manoeuvre/scenarios")    # redéfini dans main()
SEQ_DIR = pathlib.Path("tests/manoeuvre/sequences")     # redéfini dans main()


def _replay_states(initial: dict[str, bool],
                   manoeuvres: list[dict]) -> list[dict[str, bool]]:
    """États détaillés successifs obtenus en rejouant ``manoeuvres`` depuis
    ``initial``. ``states[0]`` = départ ; ``states[k]`` = état après la k-ième
    manœuvre. Fonction pure (testable sans Flask ni pypowsybl)."""
    states = [dict(initial)]
    running = dict(initial)
    for m in manoeuvres:
        running = dict(running)
        running[m["switch_id"]] = (m["action"] == "OPEN")
        states.append(running)
    return states


def _manual_manoeuvre(displayed_state: dict[str, bool], sid: str):
    """Manœuvre manuelle basculant ``sid`` depuis ``displayed_state`` (l'état
    affiché) : OUVRE s'il est fermé, FERME s'il est ouvert. ``None`` si l'organe
    est inconnu. Fonction pure (testable)."""
    cur = displayed_state.get(sid)
    if cur is None:
        return None
    return {"switch_id": sid, "action": "CLOSE" if cur else "OPEN",
            "raison": "manœuvre manuelle (expert)", "boucle": None}


def _delete_indices(manoeuvres: list[dict], indices) -> list[dict]:
    """Retourne ``manoeuvres`` privée des positions ``indices`` (1-based).
    Les indices hors bornes ou en double sont ignorés. Fonction pure."""
    drop = {int(i) for i in indices if 1 <= int(i) <= len(manoeuvres)}
    return [m for k, m in enumerate(manoeuvres, 1) if k not in drop]


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
        self.scenario_name: str | None = None  # nom du scénario lié à la cible
        # Séquence courante (calculée puis éventuellement éditée par l'expert)
        self.seq_manoeuvres: list[dict] = []     # [{switch_id, action, raison, boucle}]
        self.seq_states: list[dict[str, bool]] = []
        self.seq_highlights: list[str | None] = []
        self.seq_labels: list[str] = []
        self.seq_edited: bool = False
        self.seq_mode: str = "smooth"

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
        self.seq_manoeuvres = []
        self.seq_states, self.seq_highlights, self.seq_labels = [], [], []
        self.seq_edited = False
        self.scenario_name = None

    def reset(self):
        self.current = dict(self.initial)
        self.scenario_name = None

    def toggle(self, sid):
        if sid in self.current:
            self.current[sid] = not self.current[sid]
            self.scenario_name = None

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

    def step_view(self, i: int):
        """Vue **interactive** de l'étape i :
        ``(svg_highlighté, switches, nb, i, reached)``.

        Les organes sont renvoyés pour l'état de l'étape afin que l'expert puisse
        cliquer un organe à n'importe quelle étape (insertion de manœuvre).
        ``reached`` indique si l'état affiché **est déjà la topologie cible**
        (même partition nodale) — pour mettre en évidence la vue du poste."""
        if not self.seq_states:
            return "", [], 0, 0, False
        i = max(0, min(i, len(self.seq_states) - 1))
        state = self.seq_states[i]
        self.apply(state)
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
        meta = json.loads(svg.metadata)
        switches = self._switches_meta(meta, state)
        topo_i = TopologieNodale.from_graph(
            build_vl_graph(self.net, self.vl), self.vl)
        nb = topo_i.nb_noeuds
        self.apply(self.current)
        topo_c = TopologieNodale.from_graph(
            build_vl_graph(self.net, self.vl), self.vl)
        reached = topo_c.meme_topologie(topo_i)
        return _highlight(svg.svg, self.seq_highlights[i]), switches, nb, i, reached

    # --- séquence éditable (navigation + édition par l'expert) ------------
    def _rebuild_seq(self):
        """Recompose états / surlignages / libellés depuis ``seq_manoeuvres``."""
        svgid = self.svgid_par_switch()
        self.seq_states = _replay_states(self.initial, self.seq_manoeuvres)
        self.seq_highlights = [None] + [
            svgid.get(m["switch_id"]) for m in self.seq_manoeuvres]
        self.seq_labels = ["État de départ"] + [
            f'{i}. {m["action"]} {m["switch_id"]} — {m["raison"]}'
            for i, m in enumerate(self.seq_manoeuvres, 1)]

    def _violations_regles(self) -> list[str | None]:
        """Pour chaque manœuvre de la séquence, un message d'infraction d'une
        règle de sûreté (sectionneur manœuvré sous charge…) ou ``None``. Aligné
        sur ``seq_manoeuvres``. Permet d'alerter l'expert sur une manœuvre
        manuelle invalide."""
        if not self.seq_manoeuvres or not self.vl:
            return [None] * len(self.seq_manoeuvres)
        self.apply(self.initial)
        poste = PosteTopologique.from_graph(
            build_vl_graph(self.net, self.vl), self.vl)
        manos = [Manoeuvre(m["switch_id"], m["action"], m.get("raison", ""))
                 for m in self.seq_manoeuvres]
        viol = _sectionneurs_sous_charge_par_manoeuvre(poste, manos)
        self.apply(self.current)  # restaurer l'affichage courant
        return viol

    def _seq_payload(self) -> dict:
        """Charge utile commune (séquence + état final) renvoyée au front."""
        nb_final, matches = None, None
        if self.seq_states:
            self.apply(self.seq_states[-1])
            topo_f = TopologieNodale.from_graph(
                build_vl_graph(self.net, self.vl), self.vl)
            nb_final = topo_f.nb_noeuds
            self.apply(self.current)
            topo_c = TopologieNodale.from_graph(
                build_vl_graph(self.net, self.vl), self.vl)
            matches = topo_c.meme_topologie(topo_f)
        self.apply(self.current)  # restaurer l'affichage courant
        return {
            "manoeuvres": [dict(m) for m in self.seq_manoeuvres],
            "nb_manoeuvres": len(self.seq_manoeuvres),
            "n_steps": len(self.seq_states),
            "labels": self.seq_labels,
            "nb_final": nb_final,
            "matches_cible": matches,
            "edited": self.seq_edited,
            "mode": self.seq_mode,
            "violations": self._violations_regles(),
        }

    def seq_insert(self, step: int, sid: str) -> int:
        """Insère une manœuvre basculant ``sid`` juste **après** l'étape ``step``
        (la suite est conservée). Retourne l'index d'étape à afficher."""
        if not self.seq_states:
            return 0
        step = max(0, min(step, len(self.seq_states) - 1))
        m = _manual_manoeuvre(self.seq_states[step], sid)
        if m is None:
            return step
        self.seq_manoeuvres.insert(step, m)   # position step => nouvel état step+1
        self.seq_edited = True
        self._rebuild_seq()
        return step + 1

    def seq_delete(self, index: int) -> int:
        """Supprime la manœuvre n°``index`` (1-based). Retourne l'étape à afficher."""
        if 1 <= index <= len(self.seq_manoeuvres):
            self.seq_manoeuvres.pop(index - 1)
            self.seq_edited = True
            self._rebuild_seq()
        return max(0, min(index - 1, len(self.seq_states) - 1))

    def seq_delete_many(self, indices) -> int:
        """Supprime en une fois les manœuvres aux positions ``indices`` (1-based ;
        sélection multiple ou bloc). Retourne l'étape à afficher."""
        keep = _delete_indices(self.seq_manoeuvres, indices)
        if len(keep) != len(self.seq_manoeuvres):
            self.seq_manoeuvres = keep
            self.seq_edited = True
            self._rebuild_seq()
        valides = [int(i) for i in indices if int(i) >= 1]
        goto = (min(valides) - 1) if valides else 0
        return max(0, min(goto, len(self.seq_states) - 1))

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
        self.scenario_name = name
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
            self.scenario_name = None   # cible fraîche à redéfinir
        else:
            self.initial = {k: bool(data["depart"].get(k, base[k])) for k in base}
            self.current = {k: bool(data["cible"].get(k, base[k])) for k in base}
            self.scenario_name = name
        return data["voltage_level_id"]

    # --- calcul de séquence ----------------------------------------------
    def sequence(self, mode: str = "smooth"):
        # Poste à l'état de départ (A)
        self.apply(self.initial)
        poste = PosteTopologique.from_graph(build_vl_graph(self.net, self.vl), self.vl)
        # Topologie détaillée cible (B) imposée : on vise la barre exacte de
        # chaque départ, pas seulement la partition nodale.
        self.apply(self.current)
        cible_graph = build_vl_graph(self.net, self.vl)

        mode = "aggressive" if mode == "aggressive" else "smooth"
        self.seq_mode = mode
        res = determiner_manoeuvres_cible_detaillee(poste, cible_graph, mode=mode)

        # Séquence éditable initialisée depuis le résultat de l'algorithme.
        self.seq_manoeuvres = [{
            "switch_id": m.switch_id, "action": m.action,
            "raison": m.raison, "boucle": m.type_boucle,
        } for m in res.manoeuvres]
        self.seq_edited = False
        self._rebuild_seq()

        payload = {
            "verified": res.is_verified,
            "verified_detaillee": res.is_verified_detaillee,
            "ecarts": res.ecarts,
            "message": res.message,
            **self._seq_payload(),   # manoeuvres / n_steps / labels / nb_final / …
        }
        return payload

    def manual_start(self):
        """Démarre une **séquence manuelle vierge** : la liste de manœuvres est
        vidée et l'état courant repart de l'**état de départ** (étape 0).
        L'expert construit ensuite la séquence en cliquant les organes du schéma
        (chaque clic ajoute une manœuvre via ``seq_insert``), en visant la
        **topologie cible** affichée en référence.
        Retourne ``(svg_cible, nb_noeuds_cible)`` pour la vue de référence."""
        self.seq_manoeuvres = []
        self.seq_edited = True
        self._rebuild_seq()
        svg_c, _, nb_c = self.view(self.current)   # cible = référence à atteindre
        return svg_c, nb_c

    def save_sequence(self, name: str) -> str:
        """
        Sauvegarde la séquence **courante** (telle qu'éventuellement éditée par
        l'expert) dans un JSON autonome : topologies détaillées et nodales de
        départ/cible, lien vers le scénario éventuel, et liste ordonnée des
        manœuvres. On n'exécute **pas** de re-calcul de l'algorithme : la liste
        éditée est sérialisée telle quelle ; on recalcule seulement l'état nodal
        final atteint et la concordance avec la cible.
        """
        import re
        if not self.seq_states:   # aucune séquence calculée : on en calcule une
            self.sequence()
        info = self._seq_payload()
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) \
            or (self.scenario_name or self.vl)
        data = {
            "voltage_level_id": self.vl,
            "name": name,
            "scenario": self.scenario_name,          # lien vers les topologies
            "edited": self.seq_edited,
            "mode": self.seq_mode,
            "matches_cible": info["matches_cible"],
            "nb_final": info["nb_final"],
            "depart": self.initial,
            "cible": self.current,
            "depart_nodale": self.groups_of(self.initial),
            "cible_nodale": self.groups_of(self.current),
            "nb_manoeuvres": len(self.seq_manoeuvres),
            "manoeuvres": [
                {"ordre": i + 1, **m}
                for i, m in enumerate(self.seq_manoeuvres)
            ],
        }
        self.apply(self.current)  # restaurer l'affichage courant
        SEQ_DIR.mkdir(parents=True, exist_ok=True)
        path = SEQ_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return str(path)


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


def _safe_name(name, default):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or default


@app.post("/api/save")
def api_save():
    name = _safe_name(request.json.get("name", ""), SESSION.vl)
    if not request.json.get("overwrite") and (SCEN_DIR / f"{name}.json").exists():
        return jsonify(exists=True, name=name, path=str(SCEN_DIR / f"{name}.json"))
    path = SESSION.save_scenario(name)
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
    return jsonify(SESSION.sequence(request.json.get("mode", "smooth")))


@app.post("/api/save_sequence")
def api_save_sequence():
    name = _safe_name(request.json.get("name", ""),
                      SESSION.scenario_name or SESSION.vl)
    if not request.json.get("overwrite") and (SEQ_DIR / f"{name}.json").exists():
        return jsonify(exists=True, name=name, path=str(SEQ_DIR / f"{name}.json"))
    path = SESSION.save_sequence(name)
    return jsonify(path=path)


@app.get("/api/step")
def api_step():
    i = int(request.args.get("i", 0))
    svg, switches, nb, i, reached = SESSION.step_view(i)
    return jsonify(svg=svg, switches=switches, nb_noeuds=nb, i=i, reached=reached)


@app.post("/api/seq_insert")
def api_seq_insert():
    goto = SESSION.seq_insert(int(request.json["step"]), request.json["id"])
    return jsonify(goto=goto, **SESSION._seq_payload())


@app.post("/api/seq_delete")
def api_seq_delete():
    goto = SESSION.seq_delete(int(request.json["index"]))
    return jsonify(goto=goto, **SESSION._seq_payload())


@app.post("/api/seq_delete_many")
def api_seq_delete_many():
    goto = SESSION.seq_delete_many(request.json.get("indices", []))
    return jsonify(goto=goto, **SESSION._seq_payload())


@app.post("/api/manual_start")
def api_manual_start():
    svg_c, nb_c = SESSION.manual_start()
    return jsonify(cible_svg=_prefix_svg_ids(svg_c, "C_"), cible_nb=nb_c,
                   goto=0, **SESSION._seq_payload())


PAGE = r"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>IHM Manœuvre</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #side{width:340px;background:#f4f5f7;padding:12px;overflow:auto;border-right:1px solid #ccc}
 #main{flex:1;display:flex;flex-direction:column;overflow:hidden}
 .pane{flex:1;display:flex;flex-direction:column;min-height:0}
 .pane.collapsed{flex:0 0 auto !important}
 .pane.collapsed .diag{display:none}
 .cbtn{padding:0 7px;margin-right:6px;border:1px solid #999;border-radius:4px;background:#fff;cursor:pointer;font-size:12px}
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
 #seq{font-family:monospace;font-size:12px;background:#fff;border:1px solid #ddd;padding:8px;max-height:30vh;overflow:auto}
 #seq .head{white-space:pre}
 #seq .line{padding:0 2px;cursor:pointer;display:flex;justify-content:space-between;gap:6px;white-space:pre}
 #seq .line:hover{background:#eef2ff}
 #seq .line .txt{flex:1;overflow:hidden;text-overflow:ellipsis}
 #seq .cur{background:#fde68a}
 #seq .line.manual .txt{color:#7c3aed}
 #seq .line.violation{background:#fef2f2}
 #seq .line.violation .txt{color:#b91c1c}
 #seq .warn{color:#dc2626;font-weight:bold;padding:0 4px;cursor:help}
 #seq .head.hasviol{color:#b91c1c}
 #seq .del{color:#c0392b;font-weight:bold;cursor:pointer;padding:0 4px;border-radius:3px;visibility:hidden}
 #seq .line:hover .del{visibility:visible}
 #seq .del:hover{background:#fde2e2}
 #seq .line .chk{margin:0 4px 0 0;cursor:pointer;flex:0 0 auto}
 #seq .line.selected{box-shadow:inset 3px 0 0 #6366f1}
 #seltools{margin:4px 0;font-size:12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 #seltools button{padding:3px 8px;font-size:12px;margin:0}
 #anim{display:flex;align-items:center;gap:8px;padding:6px;background:#eef;border-top:1px solid #ccd}
 .badge{font-size:11px;padding:2px 6px;border-radius:6px;background:#ddd}
 .ok{background:#27ae60;color:#fff}.ko{background:#c0392b;color:#fff}
 .pane.reached{box-shadow:inset 0 0 0 5px #facc15;transition:box-shadow .2s}
 .pane.reached>.ttl{background:#fef9c3 !important}
 .pane.reached>.ttl::after{content:" — ✓ topologie cible atteinte";color:#a16207;font-weight:bold}
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
  <div style="font-size:12px;margin:2px 0">Mode :
    <select id="seqMode" style="width:auto;padding:3px" title="Smooth : dé-énergise au plus près, en place (peu d'ouvrages hors tension à la fois). Agressif : dé-énergise en lot (moins de manœuvres, plus d'ouvrages hors tension simultanément).">
      <option value="smooth">Smooth (sûr, en place)</option>
      <option value="aggressive">Agressif (batch, plus court)</option>
    </select>
  </div>
  <button id="bcalc" class="primary" onclick="sequence()" disabled>⚙ Calculer la séquence</button>
  <button id="bmanual" onclick="manualSeq()" disabled title="Construire la séquence à la main : cliquez les organes du schéma (départ → …), la cible s'affiche en référence">✋ Séquence manuelle</button>
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
    <div class="ttl"><button class="cbtn" onclick="togglePane('paneTop')" title="Réduire / agrandir">▾</button>
      <span id="ttlA">Topologie de départ</span> — <span id="nbA" class="badge">–</span> nœud(s)</div>
    <div class="diag" id="diagTop">Choisissez un poste…</div>
  </div>
  <div class="pane" id="paneBot">
    <div class="ttl"><button class="cbtn" onclick="togglePane('paneBot')" title="Réduire / agrandir">▾</button>
      Topologie cible (éditable, clic sur un organe) — <span id="nbB" class="badge">–</span> nœud(s)
        · animation de la séquence ici</div>
    <div class="diag" id="diagBottom"></div>
  </div>
  <div id="anim" style="display:none">
    <button id="bprev" onclick="step(-1)">◀</button>
    <button id="bplay" onclick="play()">▶ Lecture</button>
    <button id="bnext" onclick="step(1)">▶|</button>
    <span id="stepinfo" class="badge"></span>
    <span id="steplabel" style="font-family:monospace;font-size:12px"></span>
    <span style="flex:1"></span>
    <span style="font-size:11px;color:#3730a3">✎ Schéma éditable : cliquez un organe pour insérer une manœuvre après l'étape courante.</span>
  </div>
  <div id="seqwrap" style="padding:8px;display:none">
    <b>Séquence <span id="seqstatus"></span></b>
    <div id="seltools">
      <span>Sélection : <b id="selcount">0</b></span>
      <button id="bdelsel" onclick="seqDeleteSelected()" disabled>🗑 Supprimer la sélection</button>
      <button onclick="clearSel()">Désélectionner</button>
      <span style="color:#777">case à cocher = sélection · Maj+clic = bloc</span>
    </div>
    <div id="seq"></div>
    <div style="margin-top:6px">
      <input id="seqName" placeholder="nom de la séquence" style="width:45%;padding:4px">
      <button onclick="saveSeq()">💾 Sauvegarder la séquence</button>
      <span id="seqsavemsg" style="font-size:11px;color:#1a7f37"></span>
    </div>
  </div>
</div>
<script>
let S={n:0,idx:0,timer:null,labels:[],algo:{},sel:new Set(),lastSel:null,manual:false};
const api=(p,b)=>fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json());
function setValidated(v){document.getElementById('bcalc').disabled=!v;
  document.getElementById('bmanual').disabled=!v;
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
async function saveWithConfirm(url,name,msgEl){
  let r=await api(url,{name});
  while(r.exists){
    if(confirm('Le fichier « '+r.name+' » existe déjà.\n\nOK = écraser, Annuler = renommer.')){
      r=await api(url,{name:r.name,overwrite:true});
    }else{
      const nn=prompt('Nouveau nom :', r.name+'_v2');
      if(!nn){msgEl.textContent='Sauvegarde annulée.';return null;}
      r=await api(url,{name:nn});
    }
  }
  return r;
}
async function save(){const name=document.getElementById('scenName').value;
  const r=await saveWithConfirm('/api/save',name,document.getElementById('savemsg'));
  if(!r)return;
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
function hideSeq(){document.getElementById('seqwrap').style.display='none';document.getElementById('anim').style.display='none';
  document.getElementById('paneBot').classList.remove('reached');}
function show(d){
  document.getElementById('paneBot').classList.remove('reached');
  if(d.initial_svg!==undefined){document.getElementById('diagTop').innerHTML=d.initial_svg;
    document.getElementById('nbA').textContent=d.nb_initial;
    document.getElementById('ttlA').textContent='Topologie de départ';S.manual=false;}
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
async function sequence(){stopAnim();S.manual=false;
  const d=await api('/api/sequence',{mode:document.getElementById('seqMode').value});
  document.getElementById('seqwrap').style.display='block';
  S.algo={message:d.message,ecarts:d.ecarts||[],verified:d.verified,verified_detaillee:d.verified_detaillee,mode:d.mode};
  document.getElementById('seqName').value=(document.getElementById('scenName').value||'sequence');
  document.getElementById('seqsavemsg').textContent='';
  renderSeq(d);
  document.getElementById('anim').style.display='flex';
  await showStep(0);}
async function manualSeq(){stopAnim();S.manual=true;const d=await api('/api/manual_start',{});
  document.getElementById('seqwrap').style.display='block';
  S.algo={message:'Séquence manuelle : cliquez les organes du schéma du bas (état courant) pour ajouter des manœuvres ; la cible à atteindre est affichée en haut.',ecarts:[],verified:false,verified_detaillee:false};
  // Vue de référence : la CIBLE en haut.
  document.getElementById('diagTop').innerHTML=d.cible_svg;
  document.getElementById('nbA').textContent=d.cible_nb;
  document.getElementById('ttlA').textContent='Cible à atteindre';
  document.getElementById('seqName').value=(document.getElementById('scenName').value||'sequence');
  document.getElementById('seqsavemsg').textContent='';
  renderSeq(d);
  document.getElementById('anim').style.display='flex';
  await showStep(0);}
function renderSeq(d){
  S.n=d.n_steps;S.labels=d.labels;S.sel=new Set();S.lastSel=null;
  const st=document.getElementById('seqstatus');
  if(d.edited){const lbl=S.manual?'MANUELLE':'ÉDITÉE';
    st.innerHTML='<span class="badge" style="background:#7c3aed;color:#fff">'+lbl+(d.nb_final!=null?' · '+d.nb_final+' nœud(s)':'')+'</span> '+
      (d.matches_cible?'<span class="badge ok">= cible</span>':'<span class="badge ko">≠ cible</span>');}
  else if(S.algo.verified_detaillee){st.innerHTML='<span class="badge ok">DÉTAILLÉE VÉRIFIÉE</span>'+(S.algo.mode?' <span class="badge">mode '+(S.algo.mode==='aggressive'?'agressif':'smooth')+'</span>':'');}
  else if(S.algo.verified){st.innerHTML='<span class="badge" style="background:#d97706;color:#fff">NODALE OK · '+(S.algo.ecarts?S.algo.ecarts.length:'?')+' écart(s) détaillé(s)</span>';}
  else{st.innerHTML='<span class="badge ko">NON VÉRIFIÉE</span>';}
  const seq=document.getElementById('seq');seq.innerHTML='';
  let headtxt=(S.algo.message||'')+(d.manoeuvres.length?'':'\n(aucune manœuvre)');
  if(S.algo.ecarts&&S.algo.ecarts.length){headtxt+='\nÉcarts détaillés : '+S.algo.ecarts.join(' ; ');}
  if(d.edited){headtxt+='\n(séquence éditée manuellement)';}
  const viols=d.violations||[];
  const nbViol=viols.filter(Boolean).length;
  if(nbViol){headtxt+='\n⚠ '+nbViol+' manœuvre(s) enfreignant une règle de sûreté — voir ⚠ dans la liste.';}
  const head=document.createElement('div');head.className='head'+(nbViol?' hasviol':'');head.id='ln0';head.textContent=headtxt;
  head.style.cursor='pointer';head.title="Aller à l'état de départ";head.onclick=()=>stepGoto(0);seq.appendChild(head);
  d.manoeuvres.forEach((m,i)=>{const k=i+1;const ln=document.createElement('div');
    const viol=viols[i];
    ln.className='line'+(/manuelle/.test(m.raison)?' manual':'')+(viol?' violation':'');ln.id='ln'+k;
    const chk=document.createElement('input');chk.type='checkbox';chk.className='chk';
    chk.title='Sélectionner (Maj+clic = bloc)';
    chk.onclick=(e)=>{e.stopPropagation();onChkClick(k,e.shiftKey);};
    const txt=document.createElement('span');txt.className='txt';
    txt.textContent=`${String(k).padStart(2)}. ${m.action.padEnd(5)} ${m.switch_id}  (${m.raison})${m.boucle?' ['+m.boucle+']':''}`;
    if(viol){const w=document.createElement('span');w.className='warn';w.textContent='⚠';
      w.title=viol;ln.title=viol;ln.appendChild(w);}
    const del=document.createElement('span');del.className='del';del.textContent='✕';del.title='Supprimer cette manœuvre';
    del.onclick=(e)=>{e.stopPropagation();seqDelete(k);};
    ln.appendChild(chk);ln.appendChild(txt);ln.appendChild(del);
    ln.onclick=()=>stepGoto(k);seq.appendChild(ln);});
  updateSelUI();
}
function onChkClick(idx,shift){
  if(shift&&S.lastSel){const a=Math.min(S.lastSel,idx),b=Math.max(S.lastSel,idx);
    for(let k=a;k<=b;k++)S.sel.add(k);}
  else{if(S.sel.has(idx))S.sel.delete(idx);else S.sel.add(idx);}
  S.lastSel=idx;updateSelUI();}
function updateSelUI(){
  document.querySelectorAll('#seq .line').forEach(ln=>{const k=parseInt(ln.id.slice(2));
    const c=ln.querySelector('.chk');if(c)c.checked=S.sel.has(k);
    ln.classList.toggle('selected',S.sel.has(k));});
  const cnt=document.getElementById('selcount');if(cnt)cnt.textContent=S.sel.size;
  const b=document.getElementById('bdelsel');if(b)b.disabled=(S.sel.size===0);}
function clearSel(){S.sel.clear();S.lastSel=null;updateSelUI();}
async function seqDeleteSelected(){if(!S.sel.size)return;
  stopAnim();const indices=[...S.sel];
  const r=await api('/api/seq_delete_many',{indices});renderSeq(r);await showStep(r.goto);}
function stepGoto(i){stopAnim();showStep(i);}
async function seqInsert(step,id){stopAnim();const r=await api('/api/seq_insert',{step,id});renderSeq(r);await showStep(r.goto);}
async function seqDelete(k){stopAnim();const r=await api('/api/seq_delete',{index:k});renderSeq(r);await showStep(r.goto);}
async function saveSeq(){const name=document.getElementById('seqName').value;
  const msg=document.getElementById('seqsavemsg');
  const r=await saveWithConfirm('/api/save_sequence',name,msg);
  if(!r)return;
  msg.textContent='✓ '+r.path;}
function bindStep(switches){const root=document.getElementById('diagBottom');
  switches.forEach(s=>{const el=root.querySelector('[id="'+s.svgId+'"]');
    if(el){el.style.cursor='pointer';el.onclick=()=>seqInsert(S.idx,s.id);
      el.querySelectorAll('*').forEach(c=>c.style.cursor='pointer');}});}
async function showStep(i){S.idx=Math.max(0,Math.min(i,S.n-1));
  const r=await (await fetch('/api/step?i='+S.idx)).json();
  document.getElementById('diagBottom').innerHTML=r.svg;
  if(r.switches)bindStep(r.switches);
  if(r.nb_noeuds!=null)document.getElementById('nbB').textContent=r.nb_noeuds;
  document.getElementById('stepinfo').textContent=S.idx+'/'+(S.n-1);
  document.getElementById('steplabel').textContent=S.labels[S.idx]||'';
  document.querySelectorAll('#seq .cur').forEach(e=>e.classList.remove('cur'));
  const ln=document.getElementById('ln'+S.idx);if(ln)ln.classList.add('cur');
  // Halo jaune autour de la vue du poste quand l'état affiché EST la topologie cible.
  document.getElementById('paneBot').classList.toggle('reached', !!r.reached);
  document.getElementById('bprev').disabled=(S.idx<=0);document.getElementById('bnext').disabled=(S.idx>=S.n-1);}
function step(d){stopAnim();showStep(S.idx+d);}
function play(){stopAnim();S.timer=setInterval(async()=>{if(S.idx>=S.n-1){stopAnim();return;}await showStep(S.idx+1);},1000);}
function stopAnim(){if(S.timer){clearInterval(S.timer);S.timer=null;}}
function togglePane(id){const p=document.getElementById(id);p.classList.toggle('collapsed');
  const b=p.querySelector('.cbtn');if(b)b.textContent=p.classList.contains('collapsed')?'▸':'▾';}
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
    ap.add_argument("--sequences-dir", default="tests/manoeuvre/sequences",
                    help="Dossier de sauvegarde des séquences générées")
    args = ap.parse_args()

    global SCEN_DIR, SEQ_DIR
    SCEN_DIR = pathlib.Path(args.scenarios_dir)
    SEQ_DIR = pathlib.Path(args.sequences_dir)

    print(f"Chargement du réseau {args.grid} …")
    SESSION = Session(pp.network.load(args.grid))
    print(f"Postes de test disponibles : {SESSION.postes}")
    print(f"IHM Manœuvre : http://localhost:{args.port}  (Ctrl-C pour arrêter)")
    # threaded=False : sérialise les requêtes (l'état réseau pypowsybl est partagé).
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
