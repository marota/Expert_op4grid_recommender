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
import re
import sys
from contextlib import contextmanager

# Rendre le package importable quand lancé depuis la racine du dépôt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from flask import Flask, jsonify, request, Response
except ImportError:  # pragma: no cover
    sys.exit("Flask est requis pour l'IHM : pip install -e .[ihm]  (ou pip install flask)")

import networkx as nx
import pypowsybl as pp
import pypowsybl.network as ppn

from expert_op4grid_recommender.manoeuvre.graph import (
    build_vl_graph,
    busbar_nodes,
    equipment_nodes,
)
from expert_op4grid_recommender.manoeuvre.topologie import (
    PosteTopologique,
    TopologieNodale,
)
from expert_op4grid_recommender.manoeuvre.algo import (
    Manoeuvre,
    determiner_topo_complete_cible,
    determiner_manoeuvres_cible_detaillee,
    sectionneurs_sous_charge_par_manoeuvre,
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


def _normalize_groups(all_branches, groups) -> list[list[str]]:
    """Normalise une partition nodale éditée par l'expert en une partition
    **complète et disjointe** des ``all_branches`` (univers des départs).

    - une branche présente dans plusieurs groupes est conservée dans le
      **dernier** groupe où elle apparaît (la dernière affectation gagne) ;
    - les branches inconnues (hors ``all_branches``) sont ignorées ;
    - les groupes vides sont retirés ;
    - toute branche de ``all_branches`` absente des groupes est réinjectée dans
      un nœud propre regroupant ces orphelines.

    Fonction pure (testable sans Flask ni pypowsybl)."""
    universe = list(dict.fromkeys(all_branches))   # ordre stable, dédupliqué
    allowed = set(universe)

    # Dernière affectation gagnante : on parcourt les groupes dans l'ordre.
    assign: dict[str, int] = {}
    for gi, grp in enumerate(groups):
        for eq in grp:
            if eq in allowed:
                assign[eq] = gi

    # Reconstituer les groupes en respectant l'ordre des branches de l'univers.
    buckets: dict[int, list[str]] = {}
    for eq in universe:
        gi = assign.get(eq)
        if gi is not None:
            buckets.setdefault(gi, []).append(eq)

    result = [buckets[gi] for gi in sorted(buckets)]

    # Branches orphelines (jamais affectées) -> un nœud dédié.
    orphans = [eq for eq in universe if eq not in assign]
    if orphans:
        result.append(orphans)
    return result


def _decode_svg_id(s: str) -> str:
    """Décode un identifiant SVG pypowsybl (``_46_`` → ``.``, ``_95_`` → ``_``,
    ``_45_`` → ``-``…). Fonction pure."""
    return re.sub(r"_(\d+)_", lambda m: chr(int(m.group(1))), s)


def _parse_feeder_meta(svg: str) -> dict:
    """Extrait du SVG du SLD, **par départ** (clé = id décodé sans le préfixe
    ``id``), son ``label`` (libellé court), sa ``dir`` (``TOP``/``BOTTOM``) et son
    abscisse ``x`` (ordre gauche → droite).

    - direction & abscisse : groupe ``<g class="… sld-(top|bottom)-feeder …"
      id="id…" transform="translate(x, y)">`` (la classe peut être combinée, ex.
      ``sld-load sld-top-feeder``) ;
    - libellé : ``<text class="sld-label" id="id…_N_LABEL">`` (haut) ou ``_S_LABEL``
      (bas) ; ``_NW_LABEL`` (barres) est exclu.

    Fonction pure (testable sans pypowsybl)."""
    groups = re.findall(
        r'<g class="[^"]*sld-(top|bottom)-feeder[^"]*" id="(id[^"]+?)"'
        r'[^>]*transform="translate\(([0-9.]+),[0-9.]+\)"', svg)
    labs = dict(re.findall(
        r'<text class="sld-label" id="(id[^"]+?)_95_[NS]_95_LABEL"[^>]*>'
        r'([^<]*)</text>', svg))
    meta = {}
    for direction, gid, x in groups:
        core = _decode_svg_id(gid)[2:]   # retire le préfixe 'id'
        meta[core] = {"label": (labs.get(gid) or "").strip(),
                      "dir": direction.upper(), "x": float(x)}
    return meta


def _parse_node_colors(svg: str) -> dict:
    """Extrait du SVG du SLD la couleur (#hex) du **nœud électrique** de chaque
    élément (clé = id décodé sans le préfixe ``id``), via la palette
    ``.sld-vlXtoY.sld-bus-N {--sld-vl-color: #hex}`` et les classes
    ``sld-vl… sld-bus-N`` portées par les éléments. Fonction pure."""
    palette = {}
    for vlc, busc, hexc in re.findall(
            r"\.(sld-vl\w+)\.(sld-bus-\d+)\s*\{\s*--sld-vl-color:\s*"
            r"(#[0-9A-Fa-f]+)", svg):
        palette[(vlc, busc)] = hexc
    colors = {}
    for cls, gid in re.findall(r'<g class="([^"]*)" id="(id[^"]+)"', svg):
        if "sld-bus-" not in cls or "sld-vl" not in cls:
            continue
        vlm = re.search(r"sld-vl\w+", cls)
        bsm = re.search(r"sld-bus-\d+", cls)
        hexc = palette.get((vlm.group(0), bsm.group(0))) if vlm and bsm else None
        if hexc:
            colors[_decode_svg_id(gid)[2:]] = hexc
    return colors


def _isolated_assets(G: nx.Graph) -> list[str]:
    """Équipements **déconnectés** : ceux dont la composante connexe (en ne suivant
    que les switches **fermés**) ne contient **aucune barre**. Fonction pure
    (graphe NetworkX ; ni Flask ni pypowsybl)."""
    closed = nx.Graph()
    closed.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if not d.get("open", False):
            closed.add_edge(u, v)
    barres = set(busbar_nodes(G))
    eqset = set(equipment_nodes(G))
    iso = []
    for comp in nx.connected_components(closed):
        if comp & barres:
            continue   # composante reliée à une barre = nœud électrique
        for n in comp & eqset:
            eq = G.nodes[n].get("equipment_id")
            if eq:
                iso.append(eq)
    return iso


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
        # Caches mémoïsés par état détaillé (cf. _graph / _topo / _flows).
        # Le graphe NX, la topologie nodale et le résultat de load flow d'un VL
        # ne dépendent que du VL et de l'état des organes — purs vis-à-vis de
        # l'état appliqué. Invalidés au chargement d'un poste (load()).
        self._graph_cache: dict = {}
        self._topo_cache: dict = {}
        self._flow_cache: dict = {}

    # --- gestion d'état ---------------------------------------------------
    def switches_df(self, vl):
        df = self.net.get_switches(all_attributes=True)
        return df[df["voltage_level_id"] == vl]

    def load(self, vl):
        self.vl = vl
        # Nouveau poste : l'univers d'organes change → les caches mémoïsés par
        # état (graphe / topo / flux) ne sont plus valides.
        self._graph_cache.clear()
        self._topo_cache.clear()
        self._flow_cache.clear()
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

    @contextmanager
    def applied(self, state: dict[str, bool]):
        """Applique temporairement l'état détaillé ``state`` au réseau, puis
        **restaure l'état d'affichage courant** (``self.current``) en sortie —
        y compris si le corps lève. Remplace les paires ``apply(state)`` …
        ``apply(self.current)  # restaurer`` disséminées (sources d'oublis et de
        fuites d'état entre requêtes). Les lectures dépendantes de l'état appliqué
        (SLD, load flow) doivent se faire **dans** le bloc."""
        self.apply(state)
        try:
            yield
        finally:
            self.apply(self.current)

    # --- caches mémoïsés par état (graphe / topo / load flow) -------------
    def _state_key(self, state: dict[str, bool]):
        """Clé de cache d'un état détaillé : (VL, organes figés). Le VL borne la
        clé pour éviter toute collision si deux postes partagent un id d'organe."""
        return (self.vl, frozenset(state.items()))

    def _graph(self, state: dict[str, bool]) -> nx.Graph:
        """Graphe NX du VL pour ``state``, **mémoïsé** par (VL, état). Le graphe
        ne dépend que du VL et de l'état des organes ; on évite ainsi de
        reconstruire le graphe pypowsybl→NX à chaque vue. **Suppose le réseau
        déjà appliqué** à ``state`` (en cas de défaut de cache, ``build_vl_graph``
        lit l'état appliqué)."""
        key = self._state_key(state)
        G = self._graph_cache.get(key)
        if G is None:
            G = build_vl_graph(self.net, self.vl)
            self._graph_cache[key] = G
        return G

    def _topo(self, state: dict[str, bool]) -> TopologieNodale:
        """TopologieNodale du VL pour ``state``, **mémoïsée** (sur le graphe
        mémoïsé). Suppose le réseau déjà appliqué à ``state``."""
        key = self._state_key(state)
        topo = self._topo_cache.get(key)
        if topo is None:
            topo = TopologieNodale.from_graph(self._graph(state), self.vl)
            self._topo_cache[key] = topo
        return topo

    def _flows(self, state: dict[str, bool],
               types: dict[str, str | None]) -> dict[str, float]:
        """Flux actifs (MW) par branche pour ``state``, **load flow paresseux** :
        exécuté **et mémoïsé** par état (le load flow ne dépend que de la
        topologie, les injections étant constantes dans l'IHM). Suppose le réseau
        déjà appliqué à ``state``."""
        key = self._state_key(state)
        flows = self._flow_cache.get(key)
        if flows is None:
            flows = self._branch_flows(types)
            self._flow_cache[key] = flows
        return flows

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
        nb = self._topo(state).nb_noeuds
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
        with self.applied(state):
            svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
            meta = json.loads(svg.metadata)
            switches = self._switches_meta(meta, state)
            nb = self._topo(state).nb_noeuds
        # ``applied`` a restauré le réseau sur ``self.current``.
        reached = self._topo(self.current).meme_topologie(self._topo(state))
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
        with self.applied(self.initial):
            poste = PosteTopologique.from_graph(self._graph(self.initial), self.vl)
            manos = [Manoeuvre(m["switch_id"], m["action"], m.get("raison", ""))
                     for m in self.seq_manoeuvres]
            viol = sectionneurs_sous_charge_par_manoeuvre(poste, manos)
        return viol

    def _seq_payload(self) -> dict:
        """Charge utile commune (séquence + état final) renvoyée au front."""
        nb_final, matches = None, None
        if self.seq_states:
            with self.applied(self.seq_states[-1]):
                topo_f = self._topo(self.seq_states[-1])
                nb_final = topo_f.nb_noeuds
            # ``applied`` a restauré le réseau sur ``self.current``.
            matches = self._topo(self.current).meme_topologie(topo_f)
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
        topo = self._topo(state)
        return [sorted(n.equipment_ids) for n in topo.noeuds.values()]

    # --- topologie nodale (édition de la cible nodale d'intérêt) -----------
    def _short_name(self, eq: str) -> str:
        """Nom court d'un départ (préfixe VL retiré), pour l'affichage des chips."""
        name = eq
        if self.vl and name.startswith(self.vl + "_"):
            name = name[len(self.vl) + 1:]
        return name

    def _branch_flows(self, types: dict[str, str | None]) -> dict[str, float]:
        """Flux actif (MW) au terminal de chaque branche du poste, dans l'état
        **déjà appliqué** au réseau. Une charge de réseau (AC, repli DC) est
        exécutée ; le côté lu (``p1``/``p2``/``p``) est déduit du type de départ
        (``LINE_SIDE2`` → ``p2``…). Best-effort : ``{}`` si le calcul échoue."""
        try:
            res = pp.loadflow.run_ac(self.net)
            if not (res and str(res[0].status).endswith("CONVERGED")):
                pp.loadflow.run_dc(self.net)
        except Exception:
            try:
                pp.loadflow.run_dc(self.net)
            except Exception:
                return {}

        def _tbl(getter):
            try:
                return getter(all_attributes=True)
            except Exception:
                return None
        lines = _tbl(self.net.get_lines)
        twt = _tbl(self.net.get_2_windings_transformers)
        loads = _tbl(self.net.get_loads)
        gens = _tbl(self.net.get_generators)
        dls = _tbl(self.net.get_dangling_lines)

        def _val(df, eq, col):
            if df is None or eq not in df.index:
                return None
            v = df.loc[eq].get(col)
            try:
                return None if v is None or v != v else round(float(v), 1)  # NaN-safe
            except Exception:
                return None

        flows: dict[str, float] = {}
        for eq, t in types.items():
            t = t or ""
            v = None
            if t.startswith("LINE"):
                v = _val(lines, eq, "p2" if t.endswith("2") else "p1")
            elif t.startswith("TRANSFORMER"):
                v = _val(twt, eq, "p2" if t.endswith("2") else "p1")
            elif t == "LOAD":
                v = _val(loads, eq, "p")
            elif t == "GENERATOR":
                v = _val(gens, eq, "p")
            elif t == "DANGLING_LINE":
                v = _val(dls, eq, "p")
            if v is not None:
                flows[eq] = v
        return flows

    def _sld_feeder_meta(self) -> dict:
        """Libellé court, direction (TOP/BOTTOM) et abscisse de chaque départ,
        **extraits du SLD pypowsybl lui-même** — donc strictement identiques aux
        libellés de la vue détaillée. État réseau supposé **déjà appliqué**.

        Le SLD encode l'``equipmentId`` dans l'``id`` du groupe de départ
        (``id<equipmentId>`` avec ``_46_`` = ``.``, ``_95_`` = ``_``…) ; la classe
        ``sld-(top|bottom)-feeder`` donne la direction et ``translate(x, y)``
        l'ordre horizontal (gauche → droite)."""
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR).svg
        return _parse_feeder_meta(svg)

    def _sld_node_colors(self) -> dict:
        """Couleur du nœud électrique de chaque branche, **telle qu'utilisée par le
        SLD** (``topological_coloring``). État réseau supposé **déjà appliqué**.

        Le SLD définit une palette ``.sld-vlXtoY.sld-bus-N {--sld-vl-color: #hex}``
        (par classe de tension et indice de nœud) ; chaque élément porte les classes
        ``sld-vl… sld-bus-N``. Toutes les branches d'un même nœud électrique
        partagent la même ``sld-bus-N`` → même couleur."""
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR).svg
        return _parse_node_colors(svg)

    def _branch_colors(self, branch_ids) -> dict:
        """Couleur SLD (topological) résolue **par equipment_id**, dans l'état
        réseau **déjà appliqué**."""
        ncolors = self._sld_node_colors()

        def _color(eq):
            if eq in ncolors:
                return ncolors[eq]
            for core, c in ncolors.items():   # id interne de ligne contient l'eq
                if eq in core:
                    return c
            return None
        return {eq: _color(eq) for eq in branch_ids}

    def _branch_isolated(self, G: nx.Graph) -> list[str]:
        """Départs **déconnectés** : équipements dont la composante connexe (en ne
        suivant que les switches **fermés**) ne contient **aucune barre**. Ce ne
        sont pas des nœuds électriques — l'IHM les présente en liste compacte.

        (En NODE_BREAKER, la connectivité vient des switches : se baser sur la
        composante, pas sur les drapeaux ``connected`` de pypowsybl.)"""
        return _isolated_assets(G)

    def nodale_payload(self, state: dict[str, bool]) -> dict:
        """Partition nodale d'un état + métadonnées d'affichage des branches :
        ``{groups, labels, types, flows, dirs, order, colors, isolated}``.

        - ``labels`` : libellé court **identique au SLD** (cf. ``_sld_feeder_meta``) ;
        - ``flows``  : flux actif (MW) au terminal de la branche dans cet état ;
        - ``dirs``   : ``TOP``/``BOTTOM`` (côté du départ dans la vue détaillée) ;
        - ``order``  : abscisse SLD (ordre gauche → droite) ;
        - ``colors`` : couleur SLD du nœud électrique de la branche (topological) ;
        - ``isolated``: départs déconnectés (présentés en liste, non comme nœuds)."""
        with self.applied(state):
            G = self._graph(state)
            topo = self._topo(state)
            fmeta = self._sld_feeder_meta()
            ncolors = self._sld_node_colors()

            def _resolve(eq: str):
                if eq in fmeta:
                    return fmeta[eq]
                for core, m in fmeta.items():   # repli : id + suffixe de côté (_TWO…)
                    if core == eq or core.startswith(eq + "_"):
                        return m
                return None

            def _color(eq: str):
                if eq in ncolors:
                    return ncolors[eq]
                for core, c in ncolors.items():   # id interne de ligne contient l'eq
                    if eq in core:
                        return c
                return None

            groups, labels, types, dirs, order, colors = [], {}, {}, {}, {}, {}
            for noeud in topo.noeuds.values():
                grp = sorted(noeud.equipment_ids)
                groups.append(grp)
                for dep in noeud.departs:
                    eq = dep.equipment_id
                    m = _resolve(eq)
                    labels[eq] = (m["label"] if m and m["label"]
                                  else self._short_name(eq))
                    dirs[eq] = m["dir"] if m else "BOTTOM"
                    order[eq] = m["x"] if m else 0.0
                    types[eq] = dep.equipment_type.name if dep.equipment_type else None
                    colors[eq] = _color(eq)
            isolated = self._branch_isolated(G)
            flows = self._flows(state, types)
        return {"groups": groups, "labels": labels, "types": types,
                "flows": flows, "dirs": dirs, "order": order, "colors": colors,
                "isolated": isolated}

    def nodale_state(self, state: dict[str, bool]) -> dict:
        """Vue nodale **légère** d'un état détaillé : ``{groups, colors, isolated}``
        (partition + couleurs SLD topologiques + ouvrages déconnectés), **sans**
        recalcul de flux. Sert à resynchroniser le volet nodal cible lorsque la
        topologie **détaillée** est éditée (bascule d'organes) ou recalculée."""
        with self.applied(state):
            G = self._graph(state)
            topo = self._topo(state)
            groups = [sorted(n.equipment_ids) for n in topo.noeuds.values()]
            isolated = self._branch_isolated(G)
            colors = self._branch_colors([eq for g in groups for eq in g])
        return {"groups": groups, "colors": colors, "isolated": isolated}

    def nodale_to_detaillee(self, groups, isolated=None) -> dict:
        """Pont **nodal → détaillé** : calcule une topologie détaillée d'intérêt
        réalisant la partition nodale cible ``groups`` (éditée par l'expert) et la
        charge comme **cible détaillée** courante (volet du bas).

        ``isolated`` liste les départs à **laisser déconnectés** (hors partition
        cible : non placés sur un nœud ; ils conservent leur état de départ).

        Renvoie la vue détaillée mise à jour + un statut de réalisabilité
        (dégradation gracieuse de l'algorithme remontée à l'IHM)."""
        # Poste à l'état de départ.
        self.apply(self.initial)
        poste = PosteTopologique.from_graph(self._graph(self.initial), self.vl)

        iso = set(isolated or [])
        univers = [eq for grp in self.groups_of(self.initial)
                   for eq in grp if eq not in iso]
        groups = _normalize_groups(
            univers, [[e for e in g if e not in iso] for g in (groups or [])])
        topo_cible = TopologieNodale.from_node_groups(self.vl, groups)

        res = determiner_topo_complete_cible(poste, topo_cible)

        # État détaillé final = rejeu des manœuvres depuis l'état de départ.
        manos = [{"switch_id": m.switch_id, "action": m.action}
                 for m in res.manoeuvres]
        self.current = _replay_states(self.initial, manos)[-1]
        self.scenario_name = None   # cible à revalider avant calcul de séquence

        svg, switches, nb = self.view(self.current)
        return {
            "svg": svg, "switches": switches, "nb_noeuds": nb,
            "is_verified": res.is_verified,
            "message": res.message,
            "ecarts": res.ecarts,
            "noeuds_non_realisables": res.noeuds_non_realisables,
            "nb_obtenu": res.topo_obtenue.nb_noeuds if res.topo_obtenue else nb,
            "nb_vise": topo_cible.nb_noeuds,
            # Vue nodale **réalisée** (partition + couleurs + isolés) pour
            # resynchroniser le volet nodal cible avec le détail obtenu.
            "nodale": self.nodale_state(self.current),
        }

    def save_scenario(self, name: str) -> str:
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
        poste = PosteTopologique.from_graph(self._graph(self.initial), self.vl)
        # Topologie détaillée cible (B) imposée : on vise la barre exacte de
        # chaque départ, pas seulement la partition nodale.
        self.apply(self.current)
        cible_graph = self._graph(self.current)

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
                   svg=svg_c, switches=sw, nb_noeuds=nb_c,
                   nodale_depart=SESSION.nodale_payload(SESSION.initial),
                   nodale_cible=SESSION.nodale_state(SESSION.current))


@app.post("/api/toggle")
def api_toggle():
    SESSION.toggle(request.json["id"])
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb,
                   nodale=SESSION.nodale_state(SESSION.current))


@app.post("/api/reset")
def api_reset():
    SESSION.reset()
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb,
                   nodale=SESSION.nodale_state(SESSION.current))


@app.post("/api/cible")
def api_cible():
    """Vue détaillée **cible courante** (sans la modifier) + vue nodale — pour
    revenir en édition de la cible alors qu'une séquence est déjà calculée."""
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb,
                   nodale=SESSION.nodale_state(SESSION.current))


@app.post("/api/nodale")
def api_nodale():
    """Partitions nodales de départ et cible (cible initialisée = départ)."""
    nodale = SESSION.nodale_payload(SESSION.initial)
    return jsonify(nodale_depart=nodale, nodale_cible=nodale)


@app.post("/api/nodale_to_detaillee")
def api_nodale_to_detaillee():
    """Calcule la topologie détaillée d'intérêt réalisant la cible nodale éditée
    et la charge comme cible détaillée courante (volet du bas)."""
    return jsonify(SESSION.nodale_to_detaillee(
        request.json.get("groups", []), request.json.get("isolated", [])))


@app.get("/api/scenarios")
def api_scenarios():
    return jsonify(scenarios=SESSION.list_scenarios())


def _safe_name(name, default):
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
                   svg=svg_c, switches=sw, nb_noeuds=nb_c, vl=SESSION.vl,
                   nodale_depart=SESSION.nodale_payload(SESSION.initial),
                   nodale_cible=SESSION.nodale_state(SESSION.current))


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
 #seq .violmsg{color:#b91c1c;background:#fef2f2;border-left:3px solid #dc2626;
   padding:2px 6px 4px 24px;white-space:normal;font-size:11px;cursor:pointer}
 #seqstale{background:#fef2f2;border:1px solid #fca5a5;color:#b91c1c;font-size:12px;padding:6px 8px;border-radius:5px;margin-bottom:6px}
 #seq.stale{opacity:.5}
 #bedit.primary{background:#7c3aed;color:#fff;border-color:#7c3aed}
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
 /* Séparateur déplaçable entre le schéma détaillé (col. 2) et le volet nodal */
 #ndresize{flex:0 0 auto;width:6px;cursor:col-resize;background:#d1d5db}
 #ndresize:hover,#ndresize.drag{background:#6366f1}
 /* Volet nodal (3e colonne) */
 #nodal{width:330px;background:#f4f5f7;border-left:1px solid #ccc;display:flex;flex-direction:column;overflow:hidden}
 #nodal.collapsed{width:30px}
 #nodal.collapsed .nbody{display:none}
 #nodal .nhead{font-size:12px;font-weight:bold;padding:6px 8px;background:#ecfdf5;border-bottom:1px solid #ccc;display:flex;align-items:center;gap:6px}
 #nodal .nsec{display:flex;flex-direction:column;min-height:0;border-bottom:2px solid #d1fae5}
 #nodal .nsec .stitle{font-size:11px;font-weight:bold;padding:4px 8px;color:#065f46;display:flex;justify-content:space-between}
 #nodal .nbody{flex:1;overflow:auto;padding:4px}
 #nodal .ntools{padding:4px 8px;display:flex;flex-wrap:wrap;gap:4px;background:#fff;border-bottom:1px solid #e5e7eb}
 #nodal .ntools button{padding:3px 7px;font-size:11px;margin:0}
 /* Vue nodale : SVG « nœud + branches rayonnantes » */
 svg.nodalsvg{display:block;background:#fff}
 .nbranch line{stroke:#94a3b8;stroke-width:1.5}
 .nbranch .blabel{font:10px monospace;fill:#334155}
 .nbranch .bflow{font:9px sans-serif;fill:#0f766e;font-weight:bold}
 .nbranch.ed{cursor:grab}
 .nbranch.ed:hover line{stroke:#6366f1}
 .nbranch.sel line{stroke:#4f46e5;stroke-width:3}
 .nbranch.sel .blabel{fill:#4338ca;font-weight:bold}
 .nbushit{fill:transparent}
 .nbusbar{stroke-width:6;stroke-linecap:round}
 .nbus.ed .nbusbar,.nbus.ed .nbadge{cursor:grab}
 .nbadge text{font:bold 11px sans-serif;text-anchor:middle}
 .nbus.droptarget .nbushit{fill:rgba(99,102,241,.13)}
 .nbus.droptarget .nbusbar{stroke-dasharray:5 3}
 body.ndndrag,body.ndndrag *{cursor:grabbing !important}
 /* Ouvrages isolés (déconnectés) : liste compacte, pas de nœud électrique */
 .nodiso{flex:0 0 auto;padding:3px 8px 6px;border-top:1px dashed #e2e8f0;background:#fffdf7}
 .nodiso .isohd{font-size:10px;color:#b45309;font-weight:bold;margin:2px 0}
 .nodiso.ro{background:#f8fafc}
 .nodiso.ro .isohd{color:#64748b}
 .isochips{display:flex;flex-wrap:wrap;gap:4px}
 .isochip{font-family:monospace;font-size:10px;padding:2px 7px;border:1px dashed #f59e0b;border-radius:10px;background:#fffbeb;color:#92400e;cursor:grab;user-select:none}
 .nodiso.ro .isochip{cursor:default;border-color:#cbd5e1;background:#f1f5f9;color:#475569}
 .isochip.sel{background:#6366f1;color:#fff;border-color:#4f46e5;border-style:solid}
 #nodstatus{font-size:11px;padding:4px 8px;white-space:normal}
 #nodstatus.okv{color:#065f46;background:#ecfdf5}
 #nodstatus.kov{color:#92400e;background:#fffbeb}
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
  <div style="font-size:11px;color:#555">Clic sur un organe du schéma pour basculer son état (départ ➜ cible).</div>
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
    <button id="bedit" onclick="toggleEditTarget()" title="Revenir éditer la cible détaillée (la séquence calculée deviendra obsolète)">✎ Modifier la cible</button>
    <span id="stepinfo" class="badge"></span>
    <span id="steplabel" style="font-family:monospace;font-size:12px"></span>
    <span style="flex:1"></span>
    <span id="animhint" style="font-size:11px;color:#3730a3">✎ Schéma éditable : cliquez un organe pour insérer une manœuvre après l'étape courante.</span>
  </div>
  <div id="seqwrap" style="padding:8px;display:none">
    <div id="seqstale" style="display:none">⚠ La cible détaillée a été modifiée : la séquence ci-dessous <b>n'atteint plus</b> cet état cible. Re-validez puis <b>recalculez la séquence</b>.</div>
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
<div id="ndresize" title="Glisser pour élargir / réduire le volet nodal"></div>
<div id="nodal">
  <div class="nhead"><button class="cbtn" onclick="toggleNodal()" title="Réduire / agrandir">◂</button>
    Topologie nodale</div>
  <div class="nbody" style="display:flex;flex-direction:column;flex:1;min-height:0;overflow:hidden">
    <div class="nsec" style="flex:1">
      <div class="stitle"><span>Départ</span><span class="badge" id="ndDepN">–</span></div>
      <div class="nbody" id="ndDepart"></div>
      <div class="nodiso ro" id="ndDepartIso" style="display:none"></div>
    </div>
    <div class="nsec" style="flex:2">
      <div class="stitle"><span>Cible (éditable)</span><span class="badge" id="ndCibN">–</span></div>
      <div class="ntools">
        <button onclick="nodNewNode()" title="Créer un nœud vide (puis y glisser des départs ; ou avec la sélection courante)">＋ Nœud</button>
        <button onclick="nodReset()" title="Réinitialiser la cible nodale = départ">= départ</button>
        <button onclick="nodClearSel()" title="Vider la sélection">∅ Désélectionner</button>
      </div>
      <div style="font-size:10px;color:#64748b;padding:0 8px">Glisser un <b>départ</b> (ou une sélection : clic pour (dé)sélectionner) sur une autre barre = réaiguillage. Glisser une <b>barre</b> sur une autre = fusion. Flux en MW (état de départ).</div>
      <div class="nbody" id="ndCible"></div>
      <div class="nodiso ed" id="ndCibleIso" style="display:none"></div>
    </div>
    <button class="primary" style="margin:6px 8px" onclick="nodCompute()">⚙ Calculer la topologie détaillée d'intérêt</button>
    <div id="nodstatus"></div>
  </div>
</div>
<script>
let S={n:0,idx:0,timer:null,labels:[],algo:{},sel:new Set(),lastSel:null,manual:false,
  editTarget:false,stale:false};
// État de l'éditeur de topologie nodale (volet de droite).
let NOD={depart:{groups:[]},departIso:[],labels:{},types:{},flows:{},dirs:{},order:{},
  colors:{},colorsCible:{},groups:[],isolated:[],selBranches:new Set(),selNodes:new Set()};
const NODE_COLORS=['#16a34a','#2563eb','#15803d','#9333ea','#db2777','#d97706','#0891b2','#65a30d','#dc2626','#0d9488'];
const api=(p,b)=>fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json());
function setValidated(v){document.getElementById('bcalc').disabled=!v;
  document.getElementById('bmanual').disabled=!v;
  document.getElementById('calchint').style.display=v?'none':'block';}
async function init(){const r=await (await fetch('/api/postes')).json();const sel=document.getElementById('poste');
  r.postes.forEach(p=>{const o=document.createElement('option');o.value=p;o.text=p;sel.add(o);});
  sel.onchange=()=>load(sel.value); initNodalResize(); initNodalDnD(); await refreshScenarios(); if(r.postes.length) load(r.postes[0]);}
async function load(vl){stopAnim();const d=await api('/api/load',{vl});show(d);initNodale(d);hideSeq();setValidated(false);
  document.getElementById('scenName').value=vl+'_cible';}
async function reset(){stopAnim();const d=await api('/api/reset',{});show(d);
  syncNodalCible(d.nodale);hideSeq();setValidated(false);}
async function toggle(id){stopAnim();const d=await api('/api/toggle',{id});show(d);
  syncNodalCible(d.nodale);setValidated(false);
  // Si une séquence existe déjà, l'éditer de la cible la rend obsolète (au lieu
  // de masquer la séquence) : on la conserve affichée mais signalée.
  if(S.n>0){markSeqStale();}else{hideSeq();}}
// Bascule : revenir éditer la cible détaillée alors qu'une séquence est calculée.
async function toggleEditTarget(){
  S.editTarget=!S.editTarget;
  const b=document.getElementById('bedit'),hint=document.getElementById('animhint');
  if(S.editTarget){
    stopAnim();
    const d=await api('/api/cible',{});
    document.getElementById('diagBottom').innerHTML=d.svg;
    document.getElementById('nbB').textContent=d.nb_noeuds;
    document.getElementById('nbn').textContent=d.nb_noeuds;
    bind(d.switches);syncNodalCible(d.nodale);
    document.getElementById('paneBot').classList.remove('reached');
    b.textContent='▶ Revenir à la séquence';b.classList.add('primary');
    ['bprev','bplay','bnext'].forEach(id=>document.getElementById(id).disabled=true);
    hint.textContent='✎ Édition de la cible détaillée : cliquez un organe pour la modifier (la séquence deviendra obsolète).';
  }else{
    b.textContent='✎ Modifier la cible';b.classList.remove('primary');
    document.getElementById('bplay').disabled=false;
    hint.textContent="✎ Schéma éditable : cliquez un organe pour insérer une manœuvre après l'étape courante.";
    await showStep(S.idx);
  }}
function markSeqStale(){S.stale=true;
  const el=document.getElementById('seqstale');if(el)el.style.display='block';
  const seq=document.getElementById('seq');if(seq)seq.classList.add('stale');}
function clearSeqStale(){S.stale=false;
  const el=document.getElementById('seqstale');if(el)el.style.display='none';
  const seq=document.getElementById('seq');if(seq)seq.classList.remove('stale');}
function resetEditTarget(){S.editTarget=false;
  const b=document.getElementById('bedit');
  if(b){b.textContent='✎ Modifier la cible';b.classList.remove('primary');}}
// Resynchronise le volet nodal CIBLE depuis l'état détaillé courant
// ({groups, colors, isolated}) : la topologie détaillée fait foi.
function syncNodalCible(n){if(!n)return;
  NOD.isolated=(n.isolated||[]).slice();
  NOD.groups=dropIso(n.groups||[],NOD.isolated);
  NOD.colorsCible=Object.assign({},NOD.colors,n.colors||{});
  NOD.selBranches=new Set();renderNodaleCible();}
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
  document.getElementById('poste').value=d.vl;show(d);initNodale(d);hideSeq();
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
  document.getElementById('paneBot').classList.remove('reached');
  clearSeqStale();resetEditTarget();}
function show(d){
  document.getElementById('paneBot').classList.remove('reached');
  if(d.initial_svg!==undefined){document.getElementById('diagTop').innerHTML=d.initial_svg;
    document.getElementById('nbA').textContent=d.nb_initial;
    document.getElementById('ttlA').textContent='Topologie de départ';S.manual=false;}
  document.getElementById('diagBottom').innerHTML=d.svg;
  document.getElementById('nbB').textContent=d.nb_noeuds;
  document.getElementById('nbn').textContent=d.nb_noeuds;
  bind(d.switches);}
function bind(switches){const root=document.getElementById('diagBottom');
  switches.forEach(s=>{const el=root.querySelector('[id="'+s.svgId+'"]');
    if(el){el.style.cursor='pointer';el.onclick=()=>toggle(s.id);
      el.querySelectorAll('*').forEach(c=>c.style.cursor='pointer');}});}
async function sequence(){stopAnim();S.manual=false;resetEditTarget();clearSeqStale();
  const d=await api('/api/sequence',{mode:document.getElementById('seqMode').value});
  document.getElementById('seqwrap').style.display='block';
  S.algo={message:d.message,ecarts:d.ecarts||[],verified:d.verified,verified_detaillee:d.verified_detaillee,mode:d.mode};
  document.getElementById('seqName').value=(document.getElementById('scenName').value||'sequence');
  document.getElementById('seqsavemsg').textContent='';
  renderSeq(d);
  document.getElementById('anim').style.display='flex';
  await showStep(0);}
async function manualSeq(){stopAnim();S.manual=true;resetEditTarget();clearSeqStale();
  const d=await api('/api/manual_start',{});
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
    ln.onclick=()=>stepGoto(k);seq.appendChild(ln);
    if(viol){const vm=document.createElement('div');vm.className='violmsg';vm.id='viol'+k;
      vm.textContent='⚠ '+viol;vm.onclick=()=>stepGoto(k);seq.appendChild(vm);}});
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
function toggleNodal(){const p=document.getElementById('nodal');
  if(p.classList.toggle('collapsed')){p.dataset.w=p.style.width||'';p.style.width='';}
  else if(p.dataset.w){p.style.width=p.dataset.w;}
  const b=p.querySelector('.cbtn');if(b)b.textContent=p.classList.contains('collapsed')?'▸':'◂';}
function initNodalResize(){const rz=document.getElementById('ndresize');
  const nod=document.getElementById('nodal');let drag=false;
  rz.addEventListener('mousedown',e=>{if(nod.classList.contains('collapsed'))return;
    drag=true;rz.classList.add('drag');document.body.style.userSelect='none';e.preventDefault();});
  window.addEventListener('mousemove',e=>{if(!drag)return;
    const w=Math.max(220,Math.min(window.innerWidth-360,window.innerWidth-e.clientX));
    nod.style.width=w+'px';});
  window.addEventListener('mouseup',()=>{if(drag){drag=false;rz.classList.remove('drag');
    document.body.style.userSelect='';}});}

// ---- Éditeur de topologie nodale (volet de droite) ----
function cloneGroups(gs){return gs.map(g=>g.slice());}
// Retire les ouvrages isolés des groupes et supprime les groupes vides : un isolé
// n'est pas un nœud électrique (il est listé à part).
function dropIso(groups,iso){const s=new Set(iso||[]);
  return groups.map(g=>g.filter(eq=>!s.has(eq))).filter(g=>g.length);}
function initNodale(d){
  if(!d||!d.nodale_depart)return;
  const dep=d.nodale_depart, cib=d.nodale_cible||dep;
  // Champs partagés (identité SLD, flux de référence départ) : depuis le départ.
  NOD.labels=dep.labels||{};NOD.types=dep.types||{};NOD.flows=dep.flows||{};
  NOD.dirs=dep.dirs||{};NOD.order=dep.order||{};NOD.colors=dep.colors||{};
  NOD.departIso=(dep.isolated||[]).slice();
  NOD.depart={groups:dropIso(dep.groups||[],NOD.departIso)};
  // Cible : partition / couleurs / isolés de l'état détaillé courant (cib).
  NOD.isolated=(cib.isolated||[]).slice();
  NOD.groups=dropIso(cloneGroups(cib.groups||[]),NOD.isolated);
  NOD.colorsCible=Object.assign({},NOD.colors,cib.colors||{});
  NOD.selBranches=new Set();NOD.selNodes=new Set();
  document.getElementById('nodstatus').textContent='';
  document.getElementById('nodstatus').className='';
  renderNodaleDepart();renderNodaleCible();}
function nodReset(){NOD.groups=cloneGroups(NOD.depart.groups||[]);
  NOD.isolated=(NOD.departIso||[]).slice();
  NOD.colorsCible=Object.assign({},NOD.colors);  // recouleurs = départ
  NOD.selBranches=new Set();NOD.selNodes=new Set();renderNodaleCible();}
function nodNormalize(){NOD.groups=NOD.groups.filter(g=>g.length>0);}
function nodeColor(i){return NODE_COLORS[i%NODE_COLORS.length];}
// Couleur de la barre = couleur SLD du nœud (via une branche), sinon palette de repli.
function nodeFill(g,ni,colors){colors=colors||{};
  for(const eq of g){if(colors[eq])return colors[eq];}
  return nodeColor(ni);}
// Texte noir ou blanc selon la luminance du fond (lisibilité sur pastels).
function textOn(hex){const m=/^#?([0-9a-f]{6})$/i.exec(hex||'');if(!m)return '#fff';
  const n=parseInt(m[1],16),r=(n>>16)&255,gg=(n>>8)&255,b=n&255;
  return (0.299*r+0.587*gg+0.114*b)>150?'#111':'#fff';}
function xesc(s){return (s+'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmtFlow(v){if(v==null||isNaN(v))return '';const r=Math.round(v);return (r>0?'+':'')+r;}
function labLen(eqs){let m=3;eqs.forEach(eq=>{const l=(NOD.labels[eq]||eq).length;
  if(l>m)m=l;});return m;}
function isTop(eq){return (NOD.dirs[eq]||'BOTTOM')==='TOP';}
function byOrder(a,b){return (NOD.order[a]||0)-(NOD.order[b]||0);}
function allByDir(top){const out=[];
  [NOD.groups,(NOD.depart.groups||[])].forEach(gs=>gs.forEach(g=>g.forEach(eq=>{
    if(isTop(eq)===top&&out.indexOf(eq)<0)out.push(eq);})));return out;}
// Rendu « bus par nœud » : chaque nœud = une barre horizontale (couleur SLD) ;
// ses départs sont des branches VERTICALES (haut au-dessus, bas en dessous, comme
// la vue détaillée), triées gauche → droite par abscisse SLD, avec libellé (identique
// au SLD) et flux. Les nœuds sont empilés verticalement.
function buildNodaleSVG(groups, editable, colors){
  const BW=30, BUSH=6, STUB=24, PADX=14, PADY=10, GAPY=22, CW=6.1, BADGE=22;
  const topH=labLen(allByDir(true))*CW+6, botH=labLen(allByDir(false))*CW+6;
  const place=(x0,len,j,cnt)=>cnt>1?x0+(j*(len-BW)/(cnt-1))+BW/2:x0+len/2;
  let y=PADY, maxX=0; const segs=[];
  groups.forEach((g,ni)=>{
    const top=g.filter(isTop).sort(byOrder);
    const bot=g.filter(eq=>!isTop(eq)).sort(byOrder);
    const cols=Math.max(top.length,bot.length,1), busLen=Math.max(46,cols*BW);
    const x0=PADX+BADGE+6, x1=x0+busLen, busY=y+topH+STUB+BUSH/2;
    const blockH=topH+STUB+BUSH+STUB+botH;
    const col=nodeFill(g,ni,colors), tcol=textOn(col);
    let blk=`<g class="nbus${editable?' ed':''}" data-node="${ni}">`;
    blk+=`<rect class="nbushit" x="${(PADX-4).toFixed(1)}" y="${(y-PADY/2).toFixed(1)}" `
       +`width="${(x1-PADX+10).toFixed(1)}" height="${(blockH+PADY).toFixed(1)}"/>`;
    blk+=`<line class="nbusbar" x1="${x0.toFixed(1)}" y1="${busY.toFixed(1)}" `
       +`x2="${x1.toFixed(1)}" y2="${busY.toFixed(1)}" stroke="${col}"/>`;
    blk+=`<g class="nbadge"><rect x="${PADX}" y="${(busY-BADGE/2).toFixed(1)}" rx="4" `
       +`width="${BADGE}" height="${BADGE}" fill="${col}"/>`
       +`<text x="${(PADX+BADGE/2).toFixed(1)}" y="${(busY+3.8).toFixed(1)}" fill="${tcol}">N${ni}</text></g>`;
    blk+=`<title>N${ni} · ${g.length} branche(s)</title>`;
    top.forEach((eq,j)=>blk+=branchSeg(eq,place(x0,busLen,j,top.length),
      busY-BUSH/2,busY-BUSH/2-STUB,true,editable));
    bot.forEach((eq,j)=>blk+=branchSeg(eq,place(x0,busLen,j,bot.length),
      busY+BUSH/2,busY+BUSH/2+STUB,false,editable));
    blk+=`</g>`;
    segs.push(blk); maxX=Math.max(maxX,x1); y+=blockH+GAPY;});
  const width=Math.max(maxX+PADX,80), height=Math.max(y,40);
  return `<svg class="nodalsvg" width="${width}" height="${height}" `
    +`viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${segs.join('')}</svg>`;
}
// Une branche verticale sur la barre : tronc bus → extrémité (tip), tick, flux près
// du bus, libellé pivoté à l'extrémité (vers le haut/bas selon le côté).
function branchSeg(eq,bx,y0,tipY,top,editable){
  const sel=editable&&NOD.selBranches.has(eq);
  const lbl=NOD.labels[eq]||eq, fl=fmtFlow(NOD.flows[eq]);
  let s=`<g class="nbranch${editable?' ed':''}${sel?' sel':''}" data-br="${xesc(eq)}">`;
  s+=`<line x1="${bx.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${tipY.toFixed(1)}"/>`;
  s+=`<line x1="${(bx-4).toFixed(1)}" y1="${tipY.toFixed(1)}" x2="${(bx+4).toFixed(1)}" y2="${tipY.toFixed(1)}"/>`;
  if(fl!==''){const fy=top?y0-3:y0+10;
    s+=`<text class="bflow" x="${(bx+4).toFixed(1)}" y="${fy.toFixed(1)}">${fl}</text>`;}
  const ly=top?tipY-4:tipY+4, rot=top?-90:90;
  s+=`<text class="blabel" x="${(bx+3).toFixed(1)}" y="${ly.toFixed(1)}" `
    +`transform="rotate(${rot} ${(bx+3).toFixed(1)} ${ly.toFixed(1)})">${xesc(lbl)}</text>`;
  s+=`<title>${xesc(eq)} · ${xesc(NOD.types[eq]||'')}${fl!==''?' · '+fl+' MW':''}</title></g>`;
  return s;
}
function renderNodaleSVG(rootId,groups,editable,colors){
  const root=document.getElementById(rootId);
  root.innerHTML=buildNodaleSVG(groups,editable,colors);
  if(editable)attachNodalDnD(root);
}
function renderNodaleDepart(){const gs=NOD.depart.groups||[];
  document.getElementById('ndDepN').textContent=gs.length;
  renderNodaleSVG('ndDepart',gs,false,NOD.colors);
  renderIso('ndDepartIso',NOD.departIso,false);}
function renderNodaleCible(){nodNormalize();
  document.getElementById('ndCibN').textContent=NOD.groups.length;
  renderNodaleSVG('ndCible',NOD.groups,true,NOD.colorsCible);
  renderIso('ndCibleIso',NOD.isolated,true);}
// Liste compacte des ouvrages isolés (déconnectés). Éditable : chips glissables
// sur un nœud pour reconnecter (clic = (dé)sélection pour glisser un lot).
function renderIso(rootId,list,editable){
  const root=document.getElementById(rootId);
  if(!list||!list.length){root.style.display='none';root.innerHTML='';return;}
  root.style.display='block';
  let h=`<div class="isohd">⚠ Ouvrages isolés (${list.length})`
    +(editable?' — glisser sur un nœud pour reconnecter':' — déconnectés')+`</div><div class="isochips">`;
  list.slice().sort().forEach(eq=>{const sel=editable&&NOD.selBranches.has(eq);
    h+=`<span class="isochip${sel?' sel':''}" data-br="${xesc(eq)}" `
      +`title="${xesc(eq)}${NOD.flows[eq]!=null?' · '+fmtFlow(NOD.flows[eq])+' MW':''}">`
      +`${xesc(NOD.labels[eq]||eq)}</span>`;});
  h+=`</div>`; root.innerHTML=h;
  if(editable)root.querySelectorAll('.isochip').forEach(c=>c.addEventListener('mousedown',e=>{
    e.stopPropagation();ndStart(e,{type:'feeder',eq:c.getAttribute('data-br')});}));
}
// --- Drag & drop : départ(s) sur un nœud = réaiguillage ; nœud sur nœud = fusion ---
function attachNodalDnD(root){
  root.querySelectorAll('.nbranch').forEach(g=>g.addEventListener('mousedown',e=>{
    e.stopPropagation();ndStart(e,{type:'feeder',eq:g.getAttribute('data-br')});}));
  root.querySelectorAll('.nbus').forEach(g=>g.addEventListener('mousedown',e=>{
    ndStart(e,{type:'node',node:+g.getAttribute('data-node')});}));
}
function ndStart(e,info){e.preventDefault();
  NOD.dnd=Object.assign({x0:e.clientX,y0:e.clientY,moved:false},info);}
function ndNodeUnder(e){const el=document.elementFromPoint(e.clientX,e.clientY);
  const g=el&&el.closest?el.closest('#ndCible .nbus'):null;
  return g?+g.getAttribute('data-node'):null;}
function ndMove(e){const d=NOD.dnd;if(!d)return;
  if(!d.moved&&Math.hypot(e.clientX-d.x0,e.clientY-d.y0)<5)return;
  d.moved=true;document.body.classList.add('ndndrag');
  const tgt=ndNodeUnder(e);
  document.querySelectorAll('#ndCible .nbus').forEach(g=>
    g.classList.toggle('droptarget',tgt!=null&&+g.getAttribute('data-node')===tgt));
}
function ndUp(e){const d=NOD.dnd;NOD.dnd=null;
  document.body.classList.remove('ndndrag');
  document.querySelectorAll('.nbus.droptarget').forEach(g=>g.classList.remove('droptarget'));
  if(!d)return;
  if(!d.moved){   // simple clic = (dé)sélection d'un départ
    if(d.type==='feeder'){if(NOD.selBranches.has(d.eq))NOD.selBranches.delete(d.eq);
      else NOD.selBranches.add(d.eq);renderNodaleCible();}
    return;}
  const tgt=ndNodeUnder(e);
  if(tgt==null)return;
  if(d.type==='node'){if(tgt!==d.node)nodMergeNodes(d.node,tgt);}
  else{const items=NOD.selBranches.has(d.eq)?[...NOD.selBranches]:[d.eq];
    nodMoveBranchesTo(items,tgt);}
}
function initNodalDnD(){window.addEventListener('mousemove',ndMove);
  window.addEventListener('mouseup',ndUp);}
function nodMoveSelTo(target){nodMoveBranchesTo([...NOD.selBranches],target);}
function nodMoveBranchesTo(eqs,target){
  if(!eqs.length||target<0||target>=NOD.groups.length)return;
  NOD.isolated=NOD.isolated.filter(eq=>eqs.indexOf(eq)<0);   // reconnecte les isolés
  NOD.groups=NOD.groups.map(g=>g.filter(eq=>eqs.indexOf(eq)<0));
  eqs.forEach(eq=>{if(!NOD.groups[target].includes(eq))NOD.groups[target].push(eq);});
  NOD.selBranches=new Set();renderNodaleCible();}
function nodMergeNodes(src,dst){
  if(src===dst||src<0||dst<0||src>=NOD.groups.length||dst>=NOD.groups.length)return;
  NOD.groups[dst]=NOD.groups[dst].concat(
    NOD.groups[src].filter(eq=>NOD.groups[dst].indexOf(eq)<0));
  NOD.groups[src]=[];NOD.selBranches=new Set();renderNodaleCible();}
function nodNewNode(){NOD.groups.push([]);const idx=NOD.groups.length-1;
  if(NOD.selBranches.size)nodMoveSelTo(idx);else renderNodaleCible();}
function nodClearSel(){NOD.selBranches=new Set();renderNodaleCible();}
async function nodCompute(){
  const st=document.getElementById('nodstatus');st.className='';st.textContent='Calcul en cours…';
  const d=await api('/api/nodale_to_detaillee',{groups:NOD.groups,isolated:NOD.isolated});
  stopAnim();S.manual=false;
  // Charger la topologie détaillée d'intérêt comme cible (volet du bas).
  document.getElementById('diagBottom').innerHTML=d.svg;
  document.getElementById('nbB').textContent=d.nb_noeuds;
  document.getElementById('nbn').textContent=d.nb_noeuds;
  bind(d.switches);hideSeq();setValidated(false);
  // Resynchroniser la cible nodale sur la topologie détaillée RÉALISÉE
  // (partition + couleurs + isolés des nœuds obtenus).
  syncNodalCible(d.nodale);
  let msg='';
  if(d.is_verified){st.className='okv';msg='✓ Topologie détaillée d\'intérêt chargée comme cible ('+d.nb_obtenu+' nœud(s)). Validez puis calculez la séquence.';}
  else{st.className='kov';msg='⚠ Cible partiellement réalisable (obtenu '+d.nb_obtenu+' / visé '+d.nb_vise+' nœud(s)). '+(d.message||'');
    if(d.noeuds_non_realisables&&d.noeuds_non_realisables.length)msg+=' Nœuds non réalisables : '+d.noeuds_non_realisables.map(g=>g.join(',')).join(' | ')+'.';}
  if(d.ecarts&&d.ecarts.length)msg+=' Écarts : '+d.ecarts.join(' ; ')+'.';
  st.textContent=msg;}
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
