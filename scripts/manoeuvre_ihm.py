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
   la topologie de départ à la cible. Les calculs passent par la **façade
   pluggable** (``manoeuvre.plugins.PlanificateurTopologie``, vérification
   indépendante) : l'**algorithme de chaque phase** (identification nodale →
   détaillée, séquencement) se choisit dans l'IHM parmi ceux du registre —
   natifs « libtopo » et plugins tiers (``GET/POST /api/algos``).
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
import os
import pathlib
import platform
import re
import subprocess
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
    sectionneurs_sous_charge_par_manoeuvre,
)
from expert_op4grid_recommender.manoeuvre.plugins import (
    CibleDetaillee,
    PlanificateurTopologie,
    disponibles as algos_disponibles,
)
from expert_op4grid_recommender.manoeuvre.dataset import source as dataset_source
from expert_op4grid_recommender.manoeuvre.dataset import geographie
from expert_op4grid_recommender.manoeuvre.dataset import exploration
from expert_op4grid_recommender.manoeuvre.dataset.dgitt import _charger_reseau

# Postes de test retenus (intersectés avec les VL réellement présents)
POSTES_TEST = [
    "CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5P3", "BXTO5P6",
    "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6",
    "CORNIP3", "GUARBP6", "MORBRP6",
    # Postes 400 kV à **3 jeux de barres** identifiés (réseau France 28/08/2024),
    # gérés par le placement N-barres + le réalisateur connectivité-based.
    "SSV.OP7", "TAVELP7", "TRI.PP7", "ARGOEP7", "CHESNP7", "COR.PP7", "CERGYP7",
]

# ── Catalogue par TYPOLOGIE de poste ────────────────────────────────────────
# Sections d'exploration : chaque poste (identifié par son VL réel dans le réseau
# France 28/08/2024) est rangé sous une ou plusieurs typologies, pour « s'y
# retrouver » parmi les milliers de VL. Tous ces postes disposent d'une **fixture
# de test** correspondante (``tests/manoeuvre/fixtures``). L'IHM affiche ces
# sections dans le sélecteur et marque **disponible** chaque poste dont le VL est
# présent dans la situation réseau chargée (rendu SLD = pypowsybl, donc requiert
# le bon réseau ; lancer l'IHM sur ``grid.xiidm`` France les rend tous accessibles).
# Un même poste peut figurer dans plusieurs sections (navigation par typologie).
POSTES_CATALOG: list[tuple[str, list[str]]] = [
    ("3 jeux de barres — 400 kV",
     ["SSV.OP7", "TAVELP7", "TRI.PP7", "ARGOEP7", "CHESNP7", "COR.PP7", "CERGYP7"]),
    ("≥ 5 jeux de barres",
     [".OBER 7", ".VANY 7", ".ZAND 7", "MUHLBP7"]),
    ("4 jeux de barres",
     [".LAUF 7", "CPNIEP6", "GUARBP6", "MORBRP6"]),
    ("Sectionnement extrême (SJB ≫ barres)",
     [".LAUF 7", "REICHP3", ".ZAND 7", "CARRIP6"]),
    ("Faisceau de couplage partagé",
     [".OBER 7", ".ZAND 7", "MUHLBP7", "P.GASP6"]),
    ("Organes internes 2 bornes (self / réactance)",
     ["CPNIEP6", ".ZAND 7"]),
    ("Omnibus / départs multiples",
     ["ROMAIP6", "CORNIP3", "GUARBP6", "RAN.PP6", "REICHP3"]),
    ("Départs déconnectés (nœuds 0-barre)",
     [".MUHL 6"]),
    ("Gros postes (beaucoup de départs)",
     ["P.GASP6", ".OBER 7", "REICHP3"]),
    ("Postes standards / multi-sections",
     ["CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5P3", "BXTO5P6",
      "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6"]),
]

SLD_PAR = ppn.SldParameters(topological_coloring=True)
SCEN_DIR = pathlib.Path("tests/manoeuvre/scenarios")    # redéfini dans main()
SEQ_DIR = pathlib.Path("tests/manoeuvre/sequences")     # redéfini dans main()

# Instantané des coordonnées de postes (résolu/persisté par « Explorer la
# journée »). Par défaut **dans le cache** (``DGITT_CACHE_DIR``) : ainsi, pointer
# ce cache vers le **stockage persistant HF** (``DGITT_CACHE_DIR=/data/dgitt``)
# fait survivre **et** les instantanés XIIDM **et** les coordonnées aux
# redémarrages — une seule variable. ``MANOEUVRE_GEO_SNAPSHOT`` force un autre
# chemin. Coordonnées via OpenStreetMap/Overpass, actif sauf
# ``MANOEUVRE_ENABLE_OSM=0`` (alias historique ``MANOEUVRE_ENABLE_ODRE``).
GEO_SNAPSHOT = pathlib.Path(os.environ.get(
    "MANOEUVRE_GEO_SNAPSHOT",
    str(pathlib.Path(os.environ.get("DGITT_CACHE_DIR", ".cache/dgitt"))
        / "postes_rte_geo.json")))
# Plan de masse RTE committé (par VL) : source **primaire** des coordonnées de la
# carte (hors-ligne, ~98 % des postes). ``MANOEUVRE_GEO_LAYOUT`` force un autre chemin.
GEO_LAYOUT = pathlib.Path(os.environ.get("MANOEUVRE_GEO_LAYOUT",
                                         geographie.LAYOUT_DEFAUT))
# Fond de carte (frontières) déjà projeté dans le repère du plan de masse.
GEO_BASEMAP = pathlib.Path(os.environ.get("MANOEUVRE_GEO_BASEMAP",
                                          geographie.BASEMAP_DEFAUT))
OSM_ENABLED = (os.environ.get("MANOEUVRE_ENABLE_OSM")
               or os.environ.get("MANOEUVRE_ENABLE_ODRE", "1")).lower() not in (
    "0", "false", "no", "off")


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


def _nb_noeuds_reels(G: nx.Graph) -> int:
    """Nombre de **nœuds électriques réels** : composantes connexes (switches
    fermés) contenant **au moins une barre**. Les ouvrages **isolés** (déconnectés)
    ne sont **pas** comptés comme des nœuds — pour l'**affichage** (le moteur de
    séquencement conserve ``TopologieNodale.nb_noeuds``, isolés inclus). Cohérent
    avec l'éditeur nodal qui présente les isolés à part. Fonction pure."""
    closed = nx.Graph()
    closed.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if not d.get("open", False):
            closed.add_edge(u, v)
    barres = set(busbar_nodes(G))
    return sum(1 for comp in nx.connected_components(closed) if comp & barres)


class Session:
    """État serveur (mono-utilisateur)."""

    def __init__(self, network):
        self.net = network
        self.vls = set(network.get_voltage_levels().index)
        # Postes « épinglés » (jeu de test + 3 JdB identifiés) présents dans le réseau.
        self.postes = [p for p in POSTES_TEST if p in self.vls]
        # **Tous** les postes inspectables = voltage levels en topologie
        # NODE_BREAKER (ceux qui possèdent des sections de jeux de barres). Permet
        # d'inspecter / tester n'importe quel poste de la situation chargée, pas
        # seulement la liste épinglée. Trié, postes épinglés en tête.
        try:
            bbs = network.get_busbar_sections(all_attributes=True)
            nb_vls = set(bbs["voltage_level_id"]) if "voltage_level_id" in bbs else set()
        except Exception:
            nb_vls = set()
        autres = sorted(nb_vls - set(self.postes))
        self.all_postes = self.postes + autres
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
        # Algorithme sélectionné pour chaque phase pluggable (cf.
        # ``manoeuvre.plugins`` et docs/architecture/plugins.md) :
        # - "identificateur" : nodale -> détaillée (« calculer la topologie
        #   détaillée d'intérêt ») ;
        # - "sequenceur"     : détaillée -> séquence de manœuvres ;
        # - "planificateur"  : bout-en-bout (repli de composition de la façade).
        # Les plugins tiers enregistrés (registre / entry points) apparaissent
        # automatiquement dans ``algos_disponibles()`` et l'IHM.
        self.algos: dict[str, str] = {p: "libtopo" for p in
                                      ("identificateur", "sequenceur",
                                       "planificateur")}
        # Caches mémoïsés par état détaillé (cf. _graph / _topo / _flows).
        # Le graphe NX, la topologie nodale et le résultat de load flow d'un VL
        # ne dépendent que du VL et de l'état des organes — purs vis-à-vis de
        # l'état appliqué. Invalidés au chargement d'un poste (load()).
        self._graph_cache: dict = {}
        self._topo_cache: dict = {}
        self._flow_cache: dict = {}

    # --- algorithmes pluggables (sélection par phase) -----------------------
    def set_algos(self, choix: dict[str, str]) -> dict[str, str]:
        """Met à jour la sélection d'algorithme par phase. Les phases inconnues
        et les noms absents du registre sont ignorés (la sélection courante est
        retournée, le front se resynchronise dessus)."""
        dispo = algos_disponibles()
        for phase, nom in (choix or {}).items():
            if phase in self.algos and nom in dispo.get(phase, []):
                self.algos[phase] = nom
        return self.algos

    def _pipe(self) -> PlanificateurTopologie:
        """Façade d'orchestration configurée avec les algorithmes sélectionnés
        (vérification indépendante active : les verdicts affichés par l'IHM ne
        dépendent pas des déclarations de l'algorithme branché)."""
        return PlanificateurTopologie(
            identificateur=self.algos["identificateur"],
            sequenceur=self.algos["sequenceur"],
            planificateur=self.algos["planificateur"],
        )

    def catalog(self) -> list[dict]:
        """Catalogue par **typologie** : sections de postes (cf. ``POSTES_CATALOG``)
        avec, pour chaque poste, sa **disponibilité** dans la situation chargée
        (``available`` = le VL existe → sélectionnable et rendu SLD possible)."""
        out = []
        for title, vls in POSTES_CATALOG:
            postes = [{"vl": v, "available": v in self.vls} for v in vls]
            n_dispo = sum(1 for p in postes if p["available"])
            out.append({"title": title, "postes": postes, "n_available": n_dispo})
        return out

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

    def load_states(self, vl, states: dict[str, bool]):
        """Charge un poste avec un **état de départ explicite** (au lieu du
        pristine du réseau de référence) : utilisé par l'exploration de journée
        pour afficher la topologie d'un poste **à une heure donnée** (minuit /
        midi / 23 h), sans recharger un réseau France entier — on applique les
        états de l'heure visée pour ce VL sur le réseau de référence.

        ``states`` couvre les organes du VL à l'heure visée ; les organes
        absents retombent sur l'état courant du réseau (pristine)."""
        self.vl = vl
        self._graph_cache.clear()
        self._topo_cache.clear()
        self._flow_cache.clear()
        df = self.switches_df(vl)
        self.initial = {sid: bool(states.get(sid, self.pristine.get(sid, False)))
                        for sid in df.index}
        self.current = dict(self.initial)
        self.apply(self.initial)   # le SLD du VL reflète l'heure visée
        self.seq_manoeuvres = []
        self.seq_states, self.seq_highlights, self.seq_labels = [], [], []
        self.seq_edited = False
        self.scenario_name = None

    def set_target_states(self, states: dict[str, bool]):
        """Définit la **cible courante** depuis un état d'organes (p. ex. la
        topologie du poste à une autre heure de la journée : « retenir cette
        topologie comme cible »). Les organes absents gardent l'état de départ."""
        self.current = {sid: bool(states.get(sid, self.initial[sid]))
                        for sid in self.initial}
        self.scenario_name = None

    def reset(self):
        self.current = dict(self.initial)
        self.scenario_name = None

    def promote_cible(self):
        """Promeut la cible courante (état détaillé édité) en **nouvel état de
        départ** : ``initial`` prend l'ancienne ``current`` (et ``current`` en
        repart à l'identique). Séquence et lien de scénario réinitialisés. Permet
        de **chaîner** depuis une topologie éditée sans passer par un scénario
        sauvegardé. (Le VL ne change pas : les caches par état restent valides.)"""
        self.initial = dict(self.current)
        self.current = dict(self.initial)
        self.seq_manoeuvres = []
        self.seq_states, self.seq_highlights, self.seq_labels = [], [], []
        self.seq_edited = False
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
        # **Affichage** : nœuds réels (avec barre) ; les ouvrages isolés ne sont
        # pas comptés comme nœuds (le séquencement, lui, garde nb_noeuds).
        nb = _nb_noeuds_reels(self._graph(state))
        return svg.svg, switches, nb

    def svgid_par_switch(self):
        svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
        meta = json.loads(svg.metadata)
        return {nd["equipmentId"]: nd["id"] for nd in meta.get("nodes", [])
                if nd.get("componentType") in ("BREAKER", "DISCONNECTOR")
                and nd.get("equipmentId")}

    def diff_states(self):
        """Organes dont l'état **diffère entre départ et cible** courants —
        ``[{id, svgId, direction}]`` où ``direction`` vaut ``"closed"`` (organe
        ouvert au départ et fermé à la cible → mis en évidence en vert) ou
        ``"opened"`` (fermé au départ, ouvert à la cible → orange). Sert à
        visualiser la différence départ/cible sur les deux schémas (les ids du
        schéma de départ étant préfixés ``A_``). ``open=True`` ⇒ organe ouvert."""
        svgmap = self.svgid_par_switch()
        out = []
        for eq, svgid in svgmap.items():
            dep = bool(self.initial.get(eq, False))
            cur = bool(self.current.get(eq, False))
            if dep != cur:
                out.append({"id": eq, "svgId": svgid,
                            "direction": "closed" if (dep and not cur) else "opened"})
        return out

    def step_view(self, i: int):
        """Vue **interactive** de l'étape i :
        ``(svg_highlighté, switches, nb, i, reached)``.

        Les organes sont renvoyés pour l'état de l'étape afin que l'expert puisse
        cliquer un organe à n'importe quelle étape (insertion de manœuvre).
        ``reached`` indique si l'état affiché **est déjà la topologie cible**
        (même partition nodale) — pour mettre en évidence la vue du poste."""
        if not self.seq_states:
            return "", [], 0, 0, False, None
        i = max(0, min(i, len(self.seq_states) - 1))
        state = self.seq_states[i]
        with self.applied(state):
            svg = self.net.get_single_line_diagram(self.vl, parameters=SLD_PAR)
            meta = json.loads(svg.metadata)
            switches = self._switches_meta(meta, state)
            nb = _nb_noeuds_reels(self._graph(state))   # affichage : isolés exclus
        # ``applied`` a restauré le réseau sur ``self.current``.
        reached = self._topo(self.current).meme_topologie(self._topo(state))
        # Vue nodale de l'**état détaillé de l'étape** : permet à l'IHM de faire
        # « suivre » la topologie nodale (partition) au fil de l'animation.
        nodale = self.nodale_state(state)
        return (_highlight(svg.svg, self.seq_highlights[i]),
                switches, nb, i, reached, nodale)

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
                nb_final = _nb_noeuds_reels(self._graph(self.seq_states[-1]))
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

        Phase **A** de la couche pluggable (``identifier_topologie_detaillee``
        de la façade), avec l'algorithme sélectionné (``self.algos``).

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

        ident = self._pipe().identifier_topologie_detaillee(poste, topo_cible)

        # Cible détaillée identifiée -> nouvel état cible courant (les organes
        # non mentionnés par la cible gardent leur état de départ).
        if ident.cible is not None:
            etats = ident.cible.etats_organes
            self.current = {k: bool(etats.get(k, v))
                            for k, v in self.initial.items()}
        else:
            self.current = dict(self.initial)
        self.scenario_name = None   # cible à revalider avant calcul de séquence

        svg, switches, nb = self.view(self.current)
        seq = ident.sequence   # sous-produit éventuel (écarts détaillés)
        return {
            "svg": svg, "switches": switches, "nb_noeuds": nb,
            "is_verified": ident.is_realisable,
            "message": ident.message,
            "ecarts": seq.ecarts if seq is not None else [],
            "noeuds_non_realisables": ident.noeuds_non_realisables,
            "nb_obtenu": nb,
            "nb_vise": topo_cible.nb_noeuds,
            "algo": self.algos["identificateur"],
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
        """Phase **B** de la couche pluggable (``sequencer`` de la façade),
        avec l'algorithme sélectionné (``self.algos``)."""
        # Poste à l'état de départ (A)
        self.apply(self.initial)
        poste = PosteTopologique.from_graph(self._graph(self.initial), self.vl)
        # Topologie détaillée cible (B) imposée : on vise la barre exacte de
        # chaque départ, pas seulement la partition nodale.
        cible = CibleDetaillee(
            voltage_level_id=self.vl,
            etats_organes={k: bool(v) for k, v in self.current.items()})

        mode = "aggressive" if mode == "aggressive" else "smooth"
        self.seq_mode = mode
        res = self._pipe().sequencer(poste, cible, mode=mode)

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
            # Alertes de **bonne pratique** non bloquantes (mode smooth) : > 1
            # ouvrage ré-aiguillé temporairement hors tension à la fois (R10ter).
            "alertes": res.alertes,
            "message": res.message,
            "algo": self.algos["sequenceur"],
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


class DayExploration:
    """Exploration de l'**intérêt d'une journée** : trois situations (par défaut
    minuit / midi / 23 h) chargées pour une date, et le **bilan des changements
    d'OC par poste** sur la journée (cf. ``manoeuvre.dataset.exploration``).

    Mémoire : on ne garde **qu'un réseau de référence** (le 1ᵉʳ chargé), réutilisé
    par la ``Session`` pour rendre les SLD ; pour chaque heure, seuls les **états
    d'organes par VL** sont conservés (légers). Afficher la topologie d'un poste à
    une heure = appliquer ces états sur le réseau de référence (pas de rechargement
    d'un réseau France entier)."""

    def __init__(self, date: str, repo: str):
        self.date = date
        self.repo = repo
        self.heures: list[dict] = []                 # [{requested, ts, iso}]
        self.etats: dict[str, dict[str, dict[str, bool]]] = {}  # heure -> {vl:{sw:open}}
        self.kinds: dict[str, str] = {}
        self.struct: dict[str, dict] = {}            # {vl:{edges,poids}} (invariant)
        self.vl_meta: dict[str, dict] = {}
        self.sub_name: dict[str, str] = {}
        self.postes: dict[str, dict] = {}
        self.top: list[str] = []                     # top-10 substations
        self.classement: list[str] = []              # classement plus long (liste)
        self.positions: dict[str, dict] = {}
        self.coord_source: str = "aucune"
        self.coord_stats: dict = {}

    def etats_vl(self, heure: str, vl: str) -> dict[str, bool]:
        return self.etats.get(heure, {}).get(vl, {})


def construire_exploration(date: str,
                           heures: tuple[str, ...] = exploration.HEURES_DEFAUT):
    """Charge les 3 situations de ``date``, calcule le bilan par poste, résout
    les coordonnées. Retourne ``(DayExploration, reseau_de_reference)``.

    Lève ``FileNotFoundError`` si la journée est absente du dataset."""
    repo = dataset_source.repo_pour_date(DATASET["repo"], date)
    token = DATASET["token"]
    cache = DATASET["cache_dir"]
    insts = dataset_source.lister_instantanes(repo, date, token=token)
    if not insts:
        raise FileNotFoundError(f"Aucun instantané pour {date} dans {repo}.")

    de = DayExploration(date, repo)
    ref_net = None
    for h in heures:
        choisi = dataset_source.choisir_instantane(insts, h)
        local = dataset_source.telecharger_instantane(
            repo, choisi["path"], cache, token=token)
        net = _charger_reseau(local)
        etats, kinds = exploration.extraire_etats_kinds(net)
        de.etats[h] = etats
        if not de.kinds:
            de.kinds = kinds
        de.heures.append({"requested": h, "ts": choisi["ts"], "iso": choisi["iso"]})
        if ref_net is None:
            de.vl_meta, de.sub_name = exploration.structure_reseau(net)
            # structure topologique invariante (arêtes + ouvrages par nœud) pour
            # quantifier les re-groupements de nœuds (scissions / fusions).
            de.struct = exploration.extraire_structure_topo(net)
            ref_net = net
        # les réseaux des autres heures sont libérés (déréférencés) ici.

    situations = [de.etats[h] for h in heures]
    changes = exploration.changements_par_vl(situations, de.kinds)
    nodaux = exploration.changements_nodaux_par_vl(situations, de.struct)
    exploration.fusionner_nodaux(changes, nodaux)
    de.postes = exploration.agreger_par_poste(changes, de.vl_meta, de.sub_name)
    de.top = exploration.classer_postes(de.postes, 10)
    de.classement = exploration.classer_postes(de.postes, 40)
    # Coordonnées : **plan de masse RTE committé** (par VL, ~98 % des postes,
    # hors-ligne) en **primaire** ; repli **OSM/Overpass** (ref:FR:RTE =
    # substation_id) si le plan ne couvre rien, avec persistance du snapshot
    # (bouton « ⬇ coordonnées »). ``MANOEUVRE_ENABLE_OSM=0`` désactive le fetch.
    pos_layout, stats_layout = geographie.positions_from_layout(
        geographie.charger_layout(GEO_LAYOUT), de.vl_meta)
    pos_layout = {s: pos_layout[s] for s in de.postes if s in pos_layout}
    if pos_layout:
        de.positions, de.coord_source, de.coord_stats = (
            pos_layout, "layout", stats_layout)
    else:
        stats: dict = {}
        de.positions, de.coord_source = geographie.resoudre(
            list(de.postes), net=ref_net, snapshot_path=GEO_SNAPSHOT,
            cache_dir=cache, autoriser_osm=OSM_ENABLED,
            persist_path=GEO_SNAPSHOT, stats_out=stats)
        de.coord_stats = stats
    # Journalisé (logs du Space) pour diagnostiquer la carte : source + taux +
    # (si OSM) échantillons de codes/noms vs substation_id (pourquoi 0 apparié).
    print(f"[explore_day] coord_source={de.coord_source} "
          f"stats={json.dumps(de.coord_stats, ensure_ascii=False)}", flush=True)
    return de, ref_net


def _xy(pos: dict) -> tuple[float, float]:
    """Coordonnées **planaires prêtes pour l'écran** (y vers le bas, nord en
    haut) depuis une position résolue. Le **plan de masse RTE** a déjà le nord en
    haut dans son repère (y croît vers le sud) → utilisé **tel quel** ; les sources
    **lon/lat** (OSM/embarqué) sont projetées Web Mercator puis **y inversé** (le
    Mercator a le nord en y croissant)."""
    if "x" in pos:
        return float(pos["x"]), float(pos["y"])
    mx, my = geographie.merc(float(pos["lon"]), float(pos["lat"]))
    return mx, -my


def _explore_payload(de: DayExploration) -> dict:
    """Charge utile carte + classement pour le front. Le **classement et la mise
    en évidence sont au niveau voltage level** (plus fin que par poste) ; la carte
    place un disque par **poste**, mis en évidence s'il porte un VL du top-10."""

    def _kinds(d):
        return {t: d.get(t, 0) for t in exploration.TYPES_OC}

    # Classement **par voltage level** (chaque VL actif est une entrée).
    all_vls = []
    for sub, p in de.postes.items():
        for v in p["vls"]:
            if v["total"] > 0:
                all_vls.append({"vl": v["vl"], "sub": sub,
                                "name": v.get("name") or v["vl"],
                                "nv": v["nominal_v"], "total": v["total"],
                                "nodal": v.get("nodal", 0), **_kinds(v)})
    all_vls.sort(key=lambda d: (-d["total"], -d["nv"], d["vl"]))
    for i, v in enumerate(all_vls):
        v["rank"] = i + 1
    # Rang d'un poste sur la carte = meilleur rang d'un de ses VL dans le top-10.
    rang: dict[str, int] = {}
    for v in all_vls[:10]:
        rang.setdefault(v["sub"], v["rank"])
    classement = [{**v, "geo": v["sub"] in de.positions} for v in all_vls[:40]]

    postes_map = []
    for sub, p in de.postes.items():
        pos = de.positions.get(sub)
        if not pos:
            continue
        x, y = _xy(pos)
        postes_map.append({
            "sub": sub, "name": p["name"], "nv": p["nominal_v_max"],
            "x": round(x, 1), "y": round(y, 1), "total": p["total"],
            "nodal": p.get("nodal", 0), **_kinds(p), "rank": rang.get(sub),
            "top_vl": exploration.vl_le_plus_actif(p),
            "vls": [{"vl": v["vl"], "nv": v["nominal_v"], "total": v["total"],
                     "nodal": v.get("nodal", 0)} for v in p["vls"]],
        })
    n_actifs = sum(1 for v in all_vls)
    return {
        "ok": True, "date": de.date, "heures": de.heures,
        "coord_source": de.coord_source, "coord_stats": de.coord_stats,
        # un instantané committable a-t-il été persisté (fetch OSM réussi) ?
        "coord_file": de.coord_source == "osm" and GEO_SNAPSHOT.exists(),
        "n_postes": len(de.postes), "n_actifs": n_actifs,
        "n_geolocalises": len(postes_map),
        "types_oc": list(exploration.TYPES_OC),
        "postes": postes_map, "classement": classement,
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
# Exploration de journée en cours (carte des postes + bilan des changements).
# ``None`` tant qu'aucune journée n'a été explorée (« Explorer la journée »).
DAY: DayExploration = None  # type: ignore

# Configuration de la **source dataset par date** (mode déploiement Space
# HuggingFace : on charge à la demande un instantané XIIDM du dataset RTE 7000).
# Renseignée par ``main()`` / variables d'environnement. ``enabled=False`` =>
# mode local (``--grid``) : le bandeau « Dataset » de l'IHM reste masqué et le
# comportement historique est strictement préservé.
DATASET = {
    "enabled": False,
    "repo": dataset_source.REPO_DEFAUT,
    "cache_dir": ".cache/dgitt",
    "token": None,
    "default_date": dataset_source.DATES_ECHANTILLON[0],
    "default_time": "12:00",
    "sample_dates": list(dataset_source.DATES_ECHANTILLON),
    # IHM déportée (Space HuggingFace) : le système de fichiers est éphémère →
    # le front télécharge aussi en local les scénarios/séquences sauvegardés.
    # Auto-détecté (HF pose ``SPACE_ID``) ou forcé via env / ``--hosted``.
    "hosted": bool(os.environ.get("SPACE_ID")
                   or os.environ.get("MANOEUVRE_IHM_HOSTED")),
}


def _algos_courants() -> dict[str, str]:
    """Sélection d'algos courante, ou défauts « libtopo » si aucune session n'est
    encore chargée (mode dataset avant le choix d'une date)."""
    if SESSION is not None:
        return SESSION.algos
    return {p: "libtopo" for p in ("identificateur", "sequenceur",
                                   "planificateur")}


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/postes")
def api_postes():
    # ``postes`` : liste épinglée (jeu de test + 3 JdB). ``all`` : tous les postes
    # NODE_BREAKER de la situation chargée (recherche dans l'IHM). ``catalog`` :
    # sections par typologie (cf. POSTES_CATALOG) avec disponibilité par poste.
    # Mode dataset avant tout chargement : aucune situation => le front affiche
    # le bandeau de choix de date (``needs_date``).
    if SESSION is None:
        return jsonify(postes=[], all=[], catalog=[], needs_date=True)
    return jsonify(postes=SESSION.postes, all=SESSION.all_postes,
                   catalog=SESSION.catalog())


# ── Source dataset RTE 7000 (chargement d'une situation par date / heure) ────

@app.get("/api/dataset/config")
def api_dataset_config():
    """Configuration de la source dataset pour le front : si ``enabled``, l'IHM
    affiche le bandeau date/heure ; sinon (mode ``--grid`` local) il reste masqué."""
    return jsonify(enabled=DATASET["enabled"], repo=DATASET["repo"],
                   default_date=DATASET["default_date"],
                   default_time=DATASET["default_time"],
                   sample_dates=DATASET["sample_dates"],
                   hosted=DATASET["hosted"])


@app.get("/api/dataset/timestamps")
def api_dataset_timestamps():
    """Instantanés disponibles (HH:MM) pour la date ``?date=YYYY-MM-DD``, avec
    l'horodatage présélectionné par défaut (le plus proche de midi)."""
    date = (request.args.get("date") or "").strip()
    repo = dataset_source.repo_pour_date(DATASET["repo"], date)
    try:
        insts = dataset_source.lister_instantanes(
            repo, date, token=DATASET["token"])
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except Exception as exc:  # pragma: no cover - dépend du réseau HF
        return jsonify(ok=False, error=f"Listing HuggingFace impossible : {exc}"), 502
    choisi = dataset_source.choisir_instantane(insts, DATASET["default_time"])
    return jsonify(ok=True, date=date,
                   timestamps=[{"ts": d["ts"], "path": d["path"]} for d in insts],
                   default=(choisi["ts"] if choisi else None))


@app.post("/api/dataset/load")
def api_dataset_load():
    """Charge la **situation réseau** du dataset à ``{date, time}`` (téléchargement
    à la demande depuis HuggingFace + cache local), reconstruit la session et
    renvoie la liste des postes (même forme que ``/api/load_grid``)."""
    global SESSION
    body = request.json or {}
    date = (body.get("date") or "").strip()
    heure = (body.get("time") or DATASET["default_time"]).strip()
    repo = dataset_source.repo_pour_date(DATASET["repo"], date)
    try:
        net, meta = dataset_source.charger_situation(
            repo, date, DATASET["cache_dir"],
            heure=heure, token=DATASET["token"])
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except Exception as exc:  # pragma: no cover - dépend du réseau HF / fichier
        return jsonify(ok=False, error=f"Chargement impossible : {exc}"), 502
    SESSION = Session(net)
    return jsonify(ok=True, date=meta["date"], time=meta["ts"], iso=meta["iso"],
                   postes=SESSION.postes, all=SESSION.all_postes,
                   catalog=SESSION.catalog())


# ── Exploration de journée (carte des postes + bilan des changements d'OC) ──

@app.post("/api/explore_day")
def api_explore_day():
    """Explore une **journée** : charge 3 situations (minuit / midi / 23 h),
    calcule par poste le nombre d'OC dont l'état change sur la journée (ventilé
    par type), résout les coordonnées et renvoie la carte + le classement.

    Reconstruit aussi la ``Session`` sur le réseau de référence (1ʳᵉ heure) pour
    que le reste de l'IHM (vue topologique d'un poste) fonctionne ensuite."""
    global SESSION, DAY
    body = request.json or {}
    date = (body.get("date") or "").strip()
    if not date:
        return jsonify(ok=False, error="Choisir une date."), 400
    try:
        de, ref_net = construire_exploration(date)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except Exception as exc:  # pragma: no cover - dépend du réseau HF / fichier
        return jsonify(ok=False, error=f"Exploration impossible : {exc}"), 502
    DAY = de
    SESSION = Session(ref_net)
    return jsonify(_explore_payload(de))


def _sub_vls(sub: str) -> list[dict]:
    """VL d'un poste (substation) avec leurs changements — pour basculer entre
    les niveaux de tension d'un même poste dans la vue topologique."""
    p = (DAY.postes.get(sub) if DAY else None) or {}
    return [{"vl": v["vl"], "nv": v["nominal_v"], "total": v["total"],
             "nodal": v.get("nodal", 0), "name": v.get("name") or v["vl"]}
            for v in p.get("vls", [])]


@app.post("/api/explore_poste")
def api_explore_poste():
    """Passe en **vue topologique** d'un poste à une **heure** de la journée
    explorée (double-clic sur la carte). Applique les états d'organes de l'heure
    visée sur le réseau de référence (pas de rechargement). ``vl`` explicite, ou
    le VL le plus actif de la ``sub`` à défaut."""
    if DAY is None or SESSION is None:
        return jsonify(ok=False, error="Explorer une journée d'abord."), 400
    body = request.json or {}
    heure = body.get("hour") or (DAY.heures[0]["requested"] if DAY.heures else "12:00")
    vl = body.get("vl")
    sub = body.get("sub")
    if not vl and sub:
        p = DAY.postes.get(sub)
        vl = exploration.vl_le_plus_actif(p) if p else None
    if not vl:
        return jsonify(ok=False, error="Poste/VL introuvable."), 400
    if sub is None:
        sub = (DAY.vl_meta.get(vl) or {}).get("substation") or vl
    SESSION.load_states(vl, DAY.etats_vl(heure, vl))
    svg_i, _, nb_i = SESSION.view(SESSION.initial)
    svg_c, sw, nb_c = SESSION.view(SESSION.current)
    return jsonify(ok=True, initial_svg=_prefix_svg_ids(svg_i, "A_"), nb_initial=nb_i,
                   svg=svg_c, switches=sw, nb_noeuds=nb_c, vl=vl, sub=sub,
                   hour=heure, heures=DAY.heures, sub_vls=_sub_vls(sub),
                   changes=SESSION.diff_states(),
                   nodale_depart=SESSION.nodale_payload(SESSION.initial),
                   nodale_cible=SESSION.nodale_state(SESSION.current))


@app.post("/api/explore_retain_target")
def api_explore_retain_target():
    """Retient la topologie du poste **à une autre heure** comme **cible**
    courante (« retenir cette topologie comme cible »), le départ restant l'heure
    choisie. La cible reste ensuite éditable par l'utilisateur."""
    if DAY is None or SESSION is None or SESSION.vl is None:
        return jsonify(ok=False, error="Vue topologique d'un poste requise."), 400
    body = request.json or {}
    heure = body.get("hour")
    vl = body.get("vl") or SESSION.vl
    SESSION.set_target_states(DAY.etats_vl(heure, vl))
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(ok=True, svg=svg, switches=sw, nb_noeuds=nb, hour=heure,
                   changes=SESSION.diff_states(),
                   nodale=SESSION.nodale_state(SESSION.current))


_BASEMAP_SCREEN = None  # cache du fond de carte projeté écran (y inversé)


@app.get("/api/explore_basemap")
def api_explore_basemap():
    """Fond de carte (frontières départements + pays voisins) **dans le repère
    écran** (mêmes coordonnées que les disques : y inversé). Statique → mis en
    cache ; le front le récupère une fois."""
    global _BASEMAP_SCREEN
    if _BASEMAP_SCREEN is None:
        # Le fond est déjà dans le repère du plan de masse (même que les disques,
        # nord en haut) → servi tel quel, sans inversion.
        bm = geographie.charger_basemap(GEO_BASEMAP)
        _BASEMAP_SCREEN = {"depts": bm.get("depts", []),
                           "neighbors": bm.get("neighbors", [])}
    return jsonify(_BASEMAP_SCREEN)


@app.get("/api/explore_coords_file")
def api_explore_coords_file():
    """Renvoie l'instantané de coordonnées **résolu et persisté** au runtime
    (`data/postes_rte_geo.json`), en pièce jointe — pour le **committer** une fois
    et éviter de re-interroger ODRE à chaque démarrage (FS du Space éphémère)."""
    if not GEO_SNAPSHOT.exists():
        return jsonify(ok=False, error="Aucun instantané de coordonnées."), 404
    return Response(
        GEO_SNAPSHOT.read_text(encoding="utf-8"),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=postes_rte_geo.json"})


@app.post("/api/load_grid")
def api_load_grid():
    """Charge une **situation réseau** quelconque (chemin ``.xiidm`` côté serveur)
    et réinitialise la session. Permet d'inspecter/tester n'importe quel poste
    d'une situation arbitraire sans relancer le serveur."""
    global SESSION
    path = ((request.json or {}).get("path") or "").strip()
    if not path or not pathlib.Path(path).expanduser().exists():
        return jsonify(ok=False, error=f"Fichier introuvable : {path}"), 400
    try:
        net = pp.network.load(str(pathlib.Path(path).expanduser()))
    except Exception as exc:  # pragma: no cover - dépend du fichier fourni
        return jsonify(ok=False, error=f"Échec du chargement : {exc}"), 400
    SESSION = Session(net)
    return jsonify(ok=True, postes=SESSION.postes, all=SESSION.all_postes,
                   catalog=SESSION.catalog())


def _pick_grid_file_macos() -> dict:
    """Sélecteur de fichier natif macOS (osascript). Best-effort."""
    script = ('POSIX path of (choose file with prompt '
              '"Sélectionner une situation réseau (.xiidm)")')
    proc = subprocess.run(["osascript", "-e", script],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if "User canceled" in err or "User cancelled" in err or "(-128)" in err:
            return {"path": ""}            # annulation = pas une erreur
        return {"path": "", "error": err or "osascript a échoué"}
    return {"path": proc.stdout.strip()}


def _pick_grid_file_tkinter() -> dict:
    """Sélecteur de fichier natif via ``tkinter`` (sous-processus isolé). Sans
    afficheur (Space headless) ou sans tkinter, le sous-processus échoue et on
    renvoie une ``error`` que l'IHM affiche (invite à coller le chemin)."""
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk(); root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "p = filedialog.askopenfilename(\n"
        "    title='Sélectionner une situation réseau',\n"
        "    filetypes=[('Réseau XIIDM', '*.xiidm *.xiidm.bz2 *.xiidm.gz *.zip'),\n"
        "               ('Tous les fichiers', '*.*')])\n"
        "root.destroy()\n"
        "print(p or '')\n")
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"path": "",
                "error": (proc.stderr or "").strip() or "sélecteur indisponible"}
    return {"path": proc.stdout.strip()}


@app.get("/api/pick_grid_file")
def api_pick_grid_file():
    """Ouvre un sélecteur de fichier **natif** (usage local) pour choisir une
    situation réseau ``.xiidm`` et renvoie ``{path, error?}`` — ``path`` vide si
    l'utilisateur annule. Sur un serveur sans afficheur (Space), renvoie une
    ``error`` (l'IHM invite alors à coller le chemin à la main)."""
    try:
        if platform.system() == "Darwin":
            return jsonify(_pick_grid_file_macos())
        return jsonify(_pick_grid_file_tkinter())
    except subprocess.TimeoutExpired:
        return jsonify(path="", error="Sélecteur expiré (aucune sélection).")
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        return jsonify(path="", error=str(exc))


@app.post("/api/load")
def api_load():
    SESSION.load(request.json["vl"])
    svg_i, _, nb_i = SESSION.view(SESSION.initial)
    svg_c, sw, nb_c = SESSION.view(SESSION.current)
    return jsonify(initial_svg=_prefix_svg_ids(svg_i, "A_"), nb_initial=nb_i,
                   svg=svg_c, switches=sw, nb_noeuds=nb_c,
                   changes=SESSION.diff_states(),
                   nodale_depart=SESSION.nodale_payload(SESSION.initial),
                   nodale_cible=SESSION.nodale_state(SESSION.current))


@app.post("/api/toggle")
def api_toggle():
    SESSION.toggle(request.json["id"])
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb,
                   changes=SESSION.diff_states(),
                   nodale=SESSION.nodale_state(SESSION.current))


@app.post("/api/reset")
def api_reset():
    SESSION.reset()
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb,
                   changes=SESSION.diff_states(),
                   nodale=SESSION.nodale_state(SESSION.current))


@app.post("/api/promote_cible")
def api_promote_cible():
    """Promeut la cible courante en **nouvel état de départ** (chaînage sans
    passer par un scénario sauvegardé). Renvoie les deux schémas (départ + cible)
    et les vues nodales, comme ``/api/load``."""
    SESSION.promote_cible()
    svg_i, _, nb_i = SESSION.view(SESSION.initial)
    svg_c, sw, nb_c = SESSION.view(SESSION.current)
    return jsonify(initial_svg=_prefix_svg_ids(svg_i, "A_"), nb_initial=nb_i,
                   svg=svg_c, switches=sw, nb_noeuds=nb_c, vl=SESSION.vl,
                   changes=SESSION.diff_states(),
                   nodale_depart=SESSION.nodale_payload(SESSION.initial),
                   nodale_cible=SESSION.nodale_state(SESSION.current))


@app.post("/api/cible")
def api_cible():
    """Vue détaillée **cible courante** (sans la modifier) + vue nodale — pour
    revenir en édition de la cible alors qu'une séquence est déjà calculée."""
    svg, sw, nb = SESSION.view(SESSION.current)
    return jsonify(svg=svg, switches=sw, nb_noeuds=nb,
                   changes=SESSION.diff_states(),
                   nodale=SESSION.nodale_state(SESSION.current))


@app.post("/api/nodale")
def api_nodale():
    """Partitions nodales de départ et cible (cible initialisée = départ)."""
    nodale = SESSION.nodale_payload(SESSION.initial)
    return jsonify(nodale_depart=nodale, nodale_cible=nodale)


@app.get("/api/algos")
def api_algos():
    """Algorithmes pluggables **disponibles** (registre ``manoeuvre.plugins``,
    natifs « libtopo » + plugins tiers enregistrés / entry points), par phase,
    et **sélection courante** de la session."""
    return jsonify(disponibles=algos_disponibles(), selection=_algos_courants())


@app.post("/api/algos")
def api_algos_set():
    """Sélectionne l'algorithme d'une ou plusieurs phases, p. ex.
    ``{"sequenceur": "mon_algo"}``. Les noms inconnus du registre sont ignorés ;
    la sélection effective est renvoyée."""
    selection = SESSION.set_algos(request.json or {})
    return jsonify(disponibles=algos_disponibles(), selection=selection)


@app.post("/api/nodale_to_detaillee")
def api_nodale_to_detaillee():
    """Calcule la topologie détaillée d'intérêt réalisant la cible nodale éditée
    et la charge comme cible détaillée courante (volet du bas)."""
    return jsonify(SESSION.nodale_to_detaillee(
        request.json.get("groups", []), request.json.get("isolated", [])))


@app.get("/api/scenarios")
def api_scenarios():
    if SESSION is None:
        return jsonify(scenarios=[])
    return jsonify(scenarios=SESSION.list_scenarios())


def _safe_name(name, default):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or default


@app.post("/api/save")
def api_save():
    name = _safe_name(request.json.get("name", ""), SESSION.vl)
    if not request.json.get("overwrite") and (SCEN_DIR / f"{name}.json").exists():
        return jsonify(exists=True, name=name, path=str(SCEN_DIR / f"{name}.json"))
    path = SESSION.save_scenario(name)
    # ``content`` : JSON écrit, renvoyé pour le téléchargement local côté front
    # (IHM déportée — FS éphémère du Space).
    content = pathlib.Path(path).read_text(encoding="utf-8")
    return jsonify(path=path, name=name, content=content,
                   scenarios=SESSION.list_scenarios())


@app.post("/api/load_scenario")
def api_load_scenario():
    SESSION.load_scenario(request.json["name"],
                          request.json.get("mode", "both"))
    svg_i, _, nb_i = SESSION.view(SESSION.initial)
    svg_c, sw, nb_c = SESSION.view(SESSION.current)
    return jsonify(initial_svg=_prefix_svg_ids(svg_i, "A_"), nb_initial=nb_i,
                   svg=svg_c, switches=sw, nb_noeuds=nb_c, vl=SESSION.vl,
                   changes=SESSION.diff_states(),
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
    content = pathlib.Path(path).read_text(encoding="utf-8")   # téléchargement local
    return jsonify(path=path, name=name, content=content)


@app.get("/api/step")
def api_step():
    i = int(request.args.get("i", 0))
    svg, switches, nb, i, reached, nodale = SESSION.step_view(i)
    return jsonify(svg=svg, switches=switches, nb_noeuds=nb, i=i,
                   reached=reached, nodale=nodale)


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


# ---------------------------------------------------------------------------
# Front-end (HTML/CSS/JS) — externalisé dans manoeuvre_ihm_assets/index.html
# (cf. docs/manoeuvre/ihm.md). Chargé au démarrage du module ; servi tel quel
# par la route index(). Chemin résolu via __file__ (robuste au cwd).
# ---------------------------------------------------------------------------
_ASSETS_DIR = pathlib.Path(__file__).resolve().parent / "manoeuvre_ihm_assets"
PAGE = (_ASSETS_DIR / "index.html").read_text(encoding="utf-8")



def main():
    global SESSION
    ap = argparse.ArgumentParser(description=__doc__)
    # ``--grid`` optionnel : s'il est omis, l'IHM démarre en **mode dataset**
    # (source RTE 7000 par date) ; sinon comportement local historique.
    ap.add_argument("--grid", default=None,
                    help="Chemin du réseau .xiidm (mode local). Omis => mode dataset.")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="Interface d'écoute (0.0.0.0 pour un conteneur / Space).")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--scenarios-dir", default="tests/manoeuvre/scenarios",
                    help="Dossier de sauvegarde des scénarios cible")
    ap.add_argument("--sequences-dir", default="tests/manoeuvre/sequences",
                    help="Dossier de sauvegarde des séquences générées")
    # Mode dataset (source par date depuis HuggingFace).
    ap.add_argument("--dataset", action="store_true",
                    help="Activer la source dataset RTE 7000 (défaut si --grid absent).")
    ap.add_argument("--dataset-repo",
                    default=os.environ.get("DGITT_REPO", dataset_source.REPO_DEFAUT),
                    help="Dataset HuggingFace (défaut : %(default)s)")
    ap.add_argument("--cache-dir",
                    default=os.environ.get("DGITT_CACHE_DIR", ".cache/dgitt"),
                    help="Cache local des instantanés téléchargés")
    ap.add_argument("--default-date",
                    default=os.environ.get("DGITT_DEFAULT_DATE",
                                           dataset_source.DATES_ECHANTILLON[0]),
                    help="Date proposée par défaut dans l'IHM (YYYY-MM-DD)")
    ap.add_argument("--hosted", action="store_true",
                    help="IHM déportée (Space) : télécharge aussi en local les "
                         "fichiers sauvegardés (auto si SPACE_ID/MANOEUVRE_IHM_HOSTED).")
    args = ap.parse_args()

    global SCEN_DIR, SEQ_DIR
    SCEN_DIR = pathlib.Path(args.scenarios_dir)
    SEQ_DIR = pathlib.Path(args.sequences_dir)

    use_dataset = args.dataset or not args.grid
    DATASET.update(enabled=use_dataset, repo=args.dataset_repo,
                   cache_dir=args.cache_dir, token=os.environ.get("HF_TOKEN"),
                   default_date=args.default_date,
                   hosted=args.hosted or DATASET["hosted"])

    if args.grid:
        print(f"Chargement du réseau {args.grid} …")
        SESSION = Session(pp.network.load(args.grid))
        print(f"Postes de test disponibles : {SESSION.postes}")
    if use_dataset:
        print(f"Mode dataset : {DATASET['repo']} (cache {DATASET['cache_dir']}). "
              "Choisir une date/heure dans l'IHM.")
    print(f"IHM Manœuvre : http://{args.host}:{args.port}  (Ctrl-C pour arrêter)")
    # threaded=False : sérialise les requêtes (l'état réseau pypowsybl est partagé).
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
