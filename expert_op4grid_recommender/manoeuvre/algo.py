"""
manoeuvre/algo.py  —  Phase 2 : algorithme nodale → détaillée
---------------------------------------------------------------
Calcule la séquence d'organes de coupure (OC) à manœuvrer pour passer de
l'état détaillé courant d'un poste à une topologie nodale cible.

Correspondance C++ : ``Topologie::determineTopoCompleteCible`` (TOPOPoste.cc:3944)
et ses sous-routines (``connectAndDeconnectOuvrageHS``, ``evalueEtatCouplage``,
``identifySuperTronconnement``, ``getTronconnementBesoinReaiguillage2barres``,
``reaiguillage2barres``, ``listeDordre``).

Point d'entrée unique
~~~~~~~~~~~~~~~~~~~~~~~
``determiner_topo_complete_cible(poste, topo_cible)`` intègre toute la chaîne :
faisabilité → **placement automatique** nœud→sections de barres
(``_placement_automatique``) → séquenceur général
(``determiner_manoeuvres_avec_sections``) → vérification. Il gère
indifféremment 1 barre, 2 barres, et la **création de nœuds supplémentaires**
par ouverture de sectionnement, sans placement explicite.

Couverture
~~~~~~~~~~
- Postes 1 barre et 2 barres standard (couplage + ré-aiguillage boucle
  courte/longue).                                                        [OK]
- Création d'un nœud au-delà du nombre de barres par **ouverture de
  sectionnement de barre** (dé-énergisation préalable de la section).    [OK]
- Ordonnancement type ``listeDordre`` : fermeture des couplages d'abord,
  ré-aiguillages boucle courte, ouverture des sectionnements (hors tension),
  ouverture des couplages, ré-aiguillages boucle longue, puis suppression
  des manœuvres sans effet.                                              [OK]
- Contrôle de court-circuit avant fermeture d'un couplage.               [OK]
- Vérification post-manœuvre (recalcul de la topologie nodale).          [OK]

Règle du sectionneur de barre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Un sectionneur de barre ne se manœuvre que hors charge. Pour scinder une barre
en deux nœuds, la section à isoler est d'abord mise hors tension (ses départs
ré-aiguillés sur l'autre barre en boucle courte), puis le sectionnement est
ouvert, puis les départs du nouveau nœud y sont ré-aiguillés en boucle longue.

Postes multi-sections : les postes à plusieurs sections par barre (ex. CARRIP6,
2 barres × 3 sections = 6 SJB) sont gérés (ouverture de sectionnements avec
garage temporaire des départs sur une SJB équipotentielle).

Dégradation gracieuse : si une étape n'est pas réalisable en sécurité (pas de
SJB tampon, départ inatteignable…), l'algorithme **ne s'interrompt pas** : il
consigne les **écarts** et poursuit ; ``topo_obtenue`` est toujours renseignée.

Limites connues (documentées, cf. doc C++) :
- Ré-aiguillage d'omnibus complexes (départs multiples scindés)          [partiel]
- Contrôle de court-circuit : vérifie l'équipotentialité courante avant
  manœuvre de sectionneur (pas de calcul de potentiel/déphasage fin)     [simplifié]
- Topologies de couplers non chaînées (≥ 3 barres en anneau)             [partiel]
- Nœuds mêlant départs connectés et déconnectés                          [partiel]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import networkx as nx

from .models import SwitchKind
from .cellules import CelluleDepart, SwitchInfo
from .topologie import (
    TopologieNodale,
    PosteTopologique,
)
from .troncons import Troncon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de sortie
# ---------------------------------------------------------------------------

@dataclass
class Manoeuvre:
    """Une opération unitaire sur un organe de coupure."""
    switch_id: str
    action: Literal["OPEN", "CLOSE"]
    raison: str
    type_boucle: Optional[Literal["COURTE", "LONGUE"]] = None

    def __repr__(self) -> str:  # pragma: no cover - debug
        b = f" [{self.type_boucle}]" if self.type_boucle else ""
        return f"{self.action:5s} {self.switch_id}  ({self.raison}){b}"


@dataclass
class ResultatManoeuvres:
    """Résultat complet de l'algorithme."""
    voltage_level_id: str
    topo_initiale: TopologieNodale
    topo_cible: TopologieNodale
    manoeuvres: list[Manoeuvre] = field(default_factory=list)
    departs_reaiguilles: set[str] = field(default_factory=set)
    couplages_modifies: list[str] = field(default_factory=list)
    is_changed: bool = False
    is_verified: bool = False              # topologie NODALE atteinte
    is_verified_detaillee: bool = False    # topologie DÉTAILLÉE atteinte
    ecarts: list[str] = field(default_factory=list)  # écarts détaillés résiduels
    topo_obtenue: Optional[TopologieNodale] = None
    message: str = ""

    @property
    def nb_manoeuvres(self) -> int:
        return len(self.manoeuvres)

    def resume(self) -> str:
        lines = [
            f"Manœuvres VL '{self.voltage_level_id}' : {self.nb_manoeuvres} OC, "
            f"changed={self.is_changed}, verified={self.is_verified}"
        ]
        for i, m in enumerate(self.manoeuvres, 1):
            lines.append(f"  {i:2d}. {m!r}")
        if self.message:
            lines.append(f"  -> {self.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def determiner_topo_complete_cible(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
) -> ResultatManoeuvres:
    """
    Calcule la séquence de manœuvres pour atteindre ``topo_cible`` depuis
    l'état détaillé courant de ``poste``.

    Parameters
    ----------
    poste :
        Vue complète du poste (cellules, tronçonnement, topologie nodale
        courante), construite par ``PosteTopologique.from_graph``.
    topo_cible :
        Topologie nodale visée.

    Returns
    -------
    ResultatManoeuvres
    """
    res = ResultatManoeuvres(
        voltage_level_id=poste.voltage_level_id,
        topo_initiale=poste.topologie_nodale,
        topo_cible=topo_cible,
    )

    # On travaille sur une copie du graphe pour simuler les manœuvres.
    G = poste.graph.copy() if poste.graph is not None else None
    if G is None:
        res.message = "Graphe absent : impossible de calculer les manœuvres."
        return res

    # --- Court-circuit : la topologie courante satisfait déjà la cible ----
    if poste.topologie_nodale.meme_topologie(topo_cible):
        res.is_changed = False
        res.is_verified = True
        res.topo_obtenue = poste.topologie_nodale
        res.message = "La topologie courante satisfait déjà la cible (aucune manœuvre)."
        return res

    # --- Phase 2.1 : faisabilité (départs cibles présents) ----------------
    departs_poste = {c.equipment_id for c in poste.cellules.cellules_depart}
    for c in poste.cellules.cellules_depart:
        departs_poste |= set(c.shared_equipment_ids)
    manquants = set(topo_cible.noeud_par_depart) - departs_poste
    if manquants:
        res.topo_obtenue = poste.topologie_nodale
        res.message = f"Départs cibles absents du poste : {sorted(manquants)}"
        return res

    # --- Phases 2.2-2.4 : placement automatique nœud -> sections de barres -
    placement, faisable, msg = _placement_automatique(poste, topo_cible)
    if not faisable:
        res.topo_obtenue = poste.topologie_nodale
        res.message = msg
        return res

    # --- Délégation au séquenceur général (couplage + sectionnement) -------
    core = determiner_manoeuvres_avec_sections(poste, placement)
    core.topo_initiale = poste.topologie_nodale
    core.topo_cible = topo_cible
    core.is_verified = bool(
        core.topo_obtenue and topo_cible.meme_topologie(core.topo_obtenue)
    )
    core.is_changed = bool(core.manoeuvres)
    core.message = (
        "Topologie cible atteinte et vérifiée." if core.is_verified
        else "La topologie obtenue ne correspond pas à la cible "
             f"(obtenu {core.topo_obtenue.nb_noeuds if core.topo_obtenue else 0} "
             f"nœud(s), visé {topo_cible.nb_noeuds})."
    )
    return core


def _wired_busbar(cell: CelluleDepart, graph: nx.Graph) -> Optional[int]:
    """SJB sur laquelle un départ est câblé (chemin de SA fermés) dans ``graph``."""
    for bb in cell.busbar_nodes:
        sa = _sa_path_to_sjb(cell, bb)
        if sa and all(not graph.edges[u, v].get("open", False)
                      for u, v in _edges_of_switches(graph, sa)):
            return bb
    return None


def _edges_of_switches(graph: nx.Graph, switch_ids):
    out = []
    sset = set(switch_ids)
    for u, v, d in graph.edges(data=True):
        if d.get("switch_id") in sset:
            out.append((u, v))
    return out


def _departure_dj_changes(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
) -> tuple[list[tuple[CelluleDepart, SwitchInfo]],
           list[tuple[CelluleDepart, SwitchInfo]]]:
    """
    Identifie les changements d'état des **DJ de départ** entre l'état initial
    (``poste.graph``) et la cible (``cible_graph``).

    Les DJ de couplage (inter-SJB) sont exclus — ils sont gérés par le
    séquenceur général.

    Returns
    -------
    (reconnections, disconnections)
        Chaque élément est une liste de ``(cellule_départ, switch_info_du_DJ)``.
        - reconnection : DJ ouvert → fermé (mise en service)
        - disconnection : DJ fermé → ouvert (mise hors service)
    """
    G = poste.graph
    coupling_sids: set[str] = set()
    for cp in _inter_sjb_couplers(poste):
        coupling_sids.update(cp.switch_ids)

    reconnections: list[tuple[CelluleDepart, SwitchInfo]] = []
    disconnections: list[tuple[CelluleDepart, SwitchInfo]] = []

    for cell in poste.cellules.cellules_depart:
        for sw in cell.breakers:
            if sw.switch_id in coupling_sids:
                continue
            initial_open = _is_open(G, sw.switch_id)
            cible_open = _is_open(cible_graph, sw.switch_id)
            if initial_open and not cible_open:
                reconnections.append((cell, sw))
            elif not initial_open and cible_open:
                disconnections.append((cell, sw))

    return reconnections, disconnections


def _placement_avec_reconnexions(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    topo_cible: TopologieNodale,
    reconnections: list[tuple[CelluleDepart, SwitchInfo]],
) -> tuple[list[tuple[set[str], set[str]]], bool, str]:
    """
    Calcule le placement nœud→SJB en tenant compte des **reconnexions** de
    départ (DJ ouvert → fermé dans la cible).

    Le placement est calculé sur un poste **virtuel** où les DJ reconnectés sont
    fermés et leurs SA positionnés comme dans la cible : les départs reconnectés
    sont ainsi vus *connectés sur leur barre cible*, ce qui force l'affectation
    de leur SJB cible au bon nœud (via le coût de ré-aiguillage). Le séquenceur,
    lui, travaillera sur le poste **réel** (DJ encore ouverts), pour que les
    sections cibles restent hors tension lors des manœuvres de sectionnement.
    """
    vl = poste.voltage_level_id
    G_virt = poste.graph.copy()
    coupling_sids: set[str] = set()
    for cp in _inter_sjb_couplers(poste):
        coupling_sids.update(cp.switch_ids)

    for cell, dj_sw in reconnections:
        for sa in cell.disconnectors:
            if sa.switch_id in coupling_sids:
                continue
            _set_switch(G_virt, sa.switch_id, _is_open(cible_graph, sa.switch_id))
        _set_switch(G_virt, dj_sw.switch_id, False)

    poste_virt = PosteTopologique.from_graph(G_virt, vl)
    return _placement_automatique(poste_virt, topo_cible)


def determiner_manoeuvres_cible_detaillee(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
) -> ResultatManoeuvres:
    """
    Atteint une **topologie détaillée cible imposée** (état précis de chaque
    organe, donc de la barre de chaque départ), plus spécifique que la seule
    topologie nodale.

    Démarche :
    1. **séquence nodale** sûre. Si des **DJ de départ changent d'état**
       (reconnexions / déconnexions), le placement nœud→SJB est calculé sur un
       poste virtuel (reconnexions appliquées) mais le **séquenceur tourne sur
       le poste réel** (DJ encore ouverts) afin que les sections cibles restent
       hors tension pendant les manœuvres de sectionnement (règle du
       sectionneur). Sinon, on délègue à ``determiner_topo_complete_cible`` ;
    2. **raffiner** : ramener chaque départ sur sa barre exacte imposée par la
       cible (ré-aiguillage en boucle courte, équipotentiel) ;
    3. **changements de DJ de départ** : fermer les DJ des reconnexions (mise en
       service, la barre cible est désormais au bon potentiel) et ouvrir ceux
       des déconnexions (mise hors service) ;
    4. **vérifier** topologie nodale + détaillée ; consigner les **écarts**.
    """
    vl = poste.voltage_level_id
    cells = poste.cellules

    reconnections, disconnections = _departure_dj_changes(poste, cible_graph)

    # Barre cible (imposée) de chaque départ
    cible_busbar: dict[str, int] = {}
    for c in cells.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            bb = _wired_busbar(c, cible_graph)
            if bb is not None:
                cible_busbar[eq] = bb

    topo_cible = TopologieNodale.from_graph(cible_graph, vl)

    # --- Phase 1 : séquence nodale sûre ------------------------------------
    if reconnections:
        # Faisabilité : départs cibles présents
        departs_poste = {c.equipment_id for c in cells.cellules_depart}
        for c in cells.cellules_depart:
            departs_poste |= set(c.shared_equipment_ids)
        manquants = set(topo_cible.noeud_par_depart) - departs_poste

        res = ResultatManoeuvres(
            voltage_level_id=vl,
            topo_initiale=poste.topologie_nodale,
            topo_cible=topo_cible,
        )
        if manquants:
            res.topo_obtenue = poste.topologie_nodale
            res.message = ("Topologie nodale cible non atteinte : départs cibles "
                           f"absents du poste : {sorted(manquants)}")
            return res

        placement, faisable, msg = _placement_avec_reconnexions(
            poste, cible_graph, topo_cible, reconnections)
        if not faisable:
            res.topo_obtenue = poste.topologie_nodale
            res.message = "Topologie nodale cible non atteinte : " + msg
            return res

        # Séquenceur sur le poste RÉEL (DJ reconnectés encore ouverts).
        res = determiner_manoeuvres_avec_sections(poste, placement)
        res.topo_initiale = poste.topologie_nodale
        res.topo_cible = topo_cible
    else:
        res = determiner_topo_complete_cible(poste, topo_cible)
        if not res.is_verified:
            res.message = "Topologie nodale cible non atteinte : " + res.message
            return res

    # État détaillé atteint après la séquence nodale
    G = poste.graph.copy()
    for m in res.manoeuvres:
        _set_switch(G, m.switch_id, m.action == "OPEN")

    # 2. Raffinement : ramener chaque départ sur sa barre cible (boucle courte,
    #    équipotentielle puisque le nœud est déjà constitué). Pour un départ
    #    reconnecté, le DJ est encore ouvert : le ré-aiguillage SA reste sûr.
    extra: list[Manoeuvre] = []
    for eq, target in sorted(cible_busbar.items()):
        cell = cells.get_cellule_depart(eq)
        cur = _wired_busbar(cell, G)
        if cur == target:
            continue
        if cur is None:
            # Départ non câblé : on ne « garage » son SA sur la barre cible que
            # s'il est hors tension (DJ propre ouvert) — manœuvre sûre et
            # conforme à la cible détaillée (préparation de section).
            if not any(_is_open(G, b.switch_id) for b in cell.breakers):
                continue
        if _reaiguiller_vers_sjb(G, cells, eq, target, extra):
            res.departs_reaiguilles.add(eq)

    # 3. Changements de DJ de départ
    post_manoeuvres: list[Manoeuvre] = []
    for cell, dj_sw in reconnections:
        # La barre cible est désormais au bon potentiel : fermeture sûre.
        if _is_open(G, dj_sw.switch_id):
            post_manoeuvres.append(Manoeuvre(
                switch_id=dj_sw.switch_id,
                action="CLOSE",
                raison=f"mise en service départ {cell.equipment_id}",
            ))
            _set_switch(G, dj_sw.switch_id, False)
    for cell, dj_sw in disconnections:
        if not _is_open(G, dj_sw.switch_id):
            post_manoeuvres.append(Manoeuvre(
                switch_id=dj_sw.switch_id,
                action="OPEN",
                raison=f"mise hors service départ {cell.equipment_id}",
            ))
            _set_switch(G, dj_sw.switch_id, True)

    res.manoeuvres = res.manoeuvres + extra + post_manoeuvres

    # 4. Vérification nodale + détaillée + écarts
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = topo_cible.meme_topologie(res.topo_obtenue)
    res.is_changed = bool(res.manoeuvres)
    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_securite_sectionneurs(poste, res.manoeuvres))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if not res.is_verified:
        res.message = (
            "Topologie nodale cible non atteinte : la topologie obtenue ne "
            f"correspond pas à la cible (obtenu {res.topo_obtenue.nb_noeuds} "
            f"nœud(s), visé {topo_cible.nb_noeuds})."
        )
    elif res.is_verified_detaillee:
        res.message = "Topologie détaillée cible atteinte et vérifiée."
    else:
        res.message = (
            f"Topologie nodale atteinte ; {len(res.ecarts)} écart(s) détaillé(s) "
            "résiduel(s) : " + " ; ".join(res.ecarts[:6])
        )
    return res


def _ecarts_detailles(
    poste: PosteTopologique,
    G: nx.Graph,
    cible_graph: nx.Graph,
    cible_busbar: dict[str, int],
) -> list[str]:
    """Liste des écarts entre l'état détaillé obtenu ``G`` et la cible."""
    cells = poste.cellules
    sjb_id = {n: G.nodes[n].get("busbar_section_id")
              for n in poste.tronconnement.barre_par_busbar}
    ecarts: list[str] = []
    # Barre de chaque départ
    for eq, target in cible_busbar.items():
        cell = cells.get_cellule_depart(eq)
        cur = _wired_busbar(cell, G)
        if cur != target:
            ecarts.append(
                f"'{eq}' sur {sjb_id.get(cur, cur)} au lieu de {sjb_id.get(target, target)}")
    # État des couplers inter-SJB
    coupling_sids: set[str] = set()
    for cp in _inter_sjb_couplers(poste):
        for sid in cp.switch_ids:
            coupling_sids.add(sid)
            cur = _is_open(G, sid)
            tgt = any(cible_graph.edges[u, v].get("open", False)
                      for u, v in _edges_of_switches(cible_graph, [sid]))
            if cur != tgt:
                ecarts.append(f"organe {sid} {'ouvert' if cur else 'fermé'} "
                              f"au lieu de {'ouvert' if tgt else 'fermé'}")
    # État des DJ de départ (hors couplage)
    for c in cells.cellules_depart:
        for sw in c.breakers:
            if sw.switch_id in coupling_sids:
                continue
            cur = _is_open(G, sw.switch_id)
            tgt = _is_open(cible_graph, sw.switch_id)
            if cur != tgt:
                ecarts.append(
                    f"DJ {sw.switch_id} {'ouvert' if cur else 'fermé'} "
                    f"au lieu de {'ouvert' if tgt else 'fermé'}")
    return ecarts


# ---------------------------------------------------------------------------
# Phases 2.2-2.4 — Placement automatique des nœuds sur les sections de barres
# ---------------------------------------------------------------------------
#
# Généralise ``evalueEtatCouplage`` + ``identifySuperTronconnement`` +
# ``getTronconnementBesoinReaiguillage2barres`` du C++ : à partir d'une
# topologie nodale cible, on attribue à chaque nœud un **groupe de SJB**.
#
# Modèle (segments de barres) :
# - chaque nœud cible se voit attribuer un ensemble **connexe** de SJB (dans le
#   graphe des couplers). Les couplers internes au groupe sont **fermés** (les
#   barres du groupe forment un seul potentiel), ceux entre groupes **ouverts** ;
# - un nœud peut donc **occuper plusieurs barres** (couplage fermé) : c'est le
#   cas privilégié quand il y a moins de nœuds que de barres (R6) — on referme
#   le couplage plutôt que de ramener tous les départs sur une seule barre ;
# - un départ **reste** sur sa barre courante si elle appartient au groupe de
#   son nœud (ré-aiguillage évité) ;
# - créer plus de nœuds que de barres impose d'ouvrir des sectionnements (R10).
#
# On choisit l'affectation SJB→nœud (groupes connexes) qui minimise
# (ré-aiguillages + manœuvres de couplers, sectionnements pénalisés).
# ---------------------------------------------------------------------------

def _placement_automatique(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
) -> tuple[list[tuple[set[str], set[str]]], bool, str]:
    """
    Calcule un placement ``[(departs, sjb_ids)]`` réalisant ``topo_cible``.

    Returns
    -------
    (placement, faisable, message)
    """
    import itertools

    G = poste.graph
    barre_par = poste.tronconnement.barre_par_busbar
    sjb_nodes = sorted(barre_par)
    sjb_id = {n: G.nodes[n].get("busbar_section_id") for n in sjb_nodes}

    # Connexité / câblage courant **par équipement** : la connexité électrique
    # est jugée sur un chemin de switches fermés depuis le nœud propre de
    # l'équipement (gère les omnibus : un groupe isolé, DJ propre ouvert, n'est
    # pas connecté même si son co-locataire l'est).
    eq_node = {data.get("equipment_id"): n for n, data in G.nodes(data=True)
               if data.get("equipment_id")}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if not d.get("open", False):
            H.add_edge(u, v)

    R: dict[str, frozenset] = {}        # SJB atteignables (classe de position)
    connected: dict[str, bool] = {}
    wired_sjb: dict[str, Optional[int]] = {}   # SJB où le départ est câblé (SA fermé)
    for c in poste.cellules.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            R[eq] = frozenset(c.busbar_nodes)
            en = eq_node.get(eq)
            connected[eq] = bool(
                en is not None and en in H
                and any(s in H and nx.has_path(H, en, s) for s in R[eq])
            )
            wired = [bb for bb in c.busbar_nodes
                     if _sa_path_to_sjb(c, bb)
                     and all(not _is_open(G, s) for s in _sa_path_to_sjb(c, bb))]
            wired_sjb[eq] = wired[0] if wired else None

    # Nœuds à placer : ceux ayant ≥ 1 départ actuellement connecté.
    nodes: list[list[str]] = []
    for noeud in topo_cible.noeuds.values():
        deps = [e for e in noeud.equipment_ids if e in R and connected[e]]
        if deps:
            nodes.append(deps)

    if not nodes:
        return [], True, "Aucun nœud connecté à placer."

    k = len(nodes)
    if k > len(sjb_nodes):
        return ([], False,
                f"{k} nœuds pour {len(sjb_nodes)} SJB : topologie impossible.")

    # Graphe des couplers entre SJB (pour la contrainte de connexité des groupes)
    couplers = _inter_sjb_couplers(poste)
    CG = nx.Graph()
    CG.add_nodes_from(sjb_nodes)
    for cp in couplers:
        CG.add_edge(cp.sjb_a, cp.sjb_b)

    node_of_dep = {eq: i for i, deps in enumerate(nodes) for eq in deps}

    # Garde-fou combinatoire
    if k ** len(sjb_nodes) > 500_000:
        return ([], False, "Espace de placement trop grand (poste non géré).")

    best = None  # (cost, assign tuple)
    for assign in itertools.product(range(k), repeat=len(sjb_nodes)):
        node_sjbs: dict[int, set[int]] = {i: set() for i in range(k)}
        for j, ni in enumerate(assign):
            node_sjbs[ni].add(sjb_nodes[j])
        if any(not s for s in node_sjbs.values()):
            continue  # chaque nœud doit avoir ≥ 1 SJB
        # Groupes connexes dans le graphe des couplers
        if not all(nx.is_connected(CG.subgraph(s)) for s in node_sjbs.values()):
            continue
        # Faisabilité : chaque départ atteint une SJB de son nœud
        ok = True
        for i, deps in enumerate(nodes):
            sset = node_sjbs[i]
            if any(not (R[eq] & sset) for eq in deps):
                ok = False
                break
        if not ok:
            continue
        # Coût : ré-aiguillages (départ dont la barre câblée n'est pas dans son
        # groupe) + manœuvres de couplers (sectionnements pénalisés).
        reaig = sum(
            1 for i, deps in enumerate(nodes) for eq in deps
            if wired_sjb.get(eq) not in node_sjbs[i]
        )
        node_of_sjb = {sjb_nodes[j]: assign[j] for j in range(len(sjb_nodes))}
        cpl = 0
        sect = 0
        for cp in couplers:
            same = node_of_sjb[cp.sjb_a] == node_of_sjb[cp.sjb_b]
            currently_closed = all(not _is_open(G, s) for s in cp.switch_ids)
            if same and not currently_closed:
                cpl += 1
            elif (not same) and currently_closed:
                cpl += 1
                if cp.is_sectionnement:
                    sect += 1
        cost = 5 * reaig + cpl + 4 * sect
        if best is None or cost < best[0]:
            best = (cost, assign)

    if best is None:
        return [], False, "Aucune affectation de SJB réalisable (topologie impossible)."

    assign = best[1]
    node_sjbs = {i: set() for i in range(k)}
    for j, ni in enumerate(assign):
        node_sjbs[ni].add(sjb_nodes[j])
    placement = [
        (set(nodes[i]), {sjb_id[s] for s in node_sjbs[i]})
        for i in range(k)
    ]
    return placement, True, "OK"


def _set_switch(G: nx.Graph, switch_id: str, open_: bool) -> None:
    """Modifie l'état d'un switch (par son id) dans le graphe simulé."""
    for u, v, d in G.edges(data=True):
        if d.get("switch_id") == switch_id:
            d["open"] = open_
            return


# ===========================================================================
# Placement sur sections de barres — règle du sectionnement de barre
# ===========================================================================
#
# Un **sectionneur de barre** (sectionnement, DISCONNECTOR entre deux SJB) ne se
# manœuvre que **hors charge**. Pour l'ouvrir afin de créer un nœud
# supplémentaire (au-delà du nombre de jeux de barres), la section à isoler doit
# d'abord être mise **hors tension** : tous ses départs sont ré-aiguillés sur
# l'autre barre (boucle courte, tant que le couplage est fermé), puis le
# sectionnement est ouvert, puis les départs du nouveau nœud sont ré-aiguillés
# (boucle longue) sur la section désormais isolée.
#
# Un **couplage** (travée à DJ/BREAKER) coupe le courant nominal : son DJ
# s'ouvre directement.
#
# Cette logique permet de réaliser des cibles à *plus de nœuds que de barres*.
# ---------------------------------------------------------------------------


@dataclass
class _InterSjbCoupler:
    """Liaison entre deux SJB : sectionnement (SA seuls) ou couplage (avec DJ)."""
    sjb_a: int
    sjb_b: int
    switch_ids: list[str]
    breaker_ids: list[str]

    @property
    def is_sectionnement(self) -> bool:
        return not self.breaker_ids


def _inter_sjb_couplers(poste: PosteTopologique) -> list[_InterSjbCoupler]:
    """
    Recense les liaisons inter-SJB (sectionnements et couplages) d'un poste,
    en contractant les nœuds intermédiaires du sous-graphe de couplage.
    """
    G = poste.graph
    bb_nodes = set(poste.tronconnement.barre_par_busbar)

    # Sous-graphe de couplage : SJB + nœuds hors cellules de départ
    departure_internal: set[int] = set()
    for c in poste.cellules.cellules_depart:
        departure_internal.update(n for n in c.all_nodes if n not in bb_nodes)
    coupler_nodes = bb_nodes | (set(G.nodes()) - departure_internal)
    coupler_G = G.subgraph(coupler_nodes)

    couplers: list[_InterSjbCoupler] = []
    seen: set[frozenset] = set()
    bb_list = sorted(bb_nodes)
    for i, a in enumerate(bb_list):
        for b in bb_list[i + 1:]:
            key = frozenset((a, b))
            if key in seen:
                continue
            # Chemin entre a et b ne traversant aucune autre SJB
            others = bb_nodes - {a, b}
            H = coupler_G.subgraph(coupler_nodes - others)
            try:
                path = nx.shortest_path(H, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            sw_ids, brk_ids = [], []
            for u, v in zip(path, path[1:]):
                d = H.edges[u, v]
                sid = d.get("switch_id")
                if sid is None:
                    continue
                sw_ids.append(sid)
                if d.get("kind") == SwitchKind.BREAKER:
                    brk_ids.append(sid)
            if sw_ids:
                couplers.append(_InterSjbCoupler(a, b, sw_ids, brk_ids))
                seen.add(key)
    return couplers


def determiner_manoeuvres_avec_sections(
    poste: PosteTopologique,
    placement: list[tuple[set[str], set[str]]],
) -> ResultatManoeuvres:
    """
    Calcule la séquence de manœuvres pour réaliser un **placement explicite**
    de nœuds sur des sections de jeux de barres, en respectant la règle du
    sectionnement de barre (dé-énergisation avant ouverture).

    Parameters
    ----------
    poste :
        Vue complète du poste.
    placement :
        Liste de ``(departs, sjb_ids)`` : chaque entrée décrit un nœud cible,
        l'ensemble de ses départs et l'ensemble des SJB (``busbar_section_id``)
        qu'il occupe. Les départs non cités restent inchangés.

    Returns
    -------
    ResultatManoeuvres
    """
    vl = poste.voltage_level_id
    G = poste.graph.copy()
    cells = poste.cellules

    # Map busbar_section_id -> node
    sjb_node_par_id = {
        G.nodes[n].get("busbar_section_id"): n
        for n in poste.tronconnement.barre_par_busbar
    }

    # --- cible nodale (pour la vérification) -------------------------------
    # Les départs placés forment les nœuds explicites ; les départs non cités
    # conservent leur nœud courant (ex. générateurs isolés).
    groupes = [sorted(d) for d, _ in placement]
    places = {eq for d, _ in placement for eq in d}
    for noeud in poste.topologie_nodale.noeuds.values():
        reste = sorted(noeud.equipment_ids - places)
        if reste:
            groupes.append(reste)
    topo_cible = TopologieNodale.from_node_groups(vl, groupes)
    res = ResultatManoeuvres(
        voltage_level_id=vl,
        topo_initiale=poste.topologie_nodale,
        topo_cible=topo_cible,
    )

    # --- résolution départ -> SJB cible ------------------------------------
    target_sjb: dict[str, int] = {}
    node_de_sjb: dict[int, int] = {}  # sjb -> index de nœud
    for idx, (departs, sjb_ids) in enumerate(placement):
        sjb_set = {sjb_node_par_id[s] for s in sjb_ids if s in sjb_node_par_id}
        for s in sjb_set:
            node_de_sjb[s] = idx
        for eq in departs:
            cell = cells.get_cellule_depart(eq)
            if cell is None:
                continue
            reachable = cell.busbar_nodes & sjb_set
            if not reachable:
                # Non bloquant : on consigne l'écart et on laisse le départ en
                # place (la vérification finale le signalera).
                res.ecarts.append(
                    f"'{eq}' ne peut atteindre aucune SJB de son nœud cible")
                continue
            # On garde le départ sur sa barre actuelle si elle est dans le groupe
            # du nœud (évite un ré-aiguillage inutile) ; sinon on prend une SJB
            # du groupe.
            wired = [bb for bb in reachable
                     if _sa_path_to_sjb(cell, bb)
                     and all(not _is_open(G, s) for s in _sa_path_to_sjb(cell, bb))]
            target_sjb[eq] = wired[0] if wired else min(reachable)

    # --- couplers à ouvrir / fermer ----------------------------------------
    # - à ouvrir  : entre SJB de nœuds différents,
    # - à fermer  : entre SJB d'un même nœud actuellement séparées (fusion).
    couplers = _inter_sjb_couplers(poste)
    to_open: list[_InterSjbCoupler] = []
    to_close: list[_InterSjbCoupler] = []
    for cp in couplers:
        na, nb = node_de_sjb.get(cp.sjb_a), node_de_sjb.get(cp.sjb_b)
        if na is None or nb is None:
            continue
        if na != nb:
            to_open.append(cp)
        elif any(_is_open(G, sid) for sid in cp.switch_ids):
            to_close.append(cp)

    # --- groupes SJB finaux (couplers gardés fermés) -----------------------
    sjb_graph = nx.Graph()
    sjb_graph.add_nodes_from(poste.tronconnement.barre_par_busbar)
    open_ids = {sid for cp in to_open for sid in cp.switch_ids}
    for cp in couplers:
        if cp not in to_open:
            sjb_graph.add_edge(cp.sjb_a, cp.sjb_b)
    groupe_sjb = {}
    for gid, comp in enumerate(nx.connected_components(sjb_graph)):
        for s in comp:
            groupe_sjb[s] = gid

    # Référence = groupe portant le plus de départs cibles (le « tronc »)
    from collections import Counter
    poids = Counter(groupe_sjb[s] for s in target_sjb.values() if s in groupe_sjb)
    ref_group = poids.most_common(1)[0][0] if poids else None
    ref_sjbs = {s for s, g in groupe_sjb.items() if g == ref_group}

    # SJB « derrière un sectionnement » : au moment où l'on ouvre les
    # sectionnements (phase C), les couplages (DJ) sont *encore fermés*. On
    # calcule donc la connectivité en gardant fermés tous les couplers SAUF les
    # sectionnements à ouvrir. La composante ne contenant pas la référence est
    # la section à mettre hors tension.
    to_open_sect_ids = {sid for cp in to_open if cp.is_sectionnement
                        for sid in cp.switch_ids}
    energ_graph = nx.Graph()
    energ_graph.add_nodes_from(poste.tronconnement.barre_par_busbar)
    for cp in couplers:
        if any(sid in to_open_sect_ids for sid in cp.switch_ids):
            continue  # sectionnement ouvert en phase C
        energ_graph.add_edge(cp.sjb_a, cp.sjb_b)
    ref_repr = next(iter(ref_sjbs), None)
    energises = (nx.node_connected_component(energ_graph, ref_repr)
                 if ref_repr is not None else set())
    sjb_isoles = set(poste.tronconnement.barre_par_busbar) - energises

    # R10bis — « isoler par les disjoncteurs d'abord » : une section n'a besoin
    # de **parking / dé-énergisation** que si son isolement passe par l'ouverture
    # d'un **sectionnement fermé** (organe hors charge). Une section isolable par
    # simple ouverture d'un **couplage (DJ, qui coupe la charge)** — ou par un
    # sectionnement déjà ouvert — garde ses départs en place. On ne conserve donc
    # dans ``sjb_isoles`` que les sections **incidentes à un sectionnement fermé
    # destiné à s'ouvrir**.
    sect_isol_sjbs: set[int] = set()
    for cp in to_open:
        if cp.is_sectionnement and any(not _is_open(G, s) for s in cp.switch_ids):
            sect_isol_sjbs.add(cp.sjb_a)
            sect_isol_sjbs.add(cp.sjb_b)
    sjb_isoles &= sect_isol_sjbs

    manoeuvres: list[Manoeuvre] = []
    reaiguilles: set[str] = set()

    # Index nœud -> SJB et départ -> nœud (pour la dé-énergisation des stubs)
    node_sjb_sets: dict[int, set[int]] = {}
    for s, idx in node_de_sjb.items():
        node_sjb_sets.setdefault(idx, set()).add(s)
    node_de_dep = {eq: idx for idx, (deps, _) in enumerate(placement)
                   for eq in deps}

    def parking_sjb(eq: str, target: int) -> Optional[int]:
        """SJB tampon où garer temporairement le départ pendant l'ouverture du
        sectionnement isolant sa cible. Idéalement hors section isolée ; sinon
        toute SJB accessible actuellement équipotentielle (parking en boucle
        courte, même si elle sera isolée ensuite) ; en dernier recours toute SJB
        accessible distincte de la cible."""
        cell = cells.get_cellule_depart(eq)
        for bb in cell.busbar_nodes:                       # 1) hors section isolée
            if bb != target and bb not in sjb_isoles:
                return bb
        cur = _wired_busbar(cell, G)                       # 2) équipotentielle
        for bb in cell.busbar_nodes:
            if bb != target and (cur is None or _equipotentiel(bb, cur)):
                return bb
        for bb in cell.busbar_nodes:                       # 3) dernier recours
            if bb != target:
                return bb
        return None

    def _equipotentiel(a: int, b: int) -> bool:
        """True si deux SJB sont au même potentiel (chemin de switches fermés)."""
        Hc = nx.Graph()
        Hc.add_nodes_from(G.nodes())
        for u, v, dd in G.edges(data=True):
            if not dd.get("open", False):
                Hc.add_edge(u, v)
        return a in Hc and b in Hc and nx.has_path(Hc, a, b)

    def _departs_cables(s: int) -> list[str]:
        return [eq for eq in target_sjb if s in _wired_sjbs(G, cells, eq)]

    def _fermer_coupler(cp: _InterSjbCoupler, raison: str) -> None:
        for sid in cp.switch_ids:
            if _is_open(G, sid):
                _set_switch(G, sid, False)
                manoeuvres.append(Manoeuvre(sid, "CLOSE", raison))

    # --- Phase 0 : fermeture SÛRE des couplers (règle du sectionneur) -------
    # Un DJ de couplage peut relier deux potentiels différents (couplage) ; un
    # sectionneur ne se ferme que si ses deux côtés sont déjà équipotentiels ou
    # si l'un est hors tension. On ferme donc d'abord les DJ (qui équipotentient
    # leurs barres), puis les sectionneurs devenus sûrs.
    restants = [cp for cp in to_close
                if any(_is_open(G, s) for s in cp.switch_ids)]
    changed = True
    while changed and restants:
        changed = False
        for cp in list(restants):
            if cp.breaker_ids:                       # DJ -> couplage sûr
                _fermer_coupler(cp, "fermeture couplage de barres")
                restants.remove(cp); changed = True
            elif (_equipotentiel(cp.sjb_a, cp.sjb_b)
                  or not _departs_cables(cp.sjb_a)
                  or not _departs_cables(cp.sjb_b)):  # sectionneur sûr
                _fermer_coupler(cp, "fermeture sectionnement (barres équipotentielles)")
                restants.remove(cp); changed = True

    # Sectionneurs encore non sûrs : dé-énergiser le côté « stub » (moins de
    # départs) en ré-aiguillant ses départs vers une SJB du même nœud déjà
    # équipotentielle au côté conservé (manœuvre préalable), puis fermer.
    for cp in restants:
        a, b = cp.sjb_a, cp.sjb_b
        wa, wb = _departs_cables(a), _departs_cables(b)
        stub, keep = (b, a) if len(wb) <= len(wa) else (a, b)
        for eq in _departs_cables(stub):
            idx = node_de_dep.get(eq)
            cell = cells.get_cellule_depart(eq)
            alts = [bb for bb in cell.busbar_nodes
                    if idx is not None and bb in node_sjb_sets.get(idx, set())
                    and bb != stub and _equipotentiel(bb, keep)]
            if alts and _reaiguiller_vers_sjb(G, cells, eq, alts[0], manoeuvres):
                reaiguilles.add(eq)
                target_sjb[eq] = alts[0]   # éviter un retour en phase A/B
        if not _departs_cables(stub):
            _fermer_coupler(cp, "fermeture sectionnement (section mise hors tension)")

    # --- Phase A/B : ré-aiguillages boucle courte (couplage encore fermé) ---
    parkings: dict[str, int] = {}
    for eq, tgt in sorted(target_sjb.items()):
        if tgt in sjb_isoles:
            buf = parking_sjb(eq, tgt)
            if buf is None:
                # Pas de SJB tampon (ex. poste 1 barre multi-sections) : on ne
                # gare pas le départ ici. La section sera mise hors tension par
                # ouverture de ses DJ d'ouvrage en phase C (repli dé-énergisation)
                # avant d'ouvrir le sectionneur.
                continue
            parkings[eq] = tgt
            if _reaiguiller_vers_sjb(G, cells, eq, buf, manoeuvres):
                reaiguilles.add(eq)
        else:
            if _reaiguiller_vers_sjb(G, cells, eq, tgt, manoeuvres):
                reaiguilles.add(eq)

    # --- Phase C : ouverture des sectionnements (règle du sectionneur) -----
    # Règle : un sectionneur de barre ne se manœuvre que hors charge. Avant
    # chaque ouverture, on vérifie par parcours du graphe « live » (switches
    # fermés) l'état des deux côtés une fois le sectionneur ouvert :
    #   - chemin parallèle conservé        -> manœuvre en boucle (équipotentiel) ;
    #   - au moins un côté hors tension     -> ouverture directe sûre ;
    #   - deux côtés sous tension           -> dé-énergisation préalable du côté
    #     le plus petit (ouverture de ses DJ d'ouvrage), ouverture du sectionneur,
    #     puis ré-énergisation (refermeture des DJ). Coupure momentanée assumée.
    all_sjb = set(poste.tronconnement.barre_par_busbar)
    for cp in to_open:
        if not cp.is_sectionnement:
            continue
        if all(_is_open(G, sid) for sid in cp.switch_ids):
            continue

        def _ouvrir(raison: str) -> None:
            for sid in cp.switch_ids:
                if not _is_open(G, sid):
                    _set_switch(G, sid, True)
                    manoeuvres.append(Manoeuvre(sid, "OPEN", raison))

        H = _live_graph_sans(G, cp.switch_ids)
        a, b = cp.sjb_a, cp.sjb_b
        if a in H and b in H and nx.has_path(H, a, b):
            # Les deux côtés restent reliés par un chemin parallèle : ouverture
            # en boucle, sans divergence de potentiel.
            _ouvrir("ouverture sectionnement de barre (boucle, chemin parallèle)")
            continue

        side_a = (nx.node_connected_component(H, a) if a in H else {a}) & all_sjb
        side_b = (nx.node_connected_component(H, b) if b in H else {b}) & all_sjb
        liv_a = _ouvrages_energises_sur(G, cells, side_a, H)
        liv_b = _ouvrages_energises_sur(G, cells, side_b, H)
        if not liv_a or not liv_b:
            _ouvrir("ouverture sectionnement de barre (section hors tension)")
            continue

        # Deux côtés sous tension. Côté à isoler = le plus petit (en ouvrages
        # énergisés). R10bis : on l'**isole d'abord par les disjoncteurs** en
        # ouvrant les **couplages** (DJ, qui coupent la charge) destinés à
        # s'ouvrir et reliant cette section à l'extérieur. Cela réduit le résidu
        # à dé-énergiser ; on ne manœuvre les DJ d'ouvrage qu'en dernier recours.
        side_isol = side_a if len(liv_a) <= len(liv_b) else side_b

        for cpl in to_open:
            if cpl.is_sectionnement:
                continue
            # couplage touchant la section à isoler (frontière ou interne) :
            # son ouverture (DJ, hors charge) réduit la section à dé-énergiser.
            if cpl.sjb_a in side_isol or cpl.sjb_b in side_isol:
                for sid in cpl.breaker_ids:
                    if not _is_open(G, sid):
                        _set_switch(G, sid, True)
                        manoeuvres.append(Manoeuvre(
                            sid, "OPEN",
                            "ouverture couplage de barres (isolement de la section)"))

        # Recalcul du côté à isoler après ouverture des couplages adjacents.
        H2 = _live_graph_sans(G, cp.switch_ids)
        a_isol = a if a in side_isol else b
        side_isol = ((nx.node_connected_component(H2, a_isol)
                      if a_isol in H2 else {a_isol}) & all_sjb)
        liv_isol = _ouvrages_energises_sur(G, cells, side_isol, H2)
        if not liv_isol:
            _ouvrir("ouverture sectionnement de barre (section hors tension)")
            continue
        if not all(brk for _, brk in liv_isol):
            # Ouvrage sans DJ propre : dé-énergisation impossible.
            _ouvrir("ouverture sectionnement de barre (section ATTENTION sous tension)")
            continue

        djs_rouverts: list[str] = []
        for eq, brk in liv_isol:
            for sid in brk:
                if not _is_open(G, sid):
                    _set_switch(G, sid, True)
                    manoeuvres.append(Manoeuvre(
                        sid, "OPEN",
                        f"mise hors tension '{eq}' (avant ouverture sectionneur)"))
                    djs_rouverts.append(sid)
        _ouvrir("ouverture sectionnement de barre (section hors tension)")
        for sid in djs_rouverts:
            _set_switch(G, sid, False)
            manoeuvres.append(Manoeuvre(
                sid, "CLOSE", "remise sous tension (après ouverture sectionneur)"))

    # --- Phase D : ouverture des couplages (DJ) ----------------------------
    for cp in to_open:
        if cp.is_sectionnement:
            continue
        for sid in cp.breaker_ids:
            if not _is_open(G, sid):
                _set_switch(G, sid, True)
                manoeuvres.append(Manoeuvre(
                    switch_id=sid, action="OPEN",
                    raison="ouverture couplage de barres",
                ))

    # --- Phase E : ré-aiguillage boucle longue vers sections isolées -------
    for eq in sorted(parkings):
        if _reaiguiller_vers_sjb(G, cells, eq, parkings[eq], manoeuvres):
            reaiguilles.add(eq)

    # --- Optimisation : suppression des manœuvres sans effet (listeDordre) -
    manoeuvres = _optimiser_sequence(poste, manoeuvres)

    res.manoeuvres = manoeuvres
    res.departs_reaiguilles = reaiguilles
    res.couplages_modifies = [sid for cp in (to_open + to_close)
                              for sid in cp.switch_ids]

    # --- Vérification ------------------------------------------------------
    topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.topo_obtenue = topo_obtenue
    res.is_verified = topo_cible.meme_topologie(topo_obtenue)
    res.is_changed = bool(manoeuvres)
    # Sûreté des sectionneurs : signaler toute ouverture restée sous tension.
    res.ecarts += _verifier_securite_sectionneurs(poste, manoeuvres)
    res.message = (
        "Topologie cible atteinte et vérifiée."
        if res.is_verified
        else f"Cible non atteinte (obtenu {topo_obtenue.nb_noeuds} nœuds, "
             f"visé {topo_cible.nb_noeuds})."
    )
    return res


# ---------------------------------------------------------------------------
# Helpers bas niveau opérant sur le graphe « live »
# ---------------------------------------------------------------------------

def _live_graph_sans(G: nx.Graph, switch_ids) -> nx.Graph:
    """Sous-graphe des switches **fermés**, en forçant l'ouverture (le retrait)
    des switches ``switch_ids`` — utilisé pour évaluer la connectivité une fois
    un sectionneur ouvert."""
    forces = set(switch_ids)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d.get("switch_id") in forces:
            continue
        if not d.get("open", False):
            H.add_edge(u, v)
    return H


def _ouvrages_energises_sur(
    G: nx.Graph, cells, side_sjbs: set[int], H: nx.Graph
) -> list[tuple[str, list[str]]]:
    """Ouvrages **énergisant** le côté ``side_sjbs`` : ceux dont le nœud propre
    est relié, par un chemin de switches **fermés** dans ``H``, à une SJB du
    côté. ``H`` est le graphe « live » privé du sectionneur considéré, de sorte
    que la distinction des deux côtés est correcte.

    La connectivité électrique est utilisée (et non le câblage SA) pour capter
    aussi les ouvrages raccordés directement par disjoncteur sans sectionneur
    d'aiguillage (ex. côté HT d'un transformateur).

    Retourne ``[(equipment_id, [breaker_ids])]``. ``breaker_ids`` vide signale
    un ouvrage **sans DJ propre** : il ne peut être mis hors tension par
    ouverture de DJ."""
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for c in cells.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            if eq in seen:
                continue
            cell = cells.get_cellule_depart(eq)
            if cell is None:
                continue
            en = _eq_node(G, eq)
            if en is None or en not in H:
                continue
            if not any(s in H and nx.has_path(H, en, s) for s in side_sjbs):
                continue
            seen.add(eq)
            out.append((eq, [b.switch_id for b in cell.breakers]))
    return out


def _verifier_securite_sectionneurs(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Rejoue la séquence depuis l'état initial et vérifie la **règle du
    sectionneur** : à chaque ouverture d'un sectionnement de barre, au moins un
    côté doit être hors tension (ou les deux côtés rester reliés par un chemin
    parallèle). Retourne la liste des écarts (sectionneurs ouverts sous tension)."""
    sect_ids: dict[str, tuple[int, int]] = {}
    for cp in _inter_sjb_couplers(poste):
        if cp.is_sectionnement:
            for sid in cp.switch_ids:
                sect_ids[sid] = (cp.sjb_a, cp.sjb_b)
    if not sect_ids:
        return []

    cells = poste.cellules
    all_sjb = set(poste.tronconnement.barre_par_busbar)
    G = poste.graph.copy()
    ecarts: list[str] = []
    for m in manoeuvres:
        if (m.action == "OPEN" and m.switch_id in sect_ids
                and not _is_open(G, m.switch_id)):
            a, b = sect_ids[m.switch_id]
            H = _live_graph_sans(G, [m.switch_id])
            relies = a in H and b in H and nx.has_path(H, a, b)
            if not relies:
                side_a = (nx.node_connected_component(H, a)
                          if a in H else {a}) & all_sjb
                side_b = (nx.node_connected_component(H, b)
                          if b in H else {b}) & all_sjb
                if (_ouvrages_energises_sur(G, cells, side_a, H)
                        and _ouvrages_energises_sur(G, cells, side_b, H)):
                    ecarts.append(
                        f"sectionneur {m.switch_id} ouvert sous tension "
                        "(deux côtés énergisés)")
        _set_switch(G, m.switch_id, m.action == "OPEN")
    return ecarts


def _optimiser_sequence(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[Manoeuvre]:
    """
    Supprime les manœuvres sans effet en rejouant la séquence depuis l'état
    initial : une manœuvre qui place un OC dans l'état où il se trouve déjà est
    redondante et retirée. Les bascules réelles (ex. ouverture/fermeture d'un DJ
    en boucle longue) sont conservées.
    """
    G = poste.graph.copy()
    out: list[Manoeuvre] = []
    for m in manoeuvres:
        want_open = (m.action == "OPEN")
        if _is_open(G, m.switch_id) == want_open:
            continue
        _set_switch(G, m.switch_id, want_open)
        out.append(m)
    return out


def _is_open(G: nx.Graph, switch_id: str) -> bool:
    for _, _, d in G.edges(data=True):
        if d.get("switch_id") == switch_id:
            return bool(d.get("open", False))
    return True


def _eq_node(G: nx.Graph, eq_id: str) -> Optional[int]:
    for n, d in G.nodes(data=True):
        if d.get("equipment_id") == eq_id:
            return n
    return None


def _sa_path_to_sjb(cell: CelluleDepart, sjb_node: int) -> list[str]:
    """IDs des sectionneurs (SA) sur le chemin départ -> SJB."""
    return [s.switch_id for s in cell.disconnectors_vers_barre(sjb_node)]


def _own_breakers_to_sjb(
    cell: CelluleDepart, sjb_node: int, eq_id: str | None = None
) -> list[str]:
    """
    **Disjoncteur d'ensemble de la cellule** à manœuvrer pour mettre le départ
    hors tension lors d'un ré-aiguillage en boucle longue.

    Règle (cf. docs/manoeuvre_regles.md, R7) : on n'ouvre que le DJ situé
    **côté sélecteurs de barre** (entre les sectionneurs d'aiguillage et le
    reste de la cellule). Ouvrir ce seul disjoncteur dé-énergise la cellule et
    suffit pour basculer ensuite les sectionneurs ; on n'ouvre **pas** les
    disjoncteurs propres aux équipements situés en aval (cas omnibus : un même
    DJ de cellule alimente plusieurs équipements).

    Méthode : on parcourt le chemin équipement → SJB depuis le **côté barre**,
    on saute les sectionneurs (sélecteurs de barre), puis on collecte le(s)
    disjoncteur(s) en série jusqu'à un nœud de branchement (degré > 2, signe
    d'un point omnibus partagé) ou l'équipement.
    """
    if cell.subgraph is None:
        return [b.switch_id for b in cell.breakers]
    sg = cell.subgraph
    eq_id = eq_id or cell.equipment_id
    eq_node = next((n for n, d in sg.nodes(data=True)
                    if d.get("equipment_id") == eq_id), None)
    if eq_node is None:
        return [b.switch_id for b in cell.breakers]
    try:
        path = nx.shortest_path(sg, eq_node, sjb_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    res: list[str] = []
    collecting = False
    # Parcours depuis le côté barre (path[::-1] = [sjb, …, équipement])
    rev = path[::-1]
    for a, b in zip(rev, rev[1:]):
        d = sg.edges[a, b]
        kind = d.get("kind")
        sid = d.get("switch_id")
        if kind == SwitchKind.BREAKER and sid:
            res.append(sid)
            collecting = True
            # Nœud de branchement omnibus atteint -> DJ de cellule identifié.
            if b != eq_node and sg.degree(b) > 2:
                break
        elif kind == SwitchKind.DISCONNECTOR:
            if collecting:
                break          # sectionneur en aval du DJ -> fin
            continue            # sélecteurs de barre en amont -> on saute
        elif collecting:
            break
    return list(dict.fromkeys(res))


def _wired_sjbs(G: nx.Graph, cells, eq_id: str) -> set[int]:
    """SJB vers lesquelles le départ est *câblé* (SA fermés), indépendamment
    de la connectivité électrique globale."""
    cell = cells.get_cellule_depart(eq_id)
    if cell is None:
        return set()
    res = set()
    for bb in cell.busbar_nodes:
        sa = _sa_path_to_sjb(cell, bb)
        if sa and all(not _is_open(G, s) for s in sa):
            res.add(bb)
    return res


def _meme_noeud_hors_cellule(
    G: nx.Graph, cell: CelluleDepart, bb1: int, bb2: int
) -> bool:
    """
    True si les deux SJB sont au **même potentiel** (même nœud électrique) par un
    chemin **n'empruntant pas** les nœuds internes de la cellule ``cell``.

    Sert à appliquer l'invariant de sécurité des sectionneurs : fermer (ou
    ouvrir) un SA d'aiguillage qui relie ``bb1`` et ``bb2`` via la cellule n'est
    sûr que si ``bb1`` et ``bb2`` sont déjà reliés *par ailleurs* (donc déjà au
    même potentiel) — sinon le départ ponterait deux nœuds distincts.
    """
    if bb1 == bb2:
        return True
    internes = {n for n in cell.all_nodes if n not in cell.busbar_nodes}
    H = nx.Graph()
    H.add_nodes_from(n for n in G.nodes() if n not in internes)
    for u, v, d in G.edges(data=True):
        if d.get("open", False) or u in internes or v in internes:
            continue
        H.add_edge(u, v)
    return bb1 in H and bb2 in H and nx.has_path(H, bb1, bb2)


def _reaiguiller_vers_sjb(
    G: nx.Graph,
    cells,
    eq_id: str,
    target_sjb: int,
    manoeuvres: list[Manoeuvre],
    boucle: Optional[Literal["COURTE", "LONGUE"]] = None,
) -> bool:
    """
    Ré-aiguille un départ vers une SJB cible. Retourne True si des manœuvres
    ont été générées.

    Le **type de boucle est déterminé par l'invariant de sécurité des
    sectionneurs** (cf. docs/manoeuvre_regles.md) et non par une heuristique de
    phase :

    - **COURTE** si la barre cible et la (les) barre(s) actuelle(s) du départ
      sont **déjà le même nœud électrique** (reliées par ailleurs, p.ex. via le
      couplage fermé). On ferme alors le SA cible PUIS on ouvre l'ancien SA :
      les deux SA sont brièvement fermés mais entre barres équipotentielles
      (aucun court-circuit), et le départ reste sous tension.
    - **LONGUE** sinon (barres de potentiels différents) : ouvrir le DJ de
      cellule (départ hors tension, jonction morte) → **ouvrir l'ancien SA** →
      **fermer le SA cible** → refermer le DJ. On ne ferme jamais le SA cible
      tant que l'ancien SA est fermé (ponter deux potentiels = court-circuit).

    ``boucle`` peut être forcé, sinon il est déduit automatiquement.
    """
    cell = cells.get_cellule_depart(eq_id)
    if cell is None:
        return False

    sjb_id = (G.nodes[target_sjb].get("busbar_section_id") or str(target_sjb))
    sa_cible = _sa_path_to_sjb(cell, target_sjb)
    if not sa_cible:
        return False
    # Déjà câblé sur la cible et nulle part ailleurs ?
    deja = all(not _is_open(G, s) for s in sa_cible) and all(
        _is_open(G, s)
        for bb in cell.busbar_nodes if bb != target_sjb
        for s in _sa_path_to_sjb(cell, bb)
    )
    if deja:
        return False

    # Barres actuellement câblées (SA fermés) autres que la cible
    old_busbars = [
        bb for bb in cell.busbar_nodes
        if bb != target_sjb and _sa_path_to_sjb(cell, bb)
        and all(not _is_open(G, s) for s in _sa_path_to_sjb(cell, bb))
    ]
    # Invariant : boucle courte possible ssi toutes les anciennes barres sont
    # déjà au même potentiel que la cible (hors cellule).
    if boucle is None:
        boucle = ("COURTE"
                  if all(_meme_noeud_hors_cellule(G, cell, bb, target_sjb)
                         for bb in old_busbars)
                  else "LONGUE")

    n_before = len(manoeuvres)
    djs = _own_breakers_to_sjb(cell, target_sjb, eq_id)

    def _fermer_sa_cible():
        for sa in sa_cible:
            if _is_open(G, sa):
                _set_switch(G, sa, False)
                manoeuvres.append(Manoeuvre(
                    switch_id=sa, action="CLOSE",
                    raison=f"ré-aiguillage '{eq_id}' vers {sjb_id}",
                    type_boucle=boucle,
                ))

    def _ouvrir_sa_anciens():
        for bb in cell.busbar_nodes:
            if bb == target_sjb:
                continue
            for sa in _sa_path_to_sjb(cell, bb):
                if not _is_open(G, sa):
                    _set_switch(G, sa, True)
                    bb_id = G.nodes[bb].get("busbar_section_id") or str(bb)
                    manoeuvres.append(Manoeuvre(
                        switch_id=sa, action="OPEN",
                        raison=f"'{eq_id}' quitte {bb_id}",
                        type_boucle=boucle,
                    ))

    if boucle == "LONGUE":
        # 1) mise hors tension par le DJ de cellule
        for dj in djs:
            if not _is_open(G, dj):
                _set_switch(G, dj, True)
                manoeuvres.append(Manoeuvre(
                    switch_id=dj, action="OPEN",
                    raison=f"mise hors tension '{eq_id}' (boucle longue)",
                    type_boucle="LONGUE",
                ))
        # 2) ouvrir l'ancien SA AVANT 3) fermer le SA cible (jamais de pont)
        _ouvrir_sa_anciens()
        _fermer_sa_cible()
        # 4) remise sous tension
        for dj in djs:
            _set_switch(G, dj, False)
            manoeuvres.append(Manoeuvre(
                switch_id=dj, action="CLOSE",
                raison=f"remise sous tension '{eq_id}' (boucle longue)",
                type_boucle="LONGUE",
            ))
    else:  # COURTE : fermer la cible puis ouvrir l'ancien (même potentiel)
        _fermer_sa_cible()
        _ouvrir_sa_anciens()

    return len(manoeuvres) > n_before
