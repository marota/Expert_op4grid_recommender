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

Postes à > 2 jeux de barres : le placement nodal combinatoire est conçu pour le
double jeu de barres. Sur un poste à plus de 2 jeux de barres (niveaux
supplémentaires 3B/4B, organes internes à 2 bornes type self/réactance, nœuds à
0 barre), ``determiner_manoeuvres_cible_detaillee`` route vers
``_sequence_detaillee_multibarres`` qui **dérive le placement des composantes
connexes du graphe cible** (groupes exacts), laisse en place les organes à 2
bornes (``organes_fixes``), **isole** les nœuds à 0 barre, et détecte **tous les
couplages parallèles** (``_inter_sjb_couplers``). L'API nodale-only
``determiner_topo_complete_cible`` conserve un fallback : scoping aux 2 JdB
principaux + diagnostic + placement partiel best-effort.

Limites connues (documentées, cf. doc C++) :
- Ré-aiguillage d'omnibus complexes (départs multiples scindés)          [partiel]
- Contrôle de court-circuit : vérifie l'équipotentialité courante avant
  manœuvre de sectionneur (pas de calcul de potentiel/déphasage fin)     [simplifié]
- Topologies de couplers non chaînées (≥ 3 barres en anneau)             [partiel]
- Nœuds mêlant départs connectés et déconnectés                          [partiel]
- Postes > 2 jeux de barres : placement par composantes du graphe cible
  (chemin détaillé) ; fallback nodal-only = scoping aux 2 JdB            [OK/partiel]
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
    # Dégradation gracieuse : départs des nœuds cibles **non réalisables** sur ce
    # poste (cible partiellement atteinte, à compléter manuellement par l'opérateur).
    noeuds_non_realisables: list[list[str]] = field(default_factory=list)

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
    cible_busbar: Optional[dict[str, int]] = None,
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
    cible_busbar :
        Optionnel — barre cible exacte de chaque départ (transmis au séquenceur
        pour placer chaque départ directement sur sa barre finale).

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
    placement, faisable, msg, non_places = _placement_automatique(poste, topo_cible)
    if not placement:
        # Rien de plaçable, même en best-effort : aucune manœuvre possible.
        res.topo_obtenue = poste.topologie_nodale
        res.message = msg
        res.noeuds_non_realisables = non_places
        return res

    # --- Délégation au séquenceur général (couplage + sectionnement) -------
    core = determiner_manoeuvres_avec_sections(poste, placement, cible_busbar)
    core.topo_initiale = poste.topologie_nodale
    core.topo_cible = topo_cible
    core.is_verified = bool(
        core.topo_obtenue and topo_cible.meme_topologie(core.topo_obtenue)
    )
    core.is_changed = bool(core.manoeuvres)
    if faisable:
        core.message = (
            "Topologie cible atteinte et vérifiée." if core.is_verified
            else "La topologie obtenue ne correspond pas à la cible "
                 f"(obtenu {core.topo_obtenue.nb_noeuds if core.topo_obtenue else 0} "
                 f"nœud(s), visé {topo_cible.nb_noeuds})."
        )
    else:
        # Dégradation gracieuse (option 4) : placement partiel + diagnostic.
        core.noeuds_non_realisables = non_places
        core.message = msg
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
) -> tuple[list[tuple[set[str], set[str]]], bool, str, list[list[str]]]:
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


def _sequence_detaillee_aggressive(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    cible_busbar: dict[str, int],
) -> ResultatManoeuvres:
    """
    Mode **agressif** : atteint la topologie détaillée cible par une
    orchestration « batch » minimisant les bascules de disjoncteurs.

    Au lieu de dé-énergiser/ré-aiguiller **un ouvrage à la fois** (mode smooth,
    boucle longue, ré-alimentation immédiate), on **cumule** les
    dé-énergisations : on ouvre en une fois les DJ de tous les ouvrages
    concernés, on commute couplages/sectionnements et SA **hors tension**, puis
    on **ré-alimente une seule fois**. Bien moins de manœuvres, au prix de
    plusieurs ouvrages momentanément hors tension simultanément.

    Ordre (règle du sectionneur respectée : sections mortes avant ouverture) :
    1. ouvrir les DJ du **lot** (SA à changer ∪ ouvrages énergisés sur une
       section incidente à un sectionnement fermé à ouvrir) ;
    2. ouvrir les couplages (DJ) destinés à s'ouvrir ;
    3. ouvrir les sectionnements destinés à s'ouvrir (sections désormais mortes) ;
    4. fermer les couplages puis sectionnements destinés à se fermer ;
    5. positionner tous les SA de départ sur leur état cible (hors tension) ;
    6. refermer les DJ du lot / appliquer reconnexions-déconnexions (cible).
    """
    vl = poste.voltage_level_id
    cells = poste.cellules
    G = poste.graph.copy()
    couplers = _inter_sjb_couplers(poste)
    coupling_sids = {s for cp in couplers for s in cp.switch_ids}
    manoeuvres: list[Manoeuvre] = []

    res = ResultatManoeuvres(
        voltage_level_id=vl,
        topo_initiale=poste.topologie_nodale,
        topo_cible=TopologieNodale.from_graph(cible_graph, vl),
    )

    def setsw(sid: str, open_: bool, raison: str) -> None:
        if _is_open(G, sid) != open_:
            _set_switch(G, sid, open_)
            manoeuvres.append(Manoeuvre(
                sid, "OPEN" if open_ else "CLOSE", raison))

    all_sjb = set(poste.tronconnement.barre_par_busbar)

    # Sectionnements fermés destinés à s'ouvrir.
    sect_to_open = [cp for cp in couplers
                    if cp.is_sectionnement
                    and any(not _is_open(G, s) for s in cp.switch_ids)
                    and any(_is_open(cible_graph, s) for s in cp.switch_ids)]

    # --- Lot à dé-énergiser ------------------------------------------------
    # (a) tout ouvrage dont un SA change (ré-aiguillage hors tension) ;
    # (b) tout ouvrage **énergisé** (connectivité électrique réelle) sur une
    #     section isolée par un sectionnement à ouvrir — y compris les ouvrages
    #     raccordés directement par DJ sans SA (ex. transformateurs).
    batch: dict[str, list[str]] = {}
    for c in cells.cellules_depart:
        eq = c.equipment_id
        brk = [b.switch_id for b in c.breakers]
        if not brk or any(_is_open(G, b) for b in brk):
            continue  # pas de DJ propre, ou déjà hors tension
        if any(_is_open(G, s.switch_id) != _is_open(cible_graph, s.switch_id)
               for s in c.disconnectors if s.switch_id not in coupling_sids):
            batch[eq] = brk

    if sect_to_open:
        # Graphe avec les couplages destinés à s'ouvrir déjà ouverts, pour
        # délimiter correctement les sections à isoler.
        G_cut = G.copy()
        for cp in couplers:
            if cp.is_sectionnement:
                continue
            for sid in cp.breaker_ids:
                if _is_open(cible_graph, sid):
                    _set_switch(G_cut, sid, True)
        sect_sids = [s for cp in sect_to_open for s in cp.switch_ids]
        H = _live_graph_sans(G_cut, sect_sids)
        # Pour chaque sectionnement, ne dé-énergiser que **le plus petit côté**
        # (un seul côté mort suffit pour ouvrir le sectionneur) — minimise le lot.
        for cp in sect_to_open:
            sa = ((nx.node_connected_component(H, cp.sjb_a)
                   if cp.sjb_a in H else {cp.sjb_a}) & all_sjb)
            sb = ((nx.node_connected_component(H, cp.sjb_b)
                   if cp.sjb_b in H else {cp.sjb_b}) & all_sjb)
            liv_a = _ouvrages_energises_sur(G_cut, cells, sa, H)
            liv_b = _ouvrages_energises_sur(G_cut, cells, sb, H)
            petit = liv_a if len(liv_a) <= len(liv_b) else liv_b
            for eq, brk in petit:
                if brk and eq not in batch:
                    batch[eq] = brk

    # 1. Dé-énergiser le lot (ouverture des DJ d'ouvrage, une fois chacun).
    for eq, brk in batch.items():
        for sid in brk:
            setsw(sid, True, f"mise hors tension '{eq}' (dé-énergisation groupée)")

    # 2. Ouvrir les couplages (DJ) destinés à s'ouvrir.
    for cp in couplers:
        if cp.is_sectionnement:
            continue
        for sid in cp.breaker_ids:
            if _is_open(cible_graph, sid):
                setsw(sid, True, "ouverture couplage de barres")

    # 3. Ouvrir les sectionnements destinés à s'ouvrir (sections mortes).
    for cp in couplers:
        if not cp.is_sectionnement:
            continue
        for sid in cp.switch_ids:
            if _is_open(cible_graph, sid):
                setsw(sid, True,
                      "ouverture sectionnement de barre (section hors tension)")

    # 4. Fermer les couplages (DJ d'abord) puis sectionnements destinés à fermer.
    for cp in couplers:
        if cp.is_sectionnement:
            continue
        for sid in cp.switch_ids:
            if not _is_open(cible_graph, sid):
                setsw(sid, False, "fermeture couplage de barres")
    for cp in couplers:
        if not cp.is_sectionnement:
            continue
        for sid in cp.switch_ids:
            if not _is_open(cible_graph, sid):
                setsw(sid, False,
                      "fermeture sectionnement de barre (barres équipotentielles)")

    # 5. Positionner tous les SA de départ sur leur état cible (hors tension).
    for c in cells.cellules_depart:
        for s in c.disconnectors:
            if s.switch_id in coupling_sids:
                continue
            setsw(s.switch_id, _is_open(cible_graph, s.switch_id),
                  f"ré-aiguillage '{c.equipment_id}' (hors tension)")

    # 6. Ramener tous les DJ de départ à leur état cible (ré-alimentation /
    #    mise en service / mise hors service), une fois chacun.
    for c in cells.cellules_depart:
        for b in c.breakers:
            if b.switch_id in coupling_sids:
                continue
            setsw(b.switch_id, _is_open(cible_graph, b.switch_id),
                  f"remise en service '{c.equipment_id}'")

    manoeuvres = _optimiser_sequence(poste, manoeuvres)
    res.manoeuvres = manoeuvres
    res.is_changed = bool(manoeuvres)

    # --- Vérification (nodale + détaillée + sûreté des sectionneurs) --------
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = res.topo_cible.meme_topologie(res.topo_obtenue)
    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_regles(poste, manoeuvres, un_seul=False))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if not res.is_verified:
        res.message = (
            "Topologie nodale cible non atteinte (mode agressif) : obtenu "
            f"{res.topo_obtenue.nb_noeuds} nœud(s), visé {res.topo_cible.nb_noeuds}.")
    elif res.is_verified_detaillee:
        res.message = "Topologie détaillée cible atteinte et vérifiée (mode agressif)."
    else:
        res.message = (f"Topologie nodale atteinte ; {len(res.ecarts)} écart(s) : "
                       + " ; ".join(res.ecarts[:6]))
    return res


def _organes_internes_2bornes(poste: PosteTopologique) -> set[str]:
    """Équipements présents dans **plusieurs cellules de départ**.

    Détection **structurelle** (pas par identifiant) : un organe interne à 2
    bornes (typiquement une self/réactance dont les deux côtés sont câblés chacun
    sur une barre) apparaît dans deux cellules de départ. Ces organes sont laissés
    en place (ni ré-aiguillés ni signalés en écart)."""
    from collections import Counter
    occ: Counter = Counter()
    for c in poste.cellules.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            occ[eq] += 1
    return {eq for eq, n in occ.items() if n > 1}


def _appliquer_changements_dj(
    G: nx.Graph,
    reconnections: list[tuple[CelluleDepart, SwitchInfo]],
    disconnections: list[tuple[CelluleDepart, SwitchInfo]],
    skip: Optional[set[str]] = None,
) -> list[Manoeuvre]:
    """Applique sur ``G`` les changements d'état des **DJ de départ** (mise en
    service / hors service) et retourne les manœuvres correspondantes.

    ``skip`` : équipements à ignorer (ex. nœuds laissés à l'opérateur)."""
    skip = skip or set()
    out: list[Manoeuvre] = []
    for cell, dj in reconnections:
        if cell.equipment_id in skip:
            continue
        if _is_open(G, dj.switch_id):
            _set_switch(G, dj.switch_id, False)
            out.append(Manoeuvre(dj.switch_id, "CLOSE",
                                 f"mise en service départ {cell.equipment_id}"))
    for cell, dj in disconnections:
        if cell.equipment_id in skip:
            continue
        if not _is_open(G, dj.switch_id):
            _set_switch(G, dj.switch_id, True)
            out.append(Manoeuvre(dj.switch_id, "OPEN",
                                 f"mise hors service départ {cell.equipment_id}"))
    return out


def _consigner_non_realisables(
    res: ResultatManoeuvres, non_places: list[list[str]]
) -> None:
    """Renseigne la dégradation gracieuse : nœuds laissés à l'opérateur +
    écarts « nœud à compléter manuellement » (format unifié)."""
    res.noeuds_non_realisables = non_places
    for grp in non_places:
        res.ecarts.append(
            "nœud à compléter manuellement : {" + ", ".join(sorted(grp)) + "}")


def _isoler_depart_hors_barre(
    G: nx.Graph, cell: CelluleDepart, cible_graph: nx.Graph
) -> list[Manoeuvre]:
    """Isole un départ de ses barres (nœud à **0 barre** : ligne laissée sur son
    DJ) en respectant la **règle du sectionneur** — un sectionneur ne se manœuvre
    que hors charge :

    1. **dé-énergiser** : ouvrir le(s) DJ propre(s) encore fermé(s) ;
    2. **ouvrir les sectionneurs** de barre à ouvrir (désormais hors charge) ;
    3. **remettre le(s) DJ** à leur état cible (refermer si la cible les veut
       fermés — la ligne reste alors sur son DJ, isolée des barres).

    Ne fait rien (et n'ouvre aucun DJ) s'il n'y a aucun sectionneur à ouvrir.
    """
    sas = [sw for sw in cell.disconnectors
           if not _is_open(G, sw.switch_id) and _is_open(cible_graph, sw.switch_id)]
    if not sas:
        return []
    manos: list[Manoeuvre] = []
    djs_ouverts: list[str] = []
    for dj in cell.breakers:
        if not _is_open(G, dj.switch_id):
            _set_switch(G, dj.switch_id, True)
            manos.append(Manoeuvre(
                dj.switch_id, "OPEN",
                f"mise hors tension '{cell.equipment_id}' (avant ouverture sectionneur)"))
            djs_ouverts.append(dj.switch_id)
    for sw in sas:
        _set_switch(G, sw.switch_id, True)
        manos.append(Manoeuvre(
            sw.switch_id, "OPEN",
            f"isolement départ {cell.equipment_id} (nœud sans barre)"))
    for djid in djs_ouverts:
        if not _is_open(cible_graph, djid):
            _set_switch(G, djid, False)
            manos.append(Manoeuvre(
                djid, "CLOSE",
                f"remise sous tension '{cell.equipment_id}' (après ouverture sectionneur)"))
    return manos


def _sequence_detaillee_multibarres(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    topo_cible: TopologieNodale,
) -> ResultatManoeuvres:
    """
    Séquence pour un poste à **> 2 jeux de barres**.

    Le placement nodal classique (recherche combinatoire) ne couvre pas ces
    postes (nœuds à 0 barre, organes internes à 2 bornes, barres multiples). On
    dérive ici le placement **directement des composantes connexes du graphe
    cible** (chaque nœud = ses équipements + les sections de barre qu'il occupe),
    ce qui donne les groupes exacts, y compris :
    - les **nœuds à 0 barre** (départ isolé sur son DJ, SA ouverts) → isolés ;
    - les **organes internes à 2 bornes** (self/réactance) → laissés en place.

    Les départs/sections que l'algorithme ne sait pas réaliser sont consignés en
    écart (dégradation gracieuse) pour complétion manuelle.
    """
    vl = poste.voltage_level_id
    cells = poste.cellules
    G0 = poste.graph
    organes_fixes = _organes_internes_2bornes(poste)

    # Barre cible exacte de chaque départ (hors organes 2 bornes, ambigus).
    cible_busbar: dict[str, int] = {}
    for c in cells.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            if eq in organes_fixes:
                continue
            bb = _wired_busbar(c, cible_graph)
            if bb is not None:
                cible_busbar[eq] = bb

    dep_eqs = {c.equipment_id for c in cells.cellules_depart}
    for c in cells.cellules_depart:
        dep_eqs |= set(c.shared_equipment_ids)
    sjb_nodes = set(poste.tronconnement.barre_par_busbar)
    sjb_id = {n: G0.nodes[n].get("busbar_section_id") for n in sjb_nodes}

    # Placement depuis les composantes connexes (switches fermés) du graphe cible.
    closed = nx.Graph()
    closed.add_nodes_from(cible_graph.nodes(data=True))
    for u, v, d in cible_graph.edges(data=True):
        if not d.get("open", False):
            closed.add_edge(u, v)

    placement: list[tuple[set[str], set[str]]] = []
    noeuds_isoles: list[set[str]] = []
    for comp in nx.connected_components(closed):
        eqs = {cible_graph.nodes[n].get("equipment_id") for n in comp} & dep_eqs
        if not eqs:
            continue
        sjbs = {sjb_id[n] for n in comp if n in sjb_nodes}
        if sjbs:
            placement.append((eqs, sjbs))
        else:
            noeuds_isoles.append(eqs)

    res = determiner_manoeuvres_avec_sections(
        poste, placement, cible_busbar, organes_fixes=organes_fixes)
    res.topo_initiale = poste.topologie_nodale
    res.topo_cible = topo_cible

    # État détaillé après la séquence de placement.
    G = poste.graph.copy()
    for m in res.manoeuvres:
        _set_switch(G, m.switch_id, m.action == "OPEN")

    extra: list[Manoeuvre] = []
    # Isolation des nœuds à 0 barre : détacher le départ de ses barres en
    # respectant la **règle du sectionneur** (dé-énergiser par le DJ d'abord).
    for eqs in noeuds_isoles:
        for eq in sorted(eqs):
            if eq in organes_fixes:
                continue
            cell = cells.get_cellule_depart(eq)
            if cell is None:
                continue
            extra += _isoler_depart_hors_barre(G, cell, cible_graph)

    # Changements de DJ de départ (mise en service / hors service).
    reconnections, disconnections = _departure_dj_changes(poste, cible_graph)
    extra += _appliquer_changements_dj(G, reconnections, disconnections)

    res.manoeuvres = res.manoeuvres + extra

    # Vérification nodale + détaillée.
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = topo_cible.meme_topologie(res.topo_obtenue)
    res.is_changed = bool(res.manoeuvres)
    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_regles(poste, res.manoeuvres, un_seul=True))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if res.is_verified_detaillee:
        res.message = ("Topologie détaillée cible atteinte et vérifiée "
                       "(poste multi-barres).")
    elif res.is_verified:
        res.message = (f"Topologie nodale atteinte (poste multi-barres) ; "
                       f"{len(res.ecarts)} écart(s) détaillé(s) résiduel(s) : "
                       + " ; ".join(res.ecarts[:6]))
    else:
        # Dégradation gracieuse : nœuds cibles non réalisés (typiquement ceux qui
        # exigent des manœuvres sur les niveaux de barres supplémentaires — self/
        # réactance des JdB 3/4 — hors de portée de l'algorithme).
        obtenu = {frozenset(n.equipment_ids)
                  for n in res.topo_obtenue.noeuds.values()}
        non_realises: list[list[str]] = []
        seen: set[frozenset] = set()
        for n in topo_cible.noeuds.values():
            g = frozenset(n.equipment_ids)
            if g not in obtenu and g not in seen:
                seen.add(g)
                non_realises.append(sorted(g))
        _consigner_non_realisables(res, non_realises)
        res.message = (
            "Topologie cible partiellement atteinte (poste multi-barres) : "
            f"{res.topo_obtenue.nb_noeuds}/{topo_cible.nb_noeuds} nœuds atteints ; "
            f"{len(non_realises)} nœud(s) à compléter manuellement — manœuvres sur "
            "les niveaux de barres supplémentaires (self/réactance) non gérées.")
    return res


def determiner_manoeuvres_cible_detaillee(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    mode: str = "smooth",
) -> ResultatManoeuvres:
    """
    Atteint une **topologie détaillée cible imposée** (état précis de chaque
    organe, donc de la barre de chaque départ), plus spécifique que la seule
    topologie nodale.

    ``mode`` :
    - ``"smooth"`` (défaut) : dé-énergise **un ouvrage à la fois** (boucle longue,
      ré-alimentation immédiate) ; chaque départ est placé directement sur sa
      barre cible (pas de double-déplacement).
    - ``"aggressive"`` : orchestration **batch** — dé-énergise en une fois tous
      les ouvrages concernés, commute les SA hors tension, puis ré-alimente une
      seule fois (bien moins de manœuvres, plus d'ouvrages momentanément hors
      tension). Voir ``_sequence_detaillee_aggressive``.

    Démarche (mode smooth) :
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

    # Dégradation gracieuse : diagnostic + nœuds non réalisés (placement partiel).
    degradation: Optional[str] = None
    non_places: list[list[str]] = []

    # --- Mode agressif : orchestration batch (dé-énergiser une fois) --------
    if mode == "aggressive":
        return _sequence_detaillee_aggressive(poste, cible_graph, cible_busbar)

    # --- Poste à > 2 jeux de barres : placement par composantes ------------
    if len(set(poste.tronconnement.barre_par_busbar.values())) > 2:
        return _sequence_detaillee_multibarres(poste, cible_graph, topo_cible)

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

        placement, faisable, msg, np_ = _placement_avec_reconnexions(
            poste, cible_graph, topo_cible, reconnections)
        if not placement:
            res.topo_obtenue = poste.topologie_nodale
            res.message = "Topologie nodale cible non atteinte : " + msg
            res.noeuds_non_realisables = np_
            return res

        # Séquenceur sur le poste RÉEL (DJ reconnectés encore ouverts).
        res = determiner_manoeuvres_avec_sections(poste, placement, cible_busbar)
        res.topo_initiale = poste.topologie_nodale
        res.topo_cible = topo_cible
        if not faisable:
            degradation, non_places = msg, np_
    else:
        res = determiner_topo_complete_cible(poste, topo_cible, cible_busbar)
        if not res.is_verified and not res.noeuds_non_realisables:
            res.message = "Topologie nodale cible non atteinte : " + res.message
            return res
        if res.noeuds_non_realisables:
            degradation, non_places = res.message, res.noeuds_non_realisables

    # État détaillé atteint après la séquence nodale
    G = poste.graph.copy()
    for m in res.manoeuvres:
        _set_switch(G, m.switch_id, m.action == "OPEN")

    # 2. Raffinement : ramener chaque départ sur sa barre cible (boucle courte,
    #    équipotentielle puisque le nœud est déjà constitué). Pour un départ
    #    reconnecté, le DJ est encore ouvert : le ré-aiguillage SA reste sûr.
    extra: list[Manoeuvre] = []
    # Départs des nœuds non réalisés : laissés strictement en place (l'opérateur
    # complètera la séquence) — on ne les raffine ni ne touche leurs DJ.
    non_places_eqs = {eq for grp in non_places for eq in grp}
    for eq, target in sorted(cible_busbar.items()):
        if eq in non_places_eqs:
            continue
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

    # 3. Changements de DJ de départ (la barre cible est désormais au bon
    #    potentiel : fermeture/ouverture sûres ; nœuds non réalisés ignorés).
    post_manoeuvres = _appliquer_changements_dj(
        G, reconnections, disconnections, skip=non_places_eqs)

    res.manoeuvres = res.manoeuvres + extra + post_manoeuvres

    # 4. Vérification nodale + détaillée + écarts
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = topo_cible.meme_topologie(res.topo_obtenue)
    res.is_changed = bool(res.manoeuvres)
    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_regles(poste, res.manoeuvres, un_seul=True))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if degradation:
        # Dégradation gracieuse (option 4) : cible partiellement atteinte.
        _consigner_non_realisables(res, non_places)
        res.message = (
            degradation
            + f" Placement partiel : {len(non_places)} nœud(s) non réalisé(s) "
            f"laissé(s) à l'opérateur ; {res.nb_manoeuvres} manœuvre(s) "
            "partielle(s) générée(s) — complétez la séquence manuellement."
        )
    elif not res.is_verified:
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
) -> tuple[list[tuple[set[str], set[str]]], bool, str, list[list[str]]]:
    """
    Calcule un placement ``[(departs, sjb_ids)]`` réalisant ``topo_cible``.

    Si la cible est **réalisable**, retourne le placement complet
    (``faisable=True``). Sinon (option 2/4) : retourne un **diagnostic explicite**
    de l'infaisabilité dans ``message`` et un **placement partiel** « best-effort »
    plaçant le plus de nœuds possible, ``noeuds_non_places`` listant les départs
    des nœuds laissés à l'opérateur.

    Returns
    -------
    (placement, faisable, message, noeuds_non_places)
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
            # Union : un organe interne à 2 bornes (self/réactance) possède deux
            # cellules — on cumule les SJB atteignables des deux côtés.
            R[eq] = R.get(eq, frozenset()) | frozenset(c.busbar_nodes)
            en = eq_node.get(eq)
            connected[eq] = connected.get(eq, False) or bool(
                en is not None and en in H
                and any(s in H and nx.has_path(H, en, s) for s in R[eq])
            )
            if wired_sjb.get(eq) is None:
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
        return [], True, "Aucun nœud connecté à placer.", []

    # --- Scoping > 2 jeux de barres -----------------------------------------
    # L'algorithme de placement est conçu pour le double jeu de barres classique.
    # Sur un poste à > 2 jeux de barres (ex. niveaux supplémentaires 3B/4B), on se
    # restreint aux **2 jeux de barres principaux** ; les nœuds dont au moins un
    # départ n'atteint que les niveaux supplémentaires sont laissés à l'opérateur.
    forced_non_places: list[list[str]] = []
    extra_sjb: set[int] = set()
    if len(set(barre_par.values())) > 2:
        main_sjb, extra_sjb = _main_busbar_sjb(poste)
        sjb_nodes = [s for s in sjb_nodes if s in main_sjb]
        R = {eq: (rs & main_sjb) for eq, rs in R.items()}
        wired_sjb = {eq: (b if b in main_sjb else None)
                     for eq, b in wired_sjb.items()}
        kept: list[list[str]] = []
        for nd in nodes:
            if all(R.get(eq) for eq in nd):
                kept.append(nd)              # nœud entièrement sur le 2-JdB
            else:
                forced_non_places.append(nd)  # touche un niveau supplémentaire
        nodes = kept

    # Graphe des couplers entre SJB (pour la contrainte de connexité des groupes),
    # restreint aux SJB retenues (2 jeux de barres principaux si scoping).
    main_set = set(sjb_nodes)
    couplers = [cp for cp in _inter_sjb_couplers(poste)
                if cp.sjb_a in main_set and cp.sjb_b in main_set]
    CG = nx.Graph()
    CG.add_nodes_from(sjb_nodes)
    for cp in couplers:
        CG.add_edge(cp.sjb_a, cp.sjb_b)

    k = len(nodes)

    # Recherche exhaustive d'une affectation **complète** (tous les nœuds placés),
    # uniquement si elle est possible (k ≤ nb SJB) et tient dans le garde-fou.
    best = None  # (cost, assign tuple)
    if k <= len(sjb_nodes) and k ** len(sjb_nodes) <= 500_000:
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

    full_placement = None
    if best is not None:
        assign = best[1]
        node_sjbs = {i: set() for i in range(k)}
        for j, ni in enumerate(assign):
            node_sjbs[ni].add(sjb_nodes[j])
        full_placement = [
            (set(nodes[i]), {sjb_id[s] for s in node_sjbs[i]})
            for i in range(k)
        ]

    # Cas nominal : affectation complète ET aucun niveau supplémentaire écarté.
    if full_placement is not None and not forced_non_places:
        return full_placement, True, "OK", []

    # --- Dégradation : placement partiel + diagnostic explicite -------------
    raisons: list[str] = []
    if forced_non_places:
        raisons.append(_scoping_raison(sjb_id, extra_sjb, forced_non_places))
    if full_placement is not None:
        placement, partial_non_places = full_placement, []
    elif nodes:
        raisons.extend(_diagnostic_infaisabilite(nodes, R, sjb_id))
        placement, partial_non_places = _placement_best_effort(
            nodes, R, sjb_nodes, sjb_id, CG, wired_sjb)
    else:
        placement, partial_non_places = [], []
    non_places = partial_non_places + forced_non_places
    return placement, False, _message_non_realisable(raisons), non_places


def _main_busbar_sjb(poste: PosteTopologique) -> tuple[set[int], set[int]]:
    """
    Pour un poste à > 2 jeux de barres, identifie les **2 jeux de barres
    principaux** (système double-barre classique) et les niveaux supplémentaires.

    Heuristique : on regroupe les SJB par jeu de barres (``barre_par_busbar``) et
    on retient les **2 jeux de barres portant le plus de sections** (départage par
    index croissant). Les SJB des autres jeux de barres (typiquement des niveaux à
    section unique, ex. 3B/4B) sont « supplémentaires » et laissés à l'opérateur.

    Returns
    -------
    (main_sjb, extra_sjb)  ensembles de nœuds SJB.
    """
    barre_par = poste.tronconnement.barre_par_busbar
    by_barre: dict[int, set[int]] = {}
    for s, b in barre_par.items():
        by_barre.setdefault(b, set()).add(s)
    ordered = sorted(by_barre, key=lambda b: (-len(by_barre[b]), b))
    main_barres = set(ordered[:2])
    main_sjb = {s for b in main_barres for s in by_barre[b]}
    extra_sjb = set(barre_par) - main_sjb
    return main_sjb, extra_sjb


def _scoping_raison(
    sjb_id: dict[int, Optional[str]],
    extra_sjb: set[int],
    forced_non_places: list[list[str]],
) -> str:
    """Raison de dégradation liée au scoping > 2 jeux de barres."""
    noms = ", ".join(str(sjb_id.get(s, s)) for s in sorted(extra_sjb))
    return (f"niveaux de barres supplémentaires ({noms}) non gérés par "
            f"l'algorithme 2 jeux de barres : {len(forced_non_places)} nœud(s) "
            "laissé(s) à l'opérateur")


def _message_non_realisable(raisons: list[str]) -> str:
    """Assemble le message de dégradation à partir des raisons collectées."""
    if not raisons:
        return "Aucune affectation de SJB réalisable (topologie impossible)."
    return "Topologie cible non réalisable sur ce poste : " + " ; ".join(raisons) + "."


def _diagnostic_infaisabilite(
    nodes: list[list[str]],
    R: dict[str, frozenset],
    sjb_id: dict[int, Optional[str]],
) -> list[str]:
    """
    Explique **pourquoi** aucune affectation SJB complète n'existe (option 2).

    Retourne une **liste de raisons** (sans préfixe), parmi les deux modes
    d'échec réels :
    1. *Sur-réservation d'une classe d'aiguillage* (condition de Hall) : plus de
       nœuds confinés à un ensemble de sections que de sections disponibles ;
    2. *Organe interne à 2 bornes* (self/réactance ``LINE_SIDE1``+``LINE_SIDE2``)
       présent sur plusieurs nœuds alors qu'il n'atteint qu'une seule section.
    """
    from collections import Counter

    def _names(sset) -> str:
        return "{" + ", ".join(str(sjb_id.get(s, s)) for s in sorted(sset)) + "}"

    raisons: list[str] = []

    # 1. Sur-réservation (Hall) : pour chaque ensemble de sections T atteignable
    #    par un départ, compter les nœuds *forcés* d'utiliser une section de T
    #    (un de leurs départs n'atteint que des sections de T).
    classes = {R[eq] for node in nodes for eq in node if R.get(eq)}
    vus: set[frozenset] = set()
    for T in sorted(classes, key=lambda t: (len(t), sorted(t))):
        forces = [node for node in nodes
                  if any(R.get(eq, frozenset()) <= T for eq in node)]
        if len(forces) > len(T) and T not in vus:
            vus.add(T)
            raisons.append(
                f"{len(forces)} nœuds nécessitent une section parmi {_names(T)} "
                f"({len(T)} disponible(s))")

    # 2. Organe interne à 2 bornes éclaté sur plusieurs nœuds
    occurrences = Counter(eq for node in nodes for eq in node)
    for eq, nb in sorted(occurrences.items()):
        atteignables = R.get(eq, frozenset())
        if nb >= 2 and len(atteignables) < nb:
            raisons.append(
                f"organe interne '{eq}' présent sur {nb} nœuds mais seulement "
                f"{len(atteignables)} section(s) atteignable(s) {_names(atteignables)} "
                "— selfs/réactances à 2 bornes non gérées côté par côté")

    return raisons


def _placement_best_effort(
    nodes: list[list[str]],
    R: dict[str, frozenset],
    sjb_nodes: list[int],
    sjb_id: dict[int, Optional[str]],
    CG: nx.Graph,
    wired_sjb: dict[str, Optional[int]],
) -> tuple[list[tuple[set[str], set[str]]], list[list[str]]]:
    """
    Placement **partiel** « best-effort » (option 4 — dégradation gracieuse).

    Quand aucune affectation complète n'existe, place le **plus grand nombre
    possible** de nœuds cibles (chacun entièrement satisfait : sections connexes
    + tous ses départs atteignant une section de leur groupe) et laisse les
    nœuds restants à compléter par l'opérateur.

    Returns
    -------
    (placement, noeuds_non_places)
        ``placement`` = ``[(departs, sjb_ids)]`` des nœuds réalisés ;
        ``noeuds_non_places`` = liste des départs des nœuds non réalisés.
    """
    import itertools

    k = len(nodes)
    n_sjb = len(sjb_nodes)
    if k == 0:
        return [], []

    # Garde-fou : chaque SJB → un nœud OU « inutilisée » (k+1 choix par SJB).
    if (k + 1) ** n_sjb > 2_000_000:
        return _placement_greedy(nodes, R, sjb_nodes, sjb_id)

    best = None  # (score, placement, non_places)
    for assign in itertools.product(range(k + 1), repeat=n_sjb):
        node_sjbs: dict[int, set[int]] = {i: set() for i in range(k)}
        for j, ni in enumerate(assign):
            if ni < k:
                node_sjbs[ni].add(sjb_nodes[j])
        places: list[int] = []
        valide = True
        for i in range(k):
            s = node_sjbs[i]
            if not s:
                continue  # nœud abandonné (laissé à l'opérateur)
            if not nx.is_connected(CG.subgraph(s)):
                valide = False
                break
            if any(not (R.get(eq, frozenset()) & s) for eq in nodes[i]):
                valide = False
                break
            places.append(i)
        if not valide or not places:
            continue
        reaig = sum(
            1 for i in places for eq in nodes[i]
            if wired_sjb.get(eq) not in node_sjbs[i]
        )
        score = (len(places), -reaig)
        if best is None or score > best[0]:
            placement = [
                (set(nodes[i]), {sjb_id[s] for s in node_sjbs[i]}) for i in places
            ]
            non_places = [list(nodes[i]) for i in range(k) if i not in places]
            best = (score, placement, non_places)

    if best is None:
        return [], [list(n) for n in nodes]
    return best[1], best[2]


def _placement_greedy(
    nodes: list[list[str]],
    R: dict[str, frozenset],
    sjb_nodes: list[int],
    sjb_id: dict[int, Optional[str]],
) -> tuple[list[tuple[set[str], set[str]]], list[list[str]]]:
    """Repli glouton (postes très volumineux) : une SJB simple par nœud, les
    nœuds les plus contraints d'abord."""
    libres = set(sjb_nodes)
    placement: list[tuple[set[str], set[str]]] = []
    places: set[int] = set()
    ordre = sorted(
        range(len(nodes)),
        key=lambda i: min((len(R.get(eq, ())) for eq in nodes[i]), default=99),
    )
    for i in ordre:
        cand = [s for s in libres
                if all(s in R.get(eq, frozenset()) for eq in nodes[i])]
        if cand:
            s = min(cand)
            libres.discard(s)
            places.add(i)
            placement.append((set(nodes[i]), {sjb_id[s]}))
    non_places = [list(nodes[i]) for i in range(len(nodes)) if i not in places]
    return placement, non_places


def _switch_edge_index(G: nx.Graph) -> dict[str, tuple[int, int]]:
    """Index ``switch_id -> (u, v)`` mémoïsé sur le graphe (``G.graph``).

    Construit en une passe au premier accès, puis réutilisé en O(1) — il
    remplace les anciens scans linéaires de ``_is_open`` / ``_set_switch``.

    Validité : la coordonnée ``(u, v)`` d'un switch est **topologique**, donc
    stable tant que la structure du graphe ne change pas. Le séquenceur ne fait
    que basculer l'attribut ``open`` (jamais ajouter/retirer d'arête), et
    ``G.copy()`` préserve nœuds et arêtes : l'index reste valide sur les graphes
    dérivés (copies de travail, ``cible_graph`` du même réseau). Le nombre
    d'arêtes sert de garde-fou : toute modification structurelle force une
    reconstruction.
    """
    cache = G.graph.get("_switch_edge_index")
    if cache is None or cache[0] != G.number_of_edges():
        mapping = {d["switch_id"]: (u, v)
                   for u, v, d in G.edges(data=True)
                   if d.get("switch_id") is not None}
        cache = (G.number_of_edges(), mapping)
        G.graph["_switch_edge_index"] = cache
    return cache[1]


def _equipment_node_index(G: nx.Graph) -> dict[str, int]:
    """Index ``equipment_id -> node`` mémoïsé sur le graphe (cf.
    ``_switch_edge_index`` ; garde-fou sur le nombre de nœuds)."""
    cache = G.graph.get("_equipment_node_index")
    if cache is None or cache[0] != G.number_of_nodes():
        mapping = {d["equipment_id"]: n
                   for n, d in G.nodes(data=True)
                   if d.get("equipment_id") is not None}
        cache = (G.number_of_nodes(), mapping)
        G.graph["_equipment_node_index"] = cache
    return cache[1]


def _set_switch(G: nx.Graph, switch_id: str, open_: bool) -> None:
    """Modifie l'état d'un switch (par son id) dans le graphe simulé.

    No-op silencieux si l'id est inconnu (contrat historique, cf.
    ``tests/manoeuvre/test_lookup_helpers.py``)."""
    edge = _switch_edge_index(G).get(switch_id)
    if edge is not None:
        G.edges[edge]["open"] = open_


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

    **Mémoïsé sur le poste** (auparavant recalculé ~10×/analyse). Le résultat ne
    dépend que de la **topologie** (graphe + tronçonnement), pas de l'état
    ouvert/fermé des organes — propriété vérifiée par
    ``tests/manoeuvre/test_couplers_memoisation.py``. Le poste n'étant jamais
    muté structurellement, le cache reste valide toute sa durée de vie.
    """
    cached = getattr(poste, "_inter_sjb_couplers_cache", None)
    if cached is not None:
        return cached

    G = poste.graph
    bb_nodes = set(poste.tronconnement.barre_par_busbar)

    # Sous-graphe de couplage : SJB + nœuds hors cellules de départ
    departure_internal: set[int] = set()
    for c in poste.cellules.cellules_depart:
        departure_internal.update(n for n in c.all_nodes if n not in bb_nodes)
    coupler_nodes = bb_nodes | (set(G.nodes()) - departure_internal)
    coupler_G = G.subgraph(coupler_nodes)

    couplers: list[_InterSjbCoupler] = []
    bb_list = sorted(bb_nodes)
    for i, a in enumerate(bb_list):
        for b in bb_list[i + 1:]:
            # Chemins entre a et b ne traversant aucune autre SJB. On détecte
            # **tous les couplages parallèles** (ex. un couplage DJ direct ET une
            # liaison via un nœud de couplage commun à plusieurs barres) en
            # retirant itérativement les arêtes du chemin trouvé jusqu'à
            # épuisement. Sans cela, un couplage fermé masqué par un couplage
            # parallèle ouvert resterait fermé → fusion de nœuds erronée.
            others = bb_nodes - {a, b}
            H = coupler_G.subgraph(coupler_nodes - others).copy()
            while True:
                try:
                    path = nx.shortest_path(H, a, b)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    break
                sw_ids, brk_ids = [], []
                edges = list(zip(path, path[1:]))
                for u, v in edges:
                    d = H.edges[u, v]
                    sid = d.get("switch_id")
                    if sid is None:
                        continue
                    sw_ids.append(sid)
                    if d.get("kind") == SwitchKind.BREAKER:
                        brk_ids.append(sid)
                if sw_ids:
                    couplers.append(_InterSjbCoupler(a, b, sw_ids, brk_ids))
                # Retirer les arêtes du chemin pour révéler un couplage parallèle.
                H.remove_edges_from(edges)

    poste._inter_sjb_couplers_cache = couplers
    return couplers


def determiner_manoeuvres_avec_sections(
    poste: PosteTopologique,
    placement: list[tuple[set[str], set[str]]],
    cible_busbar: Optional[dict[str, int]] = None,
    organes_fixes: Optional[set[str]] = None,
) -> ResultatManoeuvres:
    """
    Calcule la séquence de manœuvres pour réaliser un **placement explicite**
    de nœuds sur des sections de jeux de barres, en respectant la règle du
    sectionnement de barre (dé-énergisation avant ouverture).

    ``organes_fixes`` (optionnel) : équipements à **laisser en place** (ni
    ré-aiguillage ni écart) — typiquement les organes internes à 2 bornes
    (self/réactance) déjà câblés sur leurs barres cibles.

    Parameters
    ----------
    poste :
        Vue complète du poste.
    placement :
        Liste de ``(departs, sjb_ids)`` : chaque entrée décrit un nœud cible,
        l'ensemble de ses départs et l'ensemble des SJB (``busbar_section_id``)
        qu'il occupe. Les départs non cités restent inchangés.
    cible_busbar :
        Optionnel — barre (SJB) **cible exacte** de chaque départ (``eq -> sjb``).
        Si fournie, ``target_sjb`` est amorcé avec cette barre quand elle est
        dans le groupe du nœud : chaque départ est alors placé **directement**
        sur sa barre finale (le raffinement R15 devient un no-op), ce qui évite
        le **double-déplacement** (placer puis ramener).

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
        # ``departs`` est un ensemble : on l'itère **trié** pour que l'ordre
        # d'insertion dans ``target_sjb`` (donc l'ordre des manœuvres qui en
        # découlent) soit reproductible d'un process à l'autre (PYTHONHASHSEED).
        for eq in sorted(departs):
            if organes_fixes and eq in organes_fixes:
                continue  # organe interne à 2 bornes : laissé en place
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
            # Barre cible exacte connue (flux détaillé) et dans le groupe : on la
            # vise directement → placement en une fois, sans retour ultérieur.
            if cible_busbar and cible_busbar.get(eq) in sjb_set:
                target_sjb[eq] = cible_busbar[eq]
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
        currently_closed = all(not _is_open(G, sid) for sid in cp.switch_ids)
        if na != nb:
            # On n'ouvre un couplage que s'il est **réellement fermé** (conducteur).
            # Sinon les deux barres sont déjà séparées : ne pas toucher (évite
            # d'ouvrir un organe partagé avec un couplage à garder fermé — ex.
            # DJ commun à deux liaisons sur trois barres).
            if currently_closed:
                to_open.append(cp)
        elif not currently_closed:
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

    def _equipotentiel(a: int, b: int) -> bool:
        """True si deux SJB sont au même potentiel (chemin de switches fermés)."""
        Hc = nx.Graph()
        Hc.add_nodes_from(G.nodes())
        for u, v, dd in G.edges(data=True):
            if not dd.get("open", False):
                Hc.add_edge(u, v)
        return a in Hc and b in Hc and nx.has_path(Hc, a, b)

    def _departs_cables(s: int) -> list[str]:
        # Tri explicite : la séquence de dé-énergisation ne doit pas dépendre de
        # l'ordre d'itération de ``target_sjb`` (cf. construction triée ci-dessus).
        return sorted(eq for eq in target_sjb if s in _wired_sjbs(G, cells, eq))

    def parking_sjb(eq: str, target: int) -> Optional[int]:
        """SJB **tampon** où garer temporairement le départ pendant l'ouverture
        du sectionnement isolant sa cible (règle « un seul ouvrage hors tension à
        la fois », R10ter). Préférences :
        1. une SJB **hors section isolée** (et atteignable) ;
        2. à défaut, une SJB **équipotentielle** (parking en **boucle courte**,
           donc *sans* coupure — même si elle sera isolée ensuite) ;
        3. en dernier recours, toute SJB accessible distincte de la cible.
        Retourne ``None`` si aucune section de parking n'existe (→ exception :
        dé-énergisation en place en phase C)."""
        cell = cells.get_cellule_depart(eq)
        if cell is None:
            return None
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
    # Règle « un seul ouvrage hors tension à la fois » (R10ter, mode smooth) :
    # un départ dont la barre cible est une section à isoler est **garé** (un par
    # un) sur une SJB tampon — en **boucle courte** si équipotentielle (sans
    # coupure), sinon boucle longue (une seule coupure, séquentielle) —, puis
    # **ramené** en phase E. Faute de tampon (``parking_sjb`` None), il est
    # dé-énergisé **en place** en phase C (exception assumée).
    parkings: dict[str, int] = {}
    for eq, tgt in sorted(target_sjb.items()):
        if tgt in sjb_isoles:
            buf = parking_sjb(eq, tgt)
            if buf is None:
                continue  # aucun tampon : dé-énergisation en place (phase C)
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
    # Sûreté des sectionneurs : signaler toute ouverture restée sous tension
    # (sectionnements de barre) ou tout sectionneur manœuvré sous charge.
    res.ecarts += _verifier_regles(poste, manoeuvres, un_seul=False)
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


def _rejeu_securite(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> tuple[list[str], list[Optional[str]], list[str]]:
    """**Passe de rejeu unique** des règles du sectionneur (au lieu de trois
    rejeux séparés). Rejoue la séquence une seule fois depuis l'état initial et
    calcule en parallèle les trois diagnostics, en partageant le graphe « live »
    construit au plus une fois par étape.

    Returns
    -------
    (securite, sous_charge_par_man, un_seul)
        - ``securite`` : sectionnements de barre ouverts **sous tension** (deux
          côtés énergisés) — règle stricte du sectionneur de barre ;
        - ``sous_charge_par_man`` : liste **alignée** sur ``manoeuvres`` ; message
          d'infraction si la manœuvre ouvre un sectionneur (DISCONNECTOR) sous
          charge, sinon ``None`` ;
        - ``un_seul`` : moments où **plus d'un** ouvrage est hors tension par
          ré-aiguillage (boucle longue) simultanément (R10ter, mode smooth).
    """
    couplers = _inter_sjb_couplers(poste)
    cells = poste.cellules
    all_sjb = set(poste.tronconnement.barre_par_busbar)

    # Sectionnements de barre : sid -> (SJB a, SJB b) (règle stricte).
    sect_pairs: dict[str, tuple[int, int]] = {}
    for cp in couplers:
        if cp.is_sectionnement:
            for sid in cp.switch_ids:
                sect_pairs[sid] = (cp.sjb_a, cp.sjb_b)
    sect_id_set = set(sect_pairs)

    # Tous les sectionneurs (DISCONNECTOR) : sid -> arête (« sous charge »).
    disc: dict[str, tuple[int, int]] = {}
    for u, v, d in poste.graph.edges(data=True):
        sid = d.get("switch_id")
        if sid and d.get("kind") == SwitchKind.DISCONNECTOR:
            disc[sid] = (u, v)
    equip = {n for n, d in poste.graph.nodes(data=True) if d.get("equipment_id")}

    # DJ de départ (hors couplage) : sid -> équipement (« un seul HS »).
    coupling_sids = {s for cp in couplers for s in cp.switch_ids}
    dj_eq: dict[str, str] = {}
    for c in cells.cellules_depart:
        for b in c.breakers:
            if b.switch_id not in coupling_sids:
                dj_eq[b.switch_id] = c.equipment_id

    securite: list[str] = []
    sous_charge: list[Optional[str]] = []
    un_seul: list[str] = []
    temp_parking: set[str] = set()

    G = poste.graph.copy()
    for m in manoeuvres:
        opening = (m.action == "OPEN" and not _is_open(G, m.switch_id))
        H = None  # graphe « live » privé du switch, construit au plus une fois

        # --- sectionneur sous charge (tout DISCONNECTOR), aligné sur la séquence
        msg: Optional[str] = None
        if opening and m.switch_id in disc:
            H = _live_graph_sans(G, [m.switch_id])
            u, v = disc[m.switch_id]
            if not (u in H and v in H and nx.has_path(H, u, v)):
                cu = nx.node_connected_component(H, u) if u in H else {u}
                cv = nx.node_connected_component(H, v) if v in H else {v}
                if (cu & equip) and (cv & equip):
                    if m.switch_id in sect_id_set:
                        msg = ("sectionneur de barre manœuvré sous charge — mettre "
                               "la section de barre hors tension (ré-aiguiller ses "
                               "départs sur l'autre section) avant d'ouvrir ce "
                               "sectionnement")
                    else:
                        msg = ("sectionneur manœuvré sous charge — dé-énergiser la "
                               "branche par son disjoncteur avant d'ouvrir ce "
                               "sectionneur")
        sous_charge.append(msg)

        # --- sectionnement de barre ouvert sous tension (règle stricte) --------
        if opening and m.switch_id in sect_pairs:
            if H is None:
                H = _live_graph_sans(G, [m.switch_id])
            a, b = sect_pairs[m.switch_id]
            if not (a in H and b in H and nx.has_path(H, a, b)):
                side_a = (nx.node_connected_component(H, a)
                          if a in H else {a}) & all_sjb
                side_b = (nx.node_connected_component(H, b)
                          if b in H else {b}) & all_sjb
                if (_ouvrages_energises_sur(G, cells, side_a, H)
                        and _ouvrages_energises_sur(G, cells, side_b, H)):
                    securite.append(
                        f"sectionneur {m.switch_id} ouvert sous tension "
                        "(deux côtés énergisés)")

        # --- un seul ouvrage hors tension par ré-aiguillage (boucle longue) ----
        eq = dj_eq.get(m.switch_id)
        if eq is not None:
            longue = (m.type_boucle == "LONGUE"
                      or "boucle longue" in (m.raison or ""))
            if m.action == "OPEN" and longue:
                temp_parking.add(eq)
            elif m.action == "CLOSE":
                temp_parking.discard(eq)
            if len(temp_parking) > 1:
                un_seul.append(
                    "plus d'un ouvrage hors tension simultanément (ré-aiguillage) : "
                    + ", ".join(sorted(temp_parking)))

        _set_switch(G, m.switch_id, m.action == "OPEN")

    return securite, sous_charge, list(dict.fromkeys(un_seul))


def _verifier_securite_sectionneurs(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Sectionnements de barre ouverts sous tension (cf. ``_rejeu_securite``)."""
    return _rejeu_securite(poste, manoeuvres)[0]


def _verifier_un_seul_hors_tension(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Règle R10ter : pas plus d'**un** ouvrage hors tension par ré-aiguillage
    (boucle longue) simultanément (cf. ``_rejeu_securite``)."""
    return _rejeu_securite(poste, manoeuvres)[2]


def _sectionneurs_sous_charge_par_manoeuvre(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[Optional[str]]:
    """Analyse, **manœuvre par manœuvre**, la règle du sectionneur (hors charge).

    Retourne une liste **alignée sur ``manoeuvres``** : pour chaque manœuvre, un
    message d'infraction si elle ouvre un **sectionneur** (DISCONNECTOR) **sous
    charge** — i.e. elle déconnecte, sans chemin parallèle, deux parties **toutes
    deux porteuses d'un ouvrage** énergisé — sinon ``None``.

    Un sectionneur ne se manœuvre que hors charge : la parade est de dé-énergiser
    la branche par son **disjoncteur** avant d'ouvrir le sectionneur.
    """
    return _rejeu_securite(poste, manoeuvres)[1]


def _verifier_regles(
    poste: PosteTopologique,
    manoeuvres: list[Manoeuvre],
    un_seul: bool = True,
) -> list[str]:
    """Agrège, en **une seule passe de rejeu** (``_rejeu_securite``), les écarts
    des règles du sectionneur dans l'ordre historique : sécurité des
    sectionnements, [un seul ouvrage hors tension,] sectionneurs hors charge.

    ``un_seul`` : inclure la règle R10ter (modes smooth / multi-barres) ; les
    orchestrations « batch » (agressif, séquenceur à sections) l'omettent.
    """
    securite, sous_charge, un = _rejeu_securite(poste, manoeuvres)
    hors_charge = list(dict.fromkeys(
        f"{m.switch_id} : {msg}"
        for m, msg in zip(manoeuvres, sous_charge) if msg))
    out = list(securite)
    if un_seul:
        out += un
    out += hors_charge
    return out


def _verifier_sectionneurs_hors_charge(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Règle générale du sectionneur : un **sectionneur** (DISCONNECTOR) ne se
    manœuvre que **hors charge** (cf. ``_sectionneurs_sous_charge_par_manoeuvre``).
    Retourne les écarts agrégés (sectionneurs manœuvrés sous charge), pour
    **tous** les sectionneurs (sélecteurs de barre comme sectionnements)."""
    par_man = _sectionneurs_sous_charge_par_manoeuvre(poste, manoeuvres)
    ecarts = [f"{m.switch_id} : {msg}"
              for m, msg in zip(manoeuvres, par_man) if msg]
    return list(dict.fromkeys(ecarts))


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
    """État d'un switch (True = ouvert). Un id inconnu est considéré **ouvert**
    (contrat historique)."""
    edge = _switch_edge_index(G).get(switch_id)
    if edge is None:
        return True
    return bool(G.edges[edge].get("open", False))


def _eq_node(G: nx.Graph, eq_id: str) -> Optional[int]:
    """Nœud de connectivité d'un équipement (ou ``None`` si inconnu)."""
    return _equipment_node_index(G).get(eq_id)


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
