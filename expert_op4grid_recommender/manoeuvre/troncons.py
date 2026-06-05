"""
manoeuvre/troncons.py  —  Étapes 1.3 et 1.4
---------------------------------------------
Tronçonnement des sections de jeux de barres (SJB) et attribution des nœuds
électriques aux tronçons.

Contexte
~~~~~~~~
Un **tronçon** est un segment de barre électriquement structurant du poste.
Le portage suit la sémantique du C++ ``Topologie::buildTronconnement``
(TOPOPoste.cc:1947) et ``CelluleBarresTopo::tronconneGraph``
(TOPOPosteCellElement.cc:639), qui construisent le tronçonnement à partir de
la **structure** du poste (indépendamment de l'état ouvert/fermé courant des
organes de coupure — cf. commentaire C++ ligne 722 « peu importe dans quelle
topologie il est »).

Concepts physiques (poste RTE double barre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Jeu de barres (barre)** : un rail électrique courant sur la longueur du
  poste (barre 1, barre 2…).
- **Sectionnement** : sectionneur (SA/DISCONNECTOR) reliant *directement* deux
  SJB d'une *même* barre → il découpe la barre en sections longitudinales.
- **Couplage** : travée à disjoncteur (DJ/BREAKER) reliant deux barres
  *différentes* → permet de transférer/paralléliser les barres.

Définitions retenues
~~~~~~~~~~~~~~~~~~~~~
- **barre** = composante connexe des SJB reliées par des *sectionnements*
  (arêtes DISCONNECTOR directes SJB↔SJB).
- **tronçon** = composante connexe des SJB dans le *sous-graphe de couplage*
  (SJB + nœuds n'appartenant à aucune cellule de départ), structurel,
  c.-à-d. en traversant aussi bien les sectionnements que les couplages.
  ``nb_jeux_barres`` du tronçon = nombre de barres distinctes qu'il contient.

Pour le poste de référence CARRIP3 (4 SJB : 1.1, 1.2, 2.1, 2.2 ;
sectionnements SS.1.12 0-1 et SS.2.12 2-3 ; couplage COUPL 0-2) :
- 2 barres : {1.1, 1.2} et {2.1, 2.2}
- 1 tronçon : {1.1, 1.2, 2.1, 2.2}, ``nb_jeux_barres = 2``
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .models import NodeType, SwitchKind
from .graph import busbar_nodes
from .cellules import CellulesVL, SwitchInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class Troncon:
    """
    Un tronçon : ensemble de SJB électriquement structurées comme un segment.

    Correspondance C++ : ``Troncon`` (TOPOPosteCellElement.h).
    """
    numero: int
    busbar_nodes: set[int] = field(default_factory=set)
    nb_jeux_barres: int = 0
    departs: set[str] = field(default_factory=set)
    # equipement -> numéro de barre fixe (départ non ré-aiguillable)
    departs_fixes: dict[str, int] = field(default_factory=dict)
    # equipement -> ensemble des barres atteignables (départ ré-aiguillable)
    departs_couplage: dict[str, set[int]] = field(default_factory=dict)
    # groupes omnibus (équipements partageant une cellule)
    departs_multiples: list[set[str]] = field(default_factory=list)
    # tronçon ne servant qu'à relier des barres (sans départ propre)
    is_couplage: bool = False
    # switches de couplage internes au tronçon (DJ + SA de couplage)
    switches_couplage: list[SwitchInfo] = field(default_factory=list)
    # noeuds électriques attribués (étape 1.4) : nom_noeud -> set(barres)
    noeuds_electriques: dict[str, set[int]] = field(default_factory=dict)

    @property
    def couplage_breakers(self) -> list[SwitchInfo]:
        """Disjoncteurs (DJ) de couplage du tronçon."""
        return [s for s in self.switches_couplage if s.is_breaker]

    @property
    def couplage_disconnectors(self) -> list[SwitchInfo]:
        """Sectionneurs (SA) de couplage du tronçon."""
        return [s for s in self.switches_couplage if s.is_disconnector]

    def __repr__(self) -> str:  # pragma: no cover - debug
        return (
            f"Troncon(n={self.numero}, sjb={sorted(self.busbar_nodes)}, "
            f"njb={self.nb_jeux_barres}, departs={sorted(self.departs)}, "
            f"couplage={self.is_couplage})"
        )


@dataclass
class Tronconnement:
    """
    Résultat complet du tronçonnement d'un voltage level.

    Attributes
    ----------
    voltage_level_id :
        Identifiant du voltage level.
    troncons :
        Map numéro -> Troncon.
    troncon_par_depart :
        Map equipement -> numéro de tronçon.
    barre_par_busbar :
        Map nœud SJB -> numéro de barre (jeu de barres).
    nb_jeux_barres :
        Nombre total de barres distinctes du poste.
    """
    voltage_level_id: str
    troncons: dict[int, Troncon] = field(default_factory=dict)
    troncon_par_depart: dict[str, int] = field(default_factory=dict)
    barre_par_busbar: dict[int, int] = field(default_factory=dict)
    nb_jeux_barres: int = 0

    @property
    def departs_par_troncon(self) -> dict[int, set[str]]:
        return {num: set(t.departs) for num, t in self.troncons.items()}

    def troncon_de(self, equipment_id: str) -> Optional[Troncon]:
        num = self.troncon_par_depart.get(equipment_id)
        return self.troncons.get(num) if num is not None else None

    def barre_de_busbar(self, bb_node: int) -> Optional[int]:
        return self.barre_par_busbar.get(bb_node)

    def resume(self) -> str:
        parts = [
            f"VL '{self.voltage_level_id}': {len(self.troncons)} tronçon(s), "
            f"{self.nb_jeux_barres} jeu(x) de barres"
        ]
        for num in sorted(self.troncons):
            t = self.troncons[num]
            parts.append(
                f"  T{num}: {len(t.busbar_nodes)} SJB, {t.nb_jeux_barres} barre(s), "
                f"{len(t.departs)} départ(s)"
                + (" [couplage]" if t.is_couplage else "")
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Étape 1.3 — Construction du tronçonnement
# ---------------------------------------------------------------------------

def construire_tronconnement(cellules: CellulesVL, G: nx.Graph) -> Tronconnement:
    """
    Construit le tronçonnement structurel d'un voltage level.

    Parameters
    ----------
    cellules :
        Résultat de ``detecter_cellules`` (étape 1.2).
    G :
        Graphe node/breaker du voltage level (étape 1.1).

    Returns
    -------
    Tronconnement
    """
    result = Tronconnement(voltage_level_id=cellules.voltage_level_id)

    bb_nodes = set(busbar_nodes(G))
    if not bb_nodes:
        logger.warning("VL '%s' sans section de jeux de barres.", cellules.voltage_level_id)
        return result

    # --- 1. Détection des barres (sectionnements directs SJB↔SJB) ----------
    barre_par_busbar = _detecter_barres(G, bb_nodes, cellules.voltage_level_id)
    result.barre_par_busbar = barre_par_busbar
    result.nb_jeux_barres = len(set(barre_par_busbar.values()))

    # --- 2. Sous-graphe de couplage (SJB + nœuds hors cellules de départ) ---
    coupler_G, coupler_switches = _build_coupler_subgraph(cellules, G, bb_nodes)

    # --- 3. Tronçons = composantes connexes des SJB dans le sous-graphe -----
    numero = 0
    sjb_to_troncon: dict[int, int] = {}
    for component in nx.connected_components(coupler_G):
        sjb_in_comp = component & bb_nodes
        if not sjb_in_comp:
            continue
        barres = {barre_par_busbar[n] for n in sjb_in_comp}
        troncon = Troncon(
            numero=numero,
            busbar_nodes=set(sjb_in_comp),
            nb_jeux_barres=len(barres),
        )
        # Switches de couplage internes à ce tronçon (arêtes du sous-graphe
        # touchant un nœud de la composante, hors sectionnements directs).
        troncon.switches_couplage = [
            sw for sw in coupler_switches
            if sw.node1 in component and sw.node2 in component
            and not (sw.node1 in bb_nodes and sw.node2 in bb_nodes
                     and barre_par_busbar.get(sw.node1) == barre_par_busbar.get(sw.node2))
        ]
        result.troncons[numero] = troncon
        for n in sjb_in_comp:
            sjb_to_troncon[n] = numero
        numero += 1

    # --- 4. Attribution des départs aux tronçons ----------------------------
    _attribuer_departs(result, cellules, sjb_to_troncon, barre_par_busbar)

    logger.info(result.resume())
    return result


def _detecter_barres(
    G: nx.Graph, bb_nodes: set[int], vl_id: str
) -> dict[int, int]:
    """
    Regroupe les SJB en barres (jeux de barres).

    Stratégie :
    1. **Nommage RTE** (primaire) : le préfixe entier du nom de SJB après le
       VL_id donne le numéro de barre ('CARRIP3_1.1' -> 1, 'NOVIOP3_2A' -> 2).
       Fiable sur le réseau RTE et physiquement correct (sections d'une même
       barre partagent le même entier).
    2. **Structure** (repli) : deux SJB sont dans la même barre si reliées par
       un chemin de *sectionnements* — c.-à-d. de switches DISCONNECTOR sans
       traverser de BREAKER (un BREAKER signale une travée de couplage).

    Returns
    -------
    dict[int, int] : nœud SJB -> numéro de barre (réindexé de 0..n-1).
    """
    name_groups = _barres_par_nommage(G, bb_nodes, vl_id)
    if name_groups is not None:
        return _reindex(name_groups)

    # Repli structurel : connectivité par sectionnements (disjoncteurs exclus)
    disc_graph = nx.Graph()
    disc_graph.add_nodes_from(bb_nodes)
    for u, v, d in G.edges(data=True):
        if d.get("kind") == SwitchKind.BREAKER:
            continue  # un DJ est une travée de couplage, jamais un sectionnement
        disc_graph.add_edge(u, v)

    barre_par_busbar: dict[int, int] = {}
    idx = 0
    for comp in nx.connected_components(disc_graph):
        sjb = comp & bb_nodes
        if not sjb:
            continue
        for n in sjb:
            barre_par_busbar[n] = idx
        idx += 1
    return barre_par_busbar


def _reindex(groups: dict[int, int]) -> dict[int, int]:
    """Réindexe des numéros de barre arbitraires en 0..n-1 (ordre croissant)."""
    labels = sorted(set(groups.values()))
    remap = {lab: i for i, lab in enumerate(labels)}
    return {node: remap[lab] for node, lab in groups.items()}


def _barres_par_nommage(
    G: nx.Graph, bb_nodes: set[int], vl_id: str
) -> Optional[dict[int, int]]:
    """
    Déduit les barres depuis le nommage des SJB (préfixe entier après VL_id).

    Ex : 'CARRIP3_1.1' -> barre 1 ; 'NOVIOP3_2A' -> barre 2.
    Retourne None si le nommage n'est pas exploitable.
    """
    groups: dict[int, int] = {}
    for n in bb_nodes:
        bbs_id = G.nodes[n].get("busbar_section_id")
        if not bbs_id:
            return None
        suffix = bbs_id
        prefix = f"{vl_id}_"
        if bbs_id.startswith(prefix):
            suffix = bbs_id[len(prefix):]
        m = re.match(r"(\d+)", suffix)
        if not m:
            return None
        groups[n] = int(m.group(1))
    return groups


def _build_coupler_subgraph(
    cellules: CellulesVL, G: nx.Graph, bb_nodes: set[int]
) -> tuple[nx.Graph, list[SwitchInfo]]:
    """
    Construit le sous-graphe de couplage : SJB + nœuds n'appartenant à aucune
    cellule de départ. Les composantes connexes de ce sous-graphe regroupant
    plusieurs SJB définissent les tronçons.

    Retourne aussi la liste des SwitchInfo des arêtes (switches) du sous-graphe.
    """
    departure_internal: set[int] = set()
    for c in cellules.cellules_depart:
        departure_internal.update(n for n in c.all_nodes if n not in bb_nodes)

    coupler_nodes = bb_nodes | (set(G.nodes()) - departure_internal)
    coupler_G = G.subgraph(coupler_nodes).copy()

    switches: list[SwitchInfo] = []
    for u, v, d in coupler_G.edges(data=True):
        if d.get("switch_id") is not None:
            switches.append(SwitchInfo(
                switch_id=d["switch_id"],
                node1=u, node2=v,
                kind=d.get("kind", SwitchKind.DISCONNECTOR),
                open=d.get("open", False),
            ))
    return coupler_G, switches


def _attribuer_departs(
    result: Tronconnement,
    cellules: CellulesVL,
    sjb_to_troncon: dict[int, int],
    barre_par_busbar: dict[int, int],
) -> None:
    """
    Rattache chaque cellule de départ à son tronçon et classe les départs
    en fixes / couplage / multiples.
    """
    for cell in cellules.cellules_depart:
        eq_ids = {cell.equipment_id} | set(cell.shared_equipment_ids)

        # Tronçon : déterminé par les SJB accessibles de la cellule
        troncon_nums = {sjb_to_troncon[n] for n in cell.busbar_nodes
                        if n in sjb_to_troncon}
        if not troncon_nums:
            logger.debug("Cellule '%s' sans tronçon rattachable.", cell.equipment_id)
            continue
        # Premier tronçon atteignable (en pratique unique pour un poste standard)
        num_troncon = min(troncon_nums)
        troncon = result.troncons[num_troncon]

        # Barres atteignables par la cellule
        barres_accessibles = {barre_par_busbar[n] for n in cell.busbar_nodes
                              if n in barre_par_busbar}

        for eq_id in eq_ids:
            troncon.departs.add(eq_id)
            result.troncon_par_depart[eq_id] = num_troncon

            if len(barres_accessibles) <= 1:
                # Départ fixe : une seule barre atteignable
                barre = next(iter(barres_accessibles)) if barres_accessibles else -1
                troncon.departs_fixes[eq_id] = barre
            else:
                # Départ ré-aiguillable entre plusieurs barres
                troncon.departs_couplage[eq_id] = set(barres_accessibles)

        if cell.is_multiple:
            troncon.departs_multiples.append(set(eq_ids))
