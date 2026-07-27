"""
manoeuvre/topologie.py  —  Étapes 1.5 et 1.6
----------------------------------------------
Représentation de la topologie *nodale* d'un poste (quels équipements sont sur
quels nœuds électriques) et assemblage de la vue complète ``PosteTopologique``.

Correspondance C++ : ``Topologie``, ``Noeud``, ``Depart`` (TOPOPoste.h) et la
phase 3 ``getNoeudTronconnement`` (attribution des nœuds aux tronçons).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .models import EquipmentType
from .graph import busbar_nodes, equipment_nodes, get_node_attrs
from .cellules import CellulesVL, detecter_cellules
from .troncons import Tronconnement, construire_tronconnement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class DepartInfo:
    """Départ (équipement) dans une topologie nodale."""
    equipment_id: str
    equipment_type: Optional[EquipmentType]
    noeud: str                     # nom du nœud électrique ("N0", "N1", …)
    troncon: Optional[int] = None  # numéro de tronçon (après attribution)
    is_connected: bool = True


@dataclass
class NoeudElectrique:
    """Nœud électrique (bus) regroupant des départs au même potentiel."""
    nom: str
    departs: list[DepartInfo] = field(default_factory=list)
    troncons: set[int] = field(default_factory=set)

    @property
    def equipment_ids(self) -> set[str]:
        return {d.equipment_id for d in self.departs}


@dataclass
class TopologieNodale:
    """
    Topologie nodale d'un voltage level : partition des départs en nœuds
    électriques.
    """
    voltage_level_id: str
    noeuds: dict[str, NoeudElectrique] = field(default_factory=dict)
    noeud_par_depart: dict[str, str] = field(default_factory=dict)
    #: Noms des « nœuds » qui ne portent **aucune barre** : ce sont des ouvrages
    #: **déconnectés** (composante 0-barre), pas de véritables nœuds électriques.
    #: Renseigné par ``from_graph`` uniquement — une topologie *cible* construite
    #: depuis une partition (``from_node_groups``) n'a pas cette notion.
    noeuds_isoles: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------

    @classmethod
    def from_graph(cls, G: nx.Graph, voltage_level_id: str) -> "TopologieNodale":
        """
        Extrait la topologie nodale *courante* depuis l'état détaillé du graphe.

        Un nœud électrique = composante connexe du sous-graphe restreint aux
        switches **fermés** (+ connexions internes), regroupant les équipements
        au même potentiel.

        Les composantes ne contenant **aucune barre** sont des ouvrages
        **déconnectés** (« ouvrages isolés » de l'IHM) : elles restent des
        entrées de ``noeuds`` (compatibilité) mais sont référencées dans
        ``noeuds_isoles`` pour que la comparaison de topologies puisse les
        distinguer d'un vrai nœud électrique.
        """
        closed_G = nx.Graph()
        closed_G.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            if not d.get("open", False):
                closed_G.add_edge(u, v)

        eq_nodes = equipment_nodes(G)
        barres = set(busbar_nodes(G))

        # Composante connexe de chaque nœud d'équipement
        comp_id: dict[int, int] = {}
        comp_avec_barre: dict[int, bool] = {}
        for idx, comp in enumerate(nx.connected_components(closed_G)):
            comp_avec_barre[idx] = bool(comp & barres)
            for n in comp:
                comp_id[n] = idx

        # Regrouper les équipements par composante
        groupes: dict[int, list[int]] = {}
        for eqn in eq_nodes:
            cid = comp_id.get(eqn)
            if cid is None:
                continue
            groupes.setdefault(cid, []).append(eqn)

        topo = cls(voltage_level_id=voltage_level_id)
        # Nommage déterministe : trier les groupes par plus petit equipment_id
        ordered = sorted(
            groupes.items(),
            key=lambda item: min(
                G.nodes[n].get("equipment_id") or "" for n in item[1]
            ),
        )
        for i, (cid, nodes) in enumerate(ordered):
            nom = f"N{i}"
            noeud = NoeudElectrique(nom=nom)
            if not comp_avec_barre.get(cid, True):
                topo.noeuds_isoles.add(nom)   # composante sans barre = ouvrage isolé
            for n in nodes:
                attrs = get_node_attrs(G, n)
                dep = DepartInfo(
                    equipment_id=attrs.equipment_id,
                    equipment_type=attrs.equipment_type,
                    noeud=nom,
                )
                noeud.departs.append(dep)
                topo.noeud_par_depart[attrs.equipment_id] = nom
            topo.noeuds[nom] = noeud
        return topo

    @classmethod
    def from_bus_assignment(
        cls,
        voltage_level_id: str,
        bus_map: dict[str, int],
        equipment_types: Optional[dict[str, EquipmentType]] = None,
    ) -> "TopologieNodale":
        """
        Construit une topologie nodale depuis une assignation équipement -> bus.

        Parameters
        ----------
        bus_map :
            equipment_id -> numéro de nœud électrique (entier arbitraire).
        equipment_types :
            optionnel : equipment_id -> EquipmentType.
        """
        equipment_types = equipment_types or {}
        topo = cls(voltage_level_id=voltage_level_id)

        # Regrouper par bus, nommage déterministe (ordre des bus)
        groupes: dict[int, list[str]] = {}
        for eq_id, bus in bus_map.items():
            groupes.setdefault(bus, []).append(eq_id)

        for i, bus in enumerate(sorted(groupes)):
            nom = f"N{i}"
            noeud = NoeudElectrique(nom=nom)
            for eq_id in sorted(groupes[bus]):
                dep = DepartInfo(
                    equipment_id=eq_id,
                    equipment_type=equipment_types.get(eq_id),
                    noeud=nom,
                )
                noeud.departs.append(dep)
                topo.noeud_par_depart[eq_id] = nom
            topo.noeuds[nom] = noeud
        return topo

    @classmethod
    def from_node_groups(
        cls,
        voltage_level_id: str,
        groups: list[list[str]],
        equipment_types: Optional[dict[str, EquipmentType]] = None,
    ) -> "TopologieNodale":
        """
        Construit une topologie nodale depuis une liste ordonnée de groupes de
        départs (un groupe = un nœud électrique). Le nœud i porte le nom 'Ni',
        ce qui permet d'exprimer directement une cible « noeud n°0 / 1 / 2 ».
        """
        bus_map: dict[str, int] = {}
        for i, grp in enumerate(groups):
            for eq_id in grp:
                bus_map[eq_id] = i
        return cls.from_bus_assignment(voltage_level_id, bus_map, equipment_types)

    # ------------------------------------------------------------------
    # Comparaison
    # ------------------------------------------------------------------

    def partition(self) -> set[frozenset[str]]:
        """Partition des équipements en ensembles (un par nœud électrique)."""
        return {frozenset(n.equipment_ids) for n in self.noeuds.values()}

    def univers(self) -> set[str]:
        """Ensemble des équipements couverts par cette topologie."""
        return set(self.noeud_par_depart)

    def partition_hors_isoles_inconnus(
        self, univers_autre: set[str]
    ) -> set[frozenset[str]]:
        """Partition restreinte au **périmètre** de ``univers_autre`` : les
        ouvrages **isolés** (composante 0-barre) dont aucun équipement
        n'apparaît dans ``univers_autre`` sont écartés.

        C'est ce qui permet de comparer une topologie *réalisée* (issue du
        graphe, qui « voit » les ouvrages déconnectés) à une topologie *cible*
        éditée par l'expert, qui ne mentionne que les ouvrages à placer sur un
        nœud : un ouvrage déjà déconnecté et laissé hors cible ne doit pas
        rendre la cible non réalisable. Un isolé **mentionné** par la cible
        reste comparé strictement (l'expert demande sa reconnexion)."""
        garde: set[frozenset[str]] = set()
        for nom, noeud in self.noeuds.items():
            ids = frozenset(noeud.equipment_ids)
            if nom in self.noeuds_isoles and not (ids & univers_autre):
                continue          # ouvrage isolé hors du périmètre comparé
            garde.add(ids)
        return garde

    def meme_topologie(self, other: "TopologieNodale") -> bool:
        """
        Compare deux topologies nodales par isomorphisme de partition
        (les noms de nœuds sont ignorés).

        Les **ouvrages isolés hors périmètre** (déconnectés d'un côté, absents
        de l'autre) sont ignorés de part et d'autre : ce ne sont pas des nœuds
        électriques et ils ne peuvent donc pas invalider une cible qui ne les
        mentionne pas. La comparaison reste **stricte** dès que les deux
        topologies parlent du même équipement.
        """
        return (self.partition_hors_isoles_inconnus(other.univers())
                == other.partition_hors_isoles_inconnus(self.univers()))

    @property
    def nb_noeuds(self) -> int:
        return len(self.noeuds)

    @property
    def nb_noeuds_reels(self) -> int:
        """Nombre de **vrais** nœuds électriques : ouvrages isolés exclus.

        (``nb_noeuds`` compte toute composante portant un équipement, ouvrages
        déconnectés compris — conservé tel quel car le placement/séquencement
        s'appuie dessus.)"""
        return len(self.noeuds) - len(self.noeuds_isoles)

    def resume(self) -> str:
        parts = [f"TopologieNodale VL '{self.voltage_level_id}': {self.nb_noeuds} nœud(s)"]
        for nom in sorted(self.noeuds):
            ids = sorted(self.noeuds[nom].equipment_ids)
            parts.append(f"  {nom}: {', '.join(ids)}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Étape 1.4 — Attribution des nœuds aux tronçons
# ---------------------------------------------------------------------------

def attribuer_noeuds(
    tronconnement: Tronconnement,
    topo: TopologieNodale,
) -> None:
    """
    Attribue à chaque tronçon les nœuds électriques qu'il porte (étape 1.4 /
    phase 3 ``getNoeudTronconnement``), avec consolidation des tronçons
    intermédiaires.

    Met à jour ``Troncon.noeuds_electriques`` (nom_noeud -> set(barres
    potentielles)) et ``DepartInfo.troncon``.
    """
    # Réinitialiser
    for t in tronconnement.troncons.values():
        t.noeuds_electriques = {}

    # 1. Mapping initial : pour chaque départ, son tronçon et son nœud
    noeud_troncons: dict[str, set[int]] = {}
    for nom, noeud in topo.noeuds.items():
        for dep in noeud.departs:
            num = tronconnement.troncon_par_depart.get(dep.equipment_id)
            if num is None:
                continue
            dep.troncon = num
            tronconnement.troncons[num].noeuds_electriques.setdefault(nom, set())
            noeud.troncons.add(num)
            noeud_troncons.setdefault(nom, set()).add(num)

    # 2. Consolidation : un nœud présent sur deux tronçons non consécutifs
    #    impose sa présence sur les tronçons intermédiaires (pas d'OC de
    #    pontage pour les isoler). On comble les trous par numéro de tronçon.
    for nom, troncons in noeud_troncons.items():
        if len(troncons) < 2:
            continue
        lo, hi = min(troncons), max(troncons)
        for num in range(lo, hi + 1):
            if num in tronconnement.troncons:
                tronconnement.troncons[num].noeuds_electriques.setdefault(nom, set())


# ---------------------------------------------------------------------------
# Étape 1.6 — PosteTopologique
# ---------------------------------------------------------------------------

@dataclass
class PosteTopologique:
    """Vue complète d'un poste : cellules + tronçonnement + topologie nodale."""
    voltage_level_id: str
    cellules: CellulesVL
    tronconnement: Tronconnement
    topologie_nodale: TopologieNodale
    graph: Optional[nx.Graph] = field(default=None, repr=False)

    @classmethod
    def from_graph(cls, G: nx.Graph, voltage_level_id: str) -> "PosteTopologique":
        """Construit la vue complète depuis un graphe node/breaker."""
        cellules = detecter_cellules(G, voltage_level_id)
        tronconnement = construire_tronconnement(cellules, G)
        topo = TopologieNodale.from_graph(G, voltage_level_id)
        attribuer_noeuds(tronconnement, topo)
        return cls(
            voltage_level_id=voltage_level_id,
            cellules=cellules,
            tronconnement=tronconnement,
            topologie_nodale=topo,
            graph=G,
        )

    @property
    def nb_jeux_barres(self) -> int:
        return self.tronconnement.nb_jeux_barres

    def resume(self) -> str:
        return (
            f"Poste '{self.voltage_level_id}'\n"
            + self.cellules.resume() + "\n"
            + self.tronconnement.resume() + "\n"
            + self.topologie_nodale.resume()
        )
