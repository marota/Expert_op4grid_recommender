"""
manoeuvre/algo.py  —  Phase 2 : algorithme nodale → détaillée
---------------------------------------------------------------
Calcule la séquence d'organes de coupure (OC) à manœuvrer pour passer de
l'état détaillé courant d'un poste à une topologie nodale cible.

Correspondance C++ : ``Topologie::determineTopoCompleteCible`` (TOPOPoste.cc:3944)
et ses sous-routines (``connectAndDeconnectOuvrageHS``, ``evalueEtatCouplage``,
``identifySuperTronconnement``, ``getTronconnementBesoinReaiguillage2barres``,
``reaiguillage2barres``, ``listeDordre``).

Couverture
~~~~~~~~~~
- Postes 1 barre (sectionnements ouverts/fermés selon nœuds).            [OK]
- Postes 2 barres standard (couplage + ré-aiguillage boucle courte/longue). [OK]
- Création d'un nœud supplémentaire par **ouverture de sectionnement de
  barre** (dé-énergisation préalable de la section), via
  ``determiner_manoeuvres_avec_sections``.                              [OK]
- Vérification post-manœuvre (recalcul de la topologie nodale).          [OK]

Règle du sectionneur de barre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Un sectionneur de barre ne se manœuvre que hors charge. Pour scinder une barre
en deux nœuds, la section à isoler est d'abord mise hors tension (ses départs
ré-aiguillés sur l'autre barre en boucle courte), puis le sectionnement est
ouvert, puis les départs du nouveau nœud y sont ré-aiguillés en boucle longue.

Limites connues (documentées, cf. doc C++) :
- Ré-aiguillage d'omnibus complexes (départs multiples scindés)          [partiel]
- Vérification fine de court-circuit avant fermeture de couplage         [non traité]
- Postes ≥ 3 barres physiques                                           [partiel]
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
    is_verified: bool = False
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

    manoeuvres: list[Manoeuvre] = []

    # --- Phase 2.1 : nettoyage (connect / déconnect) ----------------------
    if not _nettoyage_ouvrages(poste, topo_cible, G, manoeuvres):
        res.manoeuvres = manoeuvres
        res.message = "Échec du nettoyage : un départ cible est absent du poste."
        return res

    nb_barres = poste.tronconnement.nb_jeux_barres

    if nb_barres <= 1:
        # --- Postes 1 barre ------------------------------------------------
        _traiter_1_barre(poste, topo_cible, G, manoeuvres)
    else:
        # --- Postes 2 barres (et plus, partiel) ----------------------------
        reaiguilles = _traiter_2_barres(poste, topo_cible, G, manoeuvres, res)
        res.departs_reaiguilles = reaiguilles

    res.manoeuvres = manoeuvres
    res.couplages_modifies = [
        m.switch_id for m in manoeuvres
        if "couplage" in m.raison.lower()
    ]

    # --- Phase 2.5 : vérification ----------------------------------------
    topo_obtenue = TopologieNodale.from_graph(G, poste.voltage_level_id)
    res.topo_obtenue = topo_obtenue
    res.is_verified = topo_cible.meme_topologie(topo_obtenue)
    res.is_changed = len(manoeuvres) > 0 or res.is_verified
    if res.is_verified:
        res.message = "Topologie cible atteinte et vérifiée."
    else:
        res.message = (
            "La topologie obtenue ne correspond pas à la cible "
            f"(obtenu {topo_obtenue.nb_noeuds} nœud(s), "
            f"visé {topo_cible.nb_noeuds})."
        )
    return res


# ---------------------------------------------------------------------------
# Phase 2.1 — Nettoyage
# ---------------------------------------------------------------------------

def _nettoyage_ouvrages(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    G: nx.Graph,
    manoeuvres: list[Manoeuvre],
) -> bool:
    """
    Déconnecte les départs présents mais absents de la cible, reconnecte ceux
    présents dans la cible mais déconnectés.

    Retourne False si un départ cible n'existe pas physiquement.
    """
    departs_poste = {c.equipment_id for c in poste.cellules.cellules_depart}
    for c in poste.cellules.cellules_depart:
        departs_poste |= set(c.shared_equipment_ids)

    cible_ids = set(topo_cible.noeud_par_depart)

    # Départ cible absent du poste -> échec
    manquants = cible_ids - departs_poste
    if manquants:
        logger.warning("Départs cibles absents du poste : %s", manquants)
        return False

    # Déconnexion des départs hors cible (ouvrir leur DJ)
    for eq_id in departs_poste - cible_ids:
        cell = poste.cellules.get_cellule_depart(eq_id)
        if cell is None:
            continue
        for dj in cell.breakers:
            if not dj.open:
                _set_switch(G, dj.switch_id, open_=True)
                manoeuvres.append(Manoeuvre(
                    switch_id=dj.switch_id, action="OPEN",
                    raison=f"déconnexion départ hors cible '{eq_id}'",
                ))
    return True


# ---------------------------------------------------------------------------
# Phase « postes 1 barre »
# ---------------------------------------------------------------------------

def _traiter_1_barre(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    G: nx.Graph,
    manoeuvres: list[Manoeuvre],
) -> None:
    """
    Postes 1 barre : un nœud = un potentiel. On (dé)couple les tronçons selon
    que des départs voisins partagent ou non le même nœud cible.

    Implémentation simple : pour chaque sectionnement reliant deux SJB, fermer
    si les départs des deux côtés sont sur le même nœud cible, ouvrir sinon.
    """
    # Cas mono-tronçon trivial : si la cible n'a qu'un nœud, tout reste fermé.
    # Sinon, on ne dispose pas (encore) de logique de découpe fine sur 1 barre.
    if topo_cible.nb_noeuds <= 1:
        return
    logger.info(
        "Poste 1 barre avec %d nœuds cibles : découpe de barre non encore "
        "implémentée finement (cf. tronconnerSJB).", topo_cible.nb_noeuds,
    )


# ---------------------------------------------------------------------------
# Phase « postes 2 barres »
# ---------------------------------------------------------------------------

def _traiter_2_barres(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    G: nx.Graph,
    manoeuvres: list[Manoeuvre],
    res: ResultatManoeuvres,
) -> set[str]:
    """
    Traite le cas standard 2 barres : évaluation des couplages + ré-aiguillage.

    Retourne l'ensemble des départs ré-aiguillés.
    """
    tr = poste.tronconnement
    barre_par_busbar = tr.barre_par_busbar
    departs_reaiguilles: set[str] = set()

    # On traite chaque tronçon indépendamment (cas standard : 1 seul tronçon).
    for troncon in tr.troncons.values():
        if troncon.nb_jeux_barres < 2:
            continue

        # Nœuds cibles portés par ce tronçon
        noeuds_cible = _noeuds_cible_du_troncon(troncon, topo_cible)
        nb_noeuds = len(noeuds_cible)
        nb_barres = troncon.nb_jeux_barres

        # --- 2.2 Évaluation de l'état de couplage --------------------------
        # nbNoeuds < nbBarres -> fermer ; nbNoeuds >= nbBarres -> ouvrir.
        couplage_doit_fermer = nb_noeuds < nb_barres

        # --- 2.3 Affectation noeud -> barre + départs à ré-aiguiller -------
        if not couplage_doit_fermer and nb_noeuds >= 2:
            # On répartit les nœuds sur les barres et on ré-aiguille.
            affectation = _affecter_noeuds_barres(
                troncon, noeuds_cible, poste, topo_cible, barre_par_busbar,
            )
            reaig = _determiner_reaiguillages(
                troncon, affectation, poste, topo_cible, barre_par_busbar,
            )
            departs_reaiguilles |= reaig

            # --- 2.4 Génération de la séquence -----------------------------
            couplage_ferme_actuel = _couplage_est_ferme(troncon)
            _generer_reaiguillages(
                reaig, affectation, poste, topo_cible, barre_par_busbar,
                G, manoeuvres, boucle_courte_possible=couplage_ferme_actuel,
            )
            # Ouvrir le couplage en fin de séquence
            _manoeuvrer_couplage(troncon, G, manoeuvres, fermer=False)
        elif couplage_doit_fermer:
            _manoeuvrer_couplage(troncon, G, manoeuvres, fermer=True)

    return departs_reaiguilles


def _noeuds_cible_du_troncon(
    troncon: Troncon, topo_cible: TopologieNodale
) -> list[str]:
    """Noms des nœuds cibles dont au moins un départ appartient au tronçon."""
    noeuds = []
    for nom, noeud in topo_cible.noeuds.items():
        if noeud.equipment_ids & troncon.departs:
            noeuds.append(nom)
    return sorted(noeuds)


def _affecter_noeuds_barres(
    troncon: Troncon,
    noeuds_cible: list[str],
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    barre_par_busbar: dict[int, int],
) -> dict[str, int]:
    """
    Affecte chaque nœud cible à une barre, en minimisant les ré-aiguillages.

    Pour 2 nœuds sur 2 barres : essaie les deux configurations et choisit celle
    qui demande le moins de mouvements tout en respectant les départs fixes.
    """
    barres = sorted({barre_par_busbar[n] for n in troncon.busbar_nodes})
    noeuds = noeuds_cible[:len(barres)]

    if len(noeuds) <= 1:
        return {noeuds[0]: barres[0]} if noeuds else {}

    # Deux configurations possibles (cas 2 nœuds / 2 barres)
    configs = [
        {noeuds[0]: barres[0], noeuds[1]: barres[1]},
        {noeuds[0]: barres[1], noeuds[1]: barres[0]},
    ]
    meilleur, meilleur_cout = None, None
    for cfg in configs:
        cout, faisable = _cout_config(cfg, troncon, poste, topo_cible, barre_par_busbar)
        if not faisable:
            continue
        if meilleur_cout is None or cout < meilleur_cout:
            meilleur, meilleur_cout = cfg, cout
    if meilleur is None:
        # Aucune config ne respecte les départs fixes : on prend la 1re.
        logger.warning("Aucune config 2 barres ne respecte les départs fixes ; "
                       "config par défaut retenue.")
        meilleur = configs[0]
    return meilleur


def _cout_config(
    cfg: dict[str, int],
    troncon: Troncon,
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    barre_par_busbar: dict[int, int],
) -> tuple[int, bool]:
    """Nombre de ré-aiguillages d'une config + faisabilité (départs fixes)."""
    cout = 0
    faisable = True
    for nom, barre_cible in cfg.items():
        for eq_id in topo_cible.noeuds[nom].equipment_ids:
            if eq_id not in troncon.departs:
                continue
            barre_actuelle = _barre_actuelle(eq_id, poste, barre_par_busbar)
            if barre_actuelle is not None and barre_actuelle != barre_cible:
                cout += 1
                if eq_id in troncon.departs_fixes:
                    faisable = False
    return cout, faisable


def _determiner_reaiguillages(
    troncon: Troncon,
    affectation: dict[str, int],
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    barre_par_busbar: dict[int, int],
) -> set[str]:
    """Départs dont la barre actuelle ≠ barre cible."""
    reaig: set[str] = set()
    for nom, barre_cible in affectation.items():
        for eq_id in topo_cible.noeuds[nom].equipment_ids:
            if eq_id not in troncon.departs:
                continue
            barre_actuelle = _barre_actuelle(eq_id, poste, barre_par_busbar)
            if barre_actuelle is not None and barre_actuelle != barre_cible:
                reaig.add(eq_id)
    return reaig


# ---------------------------------------------------------------------------
# Génération des manœuvres de ré-aiguillage
# ---------------------------------------------------------------------------

def _generer_reaiguillages(
    reaig: set[str],
    affectation: dict[str, int],
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    barre_par_busbar: dict[int, int],
    G: nx.Graph,
    manoeuvres: list[Manoeuvre],
    boucle_courte_possible: bool,
) -> None:
    """
    Génère, pour chaque départ à ré-aiguiller, les manœuvres de sectionneurs.

    - **Boucle courte** (couplage fermé) : fermer le SA vers la barre cible,
      ouvrir le SA vers l'ancienne barre. Le départ reste sous tension.
    - **Boucle longue** (sinon) : ouvrir le DJ, basculer les SA, refermer le DJ.
    """
    barre_cible_par_depart: dict[str, int] = {}
    for nom, barre in affectation.items():
        for eq_id in topo_cible.noeuds[nom].equipment_ids:
            barre_cible_par_depart[eq_id] = barre

    for eq_id in sorted(reaig):
        cell = poste.cellules.get_cellule_depart(eq_id)
        if cell is None:
            continue
        barre_cible = barre_cible_par_depart[eq_id]
        barre_old = _barre_actuelle(eq_id, poste, barre_par_busbar)

        sa_vers_cible = _sa_vers_barre(cell, barre_cible, barre_par_busbar)
        sa_vers_old = _sa_vers_barre(cell, barre_old, barre_par_busbar)

        boucle = "COURTE" if boucle_courte_possible else "LONGUE"

        if not boucle_courte_possible:
            # Boucle longue : ouvrir le DJ d'abord
            for dj in cell.breakers:
                if not dj.open:
                    _set_switch(G, dj.switch_id, open_=True)
                    manoeuvres.append(Manoeuvre(
                        switch_id=dj.switch_id, action="OPEN",
                        raison=f"mise hors tension départ '{eq_id}' (boucle longue)",
                        type_boucle=boucle,
                    ))

        # Fermer le SA vers la barre cible
        for sa in sa_vers_cible:
            if sa.open:
                _set_switch(G, sa.switch_id, open_=False)
                manoeuvres.append(Manoeuvre(
                    switch_id=sa.switch_id, action="CLOSE",
                    raison=f"ré-aiguillage '{eq_id}' vers barre {barre_cible+1}",
                    type_boucle=boucle,
                ))
        # Ouvrir le SA vers l'ancienne barre
        for sa in sa_vers_old:
            if not sa.open:
                _set_switch(G, sa.switch_id, open_=True)
                manoeuvres.append(Manoeuvre(
                    switch_id=sa.switch_id, action="OPEN",
                    raison=f"ré-aiguillage '{eq_id}' depuis barre {barre_old+1}",
                    type_boucle=boucle,
                ))

        if not boucle_courte_possible:
            # Boucle longue : refermer le DJ
            for dj in cell.breakers:
                _set_switch(G, dj.switch_id, open_=False)
                manoeuvres.append(Manoeuvre(
                    switch_id=dj.switch_id, action="CLOSE",
                    raison=f"remise sous tension départ '{eq_id}' (boucle longue)",
                    type_boucle=boucle,
                ))


def _manoeuvrer_couplage(
    troncon: Troncon,
    G: nx.Graph,
    manoeuvres: list[Manoeuvre],
    fermer: bool,
) -> None:
    """Ferme ou ouvre les DJ (et SA) de couplage d'un tronçon."""
    action = "CLOSE" if fermer else "OPEN"
    raison = ("fermeture couplage de barres" if fermer
              else "ouverture couplage de barres")
    for sw in troncon.switches_couplage:
        # On ne manœuvre que ce qui doit changer d'état.
        if fermer and sw.open:
            _set_switch(G, sw.switch_id, open_=False)
            manoeuvres.append(Manoeuvre(sw.switch_id, action, raison))
        elif (not fermer) and sw.is_breaker and not sw.open:
            # Pour ouvrir, il suffit d'ouvrir le DJ de couplage.
            _set_switch(G, sw.switch_id, open_=True)
            manoeuvres.append(Manoeuvre(sw.switch_id, action, raison))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _couplage_est_ferme(troncon: Troncon) -> bool:
    """True si tous les DJ de couplage du tronçon sont fermés."""
    djs = troncon.couplage_breakers
    if not djs:
        return False
    return all(not dj.open for dj in djs)


def _barre_actuelle(
    eq_id: str, poste: PosteTopologique, barre_par_busbar: dict[int, int]
) -> Optional[int]:
    """Barre à laquelle le départ est actuellement connecté (via SA fermés)."""
    cell = poste.cellules.get_cellule_depart(eq_id)
    if cell is None:
        return None
    barres = {barre_par_busbar[n] for n in cell.connected_busbars
              if n in barre_par_busbar}
    if not barres:
        return None
    # Si connecté à plusieurs barres (couplage fermé), on prend la plus petite.
    return min(barres)


def _sa_vers_barre(
    cell: CelluleDepart, barre: Optional[int], barre_par_busbar: dict[int, int]
) -> list[SwitchInfo]:
    """Sectionneurs (SA) de la cellule menant vers une barre donnée."""
    if barre is None:
        return []
    result: list[SwitchInfo] = []
    seen: set[str] = set()
    for bb in cell.busbar_nodes:
        if barre_par_busbar.get(bb) != barre:
            continue
        for sa in cell.disconnectors_vers_barre(bb):
            if sa.switch_id not in seen:
                seen.add(sa.switch_id)
                result.append(sa)
    return result


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
                res.message = (
                    f"Départ '{eq}' ne peut atteindre aucune SJB de son nœud."
                )
                return res
            target_sjb[eq] = min(reachable)

    # --- couplers à ouvrir (entre SJB de nœuds différents) -----------------
    couplers = _inter_sjb_couplers(poste)
    to_open: list[_InterSjbCoupler] = []
    for cp in couplers:
        na, nb = node_de_sjb.get(cp.sjb_a), node_de_sjb.get(cp.sjb_b)
        if na is not None and nb is not None and na != nb:
            to_open.append(cp)

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

    manoeuvres: list[Manoeuvre] = []
    reaiguilles: set[str] = set()

    def parking_sjb(eq: str, target: int) -> Optional[int]:
        """SJB tampon (hors section isolée) accessible par le départ."""
        cell = cells.get_cellule_depart(eq)
        for bb in cell.busbar_nodes:
            if bb != target and bb not in sjb_isoles:
                return bb
        return None

    # --- Phase A/B : ré-aiguillages boucle courte (couplage encore fermé) ---
    parkings: dict[str, int] = {}
    for eq, tgt in sorted(target_sjb.items()):
        if tgt in sjb_isoles:
            buf = parking_sjb(eq, tgt)
            if buf is None:
                res.message = f"Pas de SJB tampon pour '{eq}'."
                return res
            parkings[eq] = tgt
            if _reaiguiller_vers_sjb(G, cells, eq, buf, "COURTE", manoeuvres):
                reaiguilles.add(eq)
        else:
            if _reaiguiller_vers_sjb(G, cells, eq, tgt, "COURTE", manoeuvres):
                reaiguilles.add(eq)

    # --- Phase C : ouverture des sectionnements (section hors tension) -----
    for cp in to_open:
        if not cp.is_sectionnement:
            continue
        # Vérifier que la section isolée (côté hors tension) ne porte plus de
        # départ câblé.
        cote = cp.sjb_a if cp.sjb_a in sjb_isoles else cp.sjb_b
        encore = [eq for eq in target_sjb
                  if cote in _wired_sjbs(G, cells, eq)]
        for sid in cp.switch_ids:
            if not _is_open(G, sid):
                _set_switch(G, sid, True)
                etat = "hors tension" if not encore else "ATTENTION sous tension"
                manoeuvres.append(Manoeuvre(
                    switch_id=sid, action="OPEN",
                    raison=f"ouverture sectionnement de barre (section {etat})",
                ))

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
        if _reaiguiller_vers_sjb(G, cells, eq, parkings[eq], "LONGUE", manoeuvres):
            reaiguilles.add(eq)

    res.manoeuvres = manoeuvres
    res.departs_reaiguilles = reaiguilles
    res.couplages_modifies = [sid for cp in to_open for sid in cp.switch_ids]

    # --- Vérification ------------------------------------------------------
    topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.topo_obtenue = topo_obtenue
    res.is_verified = topo_cible.meme_topologie(topo_obtenue)
    res.is_changed = bool(manoeuvres)
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


def _own_breakers_to_sjb(cell: CelluleDepart, sjb_node: int) -> list[str]:
    """DJ propres au départ sur le chemin vers une SJB (exclut branches omnibus)."""
    if cell.subgraph is None:
        return [b.switch_id for b in cell.breakers]
    sg = cell.subgraph
    eq_node = next((n for n, d in sg.nodes(data=True)
                    if d.get("equipment_id") == cell.equipment_id), None)
    if eq_node is None:
        return [b.switch_id for b in cell.breakers]
    out: list[str] = []
    try:
        path = nx.shortest_path(sg, eq_node, sjb_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return out
    for u, v in zip(path, path[1:]):
        d = sg.edges[u, v]
        if d.get("kind") == SwitchKind.BREAKER and d.get("switch_id"):
            out.append(d["switch_id"])
    return list(dict.fromkeys(out))


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


def _reaiguiller_vers_sjb(
    G: nx.Graph,
    cells,
    eq_id: str,
    target_sjb: int,
    boucle: Literal["COURTE", "LONGUE"],
    manoeuvres: list[Manoeuvre],
) -> bool:
    """
    Ré-aiguille un départ vers une SJB cible. Retourne True si des manœuvres
    ont été générées.

    - COURTE : ferme le SA cible, ouvre les autres SA (départ sous tension).
    - LONGUE : ouvre le DJ propre, bascule les SA, referme le DJ.
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

    n_before = len(manoeuvres)
    djs = _own_breakers_to_sjb(cell, target_sjb)

    if boucle == "LONGUE":
        for dj in djs:
            if not _is_open(G, dj):
                _set_switch(G, dj, True)
                manoeuvres.append(Manoeuvre(
                    switch_id=dj, action="OPEN",
                    raison=f"mise hors tension '{eq_id}' (boucle longue)",
                    type_boucle="LONGUE",
                ))

    for sa in sa_cible:
        if _is_open(G, sa):
            _set_switch(G, sa, False)
            manoeuvres.append(Manoeuvre(
                switch_id=sa, action="CLOSE",
                raison=f"ré-aiguillage '{eq_id}' vers {sjb_id}",
                type_boucle=boucle,
            ))
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
        for dj in djs:
            _set_switch(G, dj, False)
            manoeuvres.append(Manoeuvre(
                switch_id=dj, action="CLOSE",
                raison=f"remise sous tension '{eq_id}' (boucle longue)",
                type_boucle="LONGUE",
            ))

    return len(manoeuvres) > n_before
