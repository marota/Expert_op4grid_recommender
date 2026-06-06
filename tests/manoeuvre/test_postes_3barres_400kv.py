"""
tests/manoeuvre/test_postes_3barres_400kv.py
--------------------------------------------
Batterie de **topologies cibles à 3 et 4 nœuds** sur plusieurs **postes 400 kV à
3 jeux de barres** réels (réseau France 28/08/2024) :

    SSV.OP7, TAVELP7, TRI.PP7, ARGOEP7, CHESNP7, COR.PP7, CERGYP7
    (tous : 3 barres × 2 demi-rames, 6 SJB, 9-16 départs).

Deux niveaux d'exigence, conformes à l'état réel du moteur :

1. **Réalisation complète** (``SSV.OP7``) : plusieurs cibles 3 et 4 nœuds sont
   atteintes ET vérifiées (placement N-barres + séquençage Phase F).

2. **Innocuité** (tous les postes) : pour une large variété de cibles 3/4 nœuds,
   le moteur ne régresse jamais — graphe du poste non muté, manœuvres portant sur
   des organes existants, **aucune dégradation par scoping** « 2 jeux de barres »,
   vérificateur de sectionneurs aligné, et cohérence ``is_verified`` ⇒ 0 écart.

   Limite connue (documentée) : sur les postes en *triangle* à cellules de
   couplage **multi-barres partagées** (COUPL.A/COUPL.B/LIAIS), `_inter_sjb_couplers`
   décompose mal le faisceau partagé ⇒ certaines cibles ne sont pas réalisées
   exactement. La Phase F est **transactionnelle** : elle n'est conservée que si
   elle atteint la cible, sinon annulée — donc jamais de sur-fragmentation.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    determiner_topo_complete_cible,
    sectionneurs_sous_charge_par_manoeuvre,
)
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _wired_busbar, _wired_sjbs
from expert_op4grid_recommender.manoeuvre.algo.placement import _placement_automatique
from expert_op4grid_recommender.manoeuvre.algo.sequencing import (
    determiner_manoeuvres_par_connectivite,
)

from .fixture_loader import (
    build_graph_from_fixture,
    get_fixture_metadata,
    list_available_fixtures,
)

POSTES_3B_400 = ["SSV_OP7", "TAVELP7", "TRI_PP7", "ARGOEP7",
                 "CHESNP7", "COR_PP7", "CERGYP7"]

_DISPONIBLES = [n for n in POSTES_3B_400 if n in list_available_fixtures()]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _poste(name: str) -> PosteTopologique:
    vl = get_fixture_metadata(name)["voltage_level_id"]
    return PosteTopologique.from_graph(build_graph_from_fixture(name), vl)


def _switch_states(G):
    return {d["switch_id"]: bool(d.get("open", False))
            for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _known_switch_ids(G):
    return {d.get("switch_id") for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _departs(poste) -> list[str]:
    return sorted(e for n in poste.topologie_nodale.noeuds.values()
                  for e in n.equipment_ids)


def _par_barre(poste):
    bp = poste.tronconnement.barre_par_busbar
    out = defaultdict(list)
    for c in poste.cellules.cellules_depart:
        b = bp.get(_wired_busbar(c, poste.graph))
        if b is not None:
            out[b].append(c.equipment_id)
    return out


def _cible_separer_barres(poste) -> list[list[str]]:
    """Cible **bien formée** « séparer les barres couplées » : scinde le nœud
    principal (couplé) par barre câblée, et **préserve les autres nœuds courants**
    tels quels (les départs déjà déconnectés restent des nœuds isolés). C'est la
    manœuvre d'exploitation séparée par jeu de barres."""
    bp = poste.tronconnement.barre_par_busbar
    noeuds = list(poste.topologie_nodale.noeuds.values())
    big = max(noeuds, key=lambda n: len(n.equipment_ids))
    groups = [sorted(n.equipment_ids) for n in noeuds if n is not big]
    sub: dict = defaultdict(list)
    for eq in sorted(big.equipment_ids):   # tri → cible déterministe (indép. PYTHONHASHSEED)
        cell = poste.cellules.get_cellule_depart(eq)
        wb = _wired_busbar(cell, poste.graph) if cell else None
        sub[bp.get(wb) if wb is not None else f"iso_{eq}"].append(eq)
    groups += [sorted(v) for v in sub.values()]
    return [g for g in groups if g]


def _cible_tronconner(poste):
    """Cible bien formée = ``_cible_separer_barres`` + tronçonnage (demi-rame) de
    la barre la plus chargée. Sollicite un sectionnement intra-barre."""
    bp = poste.tronconnement.barre_par_busbar
    sep = _cible_separer_barres(poste)
    # repérer le plus gros groupe tenant sur une seule barre (≥ 2 départs).
    def _barre_unique(grp):
        bs = set()
        for eq in grp:
            cell = poste.cellules.get_cellule_depart(eq)
            wb = _wired_busbar(cell, poste.graph) if cell else None
            bs.add(bp.get(wb))
        return next(iter(bs)) if len(bs) == 1 and None not in bs else None

    cibles = [(g, _barre_unique(g)) for g in sep]
    candidats = [(g, b) for g, b in cibles if b is not None and len(g) >= 2]
    if not candidats:
        return None
    big, _b = max(candidats, key=lambda gb: len(gb[0]))
    half: dict = defaultdict(list)
    for eq in big:
        sj = _wired_sjbs(poste.graph, poste.cellules, eq)
        half[sorted(sj)[0] if sj else None].append(eq)
    if len(half) < 2:
        return None
    out = [g for g in sep if g is not big] + [sorted(v) for v in half.values() if v]
    return [g for g in out if g]


def _cibles(poste) -> dict[str, list[list[str]]]:
    """Plusieurs partitions cibles à 3 et 4 nœuds (round-robin, isolement,
    tronçonnage de la barre la plus chargée)."""
    d = _departs(poste)
    out: dict[str, list[list[str]]] = {
        "rr3": [d[i::3] for i in range(3)],
        "rr4": [d[i::4] for i in range(4)],
    }
    bb = _par_barre(poste)  # départs **câblés** uniquement, groupés par barre
    pop = sorted((b for b in bb if bb[b]), key=lambda b: -len(bb[b]))
    # "sep" : séparer les barres courantes (départs câblés maintenus en place).
    grp_sep = [list(v) for v in bb.values() if v]
    if len(grp_sep) >= 2:
        out["sep"] = grp_sep
    if pop and len(bb[pop[0]]) >= 2:
        rest = [list(bb[b]) for b in pop]
        moved = rest[0].pop()
        out["iso"] = [g for g in rest if g] + [[moved]]
        big = max((list(bb[b]) for b in pop), key=len)
        others = [list(bb[b]) for b in pop if list(bb[b]) is not big]
        out["splitbig"] = others + [big[:len(big) // 2], big[len(big) // 2:]]
        # "tron" : scinder la barre la plus chargée par demi-rame (tronçonnage).
        sub: dict[int, list[str]] = {}
        for eq in bb[pop[0]]:
            sj = _wired_sjbs(poste.graph, poste.cellules, eq)
            key = sorted(sj)[0] if sj else None
            if key is not None:
                sub.setdefault(key, []).append(eq)
        if len(sub) >= 2:
            out["tron"] = [list(bb[b]) for b in pop[1:] if bb[b]] + \
                          [list(v) for v in sub.values() if v]
    return {k: [g for g in v if g] for k, v in out.items()}


# --------------------------------------------------------------------------
# 1. Réalisation complète sur SSV.OP7 (cibles 3 et 4 nœuds)
# --------------------------------------------------------------------------

# Cibles 3/4 nœuds réalisées de bout en bout grâce au **réalisateur
# connectivité-based** (séquenceur bay-aware) : SSV.OP7 (toutes formes) et
# TAVELP7 (séparation par barre, tronçonnage). Verrouille les gains de l'étape 2.
_REALISATIONS_VERIFIEES = [
    ("SSV_OP7", "rr3"), ("SSV_OP7", "rr4"), ("SSV_OP7", "iso"), ("SSV_OP7", "tron"),
    ("TAVELP7", "sep"), ("TAVELP7", "tron"),
]


# --------------------------------------------------------------------------
# Redesign couplage : effet de la pénalité (placement mono-barre) + réalisateur
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
def test_placement_mono_barre_pour_separation(name):
    """Effet de ``POIDS_NOEUD_MULTIBARRE`` : pour une cible « séparer barres » sur
    un poste à faisceaux partagés, le placement affecte chaque nœud à **une seule
    barre** (pas de demi-rames croisées « exotiques »)."""
    poste = _poste(name)
    groups = _cible_separer_barres(poste)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    if cible.nb_noeuds < 3:
        pytest.skip(f"{name}: séparation < 3 nœuds")
    placement, faisable, msg, _np = _placement_automatique(poste, cible)
    assert faisable is True, f"{name}: {msg}"
    bp = poste.tronconnement.barre_par_busbar
    sjbid2barre = {poste.graph.nodes[n].get("busbar_section_id"): bp[n] for n in bp}
    for _deps, sjbs in placement:
        barres = {sjbid2barre[s] for s in sjbs}
        assert len(barres) == 1, f"{name}: nœud multi-barre {sjbs}"


@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
def test_realisateur_connectivite_engage_et_separe(name):
    """Sur un poste à faisceaux partagés, ``determiner_topo_complete_cible``
    bascule sur le **réalisateur connectivité-based** : la cible « séparer barres »
    est vérifiée ET au moins une manœuvre « séparation de nœuds » (Phase F) est
    émise — preuve que le repli bay-aware est engagé."""
    poste = _poste(name)
    groups = _cible_separer_barres(poste)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    if cible.nb_noeuds < 3:
        pytest.skip(f"{name}: séparation < 3 nœuds")
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified is True, f"{name}: {res.message}"
    assert any("séparation de nœuds" in m.raison for m in res.manoeuvres), \
        f"{name}: le réalisateur connectivité aurait dû séparer des nœuds"


@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
def test_realisateur_connectivite_sur_et_pur(name):
    """Contrat du réalisateur connectivité (appel direct) : il **ne mute pas** le
    graphe du poste, ses manœuvres portent sur des organes existants, et s'il
    déclare ``is_verified`` la topologie obtenue est **exactement** la cible."""
    poste = _poste(name)
    before = _switch_states(poste.graph)
    known = _known_switch_ids(poste.graph)
    groups = _cible_separer_barres(poste)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    placement, faisable, _msg, _np = _placement_automatique(poste, cible)
    if not faisable:
        pytest.skip(f"{name}: placement non faisable")

    res = determiner_manoeuvres_par_connectivite(poste, placement, cible)

    assert _switch_states(poste.graph) == before  # pureté : poste non muté
    assert all(m.switch_id in known for m in res.manoeuvres)
    if res.is_verified:
        assert cible.meme_topologie(res.topo_obtenue)


@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
def test_separer_barres_realise(name):
    """**Manœuvre d'exploitation séparée par jeu de barres** (cible bien formée,
    états de service préservés) : réalisée ET vérifiée sur **tous** les postes
    400 kV à 3 barres (3-7 nœuds selon les départs déconnectés). C'est le cœur
    opérationnel du redesign couplage (réalisateur connectivité-based)."""
    poste = _poste(name)
    before = _switch_states(poste.graph)
    groups = _cible_separer_barres(poste)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    if cible.nb_noeuds < 3:
        pytest.skip(f"{name}: cible séparation < 3 nœuds (barre de réserve)")
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified is True, f"{name} séparer-barres: {res.message}"
    assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
    assert all(m.switch_id in _known_switch_ids(poste.graph) for m in res.manoeuvres)
    assert _switch_states(poste.graph) == before


@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
def test_tronconner_barre_realise(name):
    """**Tronçonnage d'une barre** (séparation + scission d'un JdB en demi-rames)
    : cible 4+ nœuds, réalisée ET vérifiée sur tous les postes 400 kV à 3 barres."""
    poste = _poste(name)
    before = _switch_states(poste.graph)
    groups = _cible_tronconner(poste)
    if not groups or len({frozenset(g) for g in groups}) < 3:
        pytest.skip(f"{name}: pas de tronçonnage exploitable")
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified is True, f"{name} tronçonnage: {res.message}"
    assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
    assert _switch_states(poste.graph) == before


@pytest.mark.parametrize("name,shape", _REALISATIONS_VERIFIEES)
def test_realisation_3_4_noeuds_verifiee(name, shape):
    """La cible (3 ou 4 nœuds) est atteinte ET vérifiée par le réalisateur
    connectivité-based, sur un vrai poste 400 kV à 3 barres."""
    if name not in _DISPONIBLES:
        pytest.skip(f"Fixture {name} absente.")
    poste = _poste(name)
    cibles = _cibles(poste)
    if shape not in cibles:
        pytest.skip(f"forme {shape} indisponible pour {name}")
    before = _switch_states(poste.graph)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, cibles[shape])
    assert cible.nb_noeuds in (3, 4)

    res = determiner_topo_complete_cible(poste, cible)

    assert res.is_verified is True, f"{name}/{shape}: {res.message}"
    assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
    assert res.nb_manoeuvres > 0
    assert all(m.switch_id in _known_switch_ids(poste.graph) for m in res.manoeuvres)
    assert _switch_states(poste.graph) == before  # graphe non muté


@pytest.mark.skipif("SSV_OP7" not in _DISPONIBLES, reason="Fixture SSV_OP7 absente.")
@pytest.mark.parametrize("shape", ["rr3", "rr4", "iso"])
def test_ssv_op7_realise_cible(shape):
    """SSV.OP7 : la cible (3 ou 4 nœuds) est atteinte ET vérifiée, sans écart."""
    poste = _poste("SSV_OP7")
    before = _switch_states(poste.graph)
    groups = _cibles(poste)[shape]
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    assert cible.nb_noeuds in (3, 4)

    res = determiner_topo_complete_cible(poste, cible)

    assert res.is_verified is True, f"{shape}: {res.message}"
    assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
    assert res.ecarts == []
    assert res.nb_manoeuvres > 0
    assert all(m.switch_id in _known_switch_ids(poste.graph) for m in res.manoeuvres)
    assert _switch_states(poste.graph) == before  # graphe non muté


# --------------------------------------------------------------------------
# 2. Innocuité sur tous les postes 3-barres (cibles 3 et 4 nœuds)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
@pytest.mark.parametrize("shape", ["rr3", "rr4", "iso", "splitbig"])
def test_cible_3_ou_4_noeuds_innocuite(name, shape):
    """Pour une large variété de cibles 3/4 nœuds sur des postes 3-barres réels,
    le moteur reste **sûr** quel que soit le résultat de réalisation."""
    poste = _poste(name)
    cibles = _cibles(poste)
    if shape not in cibles:
        pytest.skip(f"forme {shape} indisponible pour {name}")
    before = _switch_states(poste.graph)
    known = _known_switch_ids(poste.graph)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, cibles[shape])

    res = determiner_topo_complete_cible(poste, cible)

    # (a) graphe du poste jamais muté.
    assert _switch_states(poste.graph) == before
    # (b) toute manœuvre porte sur un organe existant.
    assert all(m.switch_id in known for m in res.manoeuvres)
    # (c) plus aucune dégradation par scoping « 2 jeux de barres ».
    assert "niveaux de barres supplémentaires" not in res.message
    assert "algorithme 2 jeux de barres" not in res.message
    # (d) vérificateur de sectionneurs : sortie alignée, pas d'exception.
    viol = sectionneurs_sous_charge_par_manoeuvre(poste, res.manoeuvres)
    assert len(viol) == len(res.manoeuvres)
    assert all(v is None or isinstance(v, str) for v in viol)
    # (e) cohérence : si vérifié, alors topologie exacte et aucun écart.
    assert res.topo_obtenue is not None
    if res.is_verified:
        assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
        assert res.ecarts == []


# --------------------------------------------------------------------------
# 3. La Phase F est transactionnelle (sur SSV.OP7, placement propre)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 4. FILET DE CARACTÉRISATION — base verrouillée avant la réécriture du
#    séquenceur « bay-aware » (étape 2/2 du redesign couplage).
# --------------------------------------------------------------------------
#
# Épingle l'état ACTUEL de réalisation (``is_verified``) de cibles déterministes
# 3/4 nœuds. `SSV.OP7` réalise déjà (placement propre + Phase F) ; les postes
# *triangle* à faisceaux de couplage **partagés** (TAVELP7…) ne réalisent pas
# encore exactement — c'est l'objet de la réécriture séquenceur à venir.
#
# >>> Quand le séquenceur bay-aware débloquera ces postes, les ``False`` ci-dessous
#     deviendront ``True`` : ce filet ÉCHOUERA et devra être mis à jour
#     **consciemment** (preuve du gain). C'est le signal de progression voulu.
_REALISATION_ACTUELLE = {
    "SSV_OP7": {"rr3": True, "rr4": True},
    # TAVELP7 rr3 débloqué par le correctif « ne pas ouvrir un sectionneur
    # partagé avec la barre cible » (sectionneur de ligne SL commun).
    "TAVELP7": {"rr3": True, "rr4": False},
    "TRI_PP7": {"rr3": False, "rr4": False},
    "ARGOEP7": {"rr3": False, "rr4": False},
    "CHESNP7": {"rr3": False, "rr4": False},
    "COR_PP7": {"rr3": False, "rr4": False},
    "CERGYP7": {"rr3": False, "rr4": False},
}


@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste 3-barres 400 kV.")
@pytest.mark.parametrize("name", _DISPONIBLES)
@pytest.mark.parametrize("shape", ["rr3", "rr4"])
def test_caracterisation_realisation_sequenceur(name, shape):
    """Filet : verrouille l'état actuel de réalisation par poste/forme. Le
    placement (mono-barre) est déjà corrigé ; ce filet suit la réalisation
    SÉQUENCÉE, cible de l'étape 2/2 (séquenceur bay-aware)."""
    attendu = _REALISATION_ACTUELLE.get(name, {}).get(shape)
    if attendu is None:
        pytest.skip(f"pas d'attendu épinglé pour {name}/{shape}")
    poste = _poste(name)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, _cibles(poste)[shape])
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified is attendu, (
        f"{name}/{shape}: réalisation={res.is_verified} (épinglé {attendu}). "
        "Si le séquenceur bay-aware a progressé, mettre à jour _REALISATION_ACTUELLE.")


@pytest.mark.skipif("SSV_OP7" not in _DISPONIBLES, reason="Fixture SSV_OP7 absente.")
@pytest.mark.parametrize("shape", ["rr3", "rr4", "iso"])
def test_phase_f_transactionnelle_ssv(shape):
    """Sur SSV.OP7 (placement propre), une manœuvre « séparation de nœuds »
    (Phase F) implique que la cible est atteinte : la Phase F n'est conservée
    que si elle réalise exactement la topologie cible (sinon annulée)."""
    poste = _poste("SSV_OP7")
    groups = _cibles(poste)[shape]
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, groups)
    res = determiner_topo_complete_cible(poste, cible)
    if any("séparation de nœuds" in m.raison for m in res.manoeuvres):
        assert res.is_verified, res.message
