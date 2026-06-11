"""
tests/manoeuvre/test_sequences_4noeuds_integration.py
-----------------------------------------------------
**Intégration des séquences sauvegardées « cible 4 nœuds »** (postes à 2 jeux de
barres) qui n'étaient jusqu'ici rattachées à **aucun test**.

Chaque fichier ``tests/manoeuvre/sequences/<nom>.json`` enregistre, depuis l'IHM
ou l'algorithme, une séquence de manœuvres complète (``depart`` + ``cible`` +
``manoeuvres``). On la rejoue et on la **valide de bout en bout**, sur le même
modèle que ``test_sequences_sauvegardees_3barres.py`` :

- chaque manœuvre porte sur un **organe existant** du poste ;
- le rejeu depuis l'état de départ atteint **exactement la partition nodale
  cible** ;
- aucune manœuvre de **sectionneur sous charge** (règle de sûreté) — sauf cas
  expert explicitement caractérisé ci-dessous.

Deux familles sont couvertes :

``_SEQUENCES_PROPRES``
    Séquences saines (expert *ou* algorithme) : les trois invariants ci-dessus
    tiennent. Pour MORBRP6, la séquence ``_expert`` démontre que la cible
    *non atteignable par l'algorithme* (cf. ``test_morbrp6_degradation_gracieuse``
    dans ``test_scenarios_sauvegardes.py``) **reste réalisable manuellement**.

``SSAVOP3_cible_4noeuds_expert``
    Séquence experte qui atteint bien la cible mais contient **une** ouverture
    de **sectionneur de barre sous charge** (section ``SEC.A23``). On fige ce
    comportement par un test de caractérisation dédié, pour éviter qu'une
    régression du vérificateur ne passe inaperçue.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    sectionneurs_sous_charge_par_manoeuvre,
)
from expert_op4grid_recommender.manoeuvre.algo.results import Manoeuvre
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _set_switch

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

_SEQ_DIR = pathlib.Path(__file__).parent / "sequences"

# Séquences « cible 4 nœuds » saines (rejeu propre, sans sectionneur sous charge).
_SEQUENCES_PROPRES = [
    "MORBRP6_cible_4noeuds_expert.json",
    "PALUNP3_cible_4noeuds_expert.json",
    "PALUNP3_cible_4noeuds_algo.json",
    "SSAVOP3_cible_4noeuds_algo.json",
]

# Séquence experte atteignant la cible mais avec un sectionnement de barre sous
# charge (vérité-terrain : index 0-based de la manœuvre fautive + organe).
_SSAVOP3_EXPERT = "SSAVOP3_cible_4noeuds_expert.json"
_SSAVOP3_EXPERT_VIOL_IDX = 3
_SSAVOP3_EXPERT_VIOL_SID = "SSAVOP3_SSAVO3SEC.A23 SS.1A23_OC"


def _graph_with(stem: str, states: dict):
    G = build_graph_from_fixture(stem)
    for sid, op in states.items():
        _set_switch(G, sid, op)
    return G


def _load_sequence(seqfile: str):
    """(``poste``, ``manoeuvres``, ``known_switch_ids``, ``d``) ou ``pytest.skip``."""
    path = _SEQ_DIR / seqfile
    if not path.exists():
        pytest.skip(f"Séquence absente : {seqfile}")
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    stem = vl.replace(".", "_")
    if stem not in list_available_fixtures():
        pytest.skip(f"Fixture {stem} absente")
    poste = PosteTopologique.from_graph(_graph_with(stem, d["depart"]), vl)
    known = {dd.get("switch_id")
             for _u, _v, dd in poste.graph.edges(data=True) if dd.get("switch_id")}
    manos = [Manoeuvre(m["switch_id"], m["action"], m.get("raison", ""))
             for m in d["manoeuvres"]]
    return poste, manos, known, d, stem, vl


@pytest.mark.parametrize("seqfile", _SEQUENCES_PROPRES)
def test_sequence_4noeuds_propre(seqfile):
    """Séquence saine : organes existants, cible nodale atteinte exactement,
    aucun sectionneur manœuvré sous charge."""
    poste, manos, known, d, stem, vl = _load_sequence(seqfile)

    # 1. Toute manœuvre porte sur un organe existant.
    assert manos, f"{seqfile}: séquence vide"
    inconnus = [m.switch_id for m in manos if m.switch_id not in known]
    assert not inconnus, f"{seqfile}: organe(s) inconnu(s) : {inconnus[:5]}"

    # 2. Le rejeu atteint exactement la partition nodale cible.
    G = _graph_with(stem, d["depart"])
    for m in manos:
        _set_switch(G, m.switch_id, m.action == "OPEN")
    obtenue = TopologieNodale.from_graph(G, vl)
    cible = TopologieNodale.from_graph(_graph_with(stem, d["cible"]), vl)
    assert cible.nb_noeuds >= 4, \
        f"{seqfile}: cible « 4 nœuds » attendue à ≥ 4 nœuds (lu {cible.nb_noeuds})"
    assert cible.meme_topologie(obtenue), \
        f"{seqfile}: la séquence n'atteint pas la partition cible " \
        f"(obtenu {obtenue.nb_noeuds}, visé {cible.nb_noeuds})"

    # 3. Aucune manœuvre de sectionneur sous charge.
    viol = sectionneurs_sous_charge_par_manoeuvre(poste, manos)
    assert len(viol) == len(manos), \
        f"{seqfile}: « par manœuvre » désaligné sur la séquence"
    fautifs = [(i, v) for i, v in enumerate(viol) if v]
    assert not fautifs, \
        f"{seqfile}: sectionneur(s) sous charge : {fautifs[:3]}"


def test_ssavop3_expert_atteint_cible_avec_sectionnement_sous_charge():
    """``SSAVOP3_cible_4noeuds_expert`` : la séquence **atteint** la cible nodale,
    mais contient **exactement une** ouverture de sectionneur de barre **sous
    charge** (section ``SEC.A23``), en position connue. Test de caractérisation :
    fige à la fois l'atteinte de la cible et l'unique infraction détectée."""
    poste, manos, known, d, stem, vl = _load_sequence(_SSAVOP3_EXPERT)

    # Organes existants + cible atteinte.
    inconnus = [m.switch_id for m in manos if m.switch_id not in known]
    assert not inconnus, f"organe(s) inconnu(s) : {inconnus[:5]}"
    G = _graph_with(stem, d["depart"])
    for m in manos:
        _set_switch(G, m.switch_id, m.action == "OPEN")
    obtenue = TopologieNodale.from_graph(G, vl)
    cible = TopologieNodale.from_graph(_graph_with(stem, d["cible"]), vl)
    assert cible.meme_topologie(obtenue), \
        "la séquence experte SSAVOP3 doit atteindre la partition cible"

    # Une seule infraction, à la position et sur l'organe attendus.
    viol = sectionneurs_sous_charge_par_manoeuvre(poste, manos)
    assert len(viol) == len(manos)
    fautifs = [i for i, v in enumerate(viol) if v]
    assert fautifs == [_SSAVOP3_EXPERT_VIOL_IDX], \
        f"position(s) d'infraction inattendue(s) : {fautifs}"
    assert manos[_SSAVOP3_EXPERT_VIOL_IDX].switch_id == _SSAVOP3_EXPERT_VIOL_SID
    # Parade attendue : section de barre (sectionnement), pas disjoncteur de branche.
    assert "section de barre" in viol[_SSAVOP3_EXPERT_VIOL_IDX]
