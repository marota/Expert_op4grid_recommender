"""
tests/manoeuvre/test_sequences_sauvegardees_3barres.py
------------------------------------------------------
**Golden de séquences sauvegardées** sur des postes 400 kV à 3 jeux de barres :
une séquence de manœuvres enregistrée depuis l'IHM (``tests/manoeuvre/sequences/
<nom>.json`` : ``depart`` + ``cible`` + ``manoeuvres``) est rejouée et **validée**
de bout en bout :

- chaque manœuvre porte sur un **organe existant** du poste ;
- le rejeu depuis l'état de départ atteint **exactement la partition nodale
  cible** ;
- aucune manœuvre de **sectionneur sous charge** (règle de sûreté).

Robuste au nommage point/underscore (VL ``TRI.PP7`` ↔ fixture ``TRI_PP7``).
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

# Séquences sauvegardées **validées** (postes 3 JdB). Le fichier porte le nom du
# scénario ; le VL (avec point) est lu dans le JSON, la fixture utilise le stem
# (underscore).
_SEQUENCES_3B = ["TRI.PP7_cible_3_noeuds.json"]


def _graph_with(stem: str, states: dict):
    G = build_graph_from_fixture(stem)
    for sid, op in states.items():
        _set_switch(G, sid, op)
    return G


@pytest.mark.parametrize("seqfile", _SEQUENCES_3B)
def test_sequence_sauvegardee_atteint_la_cible(seqfile):
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

    # 1. Toute manœuvre porte sur un organe existant.
    assert manos, "séquence vide"
    assert all(m.switch_id in known for m in manos), \
        f"{seqfile}: organe(s) inconnu(s)"

    # 2. Le rejeu atteint exactement la partition nodale cible.
    G = _graph_with(stem, d["depart"])
    for m in manos:
        _set_switch(G, m.switch_id, m.action == "OPEN")
    obtenue = TopologieNodale.from_graph(G, vl)
    cible = TopologieNodale.from_graph(_graph_with(stem, d["cible"]), vl)
    assert cible.nb_noeuds >= 3, "cible attendue à ≥ 3 nœuds (poste 3 JdB)"
    assert cible.meme_topologie(obtenue), \
        f"{seqfile}: la séquence n'atteint pas la partition cible " \
        f"(obtenu {obtenue.nb_noeuds}, visé {cible.nb_noeuds})"

    # 3. Aucune manœuvre de sectionneur sous charge.
    viol = sectionneurs_sous_charge_par_manoeuvre(poste, manos)
    assert len(viol) == len(manos)
    assert all(v is None for v in viol), \
        f"{seqfile}: sectionneur(s) sous charge : {[v for v in viol if v]}"
