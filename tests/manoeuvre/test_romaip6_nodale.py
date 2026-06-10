"""
tests/manoeuvre/test_romaip6_nodale.py
--------------------------------------
**Caractérisation de la topologie nodale** sur ``ROMAIP6_cible_3noeuds`` — réponse
à la question « les barres 1_A et 2_A couplées forment-elles bien le même nœud ? ».

Conclusion (vérifiée ici) : ``TopologieNodale.from_graph`` = composantes connexes
du sous-graphe des switches **fermés** — correcte par construction. Sur cette cible :

- les barres **1.1 et 2.1**, **couplées** (``COUPL.1`` DJ+SA fermés), forment bien
  **un seul** nœud électrique (leurs départs ``FLANDL61ROMAI`` (1.1) et
  ``ROMAI6T611`` (2.1) sont dans le même nœud) — idem **1.4/2.4** via ``COUPL.2`` ;
- le total de **10** nœuds (et non 3) vient des **ouvrages dont le disjoncteur est
  ouvert** : générateurs ``ROMAIINF``/``ROMAIING``, shunt ``COND.1``, lignes
  ``AVENI``/``PLAIS``/``VLEVA`` — **électriquement isolés**, donc nœuds séparés à
  juste titre. Ce ne sont **pas** des nœuds de barre dupliqués : un ouvrage à DJ
  ouvert n'est pas « sur la barre » même si son sectionneur d'aiguillage y pointe.

Le test **verrouille** ce comportement (un couplage fermé fusionne les barres ; un
DJ ouvert isole l'ouvrage). Sans dépendance pypowsybl (fixture + scénario).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre import TopologieNodale
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _set_switch

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

_SCEN = pathlib.Path(__file__).parent / "scenarios" / "ROMAIP6_cible_3noeuds.json"
_STEM = "ROMAIP6"

# Ouvrages dont le disjoncteur est ouvert dans la cible → isolés (nœuds séparés).
_DECONNECTES = ["AVENIL61ROMAI", "PLAISL61ROMAI", "ROMAI6COND.1",
                "ROMAIINF", "ROMAIING", "ROMAIL61VLEVA"]


def _topo():
    if not _SCEN.exists() or _STEM not in list_available_fixtures():
        pytest.skip("Scénario/fixture ROMAIP6 absent")
    d = json.loads(_SCEN.read_text())
    G = build_graph_from_fixture(_STEM)
    for sid, op in d["cible"].items():
        _set_switch(G, sid, op)
    return TopologieNodale.from_graph(G, d["voltage_level_id"])


def test_barres_couplees_forment_un_seul_noeud():
    """1.1 et 2.1 (couplées par COUPL.1 fermé) : leurs départs sont dans le **même**
    nœud électrique — la topologie nodale ne les sépare pas."""
    topo = _topo()
    n_fland = topo.noeud_par_depart.get("FLANDL61ROMAI")   # sur barre 1.1
    n_t611 = topo.noeud_par_depart.get("ROMAI6T611")        # sur barre 2.1
    assert n_fland is not None and n_fland == n_t611, (
        f"1.1/2.1 couplées devraient former un seul nœud "
        f"(FLAND->{n_fland}, T611->{n_t611})")


def test_ouvrages_a_dj_ouvert_sont_des_noeuds_isoles():
    """Chaque ouvrage à **DJ ouvert** forme son propre nœud (déconnecté) : ce sont
    eux qui expliquent les 10 nœuds (et non un défaut de fusion des barres)."""
    topo = _topo()
    for eq in _DECONNECTES:
        nom = topo.noeud_par_depart.get(eq)
        if nom is None:
            continue  # absent de cette cible
        assert len(topo.noeuds[nom].equipment_ids) == 1, (
            f"{eq} (DJ ouvert) devrait être seul dans son nœud {nom}")


def test_nombre_de_noeuds_coherent():
    """4 groupes de barre (1.1+2.1, 1.2+1.3, 2.2+2.3, 1.4+2.4) + 6 ouvrages isolés
    (DJ ouvert) = 10 nœuds. Verrou de non-régression de la topologie nodale."""
    topo = _topo()
    assert topo.nb_noeuds == 10
    # Aucun des 6 ouvrages déconnectés ne partage un nœud avec un autre départ.
    isoles = {topo.noeud_par_depart[eq] for eq in _DECONNECTES
              if eq in topo.noeud_par_depart}
    assert len(isoles) == 6
