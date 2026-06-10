"""
tests/manoeuvre/test_dataset_structures_xiidm.py
-------------------------------------------------
Structures de postes extraites **du dataset lui-même** (instantané XIIDM) et
garde de **couverture d'identifiants** — le verrou découvert sur le dataset
réel RTE 7000 : les ids d'organes des fixtures du dépôt (export normalisé,
suffixe ``_OC``) ne recouvrent pas ceux des instantanés D-GITT (champs RTE à
largeur fixe). Le tagging structurel n'a de sens qu'avec une structure aux
ids des données.

Couverture :

- ``couverture_structure`` : 1.0 sur les ids de la structure, 0.0 sur des ids
  étrangers, mélange → ratio (Python pur, fixtures du dépôt) ;
- ``postes_depuis_xiidm`` : structures construites depuis un instantané réel
  (réseau pypowsybl de référence), filtre ``vls``, échecs consignés ;
- bout-en-bout : blocs détectés sur une série d'instantanés, structures
  extraites du premier instantané, **tagging structurel** cohérent
  (l'ouverture du DJ d'un départ → ``consignation_ouvrage``).
"""
from __future__ import annotations

import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre.dataset import (
    couverture_structure,
    poste_from_fixture_json,
    postes_depuis_xiidm,
    taguer_bloc,
)
from expert_op4grid_recommender.manoeuvre.dataset.dgitt import (
    charger_timelines_xiidm,
)

from .test_dataset_dgitt_xiidm import _SWITCH_BASCULE, _ecrire_serie

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# couverture_structure (Python pur)
# ---------------------------------------------------------------------------

def _poste_carrip3():
    return poste_from_fixture_json(FIXTURES / "CARRIP3.json", "CARRIP3")


def test_couverture_pleine_sur_ids_de_la_structure():
    poste = _poste_carrip3()
    from expert_op4grid_recommender.manoeuvre.algo.graph_ops import (
        _switch_edge_index,
    )
    ids = list(_switch_edge_index(poste.graph))
    assert couverture_structure(poste, {sid: False for sid in ids}) == 1.0


def test_couverture_nulle_sur_ids_etrangers():
    poste = _poste_carrip3()
    etats = {"CARRIP3_CARRI   3SEC.A34  SS.1A43": False,
             "CARRIP3_CARRI   3U.MON.2  SA.1": True}
    assert couverture_structure(poste, etats) == 0.0


def test_couverture_partielle_et_etats_vides():
    poste = _poste_carrip3()
    from expert_op4grid_recommender.manoeuvre.algo.graph_ops import (
        _switch_edge_index,
    )
    connu = next(iter(_switch_edge_index(poste.graph)))
    etats = {connu: False, "ID_INCONNU_1": True, "ID_INCONNU_2": False,
             "ID_INCONNU_3": True}
    assert couverture_structure(poste, etats) == pytest.approx(0.25)
    assert couverture_structure(poste, {}) == 0.0


# ---------------------------------------------------------------------------
# postes_depuis_xiidm (pypowsybl)
# ---------------------------------------------------------------------------

def test_postes_depuis_xiidm_construit_les_vl(tmp_path):
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)
    snapshot = sorted(racine.rglob("*.xiidm.bz2"))[0]

    postes, echecs = postes_depuis_xiidm(snapshot)
    # Tous les VL NODE_BREAKER du réseau de référence sont construits.
    assert set(postes) >= {"S1VL1", "S1VL2"}
    assert echecs == {}
    # La structure porte bien les ids des données.
    assert couverture_structure(
        postes["S1VL1"], {_SWITCH_BASCULE: False}) == 1.0


def test_postes_depuis_xiidm_filtre_vls_et_echecs(tmp_path):
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)
    snapshot = sorted(racine.rglob("*.xiidm.bz2"))[0]

    postes, echecs = postes_depuis_xiidm(snapshot, vls={"S1VL1", "VL_ABSENT"})
    assert set(postes) == {"S1VL1"}
    assert set(echecs) == {"VL_ABSENT"}


# ---------------------------------------------------------------------------
# Bout-en-bout : tagging structurel avec structure extraite du dataset
# ---------------------------------------------------------------------------

def test_tagging_structurel_depuis_snapshot(tmp_path):
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)
    snapshot = sorted(racine.rglob("*.xiidm.bz2"))[0]

    tls = {tl.voltage_level_id: tl
           for tl in charger_timelines_xiidm(racine, vl_filter={"S1VL1"})}
    blocs, _ = tls["S1VL1"].detecter_blocs(min_stabilite=2)
    assert len(blocs) == 1

    postes, _ = postes_depuis_xiidm(snapshot, vls={"S1VL1"})
    tags = taguer_bloc(blocs[0], postes["S1VL1"])
    # L'ouverture du DJ du départ LD1 isole la charge → consignation.
    assert "consignation_ouvrage" in tags
    assert "inclasse" not in tags
