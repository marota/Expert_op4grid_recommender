"""
tests/manoeuvre/test_troncons.py
----------------------------------
Tests du tronçonnement (étapes 1.3-1.4) sur les fixtures de postes réels.
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.cellules import detecter_cellules
from expert_op4grid_recommender.manoeuvre.troncons import (
    Tronconnement,
    construire_tronconnement,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


def _fixtures_available() -> bool:
    return len(list_available_fixtures()) > 0


pytestmark = pytest.mark.skipif(
    not _fixtures_available(), reason="Fixtures de postes non générées."
)


def _tronconnement(vl: str) -> Tronconnement:
    G = build_graph_from_fixture(vl)
    cells = detecter_cellules(G, vl)
    return construire_tronconnement(cells, G)


# ---------------------------------------------------------------------------
# CARRIP3 — poste de référence double barre
# ---------------------------------------------------------------------------

def test_carrip3_un_troncon_deux_barres():
    """CARRIP3 (couplage fermé) : 1 tronçon couvrant 2 barres."""
    tr = _tronconnement("CARRIP3")
    assert tr.nb_jeux_barres == 2
    assert len(tr.troncons) == 1
    troncon = next(iter(tr.troncons.values()))
    assert troncon.nb_jeux_barres == 2
    assert len(troncon.busbar_nodes) == 4


def test_carrip3_couplage_dj_identifie():
    """Le DJ de couplage de CARRIP3 est rattaché au tronçon."""
    tr = _tronconnement("CARRIP3")
    troncon = next(iter(tr.troncons.values()))
    dj_ids = [s.switch_id for s in troncon.couplage_breakers]
    assert any("COUPL" in s for s in dj_ids), dj_ids


def test_carrip3_departs_reaiguillables():
    """Sur CARRIP3, les départs atteignent les 2 barres (ré-aiguillables)."""
    tr = _tronconnement("CARRIP3")
    troncon = next(iter(tr.troncons.values()))
    # Aucun départ fixe : double barre symétrique
    assert troncon.departs_couplage, "Des départs doivent être ré-aiguillables"
    for barres in troncon.departs_couplage.values():
        assert barres == {0, 1}


def test_carrip3_chaque_depart_un_troncon():
    """Cohérence : chaque départ appartient à exactement un tronçon."""
    G = build_graph_from_fixture("CARRIP3")
    cells = detecter_cellules(G, "CARRIP3")
    tr = construire_tronconnement(cells, G)
    all_departs = set()
    for c in cells.cellules_depart:
        all_departs.add(c.equipment_id)
        all_departs |= set(c.shared_equipment_ids)
    for eq in all_departs:
        assert eq in tr.troncon_par_depart


# ---------------------------------------------------------------------------
# Autres postes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vl,nb_barres", [
    ("CARRIP3", 2),
    ("NOVIOP3", 2),
    ("CZBEVP3", 1),
    ("SSAVOP3", 2),
])
def test_nb_barres_par_poste(vl, nb_barres):
    if vl not in list_available_fixtures():
        pytest.skip(f"Fixture {vl} absente")
    tr = _tronconnement(vl)
    assert tr.nb_jeux_barres == nb_barres


def test_czbevp3_une_seule_barre():
    """CZBEVP3 : poste 1 barre (3 sections), 1 tronçon."""
    if "CZBEVP3" not in list_available_fixtures():
        pytest.skip("Fixture CZBEVP3 absente")
    tr = _tronconnement("CZBEVP3")
    assert tr.nb_jeux_barres == 1
    assert len(tr.troncons) == 1


def test_tronconnement_couvre_toutes_les_sjb():
    """Toutes les SJB du poste sont attribuées à un tronçon."""
    for vl in ["CARRIP3", "NOVIOP3", "CZBEVP3"]:
        if vl not in list_available_fixtures():
            continue
        G = build_graph_from_fixture(vl)
        cells = detecter_cellules(G, vl)
        tr = construire_tronconnement(cells, G)
        sjb_dans_troncons = set()
        for t in tr.troncons.values():
            sjb_dans_troncons |= t.busbar_nodes
        assert sjb_dans_troncons == set(tr.barre_par_busbar)
