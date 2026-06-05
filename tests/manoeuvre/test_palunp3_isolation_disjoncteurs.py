"""
tests/manoeuvre/test_palunp3_isolation_disjoncteurs.py
------------------------------------------------------
Régression R10bis — **isoler une section par les disjoncteurs d'abord**.

Sur PALUNP3 → 4 nœuds, la section 1.2 est reliée à 2.2 par le **couplage
`COUPL.2` (DJ)**. L'algorithme doit ouvrir `COUPL.2` (organe qui coupe la
charge) **avant** le sectionnement `SS.1.12`, ce qui réduit la section à
dé-énergiser à son seul résidu — au lieu de parker/ré-aiguiller inutilement les
5 départs de la section 2.2 (qui restent sur leur barre).

Avant ce correctif la séquence comptait **40 manœuvres** ; un expert en fait ~9.
On vérifie ici que la cible détaillée est atteinte, qu'aucun sectionneur n'est
ouvert sous tension, que `COUPL.2` est ouvert **avant** `SS.1.12`, et que la
séquence est nettement plus courte (< 20 manœuvres), sans ré-aiguiller les
départs de 2.2 qui n'en changent pas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_manoeuvres_cible_detaillee,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

SCEN = Path(__file__).parent / "scenarios" / "PALUNP3_cible_4noeuds.json"
COUPL2 = "PALUNP3_PALUN3COUPL.2 DJ_OC"
SS_12 = "PALUNP3_PALUN3SEC..12 SS.1.12_OC"

pytestmark = pytest.mark.skipif(
    not SCEN.exists() or "PALUNP3" not in list_available_fixtures(),
    reason="Scénario/fixture PALUNP3 absent",
)


def _graph_from_states(vl, states):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


@pytest.fixture
def resultat():
    d = json.loads(SCEN.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible_graph = _graph_from_states(vl, d["cible"])
    return determiner_manoeuvres_cible_detaillee(poste, cible_graph)


def test_cible_detaillee_atteinte(resultat):
    assert resultat.is_verified, resultat.message
    assert resultat.is_verified_detaillee, resultat.ecarts
    assert resultat.ecarts == []


def test_sequence_nettement_plus_courte(resultat):
    """L'isolement par disjoncteurs évite le parking massif (40 → ~12)."""
    assert resultat.nb_manoeuvres < 20, (
        f"{resultat.nb_manoeuvres} manœuvres (attendu < 20)")


def test_couplage_ouvert_avant_sectionnement(resultat):
    """`COUPL.2` (DJ) est ouvert AVANT `SS.1.12` pour réduire la section à
    isoler (règle « isoler par les disjoncteurs d'abord »)."""
    ids = [m.switch_id for m in resultat.manoeuvres]
    assert COUPL2 in ids and SS_12 in ids
    assert ids.index(COUPL2) < ids.index(SS_12), (
        "Le couplage incident doit être ouvert avant le sectionnement")


def test_aucun_sectionneur_sous_tension(resultat):
    for m in resultat.manoeuvres:
        if m.switch_id == SS_12 and m.action == "OPEN":
            assert "sous tension" not in m.raison.lower(), m.raison


def test_departs_stables_de_2_2_non_reaiguilles(resultat):
    """Les départs qui restent sur la section 2.2 ne sont pas manœuvrés du tout
    (plus de parking/retour inutile de AIX/GARDA.2/Y631)."""
    touched = {m.switch_id for m in resultat.manoeuvres}
    sa_prefixes = {
        "AIX  L31PALUN": "PALUNP3_PALUN3AIX.1",
        "GARDAL32PALUN": "PALUNP3_PALUN3GARDA.2",
        "PALUNY631": "PALUNP3_PALUN3TR631",
    }
    for eq, pfx in sa_prefixes.items():
        bouges = [s for s in touched if s.startswith(pfx)]
        assert not bouges, f"'{eq}' ne devrait pas être manœuvré : {bouges}"
