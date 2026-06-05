"""
tests/manoeuvre/test_czbevp3_sectionneur_hors_tension.py
--------------------------------------------------------
Régression : **règle du sectionneur de barre** sur un poste 1 barre /
3 sections **sans couplage** (CZBEVP3).

Le scénario ``CZBEVP3_cible_3noeuds`` scinde la barre unique (3 sections
chaînées par ``SS.1.12`` et ``SS.1.23``) en **3 nœuds vivants**. Chaque section
porte des ouvrages sous tension et il n'existe **aucune SJB tampon** pour
ré-aiguiller. Ouvrir un sectionneur laisserait alors deux côtés énergisés à des
potentiels différents — interdit.

Bug d'origine : la séquence ouvrait directement les deux sectionnements avec un
simple libellé « ATTENTION sous tension » tout en se déclarant vérifiée.

Correctif attendu (repli par dé-énergisation, coupure momentanée assumée) :
- avant chaque ouverture de sectionneur, la section à isoler est mise **hors
  tension** par ouverture des DJ de ses ouvrages, puis ré-énergisée après ;
- plus aucune ouverture de sectionneur « sous tension » ;
- topologie nodale (3 nœuds) atteinte et vérifiée, sans écart.
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

SCEN = Path(__file__).parent / "scenarios" / "CZBEVP3_cible_3noeuds.json"

SECTIONNEMENTS = {
    "CZBEVP3_CZBEV3CBO.1 SS.1.12_OC",
    "CZBEVP3_CZBEV3CBO.1 SS.1.23_OC",
}

pytestmark = pytest.mark.skipif(
    not SCEN.exists() or "CZBEVP3" not in list_available_fixtures(),
    reason="Scénario/fixture CZBEVP3 absent",
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


def test_trois_noeuds_atteints_et_verifies(resultat):
    assert resultat.is_verified, resultat.message
    assert resultat.topo_obtenue.nb_noeuds == 3
    assert resultat.is_verified_detaillee, resultat.ecarts
    assert resultat.ecarts == []


def test_aucun_sectionneur_ouvert_sous_tension(resultat):
    """Aucune ouverture de sectionneur ne doit être étiquetée « sous tension »."""
    for m in resultat.manoeuvres:
        if m.switch_id in SECTIONNEMENTS and m.action == "OPEN":
            assert "sous tension" not in m.raison.lower(), m.raison
            assert "hors tension" in m.raison.lower(), m.raison


def test_section_mise_hors_tension_avant_chaque_sectionneur(resultat):
    """Chaque ouverture de sectionneur est précédée d'au moins une mise hors
    tension (OPEN DJ d'ouvrage) et suivie d'une remise sous tension (CLOSE)."""
    ms = resultat.manoeuvres
    for i, m in enumerate(ms):
        if m.switch_id in SECTIONNEMENTS and m.action == "OPEN":
            avant = [x for x in ms[:i]
                     if x.action == "OPEN" and "hors tension" in x.raison.lower()
                     and x.switch_id not in SECTIONNEMENTS]
            apres = [x for x in ms[i + 1:]
                     if x.action == "CLOSE" and "sous tension" in x.raison.lower()]
            assert avant, f"Aucune dé-énergisation avant {m.switch_id}"
            assert apres, f"Aucune ré-énergisation après {m.switch_id}"


def test_etat_final_des_dj_inchange(resultat):
    """Les DJ d'ouvrage manœuvrés pour la dé-énergisation sont **refermés** :
    leur état final est identique au départ (coupure seulement momentanée)."""
    d = json.loads(SCEN.read_text())
    etat = dict(d["depart"])
    for m in resultat.manoeuvres:
        etat[m.switch_id] = (m.action == "OPEN")
    # Tous les DJ d'ouvrage doivent rester fermés (comme au départ / à la cible).
    for sid, ouvert in d["cible"].items():
        if "DJ" in sid:
            assert etat.get(sid) == ouvert, f"{sid} état final incohérent"


SEQ_PATH = Path(__file__).parent / "sequences" / "CZBEVP3_cible_3noeuds.json"


@pytest.mark.skipif(not SEQ_PATH.exists(),
                    reason="Séquence de référence CZBEVP3 absente")
def test_sequence_correspond_a_la_reference(resultat):
    """La séquence calculée correspond à la séquence de référence sauvegardée
    (mêmes organes, mêmes actions, même ordre)."""
    ref = json.loads(SEQ_PATH.read_text())["manoeuvres"]
    assert len(resultat.manoeuvres) == len(ref), (
        f"{len(resultat.manoeuvres)} manœuvres calculées vs {len(ref)} référence")
    for i, (calc, r) in enumerate(zip(resultat.manoeuvres, ref)):
        assert calc.switch_id == r["switch_id"], (
            f"Manœuvre {i + 1}: {calc.switch_id} != {r['switch_id']}")
        assert calc.action == r["action"], (
            f"Manœuvre {i + 1}: {calc.action} != {r['action']}")
