"""
tests/manoeuvre/test_scenarios_sauvegardes.py
-----------------------------------------------
Tests sur les **scénarios sauvegardés** (topologies détaillées départ/cible
créées via l'IHM) : le calcul de séquence détaillée doit atteindre la
topologie détaillée visée (barre exacte de chaque départ).

Les scénarios sont des fixtures JSON ``tests/manoeuvre/scenarios/<nom>.json``
(``depart`` / ``cible`` = états des organes, ``*_nodale`` = partitions).
Ils sont rejoués sur les fixtures de poste (sans pypowsybl).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    Manoeuvre,
    determiner_manoeuvres_cible_detaillee,
    _verifier_sectionneurs_hors_charge,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

SCEN_DIR = Path(__file__).parent / "scenarios"

# Scénarios connus comme **non réalisables en totalité** par l'algorithme : ils
# sont exclus du test de réussite complète et couverts par un test de dégradation
# gracieuse dédié. Ici : une cible MORBRP6 qui exige des manœuvres sur la self des
# jeux de barres 3/4 (niveaux supplémentaires hors de portée de l'algorithme).
DEGRADATION_SCENARIOS: set[str] = {"MORBRP6_cible_4noeuds_non_atteignable"}


def _scenarios():
    if not SCEN_DIR.exists():
        return []
    return sorted(p for p in SCEN_DIR.glob("*.json")
                  if p.stem not in DEGRADATION_SCENARIOS)


def _graph_from_states(vl, states):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


pytestmark = pytest.mark.skipif(
    not _scenarios(), reason="Aucun scénario sauvegardé.")


@pytest.mark.parametrize("path", _scenarios(), ids=lambda p: p.stem)
def test_scenario_atteint_topologie_detaillee(path):
    """Chaque scénario sauvegardé doit mener à sa topologie détaillée cible."""
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    if vl not in list_available_fixtures():
        pytest.skip(f"Fixture {vl} absente")
    # les ids d'organes doivent exister dans la fixture
    known = {dd.get("switch_id")
             for _u, _v, dd in build_graph_from_fixture(vl).edges(data=True)}
    if not set(d["depart"]) <= known:
        pytest.skip(f"Organes du scénario absents de la fixture {vl}")

    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible_graph = _graph_from_states(vl, d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)

    assert res.is_verified, f"{path.stem} : nodale non atteinte — {res.message}"
    assert res.is_verified_detaillee, \
        f"{path.stem} : détaillée non atteinte — écarts {res.ecarts}"
    assert res.ecarts == []


def test_carrip3_1noeud_requinconcage():
    """Scénario CARRIP3 → 1 nœud : la section 1.2 est dé-énergisée pour fermer
    le sectionnement, PUIS ses départs sont **requinçonçés** (ramenés) sur 1.2
    en boucle courte (manœuvres supplémentaires) pour atteindre exactement la
    topologie détaillée cible."""
    path = SCEN_DIR / "CARRIP3_cible_1noeud.json"
    if not path.exists() or "CARRIP3" not in list_available_fixtures():
        pytest.skip("Scénario/fixture CARRIP3 absent")
    d = json.loads(path.read_text())
    poste = PosteTopologique.from_graph(
        _graph_from_states("CARRIP3", d["depart"]), "CARRIP3")
    cible_graph = _graph_from_states("CARRIP3", d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)

    assert res.is_verified_detaillee, res.ecarts

    # On ferme le sectionnement après dé-énergisation...
    idx_sect = next(i for i, m in enumerate(res.manoeuvres)
                    if m.action == "CLOSE" and "sectionnement" in m.raison.lower())
    # ... puis on requinçonçe des départs (ré-aiguillage boucle courte APRÈS)
    # pour les ramener sur leur barre cible (1.2).
    retours = [m for m in res.manoeuvres[idx_sect + 1:]
               if m.type_boucle == "COURTE" and m.action == "CLOSE"
               and "1.2" in m.raison]
    assert retours, "Aucun requinçonçage (retour boucle courte) vers 1.2"
    # La vérification détaillée (ecarts == []) garantit déjà que chaque départ
    # finit sur sa barre exacte imposée par la cible.


def test_morbrp6_multibarres():
    """MORBRP6 : poste à **4 jeux de barres** (2 jeux principaux 1A/2A/1B/2B +
    niveaux 3B/4B, avec une réactance interne ``MORBRL61REAC.`` à 2 bornes et un
    nœud isolé ``ARRIGL61MORBR``).

    La cible (6 nœuds) est atteinte via le chemin **multi-barres** :
    - placement dérivé des composantes du graphe cible (groupes exacts) ;
    - **organe interne à 2 bornes** (réactance) laissé en place ;
    - **nœud à 0 barre** isolé (ouverture de SA) ;
    - **couplages parallèles** correctement ouverts (ex. COUPL.B masqué par la
      liaison SELF.1 parallèle).
    """
    path = SCEN_DIR / "MORBRP6_cible_4noeuds.json"
    if not path.exists() or "MORBRP6" not in list_available_fixtures():
        pytest.skip("Scénario/fixture MORBRP6 absent")
    d = json.loads(path.read_text())
    poste = PosteTopologique.from_graph(
        _graph_from_states("MORBRP6", d["depart"]), "MORBRP6")
    cible_graph = _graph_from_states("MORBRP6", d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)

    # Cible nodale ET détaillée atteinte (6 nœuds), sans écart résiduel.
    assert res.is_verified, res.message
    assert res.is_verified_detaillee, res.ecarts
    assert res.ecarts == []
    assert res.topo_obtenue.nb_noeuds == 6

    touched = {m.switch_id for m in res.manoeuvres}
    # Le nœud à 0 barre ARRIG.1 est isolé (ouverture d'un de ses SA de barre).
    assert any("ARRIG.1 SA" in s for s in touched), \
        "ARRIG.1 (nœud sans barre) n'a pas été isolé"
    # Le couplage parallèle COUPL.B (masqué par SELF.1) est bien ouvert.
    assert any("COUPL.B" in s for s in touched), \
        "COUPL.B (couplage parallèle) n'a pas été ouvert"
    # L'organe interne à 2 bornes (réactance) n'est PAS manœuvré (laissé en place).
    assert not any("REAC" in s for s in touched), \
        "La réactance interne à 2 bornes ne doit pas être manœuvrée"

    # Règle du sectionneur : le SA d'ARRIG.1 est ouvert HORS CHARGE — le DJ est
    # ouvert AVANT puis refermé APRÈS (ordre DJ open → SA open → DJ close).
    seq = [(m.switch_id, m.action) for m in res.manoeuvres]
    i_sa = next(i for i, (s, a) in enumerate(seq)
                if "ARRIG.1 SA.1" in s and a == "OPEN")
    assert ("MORBRP6_MORBR6ARRIG.1 DJ_OC", "OPEN") in seq[:i_sa], \
        "le DJ d'ARRIG.1 doit être ouvert avant l'ouverture de son sectionneur"
    assert ("MORBRP6_MORBR6ARRIG.1 DJ_OC", "CLOSE") in seq[i_sa + 1:], \
        "le DJ d'ARRIG.1 doit être refermé après l'ouverture de son sectionneur"


def test_verifier_sectionneur_sous_charge_detecte():
    """Le vérificateur générique signale un **sectionneur manœuvré sous charge**
    (séquence experte fautive : on ouvre directement ``ARRIG.1 SA.1`` sans
    dé-énergiser la branche par son disjoncteur d'abord)."""
    path = (Path(__file__).parent / "sequences"
            / "MORBRP6_cible_4noeuds_wrong_last_step.json")
    if not path.exists() or "MORBRP6" not in list_available_fixtures():
        pytest.skip("Séquence/fixture MORBRP6 absente")
    d = json.loads(path.read_text())
    poste = PosteTopologique.from_graph(
        _graph_from_states("MORBRP6", d["depart"]), "MORBRP6")
    manoeuvres = [Manoeuvre(m["switch_id"], m["action"], m.get("raison", ""))
                  for m in d["manoeuvres"]]
    ecarts = _verifier_sectionneurs_hors_charge(poste, manoeuvres)
    assert any("ARRIG.1 SA.1" in e for e in ecarts), \
        f"sectionneur sous charge non détecté ; écarts={ecarts}"


def test_morbrp6_degradation_gracieuse():
    """MORBRP6 : cible **non atteignable** par l'algorithme — elle exige des
    manœuvres sur la self/réactance des jeux de barres supplémentaires 3B/4B
    (séparer la réactance et V.GEO en nœuds distincts), hors de portée du modèle.

    Au lieu d'abandonner, l'algorithme doit **dégrader gracieusement** : réaliser
    ce qu'il peut sur les 2 jeux de barres principaux, puis **signaler les nœuds à
    compléter manuellement** (ceux qui dépendent des niveaux supplémentaires).
    """
    path = SCEN_DIR / "MORBRP6_cible_4noeuds_non_atteignable.json"
    if not path.exists() or "MORBRP6" not in list_available_fixtures():
        pytest.skip("Scénario/fixture MORBRP6 absent")
    d = json.loads(path.read_text())
    poste = PosteTopologique.from_graph(
        _graph_from_states("MORBRP6", d["depart"]), "MORBRP6")
    cible_graph = _graph_from_states("MORBRP6", d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)

    # Cible non atteinte en totalité...
    assert not res.is_verified
    # ... mais dégradation gracieuse : manœuvres partielles + nœuds explicitement
    # laissés à l'opérateur.
    assert res.manoeuvres, "Aucune manœuvre partielle générée"
    assert res.noeuds_non_realisables, "Aucun nœud signalé comme non réalisable"

    # Diagnostic explicite + complétion manuelle annoncée.
    assert "compléter manuellement" in res.message
    assert any("compléter manuellement" in e for e in res.ecarts)

    # La réactance interne, qu'il faudrait scinder sur 3B/4B, fait partie des
    # nœuds non réalisés laissés à l'opérateur.
    non_places_eqs = {eq for grp in res.noeuds_non_realisables for eq in grp}
    assert "MORBRL61REAC." in non_places_eqs
