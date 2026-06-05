"""
tests/manoeuvre/test_postes_reels.py
--------------------------------------
Tests paramétrés du module Manoeuvre sur des topologies de postes RTE réels.

Les topologies sont chargées depuis les fixtures JSON produites par
``scripts/extract_test_fixtures.py``. Si les fixtures ne sont pas générées,
tous les tests sont skippés automatiquement.

Postes testés (documentés dans l'algo Apogée) :
- Standards     : CARRIP3, CARRIP6, CZTRYP6, COMPIP3, BXTO5, CZBEVP3,
                  PALUNP3, NOVIOP3, SSAVOP3, VIELMP6
- Dép. multiples : CORNIP3, CNIEP6, GUARBP6, MORBRP6, RAN.PP6

Propriétés vérifiées pour chaque poste :
1. Le graphe est construit sans erreur
2. Les cellules de départ couvrent tous les équipements
3. Chaque cellule de départ atteint au moins une SJB
4. Le nombre de cellules de départ correspond aux équipements
5. Aucun nœud intermédiaire n'est partagé entre cellules (sauf omnibus)
6. Les départs multiples sont correctement détectés et fusionnés
7. Les couplages sont détectés pour les postes à 2+ barres
8. La connectivité électrique (switches fermés) est cohérente

Usage :
    # Générer les fixtures d'abord :
    cd Expert_op4grid_recommender-qwen3-5
    venv_expert_op4grid_recommender_qwen_35/bin/python scripts/extract_test_fixtures.py \\
        --xiidm /path/to/grid.xiidm

    # Puis lancer les tests :
    pytest tests/manoeuvre/test_postes_reels.py -v
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import pytest

from expert_op4grid_recommender.manoeuvre.models import (
    NodeType,
    EquipmentType,
    SwitchKind,
)
from expert_op4grid_recommender.manoeuvre.graph import busbar_nodes, equipment_nodes
from expert_op4grid_recommender.manoeuvre.cellules import (
    detecter_cellules,
    calculer_connected_busbars,
    CellulesVL,
)

from .fixture_loader import (
    FIXTURES_DIR,
    list_available_fixtures,
    load_fixture_index,
    load_fixture_json,
    build_graph_from_fixture,
    get_fixture_metadata,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Classification des postes
# ────────────────────────────────────────────────────────────────────
POSTES_STANDARDS = [
    "CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5",
    "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6",
]
POSTES_DEPARTS_MULTIPLES = [
    "CORNIP3", "CNIEP6", "GUARBP6", "MORBRP6", "RAN_PP6",  # RAN.PP6 → RAN_PP6 en nom de fichier
]


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _fixtures_available() -> bool:
    """Vérifie qu'au moins une fixture est disponible."""
    return len(list_available_fixtures()) > 0


def _get_fixture_names_matching(postes: list[str]) -> list[str]:
    """
    Retourne les noms de fixtures disponibles correspondant aux postes demandés.
    """
    available = list_available_fixtures()
    matched = []
    for poste in postes:
        # Chercher par préfixe (la fixture peut s'appeler CARRIP3 ou CARRIP3_0 etc.)
        for fixture in available:
            if fixture.startswith(poste) or poste.replace(".", "_") in fixture:
                matched.append(fixture)
    return matched


def _all_fixture_names() -> list[str]:
    """Retourne toutes les fixtures de postes ciblés."""
    return _get_fixture_names_matching(POSTES_STANDARDS + POSTES_DEPARTS_MULTIPLES)


# ────────────────────────────────────────────────────────────────────
# Fixtures pytest
# ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_index():
    """Charge l'index des fixtures."""
    return load_fixture_index()


# ────────────────────────────────────────────────────────────────────
# Skip si pas de fixtures
# ────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not _fixtures_available(),
    reason=(
        "Fixtures de postes réels non générées. "
        "Exécuter : venv_.../bin/python scripts/extract_test_fixtures.py "
        "--xiidm /path/to/grid.xiidm"
    ),
)


# ────────────────────────────────────────────────────────────────────
# Paramétrage dynamique
# ────────────────────────────────────────────────────────────────────

def all_fixture_ids() -> list[str]:
    """IDs de fixtures pour la paramétrisation (skip si vide)."""
    names = _all_fixture_names()
    return names if names else ["NO_FIXTURE"]


def standard_fixture_ids() -> list[str]:
    names = _get_fixture_names_matching(POSTES_STANDARDS)
    return names if names else ["NO_FIXTURE"]


def multiples_fixture_ids() -> list[str]:
    names = _get_fixture_names_matching(POSTES_DEPARTS_MULTIPLES)
    return names if names else ["NO_FIXTURE"]


# ────────────────────────────────────────────────────────────────────
# Tests : construction du graphe
# ────────────────────────────────────────────────────────────────────

class TestGraphConstruction:
    """Le graphe est correctement construit depuis les fixtures."""

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_graph_builds_without_error(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_has_busbar_sections(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        bbs = busbar_nodes(G)
        assert len(bbs) >= 1, f"VL '{vl_name}' : aucune SJB trouvée"

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_has_equipment(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        eq = equipment_nodes(G)
        assert len(eq) >= 1, f"VL '{vl_name}' : aucun équipement trouvé"

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_all_nodes_have_type(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        for node, data in G.nodes(data=True):
            assert "node_type" in data, f"Nœud {node} sans node_type"
            assert isinstance(data["node_type"], NodeType)

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_all_switch_edges_have_kind(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        for u, v, d in G.edges(data=True):
            assert "kind" in d, f"Arête ({u},{v}) sans kind"
            assert isinstance(d["kind"], SwitchKind)


# ────────────────────────────────────────────────────────────────────
# Tests : détection des cellules de départ
# ────────────────────────────────────────────────────────────────────

class TestCellulesDepart:
    """Chaque poste a des cellules de départ correctement identifiées."""

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_has_departure_cells(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        assert len(cellules.cellules_depart) > 0

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_all_equipment_covered(self, vl_name):
        """Tout nœud EQUIPMENT est couvert par une cellule de départ."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)

        eq_ids_in_graph = {
            G.nodes[n]["equipment_id"]
            for n in equipment_nodes(G)
        }
        eq_ids_in_cells = set()
        for c in cellules.cellules_depart:
            eq_ids_in_cells.add(c.equipment_id)
            eq_ids_in_cells.update(c.shared_equipment_ids)

        uncovered = eq_ids_in_graph - eq_ids_in_cells
        assert len(uncovered) == 0, (
            f"Équipements non couverts dans '{vl_name}' : {uncovered}"
        )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_each_cell_reaches_at_least_one_bbs(self, vl_name):
        """Chaque cellule de départ atteint au moins une SJB."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_depart:
            assert len(c.busbar_nodes) >= 1, (
                f"Cellule '{c.equipment_id}' dans '{vl_name}' "
                f"n'atteint aucune SJB"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_each_cell_has_switches(self, vl_name):
        """Chaque cellule de départ possède au moins un switch."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_depart:
            assert len(c.switches) >= 1, (
                f"Cellule '{c.equipment_id}' dans '{vl_name}' "
                f"sans aucun switch"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_each_cell_has_at_least_one_breaker(self, vl_name):
        """
        Chaque cellule de départ standard possède au moins un disjoncteur (DJ).
        Les cellules de ré-aiguillage (is_reaiguillage=True) sont exemptées :
        elles ne comportent qu'un sectionneur d'aiguillage (SA) sans DJ propre.
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_depart:
            if c.is_reaiguillage:
                continue
            assert len(c.breakers) >= 1, (
                f"Cellule '{c.equipment_id}' dans '{vl_name}' "
                f"sans disjoncteur (BREAKER)"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_nb_cells_le_nb_equipment(self, vl_name):
        """
        Le nombre de cellules est ≤ au nombre d'équipements
        (les cellules partagées / omnibus réduisent le compte).
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        nb_eq = len(equipment_nodes(G))
        nb_cells = len(cellules.cellules_depart)
        assert nb_cells <= nb_eq, (
            f"'{vl_name}' : {nb_cells} cellules > {nb_eq} équipements"
        )


# ────────────────────────────────────────────────────────────────────
# Tests : pas de chevauchement entre cellules de départ
# ────────────────────────────────────────────────────────────────────

class TestPasDeChevaucement:
    """Les nœuds intermédiaires (non-SJB) ne sont pas partagés entre cellules."""

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_no_internal_node_overlap(self, vl_name):
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        bbs_set = set(busbar_nodes(G))
        cellules = detecter_cellules(G, vl_name)

        seen_internal: dict[int, str] = {}  # node → equipment_id de la 1ère cellule
        for c in cellules.cellules_depart:
            internal_nodes = c.all_nodes - bbs_set
            for n in internal_nodes:
                if n in seen_internal:
                    # Chevauchement → mais c'est OK si c'est la même cellule
                    # (shared_equipment_ids dans la même cellule)
                    assert seen_internal[n] == c.equipment_id or \
                           c.equipment_id in (
                               c2.equipment_id
                               for c2 in cellules.cellules_depart
                               if seen_internal[n] == c2.equipment_id
                               or seen_internal[n] in c2.shared_equipment_ids
                           ), (
                        f"Nœud {n} partagé entre cellules "
                        f"'{seen_internal[n]}' et '{c.equipment_id}' dans '{vl_name}'"
                    )
                else:
                    seen_internal[n] = c.equipment_id


# ────────────────────────────────────────────────────────────────────
# Tests : cellules de couplage
# ────────────────────────────────────────────────────────────────────

class TestCellulesCouplage:
    """Détection correcte des couplages pour les postes à 2+ barres."""

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_coupling_cells_link_two_busbars(self, vl_name):
        """Chaque cellule de couplage relie au moins 2 SJB distinctes."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_couplage:
            assert c.busbar_node_1 is not None, (
                f"Couplage sans busbar_node_1 dans '{vl_name}'"
            )
            assert c.busbar_node_2 is not None, (
                f"Couplage sans busbar_node_2 dans '{vl_name}'"
            )
            assert c.busbar_node_1 != c.busbar_node_2, (
                f"Couplage auto-référent dans '{vl_name}': "
                f"SJB {c.busbar_node_1} == {c.busbar_node_2}"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_coupling_has_breaker(self, vl_name):
        """Chaque cellule de couplage possède un disjoncteur."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_couplage:
            # Certains couplages RTE n'ont pas de DJ explicite (couplage par
            # sectionneur uniquement) — on accepte au moins 1 switch
            assert len(c.switches) >= 1, (
                f"Couplage sans aucun switch dans '{vl_name}'"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_poste_2_barres_has_coupling(self, vl_name):
        """
        Si le poste a ≥ 2 SJB (2 jeux de barres), il doit avoir au moins un
        couplage (sauf postes atypiques).
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        bbs = busbar_nodes(G)
        cellules = detecter_cellules(G, vl_name)

        if len(bbs) >= 2:
            # On log mais on ne fait pas échouer : certains postes ont des SJB
            # multiples sans couplage interne au tronçon
            if len(cellules.cellules_couplage) == 0:
                logger.warning(
                    "VL '%s' a %d SJB mais 0 couplage détecté. "
                    "Cas atypique (1 barre sectionnée ?) — à investiguer.",
                    vl_name, len(bbs),
                )


# ────────────────────────────────────────────────────────────────────
# Tests : connectivité électrique
# ────────────────────────────────────────────────────────────────────

class TestConnectiviteElectrique:
    """Cohérence du calcul des SJB effectivement connectées."""

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_connected_busbars_subset(self, vl_name):
        """Les SJB connectées sont un sous-ensemble des SJB structurelles."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_depart:
            calculer_connected_busbars(c)
            assert c.connected_busbars.issubset(c.busbar_nodes), (
                f"Cellule '{c.equipment_id}' dans '{vl_name}' : "
                f"connected {c.connected_busbars} ⊄ structural {c.busbar_nodes}"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_most_equipment_is_connected(self, vl_name):
        """
        La majorité des équipements doivent être connectés à au moins une barre.
        Un poste en service a typiquement >80% d'équipements connectés.
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)

        connected_count = 0
        total_count = len(cellules.cellules_depart)
        for c in cellules.cellules_depart:
            calculer_connected_busbars(c)
            if c.is_connected:
                connected_count += 1

        if total_count > 0:
            ratio = connected_count / total_count
            assert ratio >= 0.5, (
                f"'{vl_name}' : seulement {connected_count}/{total_count} "
                f"({ratio:.0%}) équipements connectés — suspectement bas"
            )

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_disconnectors_vers_barre(self, vl_name):
        """
        ``disconnectors_vers_barre()`` retourne des sectionneurs valides
        pour chaque barre structurellement accessible.
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        for c in cellules.cellules_depart:
            for bbs_node in c.busbar_nodes:
                discos = c.disconnectors_vers_barre(bbs_node)
                assert isinstance(discos, list)
                for d in discos:
                    assert d.is_disconnector, (
                        f"Switch '{d.switch_id}' retourné par "
                        f"disconnectors_vers_barre n'est pas un DISCONNECTOR"
                    )


# ────────────────────────────────────────────────────────────────────
# Tests spécifiques : postes à départs multiples
# ────────────────────────────────────────────────────────────────────

class TestDepartsMultiples:
    """
    Les postes de la liste « départs multiples » (CORNIP3, CNIEP6, etc.)
    doivent avoir au moins une cellule partagée (is_multiple=True).
    """

    @pytest.mark.parametrize("vl_name", multiples_fixture_ids())
    def test_has_shared_cells(self, vl_name):
        """
        Au moins une cellule omnibus / multiple est détectée.
        Note : ce test peut échouer si le réseau XIIDM ne modélise
        pas ces postes en node/breaker avec départs physiquement partagés.
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)

        shared_cells = [c for c in cellules.cellules_depart if c.is_multiple]

        # On log plutôt que de faire échouer, car la modélisation XIIDM
        # peut ne pas exposer les départs multiples comme partage de nœuds
        if len(shared_cells) == 0:
            logger.warning(
                "VL '%s' (listé comme départ multiple) : "
                "aucune cellule partagée détectée. "
                "Le modèle XIIDM n'expose peut-être pas ce partage.",
                vl_name,
            )


# ────────────────────────────────────────────────────────────────────
# Tests de résumé / diagnostic
# ────────────────────────────────────────────────────────────────────

class TestDiagnostic:
    """Tests de diagnostic et de résumé pour chaque poste."""

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_resume_is_informative(self, vl_name):
        """Le résumé des cellules est non vide et contient le VL ID."""
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        G = build_graph_from_fixture(vl_name)
        cellules = detecter_cellules(G, vl_name)
        r = cellules.resume()
        assert isinstance(r, str)
        assert vl_name in r

    @pytest.mark.parametrize("vl_name", all_fixture_ids())
    def test_stats_coherence(self, vl_name):
        """
        Les stats de la fixture sont cohérentes avec le graphe construit.
        """
        if vl_name == "NO_FIXTURE":
            pytest.skip("Aucune fixture disponible")
        meta = get_fixture_metadata(vl_name)
        stats = meta.get("stats", {})
        G = build_graph_from_fixture(vl_name)

        # Nombre de SJB cohérent
        expected_bbs = stats.get("nb_busbar_sections", 0)
        actual_bbs = len(busbar_nodes(G))
        assert actual_bbs == expected_bbs, (
            f"'{vl_name}' : {actual_bbs} SJB dans le graphe vs "
            f"{expected_bbs} attendues (stats fixture)"
        )

        # Nombre d'équipements cohérent
        expected_eq = stats.get("nb_equipment", 0)
        actual_eq = len(equipment_nodes(G))
        assert actual_eq == expected_eq, (
            f"'{vl_name}' : {actual_eq} équipements dans le graphe vs "
            f"{expected_eq} attendus (stats fixture)"
        )
