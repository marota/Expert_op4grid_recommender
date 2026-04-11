"""
tests/manoeuvre/test_graph_cellules.py
----------------------------------------
Tests unitaires pour les étapes 1.1 (graph.py) et 1.2 (cellules.py).

Réseau de test
~~~~~~~~~~~~~~
On utilise ``pp.network.create_four_substations_node_breaker_network()``,
réseau de référence pypowsybl en topologie NODE_BREAKER.

Sa topologie (extrait de la documentation pypowsybl) :
- 4 sous-stations : S1, S2, S3, S4
- Voltage levels S1VL1 (132 kV), S1VL2 (380 kV), S2VL1, S3VL1, S4VL1
- S1VL2 : 2 jeux de barres (BBS1, BBS2), couplage, plusieurs lignes et un transfo

S1VL2 est le voltage level le plus riche et sert de base principale pour les tests.
Il modélise exactement la structure d'un poste 2 barres RTE :
- 2 sections de jeu de barres
- Un couplage (BREAKER + DISCONNECTORS)
- Plusieurs départs (lignes + transfo)
- Des sectionneurs double-barre

Conventions de nommage dans les assertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Les IDs des switches dans le réseau de test pypowsybl suivent le pattern :
  ``<VL>_<BBS>_<EQUIPMENT>_DISCONNECTOR`` ou ``<VL>_COUPLER_BREAKER``
"""

import pytest
import networkx as nx
import pypowsybl as pp

from expert_op4grid_recommender.manoeuvre.graph import (
    build_vl_graph,
    busbar_nodes,
    equipment_nodes,
    TopologyError,
)
from expert_op4grid_recommender.manoeuvre.cellules import (
    detecter_cellules,
    calculer_connected_busbars,
    CelluleDepart,
    CelluleCouplage,
)
from expert_op4grid_recommender.manoeuvre.models import (
    NodeType,
    SwitchKind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def four_sub_network():
    """Réseau pypowsybl 4 postes en topologie NODE_BREAKER."""
    return pp.network.create_four_substations_node_breaker_network()


@pytest.fixture(scope="module")
def graph_s1vl2(four_sub_network):
    """Graphe node/breaker du voltage level S1VL2 (380 kV, 2 barres)."""
    return build_vl_graph(four_sub_network, "S1VL2")


@pytest.fixture(scope="module")
def cellules_s1vl2(graph_s1vl2):
    """Cellules détectées pour S1VL2."""
    return detecter_cellules(graph_s1vl2, "S1VL2")


# ---------------------------------------------------------------------------
# Tests step 1.1 — build_vl_graph
# ---------------------------------------------------------------------------

class TestBuildVlGraph:
    """Tests de l'extraction du graphe node/breaker (étape 1.1)."""

    def test_graph_is_networkx(self, graph_s1vl2):
        """Le résultat est bien un graphe NetworkX."""
        assert isinstance(graph_s1vl2, nx.Graph)

    def test_graph_has_nodes(self, graph_s1vl2):
        """Le graphe contient des nœuds."""
        assert graph_s1vl2.number_of_nodes() > 0

    def test_graph_has_edges(self, graph_s1vl2):
        """Le graphe contient des arêtes (switches + internal connections)."""
        assert graph_s1vl2.number_of_edges() > 0

    def test_busbar_sections_tagged(self, graph_s1vl2):
        """Les nœuds de SJB sont bien taggés BUSBAR_SECTION."""
        bbs = busbar_nodes(graph_s1vl2)
        assert len(bbs) >= 2, (
            "S1VL2 a 2 barres (BBS1 + BBS2), on attend au moins 2 nœuds BUSBAR_SECTION"
        )
        for node in bbs:
            assert graph_s1vl2.nodes[node]["node_type"] == NodeType.BUSBAR_SECTION
            assert graph_s1vl2.nodes[node]["busbar_section_id"] is not None

    def test_equipment_nodes_tagged(self, graph_s1vl2):
        """Les nœuds d'équipements sont taggés EQUIPMENT."""
        eq_nodes = equipment_nodes(graph_s1vl2)
        assert len(eq_nodes) > 0, "S1VL2 doit avoir des équipements connectés"
        for node in eq_nodes:
            assert graph_s1vl2.nodes[node]["node_type"] == NodeType.EQUIPMENT
            assert graph_s1vl2.nodes[node]["equipment_id"] is not None

    def test_switch_edges_have_attributes(self, graph_s1vl2):
        """Chaque arête de switch a les attributs requis : switch_id, kind, open."""
        switch_edges = [
            (u, v, d) for u, v, d in graph_s1vl2.edges(data=True)
            if d.get("switch_id") is not None
        ]
        assert len(switch_edges) > 0
        for u, v, d in switch_edges:
            assert "switch_id" in d
            assert "kind" in d
            assert "open" in d
            assert isinstance(d["kind"], SwitchKind)
            assert isinstance(d["open"], bool)

    def test_internal_connections_have_attributes(self, graph_s1vl2):
        """Les internal connections sont marquées kind=INTERNAL et open=False."""
        internal_edges = [
            (u, v, d) for u, v, d in graph_s1vl2.edges(data=True)
            if d.get("kind") == SwitchKind.INTERNAL
        ]
        for u, v, d in internal_edges:
            assert d["open"] is False
            assert d["switch_id"] is None

    def test_topology_error_on_bus_breaker(self, four_sub_network):
        """Une TopologyError est levée si le VL n'est pas NODE_BREAKER."""
        # Créer un réseau bus/breaker pour tester l'erreur
        bus_breaker_net = pp.network.create_eurostag_tutorial_example1_network()
        vl_ids = bus_breaker_net.get_voltage_levels().index.tolist()
        assert len(vl_ids) > 0
        vl_id = vl_ids[0]
        with pytest.raises(TopologyError, match="NODE_BREAKER"):
            build_vl_graph(bus_breaker_net, vl_id)

    def test_unknown_voltage_level_raises(self, four_sub_network):
        """Un voltage level inexistant lève une TopologyError."""
        with pytest.raises(TopologyError, match="introuvable"):
            build_vl_graph(four_sub_network, "VL_INEXISTANT")

    def test_graph_connected_or_multi_component(self, graph_s1vl2):
        """
        Le graphe peut être connexe ou multi-composantes selon l'état des switches.
        On vérifie juste que le nombre de composantes est positif.
        """
        components = list(nx.connected_components(graph_s1vl2))
        assert len(components) >= 1

    def test_all_nodes_have_node_type(self, graph_s1vl2):
        """Tout nœud du graphe a un attribut node_type."""
        for node, data in graph_s1vl2.nodes(data=True):
            assert "node_type" in data, f"Nœud {node} sans node_type"
            assert isinstance(data["node_type"], NodeType)


# ---------------------------------------------------------------------------
# Tests step 1.2 — detecter_cellules
# ---------------------------------------------------------------------------

class TestDetecterCellules:
    """Tests de la détection de cellules (étape 1.2)."""

    def test_returns_cellules_vl(self, cellules_s1vl2):
        """Le résultat est un objet CellulesVL."""
        from expert_op4grid_recommender.manoeuvre.cellules import CellulesVL
        assert isinstance(cellules_s1vl2, CellulesVL)

    def test_has_departure_cells(self, cellules_s1vl2):
        """Au moins une cellule de départ est détectée."""
        assert len(cellules_s1vl2.cellules_depart) > 0

    def test_has_coupling_cells(self, cellules_s1vl2):
        """
        S1VL2 a un couplage → au moins une cellule de couplage est détectée.
        """
        assert len(cellules_s1vl2.cellules_couplage) >= 1, (
            "S1VL2 a 2 barres et un couplage : on attend au moins 1 cellule de couplage"
        )

    def test_departure_cell_has_equipment(self, cellules_s1vl2):
        """Chaque cellule de départ a un equipment_id non vide."""
        for c in cellules_s1vl2.cellules_depart:
            assert c.equipment_id, f"CelluleDepart sans equipment_id : {c}"

    def test_departure_cell_reaches_busbars(self, cellules_s1vl2):
        """Chaque cellule de départ atteint au moins une SJB."""
        for c in cellules_s1vl2.cellules_depart:
            assert len(c.busbar_nodes) >= 1, (
                f"Cellule de '{c.equipment_id}' n'atteint aucune SJB"
            )

    def test_departure_cell_has_switches(self, cellules_s1vl2):
        """Chaque cellule de départ contient au moins un switch."""
        for c in cellules_s1vl2.cellules_depart:
            assert len(c.switches) >= 1, (
                f"Cellule de '{c.equipment_id}' sans aucun switch"
            )

    def test_departure_cell_has_subgraph(self, cellules_s1vl2):
        """Chaque cellule de départ a un sous-graphe."""
        for c in cellules_s1vl2.cellules_depart:
            assert c.subgraph is not None
            assert isinstance(c.subgraph, nx.Graph)

    def test_coupling_cell_has_two_busbars(self, cellules_s1vl2):
        """Chaque cellule de couplage relie exactement deux SJB."""
        for c in cellules_s1vl2.cellules_couplage:
            assert c.busbar_node_1 is not None
            assert c.busbar_node_2 is not None
            assert c.busbar_node_1 != c.busbar_node_2

    def test_coupling_cell_has_breaker(self, cellules_s1vl2):
        """
        Le couplage standard d'un poste 2 barres possède un BREAKER
        (le disjoncteur de couplage).
        """
        for c in cellules_s1vl2.cellules_couplage:
            breaker_switches = [s for s in c.switches if s.is_breaker]
            assert len(breaker_switches) >= 1, (
                f"Cellule de couplage sans BREAKER : {c}"
            )

    def test_get_cellule_depart_by_equipment(self, cellules_s1vl2):
        """``get_cellule_depart`` retrouve une cellule par son equipment_id."""
        if not cellules_s1vl2.cellules_depart:
            pytest.skip("Aucune cellule de départ disponible")
        first_cell = cellules_s1vl2.cellules_depart[0]
        found = cellules_s1vl2.get_cellule_depart(first_cell.equipment_id)
        assert found is not None
        assert found.equipment_id == first_cell.equipment_id

    def test_get_cellule_depart_unknown_returns_none(self, cellules_s1vl2):
        """Un equipment_id inconnu retourne None."""
        result = cellules_s1vl2.get_cellule_depart("EQUIPEMENT_INEXISTANT")
        assert result is None

    def test_resume_string(self, cellules_s1vl2):
        """La méthode resume() retourne une chaîne non vide."""
        r = cellules_s1vl2.resume()
        assert isinstance(r, str)
        assert len(r) > 0
        assert "S1VL2" in r


# ---------------------------------------------------------------------------
# Tests de la connectivité électrique
# ---------------------------------------------------------------------------

class TestConnectedBusbars:
    """Tests du calcul de SJB effectivement connectées."""

    def test_connected_busbars_subset_of_busbar_nodes(self, cellules_s1vl2):
        """Les SJB connectées sont un sous-ensemble des SJB atteintes."""
        for c in cellules_s1vl2.cellules_depart:
            calculer_connected_busbars(c)
            assert c.connected_busbars.issubset(c.busbar_nodes), (
                f"Cellule '{c.equipment_id}' : connected_busbars ⊄ busbar_nodes"
            )

    def test_connected_busbars_type(self, cellules_s1vl2):
        """connected_busbars est un set d'entiers."""
        for c in cellules_s1vl2.cellules_depart:
            calculer_connected_busbars(c)
            assert isinstance(c.connected_busbars, set)

    def test_is_connected_property(self, cellules_s1vl2):
        """La propriété is_connected est cohérente avec connected_busbars."""
        for c in cellules_s1vl2.cellules_depart:
            calculer_connected_busbars(c)
            if len(c.connected_busbars) > 0:
                assert c.is_connected
            else:
                assert not c.is_connected

    def test_disconnectors_vers_barre_returns_list(self, cellules_s1vl2):
        """disconnectors_vers_barre retourne une liste (vide ou non)."""
        for c in cellules_s1vl2.cellules_depart:
            for bbs_node in c.busbar_nodes:
                result = c.disconnectors_vers_barre(bbs_node)
                assert isinstance(result, list)
                for s in result:
                    from expert_op4grid_recommender.manoeuvre.cellules import SwitchInfo
                    assert isinstance(s, SwitchInfo)


# ---------------------------------------------------------------------------
# Test de cohérence globale
# ---------------------------------------------------------------------------

class TestCoherenceGlobale:
    """Tests de cohérence entre le graphe et les cellules."""

    def test_equipment_nodes_covered_by_cells(self, graph_s1vl2, cellules_s1vl2):
        """
        Tout nœud EQUIPMENT du graphe est couvert par au moins une cellule
        de départ.
        """
        eq_nodes_in_graph = set(equipment_nodes(graph_s1vl2))
        eq_ids_in_graph = {
            graph_s1vl2.nodes[n]["equipment_id"]
            for n in eq_nodes_in_graph
        }
        eq_ids_in_cells = set()
        for c in cellules_s1vl2.cellules_depart:
            eq_ids_in_cells.add(c.equipment_id)
            eq_ids_in_cells.update(c.shared_equipment_ids)

        uncovered = eq_ids_in_graph - eq_ids_in_cells
        assert len(uncovered) == 0, (
            f"Équipements non couverts par une cellule : {uncovered}"
        )

    def test_no_node_in_both_depart_and_couplage(
        self, cellules_s1vl2
    ):
        """
        Un nœud ne doit pas appartenir à la fois à une cellule de départ et
        à une cellule de couplage (sauf les SJB qui sont partagées).
        """
        from expert_op4grid_recommender.manoeuvre.models import NodeType

        depart_nodes: set[int] = set()
        for c in cellules_s1vl2.cellules_depart:
            # Exclure les SJB qui peuvent être partagées
            internal_nodes = {
                n for n in c.all_nodes
                if c.subgraph is not None
                and c.subgraph.nodes[n].get("node_type") != NodeType.BUSBAR_SECTION
            }
            depart_nodes.update(internal_nodes)

        coupling_internal_nodes: set[int] = set()
        for c in cellules_s1vl2.cellules_couplage:
            if c.subgraph is not None:
                internal = {
                    n for n, d in c.subgraph.nodes(data=True)
                    if d.get("node_type") != NodeType.BUSBAR_SECTION
                }
                coupling_internal_nodes.update(internal)

        overlap = depart_nodes & coupling_internal_nodes
        assert len(overlap) == 0, (
            f"Nœuds internes partagés entre départ et couplage : {overlap}"
        )

    def test_nb_depart_cells_matches_equipment_count(
        self, four_sub_network, cellules_s1vl2
    ):
        """
        Le nombre de cellules de départ est <= au nombre d'équipements du VL
        (les départs partagés réduisent le compte).
        """
        # Compter les équipements dans S1VL2
        all_eqs = set()
        for getter, vl_col in [
            (four_sub_network.get_loads, "voltage_level_id"),
            (four_sub_network.get_generators, "voltage_level_id"),
        ]:
            try:
                df = getter()
                if vl_col in df.columns:
                    all_eqs.update(df[df[vl_col] == "S1VL2"].index.tolist())
            except Exception:
                pass

        for getter, vl_col in [
            (four_sub_network.get_lines, "voltage_level1_id"),
            (four_sub_network.get_lines, "voltage_level2_id"),
            (four_sub_network.get_2_windings_transformers, "voltage_level1_id"),
            (four_sub_network.get_2_windings_transformers, "voltage_level2_id"),
        ]:
            try:
                df = getter()
                if vl_col in df.columns:
                    all_eqs.update(df[df[vl_col] == "S1VL2"].index.tolist())
            except Exception:
                pass

        nb_cells = len(cellules_s1vl2.cellules_depart)
        nb_eqs = len(all_eqs)
        assert nb_cells <= nb_eqs, (
            f"Plus de cellules ({nb_cells}) que d'équipements ({nb_eqs}) dans S1VL2"
        )
