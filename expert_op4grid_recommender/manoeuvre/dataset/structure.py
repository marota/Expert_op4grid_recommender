"""
manoeuvre/dataset/structure.py — Chargement de la **structure** d'un poste
depuis une fixture JSON (format de ``scripts/extract_test_fixtures.py``),
sans dépendance à pypowsybl.

Le tagging structurel (``tagging.taguer_bloc``) et la conversion nodale ont
besoin du graphe node/breaker du poste ; pour le dataset historique, la
structure vient soit d'un snapshot XIIDM (via ``manoeuvre.graph.build_vl_graph``,
nécessite pypowsybl), soit — portable et léger — d'une **fixture JSON**
extraite une fois par (poste, version structurelle).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import networkx as nx

from ..models import EquipmentType, NodeType, SwitchKind
from ..topologie import PosteTopologique

_EQ_TYPE_MAP = {t.name: t for t in EquipmentType}
_SWITCH_KIND_MAP = {
    "BREAKER": SwitchKind.BREAKER,
    "DISCONNECTOR": SwitchKind.DISCONNECTOR,
    "LOAD_BREAK_SWITCH": SwitchKind.LOAD_BREAK_SWITCH,
}


def graph_from_fixture_json(path: Union[str, Path]) -> nx.Graph:
    """Reconstruit le graphe node/breaker d'un poste depuis une fixture JSON
    (même format et même sémantique que ``tests/manoeuvre/fixture_loader.py``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    G = nx.Graph()

    def ensure(n: int) -> None:
        if n not in G:
            G.add_node(n, node_type=NodeType.INTERNAL, busbar_section_id=None,
                       equipment_id=None, equipment_type=None)

    for sw in data.get("switches", []):
        n1, n2 = sw["node1"], sw["node2"]
        ensure(n1); ensure(n2)
        kind = _SWITCH_KIND_MAP.get(sw["kind"], SwitchKind.DISCONNECTOR)
        G.add_edge(n1, n2, switch_id=sw["id"], kind=kind, open=sw["open"])
    for ic in data.get("internal_connections", []):
        n1, n2 = ic["node1"], ic["node2"]
        ensure(n1); ensure(n2)
        if not G.has_edge(n1, n2):
            G.add_edge(n1, n2, switch_id=None, kind=SwitchKind.INTERNAL,
                       open=False)
    for bbs in data.get("busbar_sections", []):
        ensure(bbs["node"])
        G.nodes[bbs["node"]].update(node_type=NodeType.BUSBAR_SECTION,
                                    busbar_section_id=bbs["id"])
    for eq in data.get("equipment", []):
        ensure(eq["node"])
        G.nodes[eq["node"]].update(
            node_type=NodeType.EQUIPMENT, equipment_id=eq["id"],
            equipment_type=_EQ_TYPE_MAP.get(eq["type"], EquipmentType.UNKNOWN))
    return G


def poste_from_fixture_json(
    path: Union[str, Path], voltage_level_id: str | None = None,
) -> PosteTopologique:
    """``PosteTopologique`` construit depuis une fixture JSON. Le VL est lu de
    la fixture (champ ``voltage_level_id``) sauf s'il est imposé."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vl = voltage_level_id or data.get("voltage_level_id") or Path(path).stem
    return PosteTopologique.from_graph(graph_from_fixture_json(path), vl)
