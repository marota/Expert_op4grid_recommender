"""
manoeuvre/dataset/structure.py — Chargement de la **structure** d'un poste :
depuis une fixture JSON (format de ``scripts/extract_test_fixtures.py``, sans
dépendance à pypowsybl) ou directement depuis un **instantané XIIDM du
dataset** (``postes_depuis_xiidm``, nécessite pypowsybl).

Le tagging structurel (``tagging.taguer_bloc``) et la conversion nodale ont
besoin du graphe node/breaker du poste, **avec les mêmes identifiants
d'organes que les données analysées**. Sur le dataset RTE 7000, les ids des
fixtures du dépôt (export normalisé : espaces repliés, suffixe ``_OC``) ne
recouvrent PAS ceux des instantanés (champs RTE à largeur fixe) — la
structure doit donc être extraite du dataset lui-même (une fois par version
structurelle, cf. plan phase 1) ; ``couverture_structure`` mesure cette
compatibilité et permet d'écarter une structure inadaptée.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

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
        ensure(n1)
        ensure(n2)
        kind = _SWITCH_KIND_MAP.get(sw["kind"], SwitchKind.DISCONNECTOR)
        G.add_edge(n1, n2, switch_id=sw["id"], kind=kind, open=sw["open"])
    for ic in data.get("internal_connections", []):
        n1, n2 = ic["node1"], ic["node2"]
        ensure(n1)
        ensure(n2)
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


def postes_depuis_xiidm(
    path: Union[str, Path],
    vls: Optional[Iterable[str]] = None,
) -> tuple[dict[str, PosteTopologique], dict[str, str]]:
    """Structures de postes extraites d'un **instantané XIIDM** (du dataset) —
    les ids d'organes sont alors ceux des données, par construction.

    ``vls`` restreint aux postes voulus (défaut : tous les VL NODE_BREAKER de
    l'instantané). Retourne ``(postes, echecs)`` : les VL dont la construction
    échoue (topologie BUS_BREAKER, structure exotique…) sont consignés dans
    ``echecs`` avec leur raison — à traiter en repli « par nommage ».
    """
    from ..graph import build_vl_graph          # paresseux : exige pypowsybl
    from .dgitt import _charger_reseau

    net = _charger_reseau(Path(path))
    if vls is None:
        vlt = net.get_voltage_levels(all_attributes=True)
        vls = vlt[vlt["topology_kind"] == "NODE_BREAKER"].index.astype(str)
    postes: dict[str, PosteTopologique] = {}
    echecs: dict[str, str] = {}
    for vl in sorted(set(map(str, vls))):
        try:
            postes[vl] = PosteTopologique.from_graph(build_vl_graph(net, vl), vl)
        except Exception as exc:                            # noqa: BLE001
            echecs[vl] = str(exc)
    return postes, echecs


def couverture_structure(
    poste: PosteTopologique, etats: dict[str, bool],
) -> float:
    """Part des organes de ``etats`` connus du graphe de ``poste`` (0..1).

    Une couverture faible signale une structure **incompatible** avec les
    données (autre convention d'identifiants, autre version structurelle) :
    le tagging structurel et les partitions nodales seraient muets ou faux —
    écarter la structure et replier sur le tagging par nommage."""
    if not etats:
        return 0.0
    from ..algo.graph_ops import _switch_edge_index
    connus = set(_switch_edge_index(poste.graph))
    return len(set(etats) & connus) / len(etats)
