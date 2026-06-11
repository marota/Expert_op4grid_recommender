"""
tests/manoeuvre/fixture_loader.py
-----------------------------------
Charge les fixtures JSON de topologie et reconstruit un graphe NetworkX
exploitable par le module Manoeuvre — sans dépendance à pypowsybl.

Les fixtures sont produites par ``scripts/extract_test_fixtures.py``
et stockées dans ``tests/manoeuvre/fixtures/<vl_id>.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from expert_op4grid_recommender.manoeuvre.models import (
    NodeType,
    EquipmentType,
    SwitchKind,
)

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Mapping des types d'équipement JSON → EquipmentType
_EQ_TYPE_MAP: dict[str, EquipmentType] = {
    "LOAD": EquipmentType.LOAD,
    "GENERATOR": EquipmentType.GENERATOR,
    "LINE_SIDE1": EquipmentType.LINE_SIDE1,
    "LINE_SIDE2": EquipmentType.LINE_SIDE2,
    "TRANSFORMER_SIDE1": EquipmentType.TRANSFORMER_SIDE1,
    "TRANSFORMER_SIDE2": EquipmentType.TRANSFORMER_SIDE2,
    "DANGLING_LINE": EquipmentType.DANGLING_LINE,
    "SHUNT_COMPENSATOR": EquipmentType.SHUNT_COMPENSATOR,
    "STATIC_VAR_COMPENSATOR": EquipmentType.STATIC_VAR_COMPENSATOR,
    "BATTERY": EquipmentType.BATTERY,
    "HVDC_CONVERTER_STATION": EquipmentType.HVDC_CONVERTER_STATION,
}

_SWITCH_KIND_MAP: dict[str, SwitchKind] = {
    "BREAKER": SwitchKind.BREAKER,
    "DISCONNECTOR": SwitchKind.DISCONNECTOR,
    "LOAD_BREAK_SWITCH": SwitchKind.LOAD_BREAK_SWITCH,
}


def list_available_fixtures() -> list[str]:
    """Retourne les noms de fixtures disponibles (sans extension)."""
    return sorted(
        p.stem for p in FIXTURES_DIR.glob("*.json") if p.stem != "index"
    )


def load_fixture_index() -> dict[str, Any]:
    """Charge l'index global des fixtures (index.json)."""
    index_path = FIXTURES_DIR / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def load_fixture_json(vl_name: str) -> dict[str, Any]:
    """Charge le JSON brut d'une fixture.

    Tolère le **nommage point/underscore** : un voltage level RTE ``TRI.PP7`` (avec
    point) est stocké sous ``TRI_PP7.json`` (le point étant remplacé par ``_`` dans
    le nom de fichier, cf. ``scripts/extract_test_fixtures.py``). On essaie donc le
    nom tel quel, puis sa variante point→underscore."""
    path = FIXTURES_DIR / f"{vl_name}.json"
    if not path.exists():
        alt = FIXTURES_DIR / f"{vl_name.replace('.', '_')}.json"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Fixture introuvable : {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_fixture(vl_name: str) -> nx.Graph:
    """
    Reconstruit un graphe NetworkX à partir d'une fixture JSON.

    Le graphe produit a exactement la même structure que celui créé par
    ``manoeuvre.graph.build_vl_graph`` à partir de pypowsybl, mais
    construit exclusivement depuis les données JSON statiques.

    Parameters
    ----------
    vl_name :
        Nom de la fixture (sans extension .json). Ex: "CARRIP3".

    Returns
    -------
    nx.Graph
        Graphe node/breaker prêt à passer à ``detecter_cellules()``.
    """
    data = load_fixture_json(vl_name)
    G = nx.Graph()

    # ── Helper ────────────────────────────────────────────────────────
    def ensure_node(n: int) -> None:
        if n not in G:
            G.add_node(
                n,
                node_type=NodeType.INTERNAL,
                busbar_section_id=None,
                equipment_id=None,
                equipment_type=None,
            )

    # ── Switches ──────────────────────────────────────────────────────
    for sw in data.get("switches", []):
        n1, n2 = sw["node1"], sw["node2"]
        ensure_node(n1)
        ensure_node(n2)
        kind_str = sw["kind"]
        kind = _SWITCH_KIND_MAP.get(kind_str, SwitchKind.DISCONNECTOR)
        G.add_edge(n1, n2, switch_id=sw["id"], kind=kind, open=sw["open"])

    # ── Internal connections ──────────────────────────────────────────
    for ic in data.get("internal_connections", []):
        n1, n2 = ic["node1"], ic["node2"]
        ensure_node(n1)
        ensure_node(n2)
        if not G.has_edge(n1, n2):
            G.add_edge(n1, n2, switch_id=None, kind=SwitchKind.INTERNAL, open=False)

    # ── Busbar sections ───────────────────────────────────────────────
    for bbs in data.get("busbar_sections", []):
        node = bbs["node"]
        ensure_node(node)
        G.nodes[node].update({
            "node_type": NodeType.BUSBAR_SECTION,
            "busbar_section_id": bbs["id"],
        })

    # ── Equipment ─────────────────────────────────────────────────────
    for eq in data.get("equipment", []):
        node = eq["node"]
        ensure_node(node)
        eq_type = _EQ_TYPE_MAP.get(eq["type"], EquipmentType.UNKNOWN)
        G.nodes[node].update({
            "node_type": NodeType.EQUIPMENT,
            "equipment_id": eq["id"],
            "equipment_type": eq_type,
        })

    return G


def get_fixture_metadata(vl_name: str) -> dict[str, Any]:
    """
    Retourne les métadonnées d'une fixture (stats, substation_id, nominal_v).
    """
    data = load_fixture_json(vl_name)
    return {
        "voltage_level_id": data.get("voltage_level_id"),
        "substation_id": data.get("substation_id"),
        "nominal_v": data.get("nominal_v"),
        "stats": data.get("stats", {}),
    }
