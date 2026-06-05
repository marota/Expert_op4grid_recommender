"""
manoeuvre/models.py
-------------------
Structures de données fondamentales pour la représentation des postes
électriques en topologie détaillée (node/breaker).

Ces classes constituent le vocabulaire partagé entre graph.py, cellules.py
et les phases algorithmiques ultérieures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Énumérations
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Type d'un nœud de connectivité dans le graphe node/breaker."""
    BUSBAR_SECTION = "BUSBAR_SECTION"   # Section de jeu de barres (SJB)
    EQUIPMENT = "EQUIPMENT"             # Borne d'un équipement réseau
    INTERNAL = "INTERNAL"               # Nœud intermédiaire sans équipement


class EquipmentType(Enum):
    """Type d'équipement connecté à un nœud de connectivité."""
    LOAD = "LOAD"
    GENERATOR = "GENERATOR"
    LINE_SIDE1 = "LINE_SIDE1"
    LINE_SIDE2 = "LINE_SIDE2"
    TRANSFORMER_SIDE1 = "TRANSFORMER_SIDE1"
    TRANSFORMER_SIDE2 = "TRANSFORMER_SIDE2"
    DANGLING_LINE = "DANGLING_LINE"
    SHUNT_COMPENSATOR = "SHUNT_COMPENSATOR"
    STATIC_VAR_COMPENSATOR = "STATIC_VAR_COMPENSATOR"
    BATTERY = "BATTERY"
    HVDC_CONVERTER_STATION = "HVDC_CONVERTER_STATION"
    UNKNOWN = "UNKNOWN"


class SwitchKind(Enum):
    """
    Type d'un organe de coupure (OC).

    Correspondance avec le C++ NF/TOPO :
    - BREAKER       → DJ (disjoncteur)   — peut couper un courant de court-circuit
    - DISCONNECTOR  → SA (sectionneur)   — manœuvrable seulement hors charge
    - LOAD_BREAK_SWITCH → interrupteur  — coupe le courant nominal
    - INTERNAL      → connexion interne fixe (pas un switch réel)
    """
    BREAKER = "BREAKER"
    DISCONNECTOR = "DISCONNECTOR"
    LOAD_BREAK_SWITCH = "LOAD_BREAK_SWITCH"
    INTERNAL = "INTERNAL"   # pseudo-kind pour les internal connections


class CelluleType(Enum):
    """Type de cellule dans le poste."""
    DEPART = "DEPART"       # Cellule de départ : un équipement + ses OC vers les barres
    COUPLAGE = "COUPLAGE"   # Cellule de couplage : OC reliant deux sections de barre
    INTERNE = "INTERNE"     # Connexion interne sans départ ni couplage identifiable


# ---------------------------------------------------------------------------
# Attributs de nœuds du graphe (stockés dans networkx comme dicts)
# ---------------------------------------------------------------------------

@dataclass
class NodeAttrs:
    """
    Attributs associés à un nœud du graphe NetworkX node/breaker.

    Ces attributs sont stockés directement dans le graphe via
    ``G.nodes[node_id].update(attrs.__dict__)``.
    """
    node_type: NodeType = NodeType.INTERNAL
    busbar_section_id: Optional[str] = None   # peuplé si node_type == BUSBAR_SECTION
    equipment_id: Optional[str] = None         # peuplé si node_type == EQUIPMENT
    equipment_type: Optional[EquipmentType] = None


@dataclass
class EdgeAttrs:
    """
    Attributs associés à une arête du graphe NetworkX node/breaker.

    Chaque arête représente soit un switch réel, soit une internal connection.
    """
    switch_id: Optional[str]       # None pour les internal connections
    kind: SwitchKind
    open: bool                     # True = ouvert (non conducteur)

    @property
    def is_closed(self) -> bool:
        return not self.open

    @property
    def is_breaker(self) -> bool:
        return self.kind == SwitchKind.BREAKER

    @property
    def is_disconnector(self) -> bool:
        return self.kind == SwitchKind.DISCONNECTOR

    @property
    def is_internal(self) -> bool:
        return self.kind == SwitchKind.INTERNAL
