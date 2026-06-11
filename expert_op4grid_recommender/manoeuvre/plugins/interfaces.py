"""
manoeuvre/plugins/interfaces.py — Contrats des trois phases de calcul pluggables.

Le passage d'une **topologie nodale cible** à une **séquence de manœuvres**
se décompose en trois phases de calcul, chacune substituable par un algorithme
tiers (typage structurel :pep:`544` — aucun héritage requis, il suffit
d'exposer la bonne méthode) :

A. ``IdentificateurTopologieDetaillee`` :
   topologie **nodale** cible → topologie **détaillée** cible (état d'organes) ;
B. ``SequenceurManoeuvres`` :
   topologie **détaillée** cible → **séquence de manœuvres** ordonnée ;
C. ``PlanificateurNodal`` :
   topologie **nodale** cible → séquence + topologie détaillée, **en une passe**
   (pour les algorithmes qui ne séparent pas les deux étapes).

Un fournisseur d'algorithme peut implémenter une seule phase, deux, ou les
trois ; l'orchestrateur (``plugins.pipeline.PlanificateurTopologie``) compose
les phases manquantes (A + B ⇒ C, et C ⇒ A par rejeu).

Type pivot : ``CibleDetaillee`` — représentation **sérialisable** d'une
topologie détaillée (``switch_id -> ouvert ?``), convertible depuis/vers le
graphe node/breaker du module (``poste.graph``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, runtime_checkable

import networkx as nx

from ..topologie import PosteTopologique, TopologieNodale
from ..algo.results import Manoeuvre, ResultatManoeuvres
from ..algo.graph_ops import _set_switch, _switch_edge_index


# ---------------------------------------------------------------------------
# Type pivot : topologie détaillée cible
# ---------------------------------------------------------------------------

@dataclass
class CibleDetaillee:
    """Topologie **détaillée** cible d'un voltage level : état OUVERT/FERMÉ de
    chaque organe de coupure.

    Représentation pivot entre phases : sérialisable (dict ``switch_id ->
    open``, le format des scénarios de l'IHM), et convertible depuis/vers le
    graphe node/breaker du module. ``etats_organes`` peut être **partiel** :
    les organes absents conservent leur état courant lors de ``to_graph``.
    """
    voltage_level_id: str
    etats_organes: dict[str, bool]   # switch_id -> True si OUVERT

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------

    @classmethod
    def from_graph(cls, G: nx.Graph, voltage_level_id: str) -> "CibleDetaillee":
        """Capture l'état de tous les organes d'un graphe node/breaker."""
        etats = {sid: bool(G.edges[edge].get("open", False))
                 for sid, edge in _switch_edge_index(G).items()}
        return cls(voltage_level_id=voltage_level_id, etats_organes=etats)

    @classmethod
    def from_manoeuvres(
        cls, poste: PosteTopologique, manoeuvres: Iterable[Manoeuvre],
    ) -> "CibleDetaillee":
        """État détaillé atteint en **rejouant** une séquence depuis l'état
        courant de ``poste`` (le graphe du poste n'est pas muté)."""
        G = poste.graph.copy()
        for m in manoeuvres:
            _set_switch(G, m.switch_id, m.action == "OPEN")
        return cls.from_graph(G, poste.voltage_level_id)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def organes_inconnus(self, poste: PosteTopologique) -> list[str]:
        """Organes de la cible absents du poste (cible incompatible)."""
        idx = _switch_edge_index(poste.graph)
        return sorted(s for s in self.etats_organes if s not in idx)

    def to_graph(self, poste: PosteTopologique) -> nx.Graph:
        """Graphe node/breaker cible : copie du graphe du poste avec les états
        d'organes de la cible appliqués (les organes non mentionnés gardent
        leur état courant)."""
        G = poste.graph.copy()
        for sid, open_ in self.etats_organes.items():
            _set_switch(G, sid, open_)
        return G

    def topologie_nodale(self, poste: PosteTopologique) -> TopologieNodale:
        """Topologie nodale **induite** par cette cible détaillée."""
        return TopologieNodale.from_graph(self.to_graph(poste),
                                          self.voltage_level_id)

    def diff(self, autre: "CibleDetaillee") -> dict[str, tuple[Optional[bool], Optional[bool]]]:
        """Organes différant entre deux cibles : ``{switch_id: (soi, autre)}``."""
        out: dict[str, tuple[Optional[bool], Optional[bool]]] = {}
        for sid in set(self.etats_organes) | set(autre.etats_organes):
            a = self.etats_organes.get(sid)
            b = autre.etats_organes.get(sid)
            if a != b:
                out[sid] = (a, b)
        return out


# ---------------------------------------------------------------------------
# Structures de résultat des phases
# ---------------------------------------------------------------------------

@dataclass
class ResultatIdentification:
    """Sortie de la phase A (identification de topologie détaillée)."""
    voltage_level_id: str
    #: Cible détaillée proposée (peut être partielle/best-effort si
    #: ``is_realisable`` est faux ; ``None`` si rien n'est proposable).
    cible: Optional[CibleDetaillee] = None
    #: La cible détaillée réalise-t-elle exactement la topologie nodale visée ?
    is_realisable: bool = False
    message: str = ""
    #: Dégradation gracieuse : groupes de départs des nœuds non réalisables.
    noeuds_non_realisables: list[list[str]] = field(default_factory=list)
    #: Sous-produit éventuel : séquence déjà calculée par l'algorithme
    #: (évite un recalcul quand l'identification dérive d'un planificateur
    #: bout-en-bout). L'orchestrateur peut la réutiliser, jamais l'exiger.
    sequence: Optional[ResultatManoeuvres] = None


@dataclass
class ResultatPlanification:
    """Sortie de la phase C (planification nodale bout-en-bout) : la séquence
    **et** la topologie détaillée qu'elle atteint."""
    cible_detaillee: Optional[CibleDetaillee]
    sequence: ResultatManoeuvres

    # Délégations de confort vers la séquence -------------------------------
    @property
    def manoeuvres(self) -> list[Manoeuvre]:
        return self.sequence.manoeuvres

    @property
    def nb_manoeuvres(self) -> int:
        return self.sequence.nb_manoeuvres

    @property
    def is_verified(self) -> bool:
        return self.sequence.is_verified

    @property
    def is_verified_detaillee(self) -> bool:
        return self.sequence.is_verified_detaillee

    @property
    def message(self) -> str:
        return self.sequence.message

    def resume(self) -> str:
        return self.sequence.resume()


# ---------------------------------------------------------------------------
# Contrats des trois phases (typage structurel)
# ---------------------------------------------------------------------------

@runtime_checkable
class IdentificateurTopologieDetaillee(Protocol):
    """Phase A — identifie une topologie **détaillée** cible réalisant une
    topologie **nodale** cible.

    Contrat :
    - ne **mute jamais** ``poste`` ni ``poste.graph`` ;
    - retourne toujours un ``ResultatIdentification`` (jamais ``None``) ;
    - ``is_realisable`` ne doit être vrai que si ``cible.topologie_nodale(poste)``
      a la même partition que ``topo_cible`` (l'orchestrateur le revérifie) ;
    - ``**options`` : paramètres propres à l'algorithme (ignorer les inconnus).
    """
    nom: str

    def identifier(
        self,
        poste: PosteTopologique,
        topo_cible: TopologieNodale,
        **options,
    ) -> ResultatIdentification: ...


@runtime_checkable
class SequenceurManoeuvres(Protocol):
    """Phase B — calcule la **séquence ordonnée de manœuvres** menant de l'état
    courant de ``poste`` à une topologie **détaillée** cible.

    Contrat :
    - ne **mute jamais** ``poste`` ni ``poste.graph`` (travailler sur copie) ;
    - les ``switch_id`` émis doivent exister dans le poste (un id inconnu est
      consigné en écart par la vérification de l'orchestrateur) ;
    - les règles de sûreté (sectionneur hors charge, ouvrages hors tension)
      sont revérifiées **indépendamment** par l'orchestrateur : l'algorithme
      n'a pas à être cru sur parole ;
    - ``**options`` : p. ex. ``mode="smooth"|"aggressive"`` pour
      l'implémentation libTOPO (ignorer les options inconnues).
    """
    nom: str

    def sequencer(
        self,
        poste: PosteTopologique,
        cible: CibleDetaillee,
        **options,
    ) -> ResultatManoeuvres: ...


@runtime_checkable
class PlanificateurNodal(Protocol):
    """Phase C — calcule **directement**, depuis une topologie **nodale** cible,
    la séquence de manœuvres et la topologie détaillée atteinte.

    Contrat : mêmes exigences de non-mutation et de vérifiabilité que les
    phases A et B. ``cible_detaillee`` doit être l'état atteint par le rejeu de
    ``sequence.manoeuvres`` (l'orchestrateur peut la reconstruire par rejeu si
    l'algorithme ne la fournit pas).
    """
    nom: str

    def planifier(
        self,
        poste: PosteTopologique,
        topo_cible: TopologieNodale,
        **options,
    ) -> ResultatPlanification: ...
