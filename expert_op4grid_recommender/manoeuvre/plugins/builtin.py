"""
manoeuvre/plugins/builtin.py — Adaptateurs « libtopo » : les algorithmes
historiques du module (portage libTOPO) exposés au travers des contrats
pluggables. Enregistrés sous le nom ``"libtopo"`` pour chacune des trois
phases ; ce sont les implémentations par défaut de l'orchestrateur.

Les fonctions sous-jacentes (``determiner_topo_complete_cible``,
``determiner_manoeuvres_cible_detaillee``) restent l'API publique historique :
ces adaptateurs sont de **minces ponts** sans logique propre.
"""
from __future__ import annotations

from typing import Optional

from ..topologie import PosteTopologique, TopologieNodale
from ..algo.results import ResultatManoeuvres
from ..algo.targets import (
    determiner_manoeuvres_cible_detaillee,
    determiner_topo_complete_cible,
)
from .interfaces import (
    CibleDetaillee,
    ResultatIdentification,
    ResultatPlanification,
)
from .registry import register


@register("planificateur", "libtopo")
class LibTopoPlanificateur:
    """Phase C — ``determiner_topo_complete_cible`` (placement automatique
    nœud→SJB + séquenceur + replis transactionnels), la cible détaillée étant
    l'état atteint par rejeu de la séquence."""
    nom = "libtopo"

    def planifier(
        self,
        poste: PosteTopologique,
        topo_cible: TopologieNodale,
        cible_busbar: Optional[dict[str, int]] = None,
        **options,
    ) -> ResultatPlanification:
        res = determiner_topo_complete_cible(poste, topo_cible, cible_busbar)
        cible = CibleDetaillee.from_manoeuvres(poste, res.manoeuvres)
        return ResultatPlanification(cible_detaillee=cible, sequence=res)


@register("identificateur", "libtopo")
class LibTopoIdentificateur:
    """Phase A — dérivée de la phase C : la topologie détaillée d'intérêt est
    celle qu'atteint le planificateur libTOPO (même pont que l'endpoint
    ``/api/nodale_to_detaillee`` de l'IHM). La séquence calculée est jointe en
    sous-produit (``ResultatIdentification.sequence``)."""
    nom = "libtopo"

    def identifier(
        self,
        poste: PosteTopologique,
        topo_cible: TopologieNodale,
        **options,
    ) -> ResultatIdentification:
        plan = LibTopoPlanificateur().planifier(poste, topo_cible, **options)
        res = plan.sequence
        return ResultatIdentification(
            voltage_level_id=poste.voltage_level_id,
            cible=plan.cible_detaillee,
            is_realisable=res.is_verified,
            message=res.message,
            noeuds_non_realisables=res.noeuds_non_realisables,
            sequence=res,
        )


@register("sequenceur", "libtopo")
class LibTopoSequenceur:
    """Phase B — ``determiner_manoeuvres_cible_detaillee`` (voie principale
    smooth/aggressive/multi-barres + candidat diff minimal transactionnel)."""
    nom = "libtopo"

    def __init__(self, mode: str = "smooth"):
        self.mode = mode

    def sequencer(
        self,
        poste: PosteTopologique,
        cible: CibleDetaillee,
        mode: Optional[str] = None,
        **options,
    ) -> ResultatManoeuvres:
        return determiner_manoeuvres_cible_detaillee(
            poste, cible.to_graph(poste), mode=mode or self.mode)
