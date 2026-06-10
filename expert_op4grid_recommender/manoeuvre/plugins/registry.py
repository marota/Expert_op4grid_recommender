"""
manoeuvre/plugins/registry.py — Registre des algorithmes pluggables, par phase.

Trois registres indépendants (un par phase de calcul) :

- ``"identificateur"`` : implémentations de ``IdentificateurTopologieDetaillee`` ;
- ``"sequenceur"``     : implémentations de ``SequenceurManoeuvres`` ;
- ``"planificateur"``  : implémentations de ``PlanificateurNodal``.

Un algorithme s'enregistre sous un nom par phase, via le décorateur ::

    from expert_op4grid_recommender.manoeuvre.plugins import register

    @register("sequenceur", "mon_algo")
    class MonSequenceur:
        nom = "mon_algo"
        def sequencer(self, poste, cible, **options): ...

ou par appel direct : ``register("sequenceur", "mon_algo", MonSequenceur)``.
On enregistre une **factory sans argument** (classe ou callable) ; l'instance
n'est créée qu'à la résolution (``get``).

Les paquets externes peuvent aussi publier leurs algorithmes via les *entry
points* du groupe ``expert_op4grid_recommender.manoeuvre`` (chargés
paresseusement au premier ``get``/``disponibles``), le nom de l'entry point
encodant la phase : ``identificateur.mon_algo = mon_pkg.mod:MaClasse``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PHASES = ("identificateur", "sequenceur", "planificateur")

#: Groupe d'entry points (setuptools/pyproject) pour les plugins externes.
ENTRY_POINT_GROUP = "expert_op4grid_recommender.manoeuvre"

_registres: dict[str, dict[str, Callable[[], Any]]] = {p: {} for p in PHASES}
_entry_points_charges = False


def _verifier_phase(phase: str) -> None:
    if phase not in PHASES:
        raise ValueError(f"Phase inconnue '{phase}' (attendu : {PHASES})")


def register(
    phase: str,
    nom: str,
    factory: Optional[Callable[[], Any]] = None,
    *,
    remplacer: bool = False,
):
    """Enregistre ``factory`` (classe ou callable sans argument) comme
    algorithme ``nom`` de la ``phase``. Utilisable en décorateur.

    ``remplacer=False`` (défaut) interdit d'écraser silencieusement un
    algorithme déjà enregistré sous le même nom.
    """
    _verifier_phase(phase)

    def _do(f: Callable[[], Any]):
        if not remplacer and nom in _registres[phase]:
            raise ValueError(
                f"Un algorithme '{nom}' est déjà enregistré pour la phase "
                f"'{phase}' (utilisez remplacer=True pour le substituer).")
        _registres[phase][nom] = f
        return f

    return _do(factory) if factory is not None else _do


def get(phase: str, nom: str) -> Any:
    """Résout et **instancie** l'algorithme ``nom`` de la ``phase``."""
    _verifier_phase(phase)
    _charger_entry_points()
    factory = _registres[phase].get(nom)
    if factory is None:
        raise KeyError(
            f"Aucun algorithme '{nom}' pour la phase '{phase}'. "
            f"Disponibles : {sorted(_registres[phase]) or 'aucun'}.")
    return factory()


def disponibles(phase: Optional[str] = None) -> dict[str, list[str]]:
    """Noms des algorithmes enregistrés, par phase (ou pour une seule phase)."""
    _charger_entry_points()
    if phase is not None:
        _verifier_phase(phase)
        return {phase: sorted(_registres[phase])}
    return {p: sorted(_registres[p]) for p in PHASES}


def _charger_entry_points() -> None:
    """Charge (une fois) les plugins déclarés par entry points. Le nom de
    l'entry point est ``<phase>.<nom>`` ; sa valeur, une factory sans argument."""
    global _entry_points_charges
    if _entry_points_charges:
        return
    _entry_points_charges = True
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - environnement sans metadata
        return
    for ep in eps:
        phase, _, nom = ep.name.partition(".")
        if phase not in PHASES or not nom:
            logger.warning("Entry point '%s' ignoré (attendu '<phase>.<nom>' "
                           "avec phase dans %s)", ep.name, PHASES)
            continue
        try:
            register(phase, nom, ep.load())
        except Exception as exc:  # plugin cassé : ne bloque pas les autres
            logger.warning("Chargement du plugin '%s' impossible : %s",
                           ep.name, exc)
