"""Absence de boucle rouge : la découverte d'actions ne doit pas lever.

``Structured_Overload_Distribution_Graph.get_dispatch_edges_nodes(
only_loop_paths=True)`` (alphaDeesp) calcule ``list(set(red_loops.Path.sum()))``.
Sur un DataFrame de boucles rouges **vide**, ``.sum()`` de pandas retourne le
scalaire ``0.0`` (``numpy.float64``) au lieu d'une liste concaténée, et
``set()`` lève ``TypeError: 'numpy.float64' object is not iterable``.

Le cas n'est pas propre aux poches radiales : toute situation sans boucle rouge
parallèle le déclenche (mesuré : 84 occurrences sur 901 contingences graduées
du jeu de données Matpower, où la découverte d'actions était alors amputée).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("red_loops, doit_lever", [
    (pd.DataFrame({"Path": [[1, 2], [2, 3]]}), False),   # cas nominal
    (pd.DataFrame({"Path": []}), True),                  # aucune boucle rouge
    (pd.DataFrame({"Path": [np.nan]}), True),            # colonne non concaténable
])
def test_le_defaut_amont_est_bien_celui_qu_on_croit(red_loops, doit_lever):
    """Caractérise le défaut d'alphaDeesp que le garde-fou contourne."""
    if doit_lever:
        with pytest.raises(TypeError, match="not iterable"):
            list(set(red_loops.Path.sum()))
    else:
        assert set(list(set(red_loops.Path.sum()))) == {1, 2, 3}


def _sans_boucle_rouge(red_loops) -> bool:
    """Réplique la condition du garde-fou de ``_orchestrator``."""
    return (
        red_loops is None
        or getattr(red_loops, "empty", True)
        or "Path" not in getattr(red_loops, "columns", ())
    )


@pytest.mark.parametrize("red_loops", [
    None,
    pd.DataFrame({"Path": []}),
    pd.DataFrame({"autre": [1, 2]}),
])
def test_le_garde_fou_intercepte_avant_l_appel(red_loops):
    """Les états qui font lever alphaDeesp sont détectés en amont."""
    assert _sans_boucle_rouge(red_loops) is True


def test_le_garde_fou_laisse_passer_le_cas_nominal():
    """Un DataFrame de boucles rouges exploitable n'est pas court-circuité :
    la découverte des chemins de dispatch en boucle reste faite."""
    assert _sans_boucle_rouge(pd.DataFrame({"Path": [[1, 2]]})) is False
