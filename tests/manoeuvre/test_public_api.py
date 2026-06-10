"""
tests/manoeuvre/test_public_api.py
----------------------------------
**Verrou de surface d'API publique** du module ``manoeuvre``.

Filet de sécurité posé *avant* l'éclatement d'``algo.py`` en sous-modules
(item #7 de la revue) : un refactor structurel **déplace** des symboles entre
fichiers. Ce test garantit que **tout** ce que ``__init__`` réexporte reste
importable et appelable depuis le package *et* depuis son chemin de sous-module,
de sorte qu'un oubli de réexport casse immédiatement la CI plutôt qu'un client.

N'exige ni pypowsybl ni fixtures (pur import).
"""

from __future__ import annotations

import importlib

import pytest


PKG = "expert_op4grid_recommender.manoeuvre"

# Contrat **explicite** (et non dérivé de __all__) : la liste des symboles que le
# module s'engage à exposer. Dupliquer __all__ ici est volontaire — si un
# refactor retire un symbole de __all__, la divergence est détectée.
EXPECTED_PUBLIC = {
    # models
    "NodeType", "EquipmentType", "SwitchKind", "CelluleType",
    "NodeAttrs", "EdgeAttrs",
    # graph
    "build_vl_graph", "get_node_attrs", "get_edge_attrs",
    "busbar_nodes", "equipment_nodes", "TopologyError",
    # cellules
    "SwitchInfo", "CelluleDepart", "CelluleCouplage", "CellulesVL",
    "detecter_cellules", "calculer_connected_busbars",
    # troncons
    "Troncon", "Tronconnement", "construire_tronconnement",
    # topologie
    "DepartInfo", "NoeudElectrique", "TopologieNodale", "PosteTopologique",
    "attribuer_noeuds",
    # algo
    "Manoeuvre", "ResultatManoeuvres", "determiner_topo_complete_cible",
    "determiner_manoeuvres_avec_sections", "determiner_manoeuvres_cible_detaillee",
    "sectionneurs_sous_charge_par_manoeuvre",
    "ouvrages_simultanement_hors_tension",
    # plugins (phases de calcul pluggables)
    "CibleDetaillee", "ResultatIdentification", "ResultatPlanification",
    "IdentificateurTopologieDetaillee", "SequenceurManoeuvres",
    "PlanificateurNodal", "PlanificateurTopologie", "verifier_sequence",
}

# Sous-modules d'origine de chaque symbole (chemin de réimport attendu).
SYMBOL_SUBMODULE = {
    "NodeType": "models", "EquipmentType": "models", "SwitchKind": "models",
    "CelluleType": "models", "NodeAttrs": "models", "EdgeAttrs": "models",
    "build_vl_graph": "graph", "get_node_attrs": "graph", "get_edge_attrs": "graph",
    "busbar_nodes": "graph", "equipment_nodes": "graph", "TopologyError": "graph",
    "SwitchInfo": "cellules", "CelluleDepart": "cellules",
    "CelluleCouplage": "cellules", "CellulesVL": "cellules",
    "detecter_cellules": "cellules", "calculer_connected_busbars": "cellules",
    "Troncon": "troncons", "Tronconnement": "troncons",
    "construire_tronconnement": "troncons",
    "DepartInfo": "topologie", "NoeudElectrique": "topologie",
    "TopologieNodale": "topologie", "PosteTopologique": "topologie",
    "attribuer_noeuds": "topologie",
    "Manoeuvre": "algo", "ResultatManoeuvres": "algo",
    "determiner_topo_complete_cible": "algo",
    "determiner_manoeuvres_avec_sections": "algo",
    "determiner_manoeuvres_cible_detaillee": "algo",
    "sectionneurs_sous_charge_par_manoeuvre": "algo",
    "ouvrages_simultanement_hors_tension": "algo",
    "CibleDetaillee": "plugins", "ResultatIdentification": "plugins",
    "ResultatPlanification": "plugins",
    "IdentificateurTopologieDetaillee": "plugins",
    "SequenceurManoeuvres": "plugins", "PlanificateurNodal": "plugins",
    "PlanificateurTopologie": "plugins", "verifier_sequence": "plugins",
}


def test_all_matches_expected_contract():
    mod = importlib.import_module(PKG)
    assert set(mod.__all__) == EXPECTED_PUBLIC, (
        "La surface publique (__all__) a divergé du contrat attendu ; "
        "mise à jour intentionnelle ? alors ajuster EXPECTED_PUBLIC."
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_PUBLIC))
def test_symbol_importable_from_package(name):
    mod = importlib.import_module(PKG)
    assert hasattr(mod, name), f"{name} absent du package {PKG}"
    assert getattr(mod, name) is not None


@pytest.mark.parametrize("name", sorted(EXPECTED_PUBLIC))
def test_symbol_reachable_from_origin_submodule(name):
    """Chaque symbole reste importable depuis son sous-module d'origine et y
    est **le même objet** que celui réexporté (pas une copie/redéfinition)."""
    sub = importlib.import_module(f"{PKG}.{SYMBOL_SUBMODULE[name]}")
    pkg = importlib.import_module(PKG)
    assert hasattr(sub, name), f"{name} absent de {SYMBOL_SUBMODULE[name]}"
    assert getattr(sub, name) is getattr(pkg, name)


def test_public_entrypoints_are_callable():
    mod = importlib.import_module(PKG)
    for fn in ("build_vl_graph", "detecter_cellules", "construire_tronconnement",
               "attribuer_noeuds", "determiner_topo_complete_cible",
               "determiner_manoeuvres_avec_sections",
               "determiner_manoeuvres_cible_detaillee",
               "sectionneurs_sous_charge_par_manoeuvre"):
        assert callable(getattr(mod, fn)), f"{fn} doit être appelable"


def test_private_checker_alias_preserved():
    """L'alias privé historique du vérificateur reste disponible (compat. tests
    et imports antérieurs à la publication, cf. #4)."""
    algo = importlib.import_module(f"{PKG}.algo")
    assert algo._sectionneurs_sous_charge_par_manoeuvre is \
        algo.sectionneurs_sous_charge_par_manoeuvre
