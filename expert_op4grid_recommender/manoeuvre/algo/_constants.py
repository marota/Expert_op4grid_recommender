"""
manoeuvre/algo/_constants.py — paramètres de placement (#9).
"""

# ---------------------------------------------------------------------------
# Paramètres de la recherche de placement (phases 2.2-2.4)
# ---------------------------------------------------------------------------
# Poids du coût d'une affectation nœud -> SJB. Un **ré-aiguillage** (déplacer un
# départ d'une barre à l'autre, manœuvre lourde) est plus cher qu'une simple
# manœuvre de **coupler** ; **ouvrir un sectionnement** (organe hors charge,
# dé-énergisation préalable) est pénalisé plus qu'ouvrir un couplage (DJ).
POIDS_REAIGUILLAGE = 5
POIDS_MANOEUVRE_COUPLER = 1
POIDS_OUVERTURE_SECTIONNEMENT = 4

# Garde-fous combinatoires : au-delà, on bascule sur une heuristique (placement
# best-effort borné, puis glouton) plutôt que d'énumérer toutes les affectations.
MAX_COMBINAISONS_PLACEMENT = 500_000        # affectation complète (~k^nb_SJB)
MAX_COMBINAISONS_BEST_EFFORT = 2_000_000    # placement partiel (~(k+1)^nb_SJB)
