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

# Pénalité d'un nœud **multi-barres** (dont les SJB couvrent plusieurs jeux de
# barres), par barre supplémentaire. **Dominante** : la *réalisabilité* prime sur
# le nombre de manœuvres. Sur les postes à faisceau de couplage **partagé**
# (un DJ atteignant > 2 barres, ex. SSV.OP7/TAVELP7), la décomposition par paires
# de ``_inter_sjb_couplers`` rend faussement réalisables des nœuds « exotiques »
# (demi-rames croisées de barres différentes, ex. ``{1A,2B}``) que le séquenceur
# ne sait pas réaliser. Cette pénalité oriente le placement vers des nœuds tenant
# sur une seule barre (ou des barres entières couplées), réalisables.
# Appliquée uniquement aux postes > 2 barres → cas 2-JdB strictement inchangés.
#
# ATTENTION — la pénalité étant dominante, elle peut forcer un placement mono-barre
# qui **multiplie les ré-aiguillages** là où un nœud multi-barres *légitime* (barres
# entièrement couplées) serait plus économique. Pour ne jamais payer ce surcoût,
# ``determiner_topo_complete_cible`` réalise AUSSI le placement de **coût brut
# minimal** (pénalité désactivée, ``penaliser_multibarre=False``) et retient
# **transactionnellement** la réalisation vérifiée la moins coûteuse en manœuvres
# (le placement exotique non réalisable n'étant jamais vérifié, il est écarté).
POIDS_NOEUD_MULTIBARRE = 1000

# Garde-fous combinatoires : au-delà, on bascule sur une heuristique (placement
# best-effort borné, puis glouton) plutôt que d'énumérer toutes les affectations.
MAX_COMBINAISONS_PLACEMENT = 500_000        # affectation complète (~k^nb_SJB)
MAX_COMBINAISONS_BEST_EFFORT = 2_000_000    # placement partiel (~(k+1)^nb_SJB)
