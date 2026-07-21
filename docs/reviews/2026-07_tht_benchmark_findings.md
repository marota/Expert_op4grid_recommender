# Constats du banc RTE7000-THT (2026-07) — recommandeur sur 10 situations réelles reconstruites

Synthèse des constats mesurés sur le banc d'essai national : 10 situations 2021-2023 reconstruites
par `grid_snapshot_reconstruct` en mode **THT-only** (225/400 kV, ~1 600 bus), criblage N-1 exhaustif
≥ 380 kV (346–407 contingences/cas), découverte complète sur toutes les contingences actionnables.
Corpus final : **201 contingences notées sur 7 cas exploitables** (3 cas réfractaires au solveur par
défaut). Banc, scripts et données brutes : `marota/Grid_snapshot_reconstruct`, branche
`claude/expert-op4grid-performance-t1pdrq`, `benchmarks/expert_op4grid_recommender/`
(rapport complet §13–15).

## 1. Résultats de tête

| Indicateur | Mesure |
|---|---|
| Résolutions (action seule) | **64/201 (32 %)** — dont **94 % par la topologie seule** |
| + paires validées par simulation vraie | **72/201 (36 %)** |
| Régime nominal (ρ ≤ 1.05) | **97 %** de résolution |
| Vitesse | criblage 0.3–0.8 s/contingence ; découverte ~2–4 s (vs 22–84 s réseau complet) |

## 2. Les deux gradients de difficulté

- **Profondeur** : 97 % résolus sous ρ 1.05 → 29 % (1.05–1.20) → 17 % (1.20–1.50) → **0 % au-delà
  de ρ 1.5** (aucune action *unitaire* n'y suffit ; tout le résiduel y est partiel).
- **Nombre de surcharges** (fort sur réseau réduit) : 81 % à 1 surcharge → ~3–9 % dès 2 → 0 % à 7+.

## 3. L'espace d'actions topologique change le verdict (finding majeur)

Les familles `line_disconnection` et `open_coupling` (split de nœud) consomment `dict_action` : sans
espace prédéfini elles tournent à vide, et le délestage paraît « résolveur n°1 » **par défaut de
concurrence**. Avec un espace généré des équipements réels du réseau (~420 ouvertures de coupleurs
`{VL}COUPL…DJ` + ~1 800 déconnexions ≥ 220 kV par réseau, format Co-study4grid) :

- résolutions 54 → **64/201** (+10, 0 perdue) ;
- **39 des 43 résolutions « délestage » se requalifient** : une action topologique gratuite résout
  aussi ; l'injection coûteuse n'est *indispensable* que sur **4** contingences résolues ;
- le **split de nœud devient le 1er résolveur** (21 meilleures-et-résolvantes) et la **déconnexion
  trouve des cibles valides** sur réseau réduit (9) — le « jamais candidate » observé à espace vide
  mesurait l'absence d'espace, pas la physique ;
- le **redispatch unitaire est un faux ami** : 90× « meilleure action » (score), **0 résolution**
  (amplitude 10 MW) — il gagne la priorisation des cas partiels sans jamais conclure.

**Doctrine mesurée** : trier la priorisation par **tier de coût avant le score** — topologie
(reconnexion, déconnexion, split, fusion, TD) d'abord ; injections (délestage, curtailment,
redispatch) en **dernier recours**, réservées aux 4 cas sans solution topologique et au mode antenne.

## 4. Taxonomie des 137 non-résolus

| Nature | Volume | Caractérisation |
|---|---:|---|
| Mode antenne (injections seules) | **112** | poche isolée : pas de topologie possible par construction ; l'amplitude unitaire ne mord pas → délestage **dimensionné en MW** requis |
| Efficacité partielle hors antenne | **25** | tous « partial » ; écart médian de la meilleure topo au seuil : +0.11 |
| dont résolubles par **paire** (validé simu) | 8 | cf. §5 |
| dont « mur Paluel » | 4 | ρ 1.40 / 5 surcharges ; le split fait −0.19 (meilleure réduction du corpus) mais il faut un 2ᵉ étage |
| dont plafond PST | 4 | la meilleure « topo » est un cran de TD quasi sans effet (motif 5-surcharges) |
| dont frôlent le seuil | 7 | best topo ∈ [1.005, 1.05[ — un délestage dimensionné (dizaines de MW) ou une paire suffirait |

## 5. Paires (superposition) : puissantes, mais à valider par simulation vraie

- **8 contingences ne sont résolubles QUE par une combinaison** (aucune action seule < 1.0 ; paire
  validée en simulation à ρ 0.976–0.999) : CRENEL71REVIG, AVOI5-G.AVO/CHINX ×4, BARNAL-ROUGE ×2,
  REALTL72TAVEL. 89 contingences ont une paire TOPO+TOPO, 74 prédites résolvantes.
- Biais de prédiction GST : **méd. +0.05, stable** sur les deux campagnes → recalable.
- **MAIS** : 4 sur-promesses massives observées (prédit 0.40–0.82 → réel 1.00–1.16), toutes sur des
  paires contenant une **déconnexion/split à fort re-routage** — la superposition linéarisée
  sur-promet hors de son domaine. **La validation par simulation vraie avant recommandation de paire
  est indispensable** (elle a écarté les 4 faux positifs sur le banc, à ~1 LF par paire).

## 6. Robustesse & performance (correctifs livrés sur cette branche)

- **Garde « zéro boucle rouge »** (issue ExpertOp4Grid_marota#2) : 20 crashs `TypeError` évités sur le
  banc, dont 14 devenus des résolutions (bug amont : `Structured_Overload_Distribution_Graph.get_dispatch_edges_nodes`,
  `red_loops.Path.sum()` sur DataFrame vide). **Corrigé en amont dans expertop4grid 0.3.3** (garde
  explicite du cas sans boucle).
- **Optimisation Dijkstra alphaDeesp** (issue ExpertOp4Grid_marota#1) : −27 % sur le hotspot du graphe,
  sortie bit-identique ; bug amont documenté — le poids `capacity` est silencieusement ignoré par les
  callables sur multigraph. **Intégrée en amont dans expertop4grid 0.3.3** (poids d'arête pré-calculé,
  quirk multigraphe « blessed » par défaut / corrigeable via `capacity_weighted=True`) ; le patch
  vendu localement (`patched_alphadeesp.py`) est donc **supprimé** et le plancher passe à `>=0.3.3`.
- **Flag `USE_DC_FOR_OVERFLOW_GRAPH`** (opt-in, défaut AC préservé) : −59 % sur la construction du
  graphe d'un point raide, même `best max_rho`.
- **Solve N-1 robuste = prérequis** : 3 cas restent réfractaires au solveur par défaut et
  `ete_pic_2021` perd 239 sims N-1 avant recommandation ; le mode hybride (défaut d'abord, solveur
  étagé de `grid_snapshot_reconstruct` en fallback) débloque 9/10 bases — sans jamais substituer le
  modèle relâché quand le défaut converge (la classification d'actionnabilité y est sensible).

## 7. Indicateur de fiabilité des situations d'entrée

Une situation réelle ne présente (quasi) pas de surcharges avant contingence. Le compte de
**surcharges de base** discrimine les reconstructions : 1–21 (DC) sur THT contre 17–259 sur le réseau
complet (la saturation venait de la reconstruction HT). Cas signalés à raffiner côté données :
`hiver_pic_2023`, `ete_creux_2023` ; les 3 cas réfractaires au solveur sont corrélés à l'indicateur.
