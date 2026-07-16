# Benchmark & analyse du recommender sur ses cas d'usage (2026-07-16)

Banc systématique du **recommender** (pipeline d'analyse N-1 → graphe de
surcharge → règles expertes → actions re-simulées) sur deux familles de
cas, avec attention portée à la **profondeur des contraintes** (ρ N-1) et
à **leur nombre**, à la performance **par famille d'action**, à la
typologie des **échecs** (efficacité partielle vs inopérante), et à des
**axes d'amélioration éprouvés** (mesurés, pas seulement proposés).

Scripts : `scripts/benchmark_recommender_cases.py` (grille Dijon),
`scripts/benchmark_tht_france_cases.py` (cas France THT reconstruits),
`scripts/analyze_recommender_benchmark.py` (analyse),
`scripts/test_improvement_axes.py` (axes A/B/C).
Config : celle des tests/production (re-simulation de chaque action,
20 actions priorisées, périmètre `lignes_a_monitorer.csv`), backend
grid2op (env assistant pypowsybl2grid) pour Dijon, backend pypowsybl
pour les cas France.

## 1. Grille de cas n°1 — env Dijon (cas historiques du recommender)

**4 128 cas = 2 journées de chronics (2024-08-28, 2024-11-27) × 48 pas
½ h × 43 défauts N-1** (`lignes_a_deconnecter.csv`) — la couverture
COMPLÈTE de la grille livrée avec le dépôt. 0,66 s méd/cas sans
contrainte, 3,0 s méd/cas contraint.

### 1.1 Profil des contraintes

96 situations contraintes uniques (2,3 % de la grille) : 94 analysées +
2 crashs (§1.4). Profondeur max méd **1,074**, p90 1,192, max **1,317** ;
26 cas multi-surcharges (25×2, 1×3). Par bin de profondeur :

| profondeur N-1 | cas | résolus | partiels | sans effet |
|---|---|---|---|---|
| 1,00-1,05 | 34 | 30 | 0 | 4 |
| 1,05-1,10 | 21 | 18 | 0 | 3 |
| 1,10-1,20 | 30 | 26 | 2 | 2 |
| > 1,20 | 9 | 4 | 3 | 2 |

**La profondeur est LE gradient de difficulté** : 88 % de résolution
sous 1,05, 44 % au-delà de 1,20. Le **nombre** de surcharges joue moins :
2 surcharges se résolvent à 88 % (22/25) contre 82 % (56/68) pour une
seule — le graphe de surcharge agrège bien les contraintes multiples
d'une même poche ; le seul cas à 3 surcharges échoue (antenne, §1.4).

### 1.2 Performance par famille d'action (94 cas analysés)

| famille | présente (cas) | meilleure (cas) | résout (ρ<1) | réduit | seule à résoudre | non-convergences |
|---|---|---|---|---|---|---|
| line_disconnection | 83 | **67** | **75** | 83 | 7 | 1 |
| open_coupling (scission) | 83 | 1 | 54 | 83 | 0 | 3/408 |
| close_coupling (fusion) | 51 | 8 | 35 | 51 | 2 | 0 |
| line_reconnection | 81 | 7 | 43 | 81 | 0 | 0 |
| load_shedding | 93 | 11 | **0** | 3 | 0 | 0 |

- **La déconnexion de ligne domine** : meilleure action dans 71 % des
  cas, résout 90 % des cas où elle figure. C'est l'action qui recâble le
  report de flux à la source (souvent la déconnexion d'une ligne du
  chemin de report, y compris la surcharge elle-même quand le réseau le
  supporte).
- **Les scissions de nœud (open_coupling) réduisent partout** (83/83)
  et résolvent 65 % — mais ne sont presque jamais LA meilleure action ;
  elles sont le levier complémentaire des paires (§3.B).
- **Les fusions de nœud** sont les plus « décisives » quand elles
  existent : 2 cas où ELLES SEULES résolvent.
- **Le délestage ne résout jamais seul en mode nominal** (0/93) — MW
  unitaires trop faibles face à l'excès — et surtout il est INOPÉRANT en
  mode antenne à cause d'un bug de construction (§1.4/§3.D).

### 1.3 Combinaisons (théorème de superposition généralisé)

Sur les 94 cas, la meilleure PAIRE prédite bat la meilleure action seule
dans **83 cas** — le headroom de combinaison est généralisé, pas
marginal. Sur les 5 échecs partiels, la paire prédit ρ<1 dans 5/5 (§3.B
valide par simulation vraie).

### 1.4 Typologie des échecs (16 + 2 crashs sur 96)

- **5 partiels « analyzed »** — défauts profonds (méd 1,24 ; AISERL31MAGNY
  1,24-1,30 et MAGNY* à 2 surcharges) : la meilleure action isolée plafonne
  à ρ 1,00-1,04 (écart méd à la bande : **+3,9 pts**) → *efficacité
  partielle réelle*, résorbée par les paires (§3.B).
- **11 « antenna » sans effet** (BEON L31CPVAN, 2024-11-27, profondeurs
  1,00-1,32) : le défaut isole une poche radiale, le mode antenne propose
  5 délestages… **tous à effet strictement NUL** (ρ identique à 10⁻⁴
  près). Cause racine trouvée : `_load_shedding.py` construit l'action
  avec le dialecte pypowsybl `{"set_load_p": …}` que l'action space
  grid2op **ignore silencieusement** → action vide (le module de
  superposition le détectait déjà : « No-op injection action detected »).
  *Ce n'est pas une efficacité partielle, c'est une recommandation
  inopérante* — corrigé et mesuré en §3.D.
- **2 crashs `error_step2`** (AISERL31MAGNY ts0/ts45) : graphe de
  surcharge **sans boucle rouge** → `red_loops.Path.sum()` d'alphaDeesp
  renvoie un float sur frame vide → `TypeError`. Garde défensive mesurée
  en §3.A. (+1 `error_step1` : divergence de la simulation de défaut,
  cas non exploitable.)

## 2. Grille de cas n°2 — France THT reconstruite (Grid_snapshot_reconstruct)

Cas issus du **mode THT-only 225/400 kV** éprouvé sur 2021-2023 (10
instants canoniques reconstruits, ~1 700 nœuds, limites saisonnières
réelles du fichier), backend **pypowsybl** natif du recommender.
Criblage N-1 DC systématique (~2 050 lignes par cas, 1 400-2 000
contingences contraignantes par instant !) puis banc sur les 12 plus
profondes par cas = **108 runs** — un stress-test volontairement à
l'extrême du spectre, complémentaire de la grille Dijon.

### 2.1 Résultats

- **33 cas analysés** (16 méché + 17 antenne), 23 faux positifs DC
  (l'AC ne confirme pas — criblage DC vs état AC), **47 divergences de
  la simulation AC du défaut** (`error_step1`) et 5 crashs boucles
  vides (le MÊME bug alphaDeesp que Dijon §1.4 — le loop-guard §3.A
  s'applique tel quel).
- Contraintes BEAUCOUP plus dures que Dijon : profondeur méd **1,48**
  (méché) / **1,95** (antenne), max 2,71 ; multi-surcharges la norme
  (méd 3-4, jusqu'à 11 lignes).
- Résolution : **2 résolus** (redispatch), **31 partiels** — meilleures
  réductions partielles −0,99/−0,41 de ρ. Familles : le délestage (sur
  les charges équivalentes THT→HT) est présent dans 28/33 et meilleur
  choix 18/33 ; le redispatch 19/33 (meilleur 11, seul à résoudre) ;
  PST 10/33. **`line_disconnection` n'apparaît JAMAIS** : sur le réseau
  THT réduit, moins maillé, les candidats de déconnexion isoleraient des
  poches (17/33 cas basculent d'ailleurs en mode antenne) — les règles
  expertes les écartent à juste titre.
- **Les paires battent l'action seule dans 30/33 cas** (gains jusqu'à
  −0,35 de ρ) — le headroom de combinaison, déjà généralisé à Dijon,
  est encore plus net quand les actions unitaires plafonnent.

### 2.2 Lecture

Sur des contraintes de cette profondeur, l'efficacité n'est que
**partielle par construction** : sans actions nodales (REPAS national
absent) ni déconnexions viables, le recommender n'a que des leviers
d'injection dont l'amplitude unitaire est petite devant l'excès de flux
(50-300 %). Les échecs ne sont PAS des recommandations inopérantes
(post-correctif §3.D, les injections réduisent réellement) mais un
espace d'actions amputé + des cas hors du domaine nominal. Les 47
divergences AC du défaut montrent aussi qu'un solve N-1 robuste
(init DC → homotopie, comme le banc du mode THT) est un prérequis
d'industrialisation sur réseau national.

## 3. Axes d'amélioration ÉPROUVÉS (mesurés sur les échecs)

| axe | levier | mesure |
|---|---|---|
| **A. loop-guard** | garde défensive sur `get_dispatch_edges_nodes` (graphe sans boucle rouge → chemin de dispatch vide au lieu d'un crash) | **2/2 crashs récupérés**, 1 résolu (ρ 0,67), 1 partiel (1,17) |
| **B. paires validées** | la meilleure paire GST re-simulée en vrai (composition d'actions grid2op) sur les 5 partiels | **5/5 sauvés** (ρ simulé 0,85-0,97) ; biais de prédiction GST remarquablement stable : **+0,042 à +0,050** (la GST sous-estime ρ d'~4,7 pts, à recaler) |
| **C. budget** | N_PRIORITIZED_ACTIONS 20→40 + minima par famille | 2/16 résolus en plus (AISER ts43/ts44 → ρ 0,82/0,90 : le top-20 COUPAIT des actions gagnantes), 3 améliorés ; antennes inchangées (normal) |
| **D. correction délestage antenne** | construire le shed en `set_bus -1` du load quand l'action grid2op issue de `set_load_p` est sans impact (fix dans `_load_shedding.py`, testé) | **10/11 cas antenne RÉSOLUS** (ρ 0,57-0,85), le 11ᵉ amélioré (1,20→1,03) |

**Bilan cumulé** : résolution 78/96 (81 %) au banc nominal →
**94/96 (98 %)** avec A+B+C+D. Les 2 restants : BEON ts46 (1,20→1,03,
partiel) et AISER ts45 (crash récupéré à 1,17, partiel) — tous deux
candidats aux paires (non re-mesurés en combinaison après fix).

## 4. Recommandations (au-delà des 4 axes mesurés)

1. **Promouvoir les paires GST au rang de recommandation de premier
   ordre** quand aucune action seule ne passe sous la bande : 5/5 des
   partiels sauvés, biais de prédiction stable donc corrigeable
   (+0,05 conservatif) ; la re-simulation vraie de la meilleure paire ne
   coûte qu'un load flow.
2. **Corriger la construction du délestage** (fait ici pour le chemin
   grid2op) et **dimensionner le délestage en MW requis** plutôt qu'en
   charges unitaires : au banc, `P_shedding` est calculé puis ignoré
   dans l'action (l'action déleste tout le transformateur).
3. **Écarter le faux dilemme budget/rang** : passer à 40 actions
   priorisées coûte ~2× la re-simulation mais récupère des solutions
   réelles coupées par le top-20 (2 cas) — ou mieux, garantir par
   famille un candidat re-simulé (les minima par famille à 0 en config
   test sont un piège).
4. **Guard alphaDeesp** à remonter upstream (frame de boucles vide) —
   confirmé sur les DEUX bancs (2 cas Dijon + 5 cas France THT).
5. **Espace d'actions REPAS pour RTE7000** : sans lui, le recommender
   national (cas France THT) n'a ni scission ni fusion de nœud — les
   familles qui portaient 65 % et 8 « meilleurs choix » au banc Dijon —
   et il ne résout que 2/33 des cas profonds nationaux.
6. **Solve N-1 robuste dans le backend pypowsybl** (init DC → repli
   homotopie) : 47/108 runs France THT meurent à la simulation du
   défaut avant toute recommandation.
7. Le champ `is_rho_reduction` des paires GST est incohérent avec leur
   `max_rho` (des paires gagnantes marquées False) — signalé, à unifier
   avec la sémantique des actions simples.

## 5. Reproduction

```bash
# venv: requirements.txt + pypowsybl2grid==0.3.0 (numpy<2)
python scripts/benchmark_recommender_cases.py --out out_benchmark/cases.jsonl --ts-step 1 --jobs 3
python scripts/analyze_recommender_benchmark.py --journal out_benchmark/cases.jsonl --json out_benchmark/analysis_dijon.json
python scripts/test_improvement_axes.py --journal out_benchmark/cases.jsonl --axis loop-guard pairs budget
python scripts/benchmark_tht_france_cases.py --per-case 12   # cas France THT (xiidm du mode THT-only)
```

Artefacts : `out_benchmark/{cases.jsonl, analysis_dijon.json, axes.json,
axis_shed_fix.json, tht_screening.json, cases_tht_france.jsonl}`.
