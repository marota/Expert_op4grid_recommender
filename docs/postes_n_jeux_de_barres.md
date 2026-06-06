# Postes à N jeux de barres (≥ 3 JdB) — état & reste à faire

> Extension du module `manoeuvre` des postes à **2 jeux de barres** vers **N JdB**
> (3 et plus). Voir aussi `expert_op4grid_recommender/manoeuvre/CLAUDE.md`
> (détails internes) et `docs/manoeuvre_ihm.md` (interface de test).
>
> Suivi de la PR *« Postes à N jeux de barres : placement, séquenceur
> connectivité-based et IHM »*.

## Contexte

Le module `manoeuvre` calcule la séquence de manœuvres réalisant une topologie
nodale cible dans un poste **node-breaker**. Il était historiquement limité aux
**postes à 2 jeux de barres** : le placement nœud → section de barre était bridé
aux 2 barres principales (les nœuds touchant une 3ᵉ barre étaient laissés à
l'opérateur). Cette extension le généralise à **3 barres et plus**, en
**réutilisant** la primitive 2-JdB plutôt qu'en la dupliquant.

Sur le réseau **France 28/08/2024**, ce n'est pas un cas marginal :
**59 postes 400 kV sur 562 ont ≥ 3 barres** (36 à 3 JdB, 17 à 4, jusqu'à 8).

## Postes de référence (fixtures)

7 postes 400 kV à 3 JdB réels, extraits dans `tests/manoeuvre/fixtures/` :

> `SSV.OP7`, `TAVELP7`, `TRI.PP7`, `ARGOEP7`, `CHESNP7`, `COR.PP7`, `CERGYP7`

Topologie type : **3 barres × 2 demi-rames (A/B)**, 6 SJB, et des **faisceaux de
couplage partagés** — un même disjoncteur (ex. `COUPL.A`, `LIAIS`) atteint
plusieurs barres par sélection de sectionneurs. C'est ce partage qui met en
défaut la décomposition par paires de `_inter_sjb_couplers` et a motivé le
redesign ci-dessous.

## Ce qui est fait

### 1. Placement N-barres — `algo/placement.py`, `algo/_constants.py`
- **Retrait du scoping « 2 JdB »** : `_placement_automatique` tourne sur **toutes**
  les SJB ; recherche exacte (`_recherche_exhaustive`, lex-min).
- **Décomposition récursive** `_placement_decompose` au-delà du garde-fou
  combinatoire : par **composantes connexes** du graphe de couplage (exacte,
  séparable) puis **bissection au niveau barre** (best-effort).
- **Pénalité `POIDS_NOEUD_MULTIBARRE`** (gated > 2 barres) : oriente le placement
  vers des nœuds **mono-barre / barres entières**, en évitant les nœuds
  « exotiques » (demi-rames croisées) faussement réalisables via les faisceaux
  partagés. **Comportement 2-JdB strictement préservé.**

### 2. Séquenceur bay-aware — `algo/sequencing.py`, `algo/targets.py`
- **Phase 0** (`bridge_breakers`) : ne ferme pas un coupler « même nœud » via un
  organe partagé avec une liaison inter-nœuds (pas de re-pontage de barres).
- **Phase F** : ouverture du **lot minimal** de DJ de couplage par **connectivité
  réelle**, **transactionnelle** (annulée si la cible n'est pas exacte) →
  *no-op* sur les postes 2 barres.
- **`determiner_manoeuvres_par_connectivite`** : réalisateur connectivité-based
  (ré-aiguillage avec maintien en place + sectionnements intra-barre par état
  direct + séparation/fusion par connectivité). Branché **transactionnellement**
  par `determiner_topo_complete_cible` : retenu seulement s'il vérifie
  exactement la cible → **ne peut jamais dégrader** un résultat correct.

### 3. IHM — `scripts/manoeuvre_ihm.py`, `…_assets/index.html`
- Les 7 postes 3-JdB sont **épinglés** (`POSTES_TEST`).
- `Session.all_postes` = **tous** les VL NODE_BREAKER ; champ de **recherche**
  pour inspecter/tester n'importe quel poste de la situation.
- Endpoint **`POST /api/load_grid`** : charge dynamiquement une autre situation
  `.xiidm` sans relancer le serveur.

### Couverture vérifiée (cibles 3 ET 4 nœuds, manœuvres opérationnelles)

| Manœuvre | Nœuds | Postes vérifiés |
|---|---|---|
| **Séparer les barres couplées** | 3 à 7 | 6/7 (skip documenté : `SSV.OP7` à 2 nœuds, barre de réserve) |
| **Tronçonnage** (demi-rames) | 4 à 8 | **7/7** |

Tests : `tests/manoeuvre/test_ssv_op7_3jdb.py`, `test_placement_decomposition.py`,
`test_postes_3barres_400kv.py`, `test_ihm_postes_3barres.py`.
**Suite `manoeuvre/` : 761 passed, 0 régression** (30 goldens intacts, dont
`MORBRP6` 4-barres). Les 6 échecs résiduels (`goldens/*_cible_2noeuds*` absents)
sont **pré-existants sur `main`** et indépendants.

## Reste à faire

### Séquenceur (raffinements)
- **Reconnexion de départs déconnectés** : une cible regroupant des départs déjà
  hors-service exige une *mise en service* (DJ + SA) que le réalisateur ne fait
  pas — les regroupements arbitraires (round-robin) ne sont donc pas tous réalisés.
- **Contrôle SA-par-SA des faisceaux partagés** : le cas dégénéré « 2 nœuds +
  barre de réserve fusionnée » (`SSV.OP7` *séparation*) exige d'ouvrir/fermer des
  SA de faisceau **individuels** ; le réalisateur ne bascule que les DJ en bloc.
  (1 test skippé.)

### Modèle de données
- **`CelluleCouplage`** n'enregistre encore que **2 SJB** par composante de
  couplage (`cellules.py:474`, warning) — cosmétique : le séquenceur utilise
  `_inter_sjb_couplers`, pas `CelluleCouplage`.
- **Bug de nommage `RAN.PP6` / `RAN_PP6`** : `troncons._barres_par_nommage`
  sous-compte la fixture à 2 barres au lieu de 4 (mismatch point/underscore entre
  l'id du VL et le *stem* de la fixture).

### Discovery — chaînon **end-to-end** manquant pour le recommandeur
- `discovery/_node_splitting.py` est **binaire** (`buses = [1, 2]`, un seul
  `bus_of_interest`) : le recommandeur ne *propose* pas encore de cible 3 nœuds —
  il ne saurait que l'*exécuter*.
- `discovery/_node_merging.py` code en dur un collapse total
  (`topo_target = [1 if bus_id >= 2 else bus_id …]`) → fusions **par paires** à
  introduire.
- Alimenter `dict_action` en actions **3-bus** (le backend supporte déjà
  `n_busbar_per_sub ≥ 3`).

### IHM (ergonomie)
- **Upload `.xiidm` côté client** : `/api/load_grid` prend pour l'instant un
  chemin **côté serveur**.
- **Filtres de recherche** (par tension P3/P6/P7, par nombre de barres) sur les
  milliers de postes d'une situation complète.

## Commandes

```bash
# Tests du module (postes N barres inclus)
pytest tests/manoeuvre/ -q

# IHM : inspecter / modifier / tester n'importe quel poste d'une situation
python scripts/manoeuvre_ihm.py --grid /chemin/vers/grid.xiidm   # http://localhost:8000
```
