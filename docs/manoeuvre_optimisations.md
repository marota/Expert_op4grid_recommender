# Module Manœuvre — Revue de code & campagne d'optimisation

> Rapport de la revue critique (architecture / qualité / performance) du module
> `manoeuvre` et de son IHM, des **filets de sécurité** posés avant refactor, et
> des **refactors à iso‑comportement** réalisés, avec les **gains mesurés**.
>
> Doc connexes : [`manoeuvre_module.md`](manoeuvre_module.md) ·
> [`manoeuvre_regles.md`](manoeuvre_regles.md) ·
> [`manoeuvre_ihm.md`](manoeuvre_ihm.md) ·
> [`../expert_op4grid_recommender/manoeuvre/CLAUDE.md`](../expert_op4grid_recommender/manoeuvre/CLAUDE.md)

---

## 1. Méthode

Chaque refactor a suivi la même discipline :

1. **Diagnostic mesuré** : profilage (`cProfile`) plutôt que lecture statique
   seule — à deux reprises le profil a corrigé l'intuition.
2. **Filet de sécurité d'abord** : tests posés et **verts sur le code actuel**
   *avant* de toucher au code, de sorte que tout écart de comportement soit
   détecté immédiatement.
3. **Refactor à iso‑comportement**, validé par les filets (notamment des
   *goldens* comparés **octet pour octet**), puis commité isolément.
4. **Gain mesuré** en A/B (commit pré‑refactor vs après).

---

## 2. Diagnostic initial (revue critique)

### Architecture
- `algo.py` est un module « god » (~2 250 lignes, 40+ fonctions, 4 points
  d'entrée) mêlant placement, séquencement, vérification et helpers bas niveau.
- **Pas d'index** : `_is_open` / `_set_switch` / `_eq_node` retrouvaient un
  organe/nœud par **scan linéaire** du graphe (82 sites d'appel).
- `_inter_sjb_couplers` **recalculé ~10×/analyse** (all‑pairs SJB + retraits
  d'arêtes), alors qu'il ne dépend que de la topologie.
- IHM : god‑file mêlant Flask, rendu pypowsybl, parsing SVG et ~600 lignes de
  HTML/JS embarqué ; franchit la frontière privée (`_sectionneurs_sous_charge_…`).

### Qualité
- Modes d'échec **silencieux** (`_set_switch` no‑op, `_is_open`→True sur id
  inconnu).
- Vérificateurs rejouant chacun la séquence (3–4 rejeux).
- Constantes magiques de coût/seuils non nommées ; imports locaux dispersés.

### Performance
- **O(E) dans des boucles combinatoires** : `_placement_automatique` énumère
  k^(nb SJB) affectations en testant `nx.is_connected` à répétition.
- Reconstructions de graphes et **recalculs de chemins structurels** dans des
  boucles chaudes.

---

## 3. Filets de sécurité (posés avant tout refactor)

| Filet | Fichier | Rôle |
|---|---|---|
| **Déterminisme** | `algo.py` (`a5c7de6`) | Canonicalisation de l'ordre d'émission (itérations d'ensembles triées) — **prérequis** pour qu'un golden d'ordre exact soit stable d'un process à l'autre (`PYTHONHASHSEED`). |
| **T1 — Golden** | `tests/manoeuvre/test_golden_sequences.py` | Caractérisation de bout en bout sur `scenarios/*.json` × {smooth, aggressive} (24 goldens) : séquence ordonnée exacte + vérifications + écarts + partition. Régénération consciente via `UPDATE_GOLDENS=1`. |
| **T2 — Vérificateurs exacts** | `tests/manoeuvre/test_verificateurs_exact.py` | Égalité de liste exacte (alignement, indices d'infraction, dédup) + idempotence + non‑mutation du graphe. |
| **T3 — Contrat lookups** | `tests/manoeuvre/test_lookup_helpers.py` | Contrat de `_is_open`/`_set_switch`/`_eq_node` (id inconnu, validité sur copie, « tout switch_id émis existe »). |
| **T4 — Invariants couplers** | `tests/manoeuvre/test_couplers_memoisation.py` | Préconditions du cache : invariance à l'état des organes, idempotence, non‑mutation de `poste.graph` par le pipeline. |

> Le golden a immédiatement révélé une **nondéterminisme réelle** (ordre de blocs
> de dé‑énergisation dépendant du hash) — d'où le fix `a5c7de6` *avant* les
> refactors.

---

## 4. Refactors réalisés (à iso‑comportement)

| # | Commit | Diagnostic | Action | Garantie |
|---|---|---|---|---|
| **#1** | `bda3234` | scans O(E) des lookups | index `switch_id→arête` / `equipment_id→nœud` mémoïsés sur `G.graph` (valides sur copies : coordonnées topologiques) | contrat id‑inconnu préservé (T3) |
| **#2** | `90cd0c2` | `_inter_sjb_couplers` recalculé ~10× | mémoïsation sur le poste (invariant à l'état — T4) | goldens inchangés |
| **#3** | `ba67820` | 3 rejeux des vérificateurs | passe unique `_rejeu_securite` + agrégateur `_verifier_regles` ; fonctions publiques = délégateurs | sortie exacte (T1+T2) |
| **P1** | `b37ed42` | `nx.is_connected` recalculé ~k^n fois | connexité **mémoïsée** par sous‑ensemble + hoist de l'invariant `currently_closed` | même ordre, même tie‑break |
| **P1+** | `ff6a14f` | énumération k^n filtrée sur la connexité | génération **directe** des partitions connexes × bijections ; **tie‑break lex‑min** reproduisant `product` | ensemble de candidats identique → coût min identique ; sortie prouvée identique |
| **Q1** | `12de96b` | garde‑fou d'index `number_of_edges()` O(E) ; `_edges_of_switches` O(E) ; `disconnectors_vers_barre` recalculé 27 k× | garde O(1) (`number_of_nodes`) ; index ; **mémoïsation** du chemin SA structurel sur la cellule | structurel/invariant → sortie identique |

### Note méthodologique
Pour **P1** comme pour **Q1**, le **profil a corrigé la lecture statique** :
- la « recherche combinatoire » était bien le coût dominant des gros postes
  (P1), mais le hotspot transverse réel n'était **pas** les reconstructions de
  graphes (`_equipotentiel`…), négligeables au profilage, **mais** le
  recalcul d'un **plus‑court‑chemin structurel** (`disconnectors_vers_barre`)
  appelé des dizaines de milliers de fois (Q1) ;
- le garde‑fou O(E) introduit par **#1** lui‑même (`number_of_edges()`) annulait
  une partie de son bénéfice — corrigé en Q1.

---

## 5. Gains mesurés

A/B sur fixtures (moyenne sur N analyses, `networkx` réel) :

| Cas | pré‑#1 | post #1‑#3 | P1 | P1+ | Q1 |
|---|---|---|---|---|---|
| **SSAVOP3** (8 SJB, 62 manœuvres) | 1201 ms | 1184 ms | 200 ms | 76 ms | **44 ms** |
| **PALUNP3** (4 SJB) | 22,9 ms | 17,2 ms | 15 ms | 20 ms | **10,5 ms** |
| **Corpus** (12 scénarios × 2 modes) | — | — | — | 349,6 ms | **232,7 ms** |

- **SSAVOP3 : ~27×** (1201 → 44 ms) — porté par P1/P1+ (placement) puis Q1.
- **Corpus : ~1,5×** sur le seul Q1 (mémoïsation du chemin SA, transverse).
- **PALUNP3 : ~2,2×** — porté par les lookups O(1) (#1 + garde Q1).

---

## 6. Garanties d'iso‑comportement

- **Goldens octet‑pour‑octet** sur 12 scénarios × 2 modes, **stables sur ≥ 3
  graines de hash** — rejoués après *chaque* refactor.
- **P1+** : l'énumération par partitions connexes explore **le même ensemble de
  candidats** que `itertools.product` filtré (donc même coût minimal) ; un
  **tie‑break lex‑min** reproduit le choix de `product` même en cas d'égalité de
  coût hors corpus → équivalence **prouvée sur toute entrée**, pas seulement
  constatée.
- Mémoïsations (#2, Q1) : portées par des **invariants vérifiés par tests**
  (indépendance à l'état des organes, non‑mutation des structures).

---

## 7. Reste à faire / limites

- Le coût résiduel des gros postes est l'**énumération combinatoire** de
  `_placement_automatique` (intrinsèque) ; la franchir demanderait un placement
  non exhaustif (heuristique/branch‑and‑bound), qui **pourrait** changer la
  sortie — non entrepris pour préserver l'iso‑comportement.

### Qualité — également traité

- **#9 — constantes nommées + imports remontés** (`algo.py`, `manoeuvre_ihm.py`) :
  poids de coût et garde‑fous combinatoires extraits en constantes documentées ;
  `itertools`/`Counter`/`re` remontés en tête de module.
- **#10 — qualité en CI** : job `quality` (CircleCI) — `ruff` (lint, **bloquant**),
  `interrogate` (docstrings ≥ 80 %, **bloquant**), `radon` (complexité,
  indicatif). **Scopé au module `manoeuvre`** (maintenu propre) ; configuration
  dans `pyproject.toml` (`[tool.ruff]`, `[tool.interrogate]`). Le reste du dépôt
  (~300 violations `ruff`) reste à résorber avant d'élargir le périmètre.

### Items de la revue encore ouverts

Éclatement d'`algo.py` en sous‑modules ; externalisation du front‑end de l'IHM ;
gestionnaire de contexte `applied(state)` côté IHM ; load flow paresseux + cache
graphe dans l'IHM ; élargissement du gate `ruff` au dépôt entier.
