# CLAUDE.md — Module Manoeuvre

> Contexte pour les assistants IA travaillant sur ce module.

## Vue d'ensemble

Portage Python de `libTOPO` (C++, RTE). Analyse la topologie NODE_BREAKER
des postes electriques via pypowsybl pour identifier cellules de depart et
de couplage.

## Fichiers

| Fichier        | Role                                             |
|---------------|--------------------------------------------------|
| `models.py`    | Enums (`NodeType`, `EquipmentType`, `SwitchKind`, `CelluleType`) et dataclasses (`NodeAttrs`, `EdgeAttrs`) |
| `graph.py`     | Etape 1.1 : `build_vl_graph()` — graphe NetworkX depuis un voltage level pypowsybl |
| `cellules.py`  | Etape 1.2 : `detecter_cellules()` — BFS structurel + connectivite electrique |
| `troncons.py`  | Etapes 1.3-1.4 : `construire_tronconnement()` — barres, troncons, attribution |
| `topologie.py` | Etapes 1.5-1.6 : `TopologieNodale`, `PosteTopologique`, `attribuer_noeuds()` |
| `algo/`        | Phase 2 : `determiner_topo_complete_cible()` — package en couches (voir ci-dessous) |
| `__init__.py`  | API publique                                     |

### Package `algo/` (Phase 2, eclate depuis l'ancien `algo.py`)

Sous-modules en couches, **dependances strictement descendantes** (sans cycle) :

| Sous-module          | Role                                                       |
|----------------------|------------------------------------------------------------|
| `algo/_constants.py` | Poids de placement et garde-fous combinatoires             |
| `algo/results.py`    | `Manoeuvre`, `ResultatManoeuvres` (structures de sortie)   |
| `algo/graph_ops.py`  | Helpers bas niveau : index O(1), `_is_open`/`_set_switch`, chemins SA, `_inter_sjb_couplers` |
| `algo/placement.py`  | Placement noeud → sections (`_placement_automatique`, best-effort, glouton, `_main_busbar_sjb`) |
| `algo/verification.py`| Regle du sectionneur (`_rejeu_securite`, `sectionneurs_sous_charge_par_manoeuvre`), `_optimiser_sequence` |
| `algo/sequencing.py` | Sequenceur general (`determiner_manoeuvres_avec_sections`), re-aiguillages |
| `algo/targets.py`    | Points d'entree : `determiner_topo_complete_cible`, `determiner_manoeuvres_cible_detaillee`, modes smooth/aggressive/multi-barres |
| `algo/__init__.py`   | **Reexporte toute la surface** de l'ancien module (publics + prives) : `manoeuvre.algo.X` et `from ...algo import X` restent inchanges |

Couches : `_constants`/`results` → `graph_ops` → `placement`/`verification` →
`sequencing` → `targets`.

## Commandes

```bash
# Tests du module
pytest tests/manoeuvre/ -v

# Extraction de fixtures depuis un .xiidm
python scripts/extract_test_fixtures.py --xiidm path/to/grid.xiidm

# Rendu SLD avant/apres d'un scenario CARRIP3 (couleurs natives pypowsybl)
python scripts/render_carrip3_sld.py --grid path/to/grid.xiidm

# IHM web de test interactif (Flask, dependance optionnelle) :
#   choisir un poste, modifier DJ/SA, valider/sauver la cible, calculer +
#   animer la sequence, sauvegarder scenarios et sequences. Doc complete :
#   docs/manoeuvre_ihm.md
pip install -e ".[ihm]"   # guillemets requis sous zsh (sinon: pip install flask)
python scripts/manoeuvre_ihm.py --grid path/to/grid.xiidm   # http://localhost:8000
```

## Conventions critiques

### pypowsybl `all_attributes=True`

**Toujours** appeler les getters pypowsybl avec `all_attributes=True` :
- `get_voltage_levels(all_attributes=True)` pour `topology_kind`
- `get_switches(all_attributes=True)` pour `node1`, `node2`
- `get_busbar_sections(all_attributes=True)` pour `node`
- `get_loads(...)`, `get_generators(...)`, etc. pour `node`

Sans ce flag, ces colonnes sont absentes et le code leve `KeyError`.
L'utilitaire `_safe_get()` dans `graph.py` gere cela automatiquement
pour les equipements ; les appels directs doivent le specifier.

### Structurel vs. Electrique

Deux niveaux d'analyse distincts :
- **Structurel** (BFS traversant tous les switches) -> `busbar_nodes`
- **Electrique** (uniquement switches fermes) -> `connected_busbars`

Ne pas confondre : une cellule peut atteindre 4 barres structurellement
mais n'etre connectee qu'a 1 electriquement.

### Topologies RTE specifiques

- **Re-aiguillage** : equipement raccorde directement a la barre via un
  seul sectionneur (SA), sans disjoncteur propre. Detecte par
  `CelluleDepart.is_reaiguillage`. Ne **pas** exiger de DJ sur ces cellules.

- **Omnibus / departs multiples** : equipements partageant les memes noeuds
  intermediaires. Signale par `CelluleDepart.is_multiple` et
  `shared_equipment_ids`.

- **Postes >= 3 barres** : le **placement** (`_placement_automatique`) gere
  desormais N jeux de barres (Etape 1+2, cf. ci-dessous). En revanche, le modele
  de cellule `CelluleCouplage` ne retient encore que 2 SJB par composante de
  couplage (warning sur > 2 SJB) et le **sequenceur** ne realise pas encore une
  cible > 2 noeuds passant par un couplage multi-barres partage : frontiere
  suivante.

## Tests

### Fixtures

Les tests `test_postes_reels.py` utilisent des fixtures JSON dans
`tests/manoeuvre/fixtures/`. Elles sont generees par
`scripts/extract_test_fixtures.py` et reconstruisent le graphe NX
**sans pypowsybl** a l'execution (independance CI).

Le `fixture_loader.py` contient les fonctions de chargement :
`build_graph_from_fixture(vl_name)` est le point d'entree principal.

### Reseau de reference

Les tests unitaires (`test_graph_cellules.py`) utilisent le reseau standard
`pp.network.create_four_substations_node_breaker_network()`, cible `S1VL2`.

### Filets de regression (refactors a iso-comportement)

Poses avant la campagne d'optimisation (cf. `docs/manoeuvre_optimisations.md`),
ces tests garantissent l'invariance du comportement :

- `test_golden_sequences.py` — **golden** de bout en bout : sequence ordonnee
  exacte + verifications + ecarts + partition, pour chaque `scenarios/*.json` x
  {smooth, aggressive}. Reference dans `tests/manoeuvre/goldens/`.
  **Regenerer consciemment** apres un changement de comportement assume :
  `UPDATE_GOLDENS=1 pytest tests/manoeuvre/test_golden_sequences.py`.
- `test_verificateurs_exact.py` — sorties exactes des verificateurs du
  sectionneur (alignement, indices, dedup, idempotence, non-mutation).
- `test_lookup_helpers.py` — contrat de `_is_open`/`_set_switch`/`_eq_node`
  (id inconnu, validite sur copie de graphe).
- `test_couplers_memoisation.py` — invariance a l'etat des couplers, purete,
  non-mutation de `poste.graph` par le pipeline.

### Filets pre-eclatement d'`algo.py` (#7)

Poses avant de scinder `algo.py` en sous-modules (un refactor structurel
**deplace** des symboles) ; couverture du module ~92 % :

- `test_public_api.py` — **verrou de surface publique** : chaque symbole de
  `__all__` reste importable depuis le package *et* depuis son sous-module
  d'origine (meme objet), entrypoints appelables, alias prive du verificateur
  conserve. Un oubli de reexport casse la CI immediatement.
- `test_multibarres_placement.py` — caracterisation du chemin **> 2 jeux de
  barres** (CORNIP3/GUARBP6/MORBRP6, 4 barres), non couvert par les goldens :
  `determiner_topo_complete_cible` (cible identite + cible scindee),
  `_main_busbar_sjb`, degradation gracieuse (`noeuds_non_realisables`), rendu
  `resume()`, non-mutation du graphe, organes emis existants.
- `test_algo_entrypoint_guards.py` — garde-fous de faisabilite
  (`determiner_topo_complete_cible`) : graphe absent, departs cibles absents.
- `test_graph_helpers.py` — `graph._safe_get` (repli `all_attributes`, erreurs).

## Performance & invariants internes

Plusieurs chemins chauds sont **memoises** ; toute modification doit preserver
ces invariants (sinon les goldens cassent). Detail et gains :
`docs/manoeuvre_optimisations.md`.

- **Index lookups** (`_switch_edge_index`, `_equipment_node_index`) : caches
  `G.graph` construits en une passe ; coordonnees **topologiques** (valides sur
  les copies). Garde-fou **O(1)** sur `number_of_nodes()` — ne **jamais**
  rebrancher sur `number_of_edges()` (O(aretes), annule le gain).
- **Couplers** (`_inter_sjb_couplers`) : memoise sur le poste — invariant a
  l'etat ouvert/ferme des organes (ne lit que la topologie + le tronconnement).
- **Verificateurs** : une **passe de rejeu unique** (`_rejeu_securite`) ; les
  fonctions publiques sont de minces delegateurs.
- **Placement** (`_placement_automatique`) : enumeration par **partitions
  connexes** (`_assignations_connexes`) + connexite memoisee + tie-break
  **lex-min** reproduisant l'ancienne enumeration `itertools.product`.
- **Chemins SA** (`CelluleDepart.disconnectors_vers_barre`) : memoise par SJB
  (sous-graphe de cellule fige -> resultat structurel invariant).

Invariant cle exploite partout : **`poste.graph` n'est jamais mute
structurellement** ; le sequenceur travaille sur des **copies** et ne bascule
que l'attribut `open`.

## Etat algo (phase 2)

`determiner_topo_complete_cible(poste, topo_cible)` traite :
- postes 1 barre (cas trivial) ;
- postes 2 barres standard : evaluation de couplage (`nbNoeuds`/`nbBarres`),
  affectation noeud->barre par cout minimal, re-aiguillage boucle courte
  (couplage ferme) ou longue, ouverture/fermeture de couplage, verification
  post-manoeuvre par recalcul de `TopologieNodale.from_graph`.

Etape 1+2 (placement N jeux de barres) :
- le **scoping 2-JdB** de `_placement_automatique` a ete retire : la recherche
  exacte (`_recherche_exhaustive`, lex-min) tourne sur **toutes** les SJB et
  **tous** les couplers — couvre les postes 3B/4B reels dans le garde-fou
  (`MAX_COMBINAISONS_PLACEMENT`). Comportement 2-barres strictement preserve.
- au-dela du garde-fou : **decomposition recursive** (`_placement_decompose`)
  le long du graphe de couplage — par **composantes connexes** (exacte,
  separable) puis **bissection au niveau barre** (best-effort) ; tout placement
  declare complet est revérifie faisable (`_placement_est_faisable`).

Limites connues (cf. docstring `algo.py`) :
- re-aiguillage d'omnibus complexes : partiel ;
- pas de verification fine de court-circuit avant fermeture de couplage ;
- postes >= 3 barres : le **placement** est gere ; la realisation **sequencee**
  d'une cible > 2 noeuds via un couplage multi-barres partage reste partielle
  (frontiere sequenceur). La bissection best-effort peut degrader gracieusement
  sur des postes a demi-rames tres maillees (la voie exacte, elle, les gere).

### Specification des regles

Toutes les regles metier du sequencement (R1-R14 : faisabilite, distinction
sectionnement/couplage, tronconnement, placement noeud->SJB, boucle courte/
longue, DJ d'ensemble de cellule, regle du sectionneur de barre, ordonnancement
listeDordre, controle court-circuit, verification) sont tracees avec leurs
fonctions et tests dans **`docs/manoeuvre_regles.md`**.

### Convention de detection des barres

`troncons.py` distingue **sectionnement** (SA reliant deux SJB d'une meme barre)
de **couplage** (travee a DJ reliant deux barres). La barre d'une SJB est
deduite en priorite du **nommage RTE** (entier de tete apres `VL_id_`, ex.
`CARRIP3_1.1` -> barre 1) ; repli structurel par connectivite sectionnement
(chemins sans BREAKER) si le nommage est inexploitable.

## Dependances internes

```
models.py  <--  graph.py  <--  cellules.py  <--  troncons.py  <--  topologie.py
                                                                       ^
                                                                       |
   algo/  (results <- graph_ops <- placement/verification <- sequencing <- targets)
                                                                       |
                                                  __init__.py  (reexporte tout)
```

Les seules dependances externes du module sont `pypowsybl`, `networkx`,
et `pandas`. Pas de dependance a `grid2op` ni au reste du recommandeur.
