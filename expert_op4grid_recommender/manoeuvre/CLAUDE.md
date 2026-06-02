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
| `algo.py`      | Phase 2 : `determiner_topo_complete_cible()` — sequence de manoeuvres |
| `__init__.py`  | API publique                                     |

## Commandes

```bash
# Tests du module
pytest tests/manoeuvre/ -v

# Extraction de fixtures depuis un .xiidm
python scripts/extract_test_fixtures.py --xiidm path/to/grid.xiidm

# Rendu SLD avant/apres d'un scenario CARRIP3 (couleurs natives pypowsybl)
python scripts/render_carrip3_sld.py --grid path/to/grid.xiidm

# IHM web de test interactif (Flask, dependance optionnelle) :
#   choisir un poste, modifier DJ/SA, calculer + animer la sequence
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

- **Postes >= 3 barres** : les couplages avec > 2 SJB emettent un warning ;
  seules les 2 premieres sont enregistrees. A ameliorer dans une version
  ulterieure.

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

## Etat algo (phase 2)

`determiner_topo_complete_cible(poste, topo_cible)` traite :
- postes 1 barre (cas trivial) ;
- postes 2 barres standard : evaluation de couplage (`nbNoeuds`/`nbBarres`),
  affectation noeud->barre par cout minimal, re-aiguillage boucle courte
  (couplage ferme) ou longue, ouverture/fermeture de couplage, verification
  post-manoeuvre par recalcul de `TopologieNodale.from_graph`.

Limites connues (cf. docstring `algo.py`) :
- re-aiguillage d'omnibus complexes : partiel ;
- pas de verification fine de court-circuit avant fermeture de couplage ;
- postes >= 3 barres : partiel (une cible a plus de noeuds que de barres est
  signalee comme non verifiee).

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
models.py  <--  graph.py  <--  cellules.py
                  ^                |
                  |                |
              __init__.py  (reexporte tout)
```

Les seules dependances externes du module sont `pypowsybl`, `networkx`,
et `pandas`. Pas de dependance a `grid2op` ni au reste du recommandeur.
