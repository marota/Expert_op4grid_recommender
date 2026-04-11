# Module Manoeuvre — Documentation

> **Package** : `expert_op4grid_recommender.manoeuvre`
> **Origine** : Portage Python de la bibliothèque C++ `libTOPO` (projet TOPO Apogee, RTE)
> **Backend reseau** : pypowsybl (topologie NODE_BREAKER)
> **Statut** : Etapes 1.1 et 1.2 implementees ; etapes 1.3+ a venir

---

## 1. Objectif

Le module **manoeuvre** analyse la topologie detaillee des postes electriques
au format IIDM/CGMES (via pypowsybl). A partir du graphe node/breaker d'un
voltage level, il identifie :

- les **cellules de depart** : sous-graphes reliant un equipement (ligne,
  transformateur, groupe, charge...) a ses jeux de barres accessibles ;
- les **cellules de couplage** : organes de coupure reliant deux sections
  de jeux de barres entre elles.

Ces structures sont les briques de base de l'algorithme *nodale -> detaillee*
qui, a terme, generera les manoeuvres topologiques candidates pour le
recommandeur expert.

---

## 2. Pipeline

```
  reseau pypowsybl (.xiidm / .cgmes)
          |
          v
  [Etape 1.1]  build_vl_graph()          graph.py
          |    Extraction du graphe NX
          |    (noeuds = connectivity nodes,
          |     aretes = switches + internal connections)
          v
  [Etape 1.2]  detecter_cellules()       cellules.py
          |    Detection structurelle BFS
          |    + connectivite electrique
          v
  CellulesVL
    .cellules_depart     : list[CelluleDepart]
    .cellules_couplage   : list[CelluleCouplage]
    .noeuds_non_attribues: set[int]
          |
          v
  [Etapes 1.3+]  (a venir)
    troncons.py   — Attribution des noeuds, tronconnement
    topologie.py  — TopologieNodale, PosteTopologique
    algo.py       — Algorithme nodale -> detaillee
```

---

## 3. Architecture des fichiers

```
expert_op4grid_recommender/manoeuvre/
  __init__.py      Public API (19 symboles exportes)
  models.py        Enumerations et dataclasses partagees
  graph.py         Etape 1.1 — construction du graphe NetworkX
  cellules.py      Etape 1.2 — detection des cellules

tests/manoeuvre/
  fixture_loader.py       Chargement des fixtures JSON (sans pypowsybl)
  fixtures/               Topologies reelles RTE serialisees
    index.json            Index des fixtures disponibles
    CARRIP3.json, ...     15 voltage levels de postes RTE
  test_graph_cellules.py  Tests unitaires (reseau 4-postes pypowsybl)
  test_postes_reels.py    Tests parametrises sur postes RTE reels

scripts/
  extract_test_fixtures.py  Extraction de fixtures depuis un .xiidm
```

---

## 4. Modeles de donnees (`models.py`)

### Enumerations

| Enum            | Valeurs principales                                       | Usage                              |
|-----------------|-----------------------------------------------------------|-------------------------------------|
| `NodeType`      | `BUSBAR_SECTION`, `EQUIPMENT`, `INTERNAL`                 | Type de noeud dans le graphe NX     |
| `EquipmentType` | `LOAD`, `GENERATOR`, `LINE_SIDE1/2`, `TRANSFORMER_SIDE1/2`, `SHUNT_COMPENSATOR`, `BATTERY`, ... (11 valeurs) | Type d'equipement                   |
| `SwitchKind`    | `BREAKER` (DJ), `DISCONNECTOR` (SA), `LOAD_BREAK_SWITCH`, `INTERNAL` | Type d'organe de coupure            |
| `CelluleType`   | `DEPART`, `COUPLAGE`, `INTERNE`                           | Classification des cellules         |

### Dataclasses

```python
@dataclass
class NodeAttrs:
    node_type: NodeType
    busbar_section_id: Optional[str]   # si BUSBAR_SECTION
    equipment_id: Optional[str]        # si EQUIPMENT
    equipment_type: Optional[EquipmentType]

@dataclass
class EdgeAttrs:
    switch_id: Optional[str]           # None pour internal connection
    kind: SwitchKind
    open: bool                         # True = non conducteur
    # Proprietes : is_closed, is_breaker, is_disconnector, is_internal
```

---

## 5. Etape 1.1 — Construction du graphe (`graph.py`)

### Fonction principale

```python
build_vl_graph(network: pp.network.Network, voltage_level_id: str) -> nx.Graph
```

Construit un graphe non oriente dont :
- **Noeuds** = connectivity nodes IIDM (entiers) ;
- **Aretes** = switches (BREAKER, DISCONNECTOR, LOAD_BREAK_SWITCH) et internal connections.

#### Deroulement

1. `_assert_node_breaker()` — verifie que le VL est en topologie NODE_BREAKER.
2. `_add_switch_edges()` — ajoute les aretes switch avec attributs `kind`, `open`, `switch_id`.
3. `_add_internal_connection_edges()` — ajoute les connexions internes (toujours fermees, `kind=INTERNAL`).
4. `_tag_busbar_section_nodes()` — marque les noeuds SJB (`node_type=BUSBAR_SECTION`).
5. `_tag_equipment_nodes()` — marque les bornes d'equipements (`node_type=EQUIPMENT`).

#### Equipements traites

| Categorie    | Getters pypowsybl                                               | Noeuds                |
|-------------|------------------------------------------------------------------|-----------------------|
| Injections   | `get_loads`, `get_generators`, `get_shunt_compensators`, `get_static_var_compensators`, `get_batteries`, `get_dangling_lines` | 1 noeud (`node`)      |
| Branches     | `get_lines`, `get_2_windings_transformers`                       | 2 noeuds (`node1`, `node2`) |

> **Note** : tous les appels utilisent `all_attributes=True` pour acceder aux
> colonnes `node`, `node1`, `node2`, `topology_kind` qui ne sont pas exposees
> par defaut dans pypowsybl.

### Fonctions utilitaires

```python
busbar_nodes(G) -> list[int]       # Noeuds BUSBAR_SECTION
equipment_nodes(G) -> list[int]    # Noeuds EQUIPMENT
get_node_attrs(G, node) -> NodeAttrs
get_edge_attrs(G, u, v) -> EdgeAttrs
```

### Exception

```python
class TopologyError(Exception):
    """Leve si le VL n'est pas en NODE_BREAKER ou si l'API ne retourne pas
    les colonnes attendues."""
```

---

## 6. Etape 1.2 — Detection des cellules (`cellules.py`)

### Fonction principale

```python
detecter_cellules(G: nx.Graph, voltage_level_id: str) -> CellulesVL
```

#### Algorithme en 3 phases

**Phase A — Cellules de depart** (BFS structurel)

Pour chaque noeud `EQUIPMENT` :
1. BFS depuis le noeud d'equipement, traversant **tous** les switches
   (ouverts ou fermes) et internal connections ;
2. Arret a chaque noeud `BUSBAR_SECTION` atteint (sans le traverser) ;
3. Les noeuds visites forment le sous-graphe de la cellule.

Si deux cellules partagent des noeuds intermediaires (topologie omnibus),
elles sont fusionnees et marquees via `shared_equipment_ids`.

**Phase B — Cellules de couplage**

1. Graphe restreint = noeuds SJB + noeuds internes non attribues en phase A ;
2. Composantes connexes avec >= 2 noeuds SJB = cellules de couplage ;
3. Le disjoncteur principal est identifie (`main_breaker`).

**Phase C — Noeuds non attribues**

Noeuds restants (ni depart, ni couplage) — typiquement neutres ou reserves.

### Classes de sortie

#### `CelluleDepart`

```python
@dataclass
class CelluleDepart:
    equipment_id: str
    equipment_type: EquipmentType
    all_nodes: set[int]
    busbar_nodes: set[int]           # accessibilite structurelle
    switches: list[SwitchInfo]
    connected_busbars: set[int]      # accessibilite electrique (switches fermes)
    shared_equipment_ids: set[str]   # vide sauf cellule omnibus
    subgraph: Optional[nx.Graph]
```

Proprietes notables :

| Propriete                 | Type         | Description                                              |
|---------------------------|-------------|----------------------------------------------------------|
| `nb_barres_accessibles`   | `int`        | Nombre de SJB atteignables (structurellement)             |
| `is_connected`            | `bool`       | Au moins une barre connectee electriquement               |
| `is_multiple`             | `bool`       | Cellule partagee (omnibus)                                |
| `breakers`                | `list`       | Disjoncteurs (DJ) de la cellule                           |
| `disconnectors`           | `list`       | Sectionneurs (SA) de la cellule                           |
| `is_reaiguillage`         | `bool`       | Cellule de re-aiguillage (1 seul SA, pas de DJ)           |
| `busbar_section_ids`      | `set[str]`   | IDs (string) des SJB atteintes                            |
| `connected_busbar_section_ids` | `set[str]` | IDs des SJB effectivement connectees                 |

#### `CelluleCouplage`

```python
@dataclass
class CelluleCouplage:
    switches: list[SwitchInfo]
    busbar_node_1: int
    busbar_node_2: int
    subgraph: Optional[nx.Graph]
```

| Propriete      | Type   | Description                               |
|----------------|--------|-------------------------------------------|
| `is_closed`    | `bool` | Tous les OC actifs fermes                 |
| `main_breaker` | `Optional[SwitchInfo]` | DJ principal du couplage |

#### `CellulesVL`

```python
@dataclass
class CellulesVL:
    voltage_level_id: str
    cellules_depart: list[CelluleDepart]
    cellules_couplage: list[CelluleCouplage]
    noeuds_non_attribues: set[int]
```

| Methode                          | Retour                  | Description                                |
|----------------------------------|-------------------------|--------------------------------------------|
| `get_cellule_depart(eq_id)`      | `Optional[CelluleDepart]` | Recherche par ID d'equipement           |
| `resume()`                       | `str`                   | Chaine de synthese lisible                 |

### Connectivite electrique

```python
calculer_connected_busbars(cellule: CelluleDepart) -> set[int]
```

Calcule les SJB connectees via un chemin de switches **tous fermes**
(composante connexe dans le sous-graphe restreint aux aretes fermees).
Met a jour `cellule.connected_busbars` en place.

---

## 7. Concepts cles du domaine

### Structurel vs. Electrique

| Dimension     | Traversee             | Resultat                  | Usage                              |
|---------------|----------------------|---------------------------|------------------------------------|
| **Structurelle** | Tous switches (ouverts/fermes) | `busbar_nodes`          | Quelles barres l'equipement *peut* atteindre |
| **Electrique**   | Uniquement switches fermes     | `connected_busbars`     | Quelles barres l'equipement *atteint actuellement* |

### Topologies RTE specifiques

#### Re-aiguillage

Raccordement direct d'un equipement a une barre via un unique sectionneur
d'aiguillage (SA), sans disjoncteur propre. Identifiable via :
- `CelluleDepart.is_reaiguillage == True`
- 1 seul switch de type `DISCONNECTOR`, 0 `BREAKER`
- Nommage typique : `*SA_F` (sectionneur d'aiguillage, fictif)

#### Cellule omnibus / departs multiples

Plusieurs equipements partagent les memes noeuds intermediaires et le meme
disjoncteur. Identifies via `CelluleDepart.is_multiple` et
`shared_equipment_ids`.

#### Postes >= 3 barres

Les cellules de couplage avec > 2 SJB emettent un warning. Seules les
2 premieres SJB sont enregistrees dans `busbar_node_1`/`busbar_node_2`.

---

## 8. Usage rapide

```python
import pypowsybl as pp
from expert_op4grid_recommender.manoeuvre import (
    build_vl_graph,
    detecter_cellules,
    busbar_nodes,
    equipment_nodes,
)

# Charger un reseau
network = pp.network.load("grid.xiidm")

# Construire le graphe d'un voltage level
G = build_vl_graph(network, "CARRIP3")

# Detecter les cellules
cellules = detecter_cellules(G, "CARRIP3")
print(cellules.resume())

# Inspecter une cellule de depart
for c in cellules.cellules_depart:
    print(f"{c.equipment_id} -> {c.nb_barres_accessibles} barres, "
          f"connecte={c.is_connected}, reaig={c.is_reaiguillage}")

# Inspecter les couplages
for coup in cellules.cellules_couplage:
    print(f"Couplage {coup.busbar_node_1}-{coup.busbar_node_2}, "
          f"ferme={coup.is_closed}")
```

---

## 9. Tests

### Tests unitaires (`test_graph_cellules.py`)

Reseau de reference : `pp.network.create_four_substations_node_breaker_network()`
cible le voltage level `S1VL2` (380 kV, 2 barres).

| Classe                   | Tests | Couverture                                       |
|--------------------------|-------|--------------------------------------------------|
| `TestBuildVlGraph`       | 9     | Construction, tagging, attributs, erreurs         |
| `TestDetecterCellules`   | 11    | Detection, proprietes, sous-graphes               |
| `TestConnectedBusbars`   | 4     | Connectivite electrique, sous-ensemble SJB        |
| `TestCoherenceGlobale`   | 3     | Coherence graphe/cellules, pas de recouvrement    |

### Tests sur postes reels (`test_postes_reels.py`)

15 voltage levels de postes RTE reels, serialises en fixtures JSON.
Pas de dependance a pypowsybl a l'execution des tests (reconstruction
du graphe NX depuis le JSON).

| Classe                       | Tests | Couverture                                    |
|------------------------------|-------|-----------------------------------------------|
| `TestGraphConstruction`      | 5     | Validite du graphe, SJB, equipements, types   |
| `TestCellulesDepart`         | 8     | Couverture equipements, SJB, switches, DJ     |
| `TestPasDeChevaucement`      | 1     | Pas de recouvrement de noeuds internes        |
| `TestCellulesCouplage`       | 3     | Structure couplage, DJ, liens >= 2 SJB        |
| `TestConnectiviteElectrique` | 3     | Sous-ensemble SJB, taux connectivite > 50%    |
| `TestDepartsMultiples`       | 1     | Detection cellules omnibus                    |
| `TestDiagnostic`             | 2     | Resume, coherence stats/fixtures              |

### Extraction des fixtures

```bash
python scripts/extract_test_fixtures.py \
    --xiidm path/to/grid.xiidm
```

Genere un fichier JSON par voltage level dans `tests/manoeuvre/fixtures/`
avec un `index.json` de synthese.

---

## 10. Prochaines etapes (roadmap)

| Etape   | Fichier          | Description                                              |
|---------|-----------------|----------------------------------------------------------|
| 1.3     | `troncons.py`    | Attribution des noeuds aux cellules, tronconnement        |
| 1.4     | `troncons.py`    | Regroupement en troncons electriques                      |
| 1.5     | `topologie.py`   | Construction de `TopologieNodale`                         |
| 1.6     | `topologie.py`   | Construction de `PosteTopologique`                        |
| Phase 2 | `algo.py`        | Algorithme nodale -> detaillee (generation de manoeuvres) |

---

## 11. Dependances

| Package      | Role                                        | Version min |
|-------------|---------------------------------------------|-------------|
| `pypowsybl` | Modele de donnees reseau (IIDM/CGMES)       | >= 1.13.0   |
| `networkx`   | Representation et algorithmes de graphes     | —           |
| `pandas`     | DataFrames pypowsybl                        | —           |
