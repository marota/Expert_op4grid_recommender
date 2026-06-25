# Carte géographique des postes — du `grid_layout.json` au rendu France

> **But du document** : décrire l'**approche complète** retenue pour afficher les
> postes du réseau RTE 7000 sur une carte de France dans l'IHM de manœuvre
> (« Explorer la journée »), **à partir du `grid_layout.json` fourni**.
>
> Vue d'ensemble de la fonctionnalité : `docs/manoeuvre_ihm.md` (§ 1ter).
> Code : `manoeuvre/dataset/geographie.py`, `scripts/build_france_basemap.py`,
> `scripts/manoeuvre_ihm.py` (`construire_exploration`, `_xy`, endpoints), et le
> front-end `scripts/manoeuvre_ihm_assets/index.html` (`buildMap`).

---

## 1. Le problème

Les instantanés du dataset RTE 7000 (`.xiidm`) **ne portent aucune coordonnée
géographique** : pas d'extension pypowsybl `substationPosition`, pas de lat/lon
(vérifié). Le jeu ODRE `postes-electriques-rte`, lui, est **purement tabulaire**
(code, nom, tension, département — **sans géométrie**, vérifié via l'API records).
Il faut donc une **autre source de positions** pour placer ~4 800 postes sur une
carte.

La solution **primaire** retenue est le **plan de masse RTE fourni**
(`grid_layout.json`), parce qu'il :

- couvre **~98 %** des postes du réseau,
- est **hors-ligne** (rien à télécharger, rien à configurer sur le Space),
- utilise des **noms de voltage levels RTE** directement recoupables avec le
  réseau chargé.

---

## 2. Le `grid_layout.json` fourni

### Format

```json
{
  "CHEVSL61MERLA": [702345.1, -2587110.4],
  "MERLAP6":       [701980.0, -2586740.2],
  "...":           [x, y]
}
```

- **Clé** = identifiant de **voltage level** RTE (mnémonique du poste + code de
  tension, p. ex. `MERLAP6` = poste `MERLA`, palier `P6` = 225 kV).
- **Valeur** = `[x, y]` dans une **projection planaire RTE** (repère métrique
  propre à RTE, *non* lon/lat, *non* une projection standard documentée).

### Nature et propriétés exploitées

- C'est un **plan de masse par VL** : plusieurs VL d'un même poste ont des
  coordonnées très proches (l'écart intra-poste est négligeable à l'échelle de la
  France).
- Le repère a le **nord en haut** mais l'axe **y croît vers le sud** (comme un
  repère écran). C'est la clé de la projection écran (cf. § 5).
- Extrait d'**un** instantané : la couverture peut être légèrement < 100 % pour
  une date donnée (postes apparus/disparus), d'où le rôle de **repli** des autres
  sources (§ 4).

### Intégration dans le dépôt

Le fichier fourni est **committé** sous
`expert_op4grid_recommender/manoeuvre/dataset/grid_layout_rte.json` et **embarqué
dans le package** (`pyproject.toml`, `tool.setuptools.package-data`) — il est donc
présent dans l'image du Space sans configuration. Constante :
`geographie.LAYOUT_DEFAUT`.

---

## 3. Pipeline — vue d'ensemble

```
grid_layout.json (fourni)                         ODRE postes-electriques-rte
  {nom_VL: [x,y]}  (plan de masse RTE)               (centroïdes lon/lat
        │                                             + département par code)
        │ committé → grid_layout_rte.json                     │
        ▼                                                      │
 [1] charger_layout()                                          │ (hors-ligne, une fois)
        │                                                      ▼
 [2] positions_from_layout(layout, vl_meta)        scripts/build_france_basemap.py
        │  → {substation: {x,y,source:'layout'}}    ajustement affine (lon,lat)→(x,y)
        │     (VL de plus haute tension par poste)  des frontières dép. + voisins
        │                                                      │
 [3] construire_exploration():                                 ▼
        layout primaire, sinon resoudre()           france_basemap.json
        (snapshot → OSM)                             {depts:[…], neighbors:[…]}
        │                                            (repère du PLAN DE MASSE)
        ▼                                                      │
 [4] _xy(pos) → (x_écran, y_écran)            GET /api/explore_basemap (tel quel)
        │  layout : (x, y) tel quel (nord en haut)             │
        │  lon/lat: Mercator puis y inversé                    │
        ▼                                                      ▼
 [5] _explore_payload() → postes[{sub,x,y,nv,…}]   ◀───────────┘
        │
        ▼
 [6] buildMap() (front) : disques colorés par tension + fond de carte
     (dessiné si coord_source=='layout', sinon enveloppe convexe) + zoom/pan
```

---

## 4. Étapes côté serveur

### [1] Chargement — `charger_layout(path)`

Lecture best-effort du JSON committé → `{nom_VL: [x, y]}` (ou `{}` si
absent/illisible). Aucune dépendance réseau.

### [2] Agrégation par poste — `positions_from_layout(layout, vl_meta)`

La carte affiche **un disque par poste (substation)**, alors que le plan de masse
est **par VL**. Pour chaque poste, on retient le **VL de plus haute tension**
présent dans le plan (coordonnée la plus représentative) :

```python
best[sub] = (nominal_v, vl)   # max nominal_v parmi les VL du poste présents au plan
positions[sub] = {"x": layout[vl][0], "y": layout[vl][1], "source": "layout"}
```

Retourne aussi des **statistiques de couverture** (`n_substations`, `n_apparies`,
`taux`) affichées dans l'en-tête de la carte (`coord. : layout (4723/4811, 98%)`).
Fonction **pure** (testée sans pypowsybl).

### [3] Choix de la source — `construire_exploration()` puis `resoudre()`

Dans `scripts/manoeuvre_ihm.py`, le **plan de masse est essayé en premier** ;
restreint aux postes **actifs** de la journée. S'il en localise au moins un, c'est
la source `'layout'`. Sinon, repli sur la **chaîne `resoudre()`** :

| Ordre | Source        | Clé d'appariement                       | Réseau ? |
|------:|---------------|------------------------------------------|:--------:|
| 0     | `layout`      | nom de VL (plan de masse fourni)         | non      |
| 1     | `xiidm`       | extension `substationPosition` (absente) | non      |
| 2     | `snapshot`    | `substation_id` (`data/postes_rte_geo.json`) | non  |
| 3     | `osm`         | `ref:FR:RTE` = `substation_id` (Overpass) | oui     |
| —     | `aucune`      | → bascule sur le **classement en liste** | —        |

> En pratique, le plan de masse fourni couvrant ~98 % des postes, la source est
> **toujours `layout`** ; les sources 1–3 sont des filets de sécurité (réseau futur
> portant la géométrie, instantané committé, ou découverte OSM persistée).

### [4] Projection écran — `_xy(pos)`

Convertit une position résolue en **coordonnées planaires prêtes pour l'écran**
(y vers le bas, **nord en haut**) :

```python
def _xy(pos):
    if "x" in pos:                       # plan de masse RTE
        return float(pos["x"]), float(pos["y"])      # tel quel (nord déjà en haut)
    mx, my = geographie.merc(pos["lon"], pos["lat"]) # sources lon/lat (OSM)
    return mx, -my                       # Mercator a le nord en y croissant → inversion
```

**Point crucial d'orientation** : le plan de masse a le nord en haut dans son
repère (y croît vers le sud, comme l'écran). On l'utilise donc **sans inverser
l'axe y**. Seules les sources **lon/lat** (OSM) sont projetées en **Web Mercator**
(`merc`) puis **inversées** (le Mercator place le nord en y croissant). *(Une
inversion erronée du plan de masse donnait précédemment une France « à l'envers ».)*

### [5] Charge utile — `_explore_payload()`

Produit la liste `postes = [{sub, name, nv, x, y, total, rank, …}]` (un élément par
poste géolocalisé, coordonnées arrondies) + le `classement` par VL + `coord_source`
+ `coord_stats`, consommée par le front.

---

## 5. Le fond de carte calibré (silhouette France)

### Pourquoi une calibration ?

Le plan de masse RTE est une **projection planaire propriétaire non documentée**.
Pour dessiner les frontières de la France **dans le même repère que les disques**
(sans reprojection au runtime), on **calibre empiriquement** la transformation
`(lon, lat) → (x, y)_plan de masse`.

### Méthode — `scripts/build_france_basemap.py`

1. **centroïdes des départements** (GeoJSON public, en lon/lat) ;
2. **ODRE** `postes-electriques-rte` → `code_poste → département` ;
3. réseau RTE 7000 → `substation_id` (= `code_poste`) → coordonnée du **plan de
   masse** ; chaque poste est apparié au **centroïde de son département** ;
4. **ajustement affine par moindres carrés** sur ces paires :

   ```
   x ≈ a·lon + b·lat + c
   y ≈ d·lon + e·lat + f      (e < 0 : nord = y le plus négatif → nord en haut)
   ```

   (le résidu est dominé par l'étalement intra-département, pas par l'erreur de
   projection ; la plage lon/lat recouvrée couvre bien la France métropolitaine) ;
5. on **transforme** les frontières (départements + pays voisins) par cet affine,
   on **décime** les anneaux, et on écrit
   `manoeuvre/dataset/france_basemap.json` :

   ```json
   { "depts":     [ [[x,y], [x,y], …], … ],
     "neighbors": [ [[x,y], …], … ] }
   ```

Ce fichier est **committé et embarqué** (package-data) : il est déjà dans le repère
du plan de masse, donc **aucune reprojection au runtime**. Servi tel quel par
`GET /api/explore_basemap` (`charger_basemap`), récupéré une fois par le front.

> Réseau requis **uniquement** pour (re)générer ce fichier (ODRE + GeoJSON GitHub) —
> à lancer hors du Space si la sortie réseau y est bloquée, puis committer le JSON.

---

## 6. Rendu front-end — `buildMap(postes)`

SVG **autonome** (sans tuiles ni librairie de carto externe), un seul `<svg>`
manipulé par `viewBox` :

1. **Boîte englobante** = postes (+ frontières des départements **si** le fond est
   utilisé). Le fond géographique réel n'est dessiné **que** si
   `coord_source === 'layout'` (repère calibré) :
   ```js
   const useBasemap = depts.length && MAP.data && MAP.data.coord_source==='layout';
   ```
   Sinon (OSM/Mercator, repère non calibré sur ce fond), repli **silhouette** =
   **enveloppe convexe** du nuage de postes (`convexHull`, Andrew monotone chain).
2. **Fond** : polygones `.nbr` (pays voisins, assombris pour contraster avec les
   zones maritimes) puis `.dept` (départements France) — dessinés **avant** les
   disques.
3. **Disques** : un cercle par poste, **couleur par niveau de tension**
   (`voltColor`/`VOLT_BANDS` : 400 kV rouge, 225 kV vert, 90 kV orange, 63 kV
   violet…). Rayon `span/520` ; les **10 VL les plus actifs** sont plus gros
   (`span/200`) avec un **halo**. Opacité plus forte si le poste a bougé.
4. **Interactions** : **molette** = zoom (recadrage du `viewBox`), **glisser** =
   déplacement, **clic** = bulle d'info, **double-clic** = vue topologique du
   poste. **Légende filtrante** des tensions (`voltToggle`/`voltAll`, attribut
   `data-vb`) pour afficher/masquer un niveau. Délégation d'événements → fluide
   jusqu'à ~6 000 disques (aucun re-rendu au pan/zoom).

---

## 7. Robustesse / dégradation gracieuse

- **Plan de masse présent** (cas nominal) : carte complète + fond France réel,
  **sans réseau ni configuration**.
- **Plan de masse absent / incomplet** : repli `snapshot` puis `osm` ; si la source
  finale n'est pas `layout`, le fond réel laisse place à l'**enveloppe convexe**.
- **Aucune coordonnée** (`source = 'aucune'`) : la carte est masquée, l'IHM reste
  utile via le **classement en liste** (cliquable) + un **diagnostic**
  d'appariement dans le bandeau.

Toutes les fonctions de `geographie.py` sont **best-effort** (ne lèvent pas) :
un échec de source n'interrompt jamais « Explorer la journée ».

---

## 8. Régénérer les artefacts

```bash
# Fond de carte (frontières) dans le repère du plan de masse — réseau requis
# (ODRE + GeoJSON). À committer ensuite.
python scripts/build_france_basemap.py --date 2021-01-03
#   ou : --xiidm chemin/instantane.xiidm.bz2

# Mettre à jour le plan de masse : remplacer le JSON fourni
#   expert_op4grid_recommender/manoeuvre/dataset/grid_layout_rte.json
# (format {nom_VL: [x, y]}). positions_from_layout s'y adapte automatiquement.
```

---

## 9. Fichiers & symboles de référence

| Rôle | Emplacement |
|------|-------------|
| Plan de masse fourni (committé, **primaire**) | `manoeuvre/dataset/grid_layout_rte.json` |
| Chargement / agrégation par poste | `geographie.charger_layout`, `geographie.positions_from_layout` |
| Chaîne de résolution (replis) | `geographie.resoudre` (`xiidm`/`snapshot`/`osm`) |
| Projection écran (orientation) | `manoeuvre_ihm._xy`, `geographie.merc` |
| Choix de source + charge utile | `manoeuvre_ihm.construire_exploration`, `_explore_payload` |
| Fond de carte calibré | `scripts/build_france_basemap.py`, `manoeuvre/dataset/france_basemap.json`, `geographie.charger_basemap`, `GET /api/explore_basemap` |
| Rendu carte (front) | `manoeuvre_ihm_assets/index.html` : `buildMap`, `voltColor`, `convexHull`, `voltToggle` |
| Tests | `tests/manoeuvre/test_geographie.py`, `tests/manoeuvre/test_ihm_explore.py` |

---

## 10. Limites connues

- Le plan de masse est une **projection planaire RTE** : les positions sont
  **relatives cohérentes**, pas des lon/lat vrais. Suffisant pour situer/comparer
  des postes, pas pour des mesures géodésiques.
- La calibration du fond de carte est **affine** (le repère RTE n'est pas une
  projection standard) : le résidu est dominé par l'étalement intra-département ;
  l'alignement frontières/disques est **visuellement correct** mais non exact au
  poste près.
- Couverture du plan **< 100 %** possible pour une date (postes apparus/disparus) ;
  les postes non localisés restent accessibles via le **classement en liste**.
