# IHM de test du module manœuvre

> Interface web légère pour **tester interactivement** le module
> `expert_op4grid_recommender.manoeuvre` sur les postes de test : éditer une
> topologie détaillée cible, calculer la séquence de manœuvres, l'animer sur le
> schéma unifilaire, et **sauvegarder scénarios et séquences** pour l'analyse et
> la création de tests.

Script : `scripts/manoeuvre_ihm.py` — Règles métier : `docs/manoeuvre_regles.md`.

---

## 1. Installation et lancement

L'IHM repose sur **Flask** (dépendance **optionnelle**) :

```bash
pip install -e ".[ihm]"        # guillemets requis sous zsh ; ou : pip install flask
```

Lancement :

```bash
python scripts/manoeuvre_ihm.py --grid /chemin/vers/grid.xiidm
# puis ouvrir http://localhost:8000
```

| Argument | Défaut | Rôle |
|----------|--------|------|
| `--grid` | *(requis)* | Réseau `.xiidm` contenant les postes de test |
| `--port` | `8000` | Port HTTP |
| `--scenarios-dir` | `tests/manoeuvre/scenarios` | Dossier de sauvegarde des **scénarios** (cibles) |
| `--sequences-dir` | `tests/manoeuvre/sequences` | Dossier de sauvegarde des **séquences** générées |

> Le serveur charge le réseau une fois (≈ 15 s pour un grand `.xiidm`) puis
> affiche les postes de test disponibles. Il est **mono-utilisateur** et
> **mono-thread** (l'état réseau pypowsybl est partagé).

**Postes de test** (intersection des fixtures et du réseau) : CARRIP3, CARRIP6,
CZTRYP6, COMPIP3, BXTO5P3, BXTO5P6, CZBEVP3, PALUNP3, NOVIOP3, SSAVOP3, VIELMP6,
CORNIP3, GUARBP6, MORBRP6.

---

## 2. Disposition de l'interface

```
┌───────────────────────────┬───────────────────────────────────────────────┐
│ PANNEAU LATÉRAL            │  SCHÉMA — TOPOLOGIE DE DÉPART (fixe)           │
│                            │  (SLD pypowsybl, couleurs natives par nœud)   │
│ • Poste (sélecteur)        ├───────────────────────────────────────────────┤
│ • ↺ État de départ         │  SCHÉMA — TOPOLOGIE CIBLE (éditable)           │
│ 1 · Valider la cible       │  clic sur un organe = bascule ; animation ici │
│ 2 · Calculer la séquence   ├───────────────────────────────────────────────┤
│ • Scénarios sauvegardés    │  Contrôles d'animation ◀ ▶ Lecture ▶|         │
│ • Liste DJ / SA + états    │  Séquence (texte) + 💾 Sauvegarder la séquence │
└───────────────────────────┴───────────────────────────────────────────────┘
```

- **Schéma de départ** (haut, bandeau bleu) : topologie détaillée initiale,
  **non modifiable** ; sert de référence.
- **Schéma cible** (bas, bandeau orange) : topologie détaillée **éditable** ;
  c'est aussi là que se déroule l'animation de la séquence.
- Le **nombre de nœuds électriques** est affiché par schéma et se met à jour à
  chaque modification.
- Chaque en-tête porte un bouton **▾ / ▸** pour **replier** son schéma : l'autre
  schéma occupe alors tout l'espace. Utile sur les grands postes pour observer
  l'animation de la séquence en plein écran sur la cible.

Les **couleurs** sont celles de pypowsybl (`topological_coloring`) : couleur de
base par niveau de tension (≈ violet pour le 63 kV), déclinée en teintes
distinctes par nœud électrique. Le navigateur rend nativement le SVG (résolution
des variables CSS).

---

## 3. Flux de travail

1. **Choisir un poste** → les deux schémas affichent l'état de départ (pristine).
2. **Éditer la cible** : cliquer un disjoncteur/sectionneur dans le schéma du
   bas (ou dans la liste latérale) pour basculer son état (ouvert/fermé). Le
   nombre de nœuds cible évolue en direct.
3. **Valider & sauvegarder la cible** (étape 1) : nomme et persiste le scénario
   (départ + cible). **Obligatoire** avant le calcul.
4. **Calculer la séquence** (étape 2, débloqué après validation) : le module
   calcule la séquence départ → cible ; statut **VÉRIFIÉE / NON VÉRIFIÉE**.
5. **Lire / animer** : la séquence s'affiche en texte ; les contrôles
   d'animation rejouent les manœuvres une par une sur le schéma cible, l'organe
   manœuvré **surligné en rouge**, la ligne de séquence courante surlignée.
6. **Sauvegarder la séquence** : 💾 écrit un JSON autonome (topologies +
   séquence + lien scénario) pour l'analyse et les tests. Si le fichier existe
   déjà, une **confirmation** est demandée (écraser) ou un **nouveau nom** est
   proposé (idem pour la sauvegarde de scénario).

> La cible éditée étant une **topologie détaillée** (barre exacte de chaque
> départ), l'IHM vise cet état exact : après avoir atteint la partition nodale,
> les départs sont ramenés sur leur barre imposée (manœuvres supplémentaires),
> et la **topologie détaillée est vérifiée**. Le statut affiche
> « DÉTAILLÉE VÉRIFIÉE », ou « NODALE OK · N écart(s) détaillé(s) » avec la
> liste des écarts résiduels.

### Naviguer et éditer la séquence (expert)
La séquence calculée peut être **parcourue et modifiée** directement, sans avoir
à balayer toutes les étapes :

- **Aller à un état** : cliquer une **ligne** de la séquence (ou l'en-tête
  « État de départ ») saute directement à cet état sur le schéma.
- **Ajouter une manœuvre** : à l'étape affichée, le schéma cible **redevient
  interactif** ; cliquer un organe **insère** une manœuvre (bascule de l'organe)
  **juste après** l'étape courante — la suite de la séquence est **conservée**.
  Les manœuvres ajoutées sont libellées « manœuvre manuelle (expert) »
  (affichées en violet).
- **Supprimer une manœuvre** : bouton **✕** au survol d'une ligne.
- **Supprimer plusieurs manœuvres / un bloc** : **cocher** les cases des lignes
  voulues (**Maj+clic** sur une case sélectionne un **bloc** contigu), puis
  **🗑 Supprimer la sélection**.
- Après toute édition, le statut passe à **ÉDITÉE · N nœud(s)** avec un
  indicateur **= cible** / **≠ cible** (l'état nodal final est recalculé). La
  **sauvegarde** de séquence enregistre alors la liste **éditée telle quelle**
  (aucun re-calcul de l'algorithme).

### Réutiliser des scénarios
- **▷ Rejouer** : recharge un scénario (départ **et** cible sauvegardés).
- **⇧ Comme départ** : la cible sauvegardée devient le **nouvel état de
  départ** — permet de **chaîner** les scénarios (repartir d'une topologie
  validée plutôt que de l'état de base du réseau), puis d'éditer une nouvelle
  cible par-dessus.

---

## 4. API HTTP

Toutes les réponses sont en JSON. Le SVG est renvoyé en ligne (rendu par le
navigateur).

| Méthode & route | Corps | Réponse | Rôle |
|-----------------|-------|---------|------|
| `GET /` | — | HTML | Page de l'IHM |
| `GET /api/postes` | — | `{postes:[…]}` | Liste des postes de test |
| `POST /api/load` | `{vl}` | `{initial_svg, nb_initial, svg, switches, nb_noeuds}` | Charge un poste (départ pristine) |
| `POST /api/toggle` | `{id}` | `{svg, switches, nb_noeuds}` | Bascule un OC (cible) |
| `POST /api/reset` | — | `{svg, switches, nb_noeuds}` | Réinitialise la cible = départ |
| `POST /api/sequence` | — | `{verified, verified_detaillee, ecarts[], message, nb_manoeuvres, manoeuvres[], n_steps, labels[], nb_final, matches_cible, edited}` | Calcule la séquence (cible **détaillée**) ; initialise la séquence **éditable** |
| `GET /api/step?i=k` | — | `{svg, switches[], nb_noeuds, i}` | Image d'animation de l'étape *k* (surlignée) **+ organes cliquables** de l'étape |
| `POST /api/seq_insert` | `{step, id}` | `{goto, manoeuvres[], n_steps, labels[], nb_final, matches_cible, edited}` | Insère une manœuvre basculant `id` **après** l'étape `step` (conserve la suite) |
| `POST /api/seq_delete` | `{index}` | idem `seq_insert` | Supprime la manœuvre n°`index` (1-based) |
| `POST /api/seq_delete_many` | `{indices:[…]}` | idem `seq_insert` | Supprime en une fois plusieurs manœuvres (sélection / bloc) |
| `GET /api/scenarios` | — | `{scenarios:[…]}` | Liste des scénarios sauvegardés |
| `POST /api/save` | `{name, overwrite?}` | `{path, scenarios[]}` ou `{exists:true, name, path}` | Sauvegarde le scénario cible |
| `POST /api/load_scenario` | `{name, mode}` | `{initial_svg, nb_initial, svg, switches, nb_noeuds, vl}` | Recharge (`mode="both"` ou `"as_depart"`) |
| `POST /api/save_sequence` | `{name, overwrite?}` | `{path}` ou `{exists:true, name, path}` | Sauvegarde la séquence **courante** (éditée telle quelle) |

> **Avertissement d'écrasement** : sans `overwrite:true`, si le fichier existe
> déjà, `/api/save` et `/api/save_sequence` renvoient `{exists:true}` (sans
> écrire) ; l'IHM demande alors confirmation d'écrasement ou un nouveau nom.

`switches` : liste d'objets `{id, name, svgId, kind, open}` (DJ/SA), où `svgId`
est l'identifiant de l'élément dans le SVG (clic et surlignage).

---

## 5. Formats de fichiers

### Scénario (`--scenarios-dir/<nom>.json`)
Topologie cible validée, réutilisable comme cible (rejeu) ou comme départ.

```json
{
  "voltage_level_id": "CARRIP3",
  "name": "scenA",
  "depart": {"<switch_id>": false, "...": true},
  "cible":  {"<switch_id>": true,  "...": false},
  "depart_nodale": [["DEP1", "DEP2"], ["..."]],
  "cible_nodale":  [["..."], ["..."]]
}
```

### Séquence (`--sequences-dir/<nom>.json`)
Séquence **courante** (telle qu'éventuellement éditée par l'expert) + topologies
+ **lien vers le scénario**. La liste de manœuvres est sérialisée telle quelle ;
seuls l'état nodal final (`nb_final`) et sa concordance avec la cible
(`matches_cible`) sont recalculés. `edited` vaut `true` si la séquence a été
modifiée à la main (insertions / suppressions).

```json
{
  "voltage_level_id": "CARRIP3",
  "name": "seqA",
  "scenario": "scenA",
  "edited": false,
  "matches_cible": true,
  "nb_final": 2,
  "depart": {"<switch_id>": false},
  "cible":  {"<switch_id>": true},
  "depart_nodale": [["..."]],
  "cible_nodale":  [["..."]],
  "nb_manoeuvres": 1,
  "manoeuvres": [
    {"ordre": 1, "switch_id": "CARRIP3_CARRI3COUPL.1 DJ_OC",
     "action": "OPEN", "raison": "ouverture couplage de barres", "boucle": null}
  ]
}
```

> Une manœuvre **ajoutée manuellement** a pour `raison` « manœuvre manuelle
> (expert) » et `boucle: null`.

### Exploitation pour un test
Un fichier de séquence (ou de scénario) est **autonome**. Pour rejouer la cible
**détaillée** (barre exacte de chaque départ) :

```python
import pypowsybl as pp, json
from expert_op4grid_recommender.manoeuvre.graph import build_vl_graph
from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import determiner_manoeuvres_cible_detaillee

d = json.load(open("tests/manoeuvre/sequences/seqA.json"))
n = pp.network.load("…/grid.xiidm")
vl = d["voltage_level_id"]

n.update_switches(id=list(d["depart"]), open=list(d["depart"].values()))
poste = PosteTopologique.from_graph(build_vl_graph(n, vl), vl)
n.update_switches(id=list(d["cible"]),  open=list(d["cible"].values()))
cible_graph = build_vl_graph(n, vl)

res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)
assert res.is_verified_detaillee     # topologie détaillée atteinte
assert res.ecarts == []
```

> Les fixtures `tests/manoeuvre/scenarios/*.json` sont rejouées **sans
> pypowsybl** (via `build_graph_from_fixture`) par
> `tests/manoeuvre/test_scenarios_sauvegardes.py`.

---

## 6. Spécifications demandées et couverture

| Spécification | Couverture |
|---------------|-----------|
| Modifier **manuellement et interactivement** l'état des DJ/SA depuis une topologie de départ | Clic sur l'organe (schéma cible) ou liste latérale → `/api/toggle` |
| **Valider** la topologie cible avant de calculer | Étape 1 « Valider & sauvegarder » ; le bouton « Calculer » reste verrouillé tant que la cible n'est pas validée |
| **Sauvegarder** la cible pour des tests par ailleurs | Scénario JSON (`/api/save`) avec topologies détaillées + nodales |
| Demander la **séquence de manœuvres** départ → cible | `/api/sequence` (module `determiner_topo_complete_cible`) |
| Affichage **textuel** de la séquence | Panneau « Séquence » |
| **Animation** sur le SLD, manœuvre par manœuvre, organe mis en évidence | Contrôles ◀ ▶ ▶| + surlignage rouge (`/api/step`) |
| Voir **départ en haut** et **cible éditable en bas** | Deux schémas empilés |
| Couleurs **natives** par niveau de tension (63 kV violet) | `topological_coloring`, rendu navigateur |
| **Recharger** une cible sauvegardée pour recalculer | « ▷ Rejouer » |
| Choisir l'**état de départ** depuis une topologie sauvegardée | « ⇧ Comme départ » |
| **Sauvegarder la séquence** avec lien vers ses topologies | `/api/save_sequence` (JSON autonome + champ `scenario`) |
| Atteindre la **topologie détaillée** imposée (barre exacte) + vérification | `determiner_manoeuvres_cible_detaillee` ; statut « DÉTAILLÉE VÉRIFIÉE » / « NODALE OK · N écart(s) » |
| **Avertissement d'écrasement** d'un fichier de sauvegarde existant | Confirmation / renommage (réponse `{exists:true}`) |
| **Replier** un schéma pour agrandir l'autre (grands postes) | Bouton ▾/▸ par en-tête de schéma |
| **Aller directement** à l'état d'une manœuvre (sans balayer) | Clic sur une ligne de séquence → `/api/step` |
| **Ajouter** une manœuvre depuis l'état affiché (schéma interactif) | Clic sur un organe en mode séquence → `/api/seq_insert` (insertion, suite conservée) |
| **Supprimer** une manœuvre / **plusieurs** / un **bloc** | ✕ par ligne (`/api/seq_delete`) ; cases à cocher + Maj+clic + 🗑 (`/api/seq_delete_many`) |
| **Sauvegarder la séquence éditée** telle quelle (sans re-calcul) | `/api/save_sequence` sérialise `seq_manoeuvres` ; champs `edited`, `matches_cible`, `nb_final` |

---

## 7. Notes techniques

- **Couleurs** : pypowsybl encode les couleurs via des variables CSS
  (`var(--sld-vl-color)`). Le navigateur les résout nativement ; aucune palette
  maison n'est appliquée. (Pour un export PNG hors navigateur, voir
  `scripts/render_carrip3_sld.py` qui utilise Chrome headless.)
- **Deux SVG d'un même poste** : les identifiants d'éléments seraient en
  collision dans le DOM. Le schéma de départ a ses ids **préfixés** ; le schéma
  cible conserve ses ids d'origine (cohérents avec le mapping `switch → svgId`
  pour le clic et le surlignage).
- **Animation paresseuse** : `/api/sequence` ne renvoie que les libellés et le
  nombre d'étapes ; chaque image est récupérée à la demande via `/api/step`
  (états pré-calculés côté serveur). Réponse initiale légère.
- **État de départ pristine** : capturé à l'initialisation, insensible aux
  modifications de session (pas de fuite d'état entre postes/scénarios).
- **Séquence éditable côté serveur** : la session conserve une liste
  `seq_manoeuvres` (manœuvres ordonnées) ; insertions/suppressions la modifient
  puis `_rebuild_seq` recompose les états successifs (helper pur `_replay_states`)
  et les surlignages. La dérivation d'une manœuvre manuelle (`_manual_manoeuvre`)
  et la suppression multiple (`_delete_indices`) sont des **fonctions pures**,
  testées sans Flask ni pypowsybl (`tests/manoeuvre/test_ihm_sequence_edit.py`).

---

## 8. Limites

- Mono-utilisateur (serveur de développement Flask, mono-thread).
- `RAN_PP6` (fixture) est absent du réseau France de référence (nommé
  `RAN.PP6`) : 14 des 15 postes de test sont disponibles.
- Les postes multi-sections (ex. CARRIP6, 2 barres × 3 sections) sont gérés ;
  les écarts détaillés résiduels éventuels sont affichés (dégradation gracieuse).
- Les limites de l'algorithme lui-même sont documentées dans
  `docs/manoeuvre_regles.md` (omnibus complexes, couplers non chaînés, etc.).
