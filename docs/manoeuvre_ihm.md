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
| `POST /api/sequence` | — | `{verified, message, nb_manoeuvres, manoeuvres[], n_steps, labels[]}` | Calcule la séquence |
| `GET /api/step?i=k` | — | `{svg}` | Image d'animation de l'étape *k* (surlignée) |
| `GET /api/scenarios` | — | `{scenarios:[…]}` | Liste des scénarios sauvegardés |
| `POST /api/save` | `{name}` | `{path, scenarios[]}` | Sauvegarde le scénario cible |
| `POST /api/load_scenario` | `{name, mode}` | `{initial_svg, nb_initial, svg, switches, nb_noeuds, vl}` | Recharge (`mode="both"` ou `"as_depart"`) |
| `POST /api/save_sequence` | `{name}` | `{path}` | Sauvegarde la séquence générée |

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
Séquence générée + topologies + **lien vers le scénario**.

```json
{
  "voltage_level_id": "CARRIP3",
  "name": "seqA",
  "scenario": "scenA",
  "verified": true,
  "message": "Topologie cible atteinte et vérifiée.",
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

### Exploitation pour un test
Un fichier de séquence est **autonome** :

```python
import pypowsybl as pp, json
from expert_op4grid_recommender.manoeuvre.graph import build_vl_graph
from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique, TopologieNodale
from expert_op4grid_recommender.manoeuvre.algo import determiner_topo_complete_cible

d = json.load(open("tests/manoeuvre/sequences/seqA.json"))
n = pp.network.load("…/grid.xiidm")
n.update_switches(id=list(d["depart"]), open=list(d["depart"].values()))
poste = PosteTopologique.from_graph(build_vl_graph(n, d["voltage_level_id"]), d["voltage_level_id"])
cible = TopologieNodale.from_node_groups(d["voltage_level_id"], d["cible_nodale"])
res = determiner_topo_complete_cible(poste, cible)
assert res.is_verified == d["verified"]
assert res.nb_manoeuvres == d["nb_manoeuvres"]
```

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

---

## 8. Limites

- Mono-utilisateur (serveur de développement Flask, mono-thread).
- `RAN_PP6` (fixture) est absent du réseau France de référence (nommé
  `RAN.PP6`) : 14 des 15 postes de test sont disponibles.
- Les limites de l'algorithme lui-même sont documentées dans
  `docs/manoeuvre_regles.md` (omnibus complexes, ≥ 3 barres, etc.).
