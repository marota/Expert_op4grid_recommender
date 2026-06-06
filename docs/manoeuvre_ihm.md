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
┌──────────────────────┬─────────────────────────────────┬────────────────────┐
│ PANNEAU LATÉRAL      │ SCHÉMA — TOPOLOGIE DE DÉPART     │ VOLET NODAL        │
│                      │ (SLD pypowsybl, couleurs nœud)  │                    │
│ • Poste (sélecteur)  ├─────────────────────────────────┤ Topo nodale DÉPART │
│ • ↺ État de départ   │ SCHÉMA — TOPOLOGIE CIBLE        │ (lecture seule)    │
│ 1 · Valider la cible │ clic organe = bascule ; anim.   ├────────────────────┤
│ 2 · Calculer séq.    ├─────────────────────────────────┤ Topo nodale CIBLE  │
│ • Scénarios          │ Contrôles ◀ ▶ Lecture ▶|        │ (éditable : chips) │
│ • Nœuds électriques  │ Séquence (texte) + 💾           │ ⚙ Calculer la      │
│                      │                                 │ topo détaillée     │
└──────────────────────┴─────────────────────────────────┴────────────────────┘
```

- **Schéma de départ** (centre haut, bandeau bleu) : topologie détaillée
  initiale, **non modifiable** ; sert de référence.
- **Schéma cible** (centre bas, bandeau orange) : topologie détaillée
  **éditable** ; c'est aussi là que se déroule l'animation de la séquence.
- **Volet nodal** (droite) : représentation **schématique en « vue bus »** (un
  **nœud** = barre horizontale colorée ; chaque **branche** = départ vertical
  portant son **libellé** détaillé et sa **valeur de flux** en MW) de la topologie
  nodale de départ et d'une **cible nodale éditable par glisser-déposer**. Voir §3
  *Éditer une topologie nodale cible*. Repliable via ◂.
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
   bas pour basculer son état (ouvert/fermé). Le nombre de nœuds cible évolue en
   direct.
3. **Valider & sauvegarder la cible** (étape 1) : nomme et persiste le scénario
   (départ + cible). **Obligatoire** avant le calcul.
4. **Calculer la séquence** (étape 2, débloqué après validation) : choisir le
   **mode** (Smooth, défaut, ou Agressif) puis lancer le calcul départ → cible ;
   statut **VÉRIFIÉE / NON VÉRIFIÉE** (+ badge `mode`).
   - **Smooth** : **un seul ouvrage hors tension à la fois** — chaque branche
     est garée (ré-aiguillée) une par une sur une autre section avant d'ouvrir le
     sectionneur ; séquence plus longue mais douce.
   - **Agressif** : dé-énergise en lot (moins de manœuvres, mais plusieurs
     ouvrages momentanément hors tension simultanément).
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

### Éditer une topologie nodale cible (étape intermédiaire)
Plutôt que de basculer les organes un à un, l'expert peut raisonner au niveau
**nodal** (quelles branches sur quel nœud électrique) dans le **volet de droite**,
puis demander un **calcul de la topologie détaillée d'intérêt** réalisant cette
partition. Le pont nodal → détaillé s'appuie sur l'algorithme
`determiner_topo_complete_cible(poste, topo_cible)`.

- **Représentation** : rendu **schématique en SVG, en « vue bus »** comparable au
  schéma détaillé. Chaque nœud électrique est une **barre horizontale** dont la
  **couleur est celle utilisée par la vue détaillée** (`topological_coloring` du
  SLD : même nœud électrique ⇒ même couleur), repérée par un badge `N0`, `N1`… Ses
  branches sont des **départs verticaux** — *du haut* au-dessus de la barre, *du
  bas* en dessous, comme dans la vue détaillée — **triés de gauche à droite** par
  leur abscisse SLD. Chaque branche porte son **libellé** (extrait du SLD, donc
  **strictement identique** au schéma détaillé : `C.REG1`, `AT762`, `TR634`…) et sa
  **valeur de flux** (P en MW, état de départ). Les nœuds sont **empilés
  verticalement**. Le volet montre la topologie nodale de **départ** (lecture
  seule) et une **cible nodale éditable** initialisée **avec la topologie de départ**.
- **Élargir le volet** : un **séparateur déplaçable** (entre le schéma détaillé et
  le volet nodal) permet d'élargir la 3ᵉ colonne en réduisant la 2ᵉ, afin
  d'afficher tous les nœuds de la partition dans la largeur.

L'édition se fait par **glisser-déposer** :
- **Réaiguiller des départs** : glisser une branche sur une autre barre l'y
  déplace. Pour en déplacer plusieurs d'un coup, **cliquer** d'abord les branches
  voulues (sélection surlignée, inter-nœuds) puis glisser l'une d'elles.
- **Fusionner des nœuds** : glisser une **barre** sur une autre barre (fusion).
- **Créer un nœud** : bouton **＋ Nœud** (nœud vide, ou contenant la sélection
  courante) — puis y glisser des départs.
- **Réinitialiser** : **= départ** ramène la cible à la partition de départ ;
  **∅ Désélectionner** vide la sélection.

Les **ouvrages déconnectés** (organe ouvert → non raccordés à une barre) ne sont
**pas** des nœuds électriques : ils sont listés à part (chips **⚠ Ouvrages
isolés**, sous chaque schéma) et n'occupent pas d'espace en barre. Pour en
**reconnecter** un, le **glisser sur une barre** (il rejoint ce nœud) ; côté
serveur, les isolés restants sont **laissés déconnectés** (hors partition cible).
Le compteur de nœuds exclut les isolés.
- **⚙ Calculer la topologie détaillée d'intérêt** : envoie la partition nodale
  (`/api/nodale_to_detaillee`) ; l'IHM **charge l'état détaillé réalisant** cette
  cible dans le schéma du bas (devient la nouvelle cible détaillée, à **valider**
  avant calcul de séquence). Le statut indique **✓** si la cible nodale est
  intégralement réalisable, ou un message de **réalisation partielle** (nœuds non
  réalisables, écarts) en cas de dégradation gracieuse de l'algorithme.
  Le volet nodal cible est alors **resynchronisé sur la topologie réalisée**
  (partition, **couleurs** topological_coloring et ouvrages isolés renvoyés dans
  `nodale` par `/api/nodale_to_detaillee`) : il prend les mêmes nœuds et couleurs
  que le schéma détaillé obtenu.

> Les nœuds vides sont automatiquement retirés et la partition renumérotée
> `N0…Nk` après chaque opération ; le nombre de nœuds cible est affiché en direct.

**Synchronisation détaillé ↔ nodal.** La topologie **détaillée** fait foi : le
volet nodal cible **reflète** la partition de l'état détaillé courant. Toute
**édition du détail** (bascule d'un organe, `↺ État de départ`) **ou rechargement
de scénario** (`▷ Rejouer`) **resynchronise** la cible nodale (partition, couleurs,
ouvrages isolés) — par ex. ouvrir le DJ d'un départ le fait passer en *ouvrage
isolé* dans le volet nodal, et recharger un scénario à N nœuds affiche bien ces N
nœuds côté nodal. Le
glisser-déposer nodal et `= départ` sont des **propositions** de partition
(le détail n'est pas encore modifié) ; **⚙ Calculer…** les réalise, charge la
cible détaillée et resynchronise le volet nodal sur la topologie **obtenue**
(d'où, en cas de réalisation partielle, un volet nodal qui correspond bien au
détail réalisé et non à la proposition).

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

### Re-éditer la cible détaillée après calcul
Une séquence calculée met le schéma du bas en mode **animation** (clic = insertion
de manœuvre). Pour **revenir éditer la cible détaillée** sans repartir de zéro, le
bouton **✎ Modifier la cible** (barre d'animation) bascule le schéma en mode
**édition** : les contrôles d'animation sont désactivés et chaque clic sur un
organe modifie de nouveau la cible. Dès qu'un organe est modifié, un **bandeau
rouge** signale que *« la séquence ci-dessous n'atteint plus cet état cible »* et
la liste de séquence est grisée. Il faut alors **re-valider** la cible puis
**recalculer la séquence** (qui efface l'obsolescence). Le bouton **▶ Revenir à la
séquence** ré-affiche l'animation de la séquence (obsolète) sans rien recalculer.

### Construire une séquence **entièrement manuelle**
Une fois la cible **validée**, le bouton **✋ Séquence manuelle** démarre une
séquence **vierge** : l'expert la construit lui-même, organe par organe.
- Le schéma **du bas** part de l'**état de départ** et est **interactif** :
  chaque clic sur un organe **ajoute une manœuvre** (bascule depuis l'état
  courant) et fait avancer la vue d'un cran.
- Le schéma **du haut** affiche la **cible à atteindre** (référence statique).
- Le statut indique **MANUELLE · N nœud(s)** avec **= cible** / **≠ cible** ;
  on suit ainsi la convergence vers la topologie visée.
- Navigation (clic sur une ligne), suppression (✕ / sélection multiple) et
  **sauvegarde** fonctionnent comme pour une séquence calculée.

### Mise en évidence « topologie cible atteinte »
Dès que l'**état affiché** (animation automatique **ou** construction manuelle)
**est** la topologie **cible** — même partition nodale —, la vue du poste (bas)
est **encadrée d'un halo jaune** et le bandeau affiche « ✓ topologie cible
atteinte ». L'indicateur est recalculé **à chaque étape** côté serveur
(`step_view` renvoie `reached = meme_topologie(état affiché, cible)`), de sorte
qu'il s'allume exactement lorsque la cible est atteinte et s'éteint sinon.

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
| `GET /api/postes` | — | `{postes:[…], all:[…]}` | `postes` = liste **épinglée** (jeu de test + 7 postes 400 kV à **3 jeux de barres** identifiés) ; `all` = **tous** les postes NODE_BREAKER de la situation (champ de recherche) |
| `POST /api/load_grid` | `{path}` | `{ok, postes:[…], all:[…]}` / `400 {ok:false, error}` | Charge **dynamiquement** une autre situation réseau `.xiidm` (chemin **côté serveur**) et réinitialise la session ; 400 propre si fichier introuvable/illisible (session inchangée) |
| `POST /api/load` | `{vl}` | `{initial_svg, nb_initial, svg, switches, nb_noeuds, nodale_depart, nodale_cible}` | Charge un poste (départ pristine) — **n'importe quel** VL NODE_BREAKER ; inclut les partitions nodales |
| `POST /api/toggle` | `{id}` | `{svg, switches, nb_noeuds, nodale}` | Bascule un OC (cible) ; `nodale` = vue nodale resynchronisée |
| `POST /api/reset` | — | `{svg, switches, nb_noeuds, nodale}` | Réinitialise la cible = départ |
| `POST /api/cible` | — | `{svg, switches, nb_noeuds, nodale}` | Vue de la cible détaillée **courante** (sans la modifier) — pour revenir l'éditer alors qu'une séquence est calculée |
| `POST /api/nodale` | — | `{nodale_depart, nodale_cible}` | Partitions nodales (cible initialisée = départ) ; `nodale_*` = `{groups[[…]], labels{id:nom}, types{id:type}, flows{id:MW}, dirs{id:TOP\|BOTTOM}, order{id:x}, colors{id:#hex}, isolated[…]}`. `labels`/`dirs`/`order`/`colors` sont **extraits du SLD** (libellés, côté, ordre horizontal et **couleur du nœud `topological_coloring`** identiques à la vue détaillée) ; `flows` provient d'une charge de réseau ; `isolated` liste les départs **déconnectés** (composante sans barre) |
| `POST /api/nodale_to_detaillee` | `{groups, isolated?}` | `{svg, switches, nb_noeuds, is_verified, message, ecarts[], noeuds_non_realisables[[…]], nb_obtenu, nb_vise, nodale}` | Calcule la **topologie détaillée d'intérêt** réalisant la cible nodale `groups` (les `isolated` sont laissés hors partition) et la charge comme cible détaillée ; `nodale` = `{groups, colors, isolated}` de la topologie **réalisée** (resynchronise le volet nodal) |
| `POST /api/sequence` | `{mode?}` | `{verified, verified_detaillee, ecarts[], message, nb_manoeuvres, manoeuvres[], n_steps, labels[], nb_final, matches_cible, edited, mode}` | Calcule la séquence (cible **détaillée**) ; `mode` = `"smooth"` (défaut) ou `"aggressive"` |
| `GET /api/step?i=k` | — | `{svg, switches[], nb_noeuds, i, reached}` | Image d'animation de l'étape *k* (surlignée) **+ organes cliquables** ; `reached` = l'état affiché est la topologie cible |
| `POST /api/seq_insert` | `{step, id}` | `{goto, manoeuvres[], n_steps, labels[], nb_final, matches_cible, edited}` | Insère une manœuvre basculant `id` **après** l'étape `step` (conserve la suite) |
| `POST /api/seq_delete` | `{index}` | idem `seq_insert` | Supprime la manœuvre n°`index` (1-based) |
| `POST /api/seq_delete_many` | `{indices:[…]}` | idem `seq_insert` | Supprime en une fois plusieurs manœuvres (sélection / bloc) |
| `POST /api/manual_start` | — | `{cible_svg, cible_nb, goto, manoeuvres[], n_steps, labels[], …}` | Démarre une séquence **manuelle vierge** (départ → cible affichée en référence) |
| `GET /api/scenarios` | — | `{scenarios:[…]}` | Liste des scénarios sauvegardés |
| `POST /api/save` | `{name, overwrite?}` | `{path, scenarios[]}` ou `{exists:true, name, path}` | Sauvegarde le scénario cible |
| `POST /api/load_scenario` | `{name, mode}` | `{initial_svg, nb_initial, svg, switches, nb_noeuds, vl, nodale_depart, nodale_cible}` | Recharge (`mode="both"` ou `"as_depart"`) ; `nodale_cible` = vue nodale de la cible chargée |
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
| Modifier **manuellement et interactivement** l'état des DJ/SA depuis une topologie de départ | Clic sur l'organe du schéma cible → `/api/toggle` |
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
| **Construire une séquence entièrement manuelle** (interagir avec le départ, cible en référence) | Bouton ✋ → `/api/manual_start` (séquence vierge) puis clics organe → `/api/seq_insert` |
| **Signaler visuellement** l'atteinte de la topologie cible | Halo jaune sur la vue du poste + « ✓ topologie cible atteinte » (champ `reached` de `/api/step`) |
| **Éditer une topologie nodale cible** (réaiguillage / fusion / nouveau nœud) | Volet nodal SVG « vue bus » (barre + départs verticaux, libellés SLD + flux MW) ; **glisser-déposer** (départ→barre = réaiguillage, barre→barre = fusion) côté client |
| **Élargir** le volet nodal pour afficher tous les nœuds | Séparateur déplaçable entre colonnes 2 et 3 (`#ndresize`) |
| **Ouvrages déconnectés** présentés en liste (pas en nœud), reconnectables | Liste « ⚠ Ouvrages isolés » (chips) ; glisser sur une barre = reconnexion ; `isolated` dans `/api/nodale*` |
| **Re-éditer la cible détaillée** après calcul de séquence + signaler l'obsolescence | Bouton **✎ Modifier la cible** (`/api/cible`) ; bandeau « la séquence n'atteint plus cet état cible » |
| Demander un **calcul de topologie détaillée d'intérêt** depuis la cible nodale | `/api/nodale_to_detaillee` → `determiner_topo_complete_cible` + rejeu, chargé comme cible détaillée |

---

## 7. Notes techniques

- **Front-end externalisé** : le HTML/CSS/JS de l'IHM vit dans
  `scripts/manoeuvre_ihm_assets/index.html` (≈ 600 lignes), chargé au démarrage
  du module et servi tel quel par la route `GET /` (`PAGE`). Le `.py` ne contient
  plus de bloc HTML embarqué — édition du front sans toucher au serveur Python.
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
- **Vue nodale — logique pure testable** : le parsing du SLD (`_parse_feeder_meta`
  pour libellés/direction/abscisse, `_parse_node_colors` pour les couleurs
  topologiques, `_decode_svg_id`), la détection des ouvrages isolés
  (`_isolated_assets`, sur graphe NX) et la normalisation de partition
  (`_normalize_groups`) sont des **fonctions de module pures**, testées sans réseau
  (`tests/manoeuvre/test_ihm_nodale_edit.py`). Le pipeline complet
  (`nodale_payload` / `nodale_state` / `nodale_to_detaillee`, cohérence détaillé ↔
  nodal, isolés, rechargement de scénario) est couvert en **intégration** sur le
  réseau `four_substations` (`tests/manoeuvre/test_ihm_nodale_integration.py`).

---

## 8. Limites

- Mono-utilisateur (serveur de développement Flask, mono-thread).
- **Postes à ≥ 3 jeux de barres** : gérés (placement N-barres + réalisateur
  connectivité-based). Les 7 postes 400 kV à 3 JdB identifiés (`SSV.OP7`,
  `TAVELP7`, `TRI.PP7`, `ARGOEP7`, `CHESNP7`, `COR.PP7`, `CERGYP7`) sont épinglés ;
  le champ de **recherche** donne accès à tout poste NODE_BREAKER de la situation.
  État détaillé et limites restantes : `docs/postes_n_jeux_de_barres.md`.
- **Chargement de situation** : `/api/load_grid` recharge un `.xiidm` côté
  serveur ; l'upload de fichier depuis le navigateur n'est pas (encore) géré.
- Les postes multi-sections (ex. CARRIP6, 2 barres × 3 sections) sont gérés ;
  les écarts détaillés résiduels éventuels sont affichés (dégradation gracieuse).
- Les limites de l'algorithme lui-même sont documentées dans
  `docs/manoeuvre_regles.md` (omnibus complexes, couplers non chaînés, etc.) et
  `docs/postes_n_jeux_de_barres.md` (reste à faire séquenceur / discovery).
