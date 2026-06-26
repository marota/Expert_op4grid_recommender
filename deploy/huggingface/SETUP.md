# Déployer l'IHM de manœuvre sur un HuggingFace Docker Space

Ce dossier scaffolde un déploiement **mono-conteneur** : un process Flask sert
l'IHM de manœuvre (HTML statique + API) sur le port **7860**, en **mode
dataset** — les situations réseau (instantanés XIIDM du dataset RTE 7000) sont
**téléchargées à la demande** depuis HuggingFace. Rien de lourd n'est embarqué
dans l'image.

Dimensionné pour **un utilisateur par Space** (l'IHM garde un unique état réseau
en mémoire). Pour plusieurs personnes, chacun clique **Duplicate this Space**.

## Ce qui a été câblé

| Fichier | Rôle |
|---|---|
| `Dockerfile` (racine) | Image mono-stage : Python + `flask`/`pypowsybl`/`networkx`/`pandas`, lance l'IHM sur `:7860` en mode dataset. |
| `.dockerignore` (racine) | Réduit le contexte de build (`.git`, `tests`, `docs`, `data`, …). |
| `deploy/huggingface/README.md` | Le **README du Space** (frontmatter `sdk: docker`, `app_port: 7860`) + page d'accueil. |
| `scripts/manoeuvre_ihm.py` | `--grid` optionnel + `--dataset` / `--host` / `--port` (et env `PORT`, `DGITT_*`, `HF_TOKEN`) ; endpoints `/api/dataset/{config,timestamps,load}`. |
| `expert_op4grid_recommender/manoeuvre/dataset/source.py` | Couche « instantané par date » : liste + télécharge un snapshot HF (stdlib pur, cache local, md5). |

Le mode dataset est activé par `--dataset` (ou dès que `--grid` est absent). En
local sans `--grid`, l'IHM démarre directement en mode dataset ; avec `--grid`,
le comportement historique (réseau local) est strictement préservé et le bandeau
dataset reste masqué.

## Pas de Git LFS nécessaire

Contrairement à Co-Study4Grid, **aucun binaire n'est embarqué** : le dataset est
récupéré au runtime depuis HuggingFace. Il n'y a donc ni `.gitattributes` LFS ni
gros fichier dans l'historique à gérer.

## Variables d'environnement (configurables côté Space)

| Variable | Défaut | Rôle |
|---|---|---|
| `DGITT_REPO` | `OpenSynth/D-GITT-RTE7000-2021` | Dataset HuggingFace source (2021 ; existe aussi en 2022 / 2023). |
| `DGITT_DEFAULT_DATE` | `2021-01-03` | Date proposée par défaut dans l'IHM. |
| `DGITT_CACHE_DIR` | `/home/user/app/.cache/dgitt` | Cache local des instantanés (éphémère sur un Space). |
| `HF_TOKEN` | *(absent)* | **Optionnel** : jeton de lecture HF pour desserrer le rate-limit anonyme du CDN. Le mettre en **secret** du Space. |
| `MANOEUVRE_ENABLE_OSM` | `1` | « Explorer la journée » : **repli OSM/Overpass** pour les coordonnées absentes du plan de masse committé. `0` le désactive (la carte reste alimentée par le plan de masse). |
| `MANOEUVRE_GEO_SNAPSHOT` | `$DGITT_CACHE_DIR/postes_rte_geo.json` | Chemin de l'instantané de coordonnées **OSM** résolu/persisté. **Par défaut dans le cache** → pointer `DGITT_CACHE_DIR` sur le stockage persistant suffit à le faire survivre. |
| `PORT` | `7860` | Port d'écoute (HF expose `:7860`). |

### « Explorer la journée » — carte des postes

**Rien à configurer, aucun accès réseau requis** : les coordonnées viennent du
**plan de masse RTE committé** (`manoeuvre/dataset/grid_layout_rte.json`, ~98 % des
postes, embarqué dans l'image). La carte s'affiche dès la 1ʳᵉ exploration ;
l'en-tête indique la source et la couverture (`coord. : layout (4723/4811, 98%)`).
Notes :

- **Repli OSM** : pour les ~2 % de postes absents du plan (ou si le plan est
  retiré), l'IHM interroge **OpenStreetMap / Overpass** (`overpass-api.de`,
  `ref:FR:RTE` = `substation_id`), persiste le résultat et l'offre au
  téléchargement (bouton **⬇ coordonnées**). Adosser `DGITT_CACHE_DIR=/data/dgitt`
  au **stockage persistant HF** fait survivre ce cache (et les instantanés XIIDM)
  aux redémarrages.
- **Mémoire** : l'exploration charge **3 réseaux France** (minuit/midi/23 h)
  séquentiellement ; le pic reste sous ~2 Go → le tier **CPU basic (16 Go)** suffit.
- **Diagnostic** : si la carte reste vide (cas rare : plan absent **et** OSM
  bloqué), le bandeau indique la cause ; l'IHM reste utilisable via le **classement
  en liste**.

## Étapes de déploiement

1. **Créer le Space** — sur huggingface.co : *New → Space → Docker → Blank*.

2. **Pousser un snapshot orphelin vers le Space** (un seul commit, sans
   historique) :

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>   # une fois

   git checkout --orphan hf-deploy
   cp deploy/huggingface/README.md README.md   # HF attend le frontmatter à la racine
   # L'image ne COPIE que expert_op4grid_recommender/ + scripts/. docs/ et data/
   # portent des binaires (PNG) inutiles au Space et rejetés par le Hub HF (hors
   # Xet/LFS) ; on les retire du snapshot (avec tests/) → push binaire-free et léger.
   rm -rf docs data tests
   git add -A
   # Filet : retirer du snapshot indexé d'éventuels binaires résiduels (venv,
   # profils, pptx) que le Hub HF rejette hors Xet/LFS (no-op si absents).
   git rm -r --cached --ignore-unmatch 'venv*' '*.prof' '*.pptx'
   git commit -m "Deploy Expert Op4Grid — IHM Manœuvre"
   git log --oneline hf-deploy                  # DOIT être un commit unique
   git -c protocol.version=0 push -f space hf-deploy:main
   git checkout -f claude/zen-ptolemy-i7qdog
   git branch -D hf-deploy
   ```

   (`protocol.version=0` contourne une erreur de négociation que certains
   réseaux rencontrent contre HF.) HuggingFace lit le `Dockerfile` racine + le
   frontmatter du README et construit l'image ; le premier build est long (wheel
   `pypowsybl`), les suivants réutilisent les couches.

3. **(Optionnel) Secret `HF_TOKEN`** — Space → *Settings → Variables and
   secrets* → ajouter le secret `HF_TOKEN` (jeton de lecture) pour fiabiliser les
   téléchargements.

4. **Utiliser** — ouvrir le Space, choisir une date/heure → **Charger la
   situation** → sélectionner un poste → éditer/séquencer.

## Redéploiement automatique (GitHub Action)

`.github/workflows/deploy-huggingface.yml` rejoue le push orphelin
automatiquement à chaque push — sur `main` **ou** sur une branche de dev
`claude/**` — qui **touche l'interface** : le script/les assets de l'IHM
(`scripts/manoeuvre_ihm.py`, `scripts/manoeuvre_ihm_assets/**`), le module
`expert_op4grid_recommender/manoeuvre/**` sur lequel tourne l'IHM, le `Dockerfile`
ou la config de déploiement (`deploy/huggingface/**`). Les push qui ne modifient
que la doc/les tests **ne redéploient pas**. Un push sur une branche de dev donne
une **preview live avant merge** (le Space reflète la **dernière** branche poussée ;
re-pousser `main` le ramène à la prod). `workflow_dispatch` permet aussi un
redéploiement manuel de n'importe quelle ref. Il est **inerte** tant que les
éléments suivants ne sont pas définis dans **Settings → Secrets and variables →
Actions** du dépôt GitHub :

| Type | Nom | Valeur |
|---|---|---|
| Secret | `HF_TOKEN` | jeton HuggingFace **write** ayant accès au Space |
| Variable | `HF_SPACE` | chemin du Space, ex. `your-user/expert-op4grid-ihm` |
| Variable | `HF_USERNAME` | *(optionnel)* le compte HF propriétaire du jeton, si le Space est sous une **org** |

## Tester l'image localement (recommandé)

```bash
docker build -t eo4g-ihm .
docker run --rm -p 7860:7860 eo4g-ihm
# → ouvrir http://localhost:7860, choisir une date, charger, sélectionner un poste
```

Le conteneur a besoin d'un **accès internet sortant** vers `huggingface.co` pour
lister et télécharger les instantanés.
