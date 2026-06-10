# Passation de session — façade pluggable, IHM, dataset RTE 7000

> Document de reprise pour une nouvelle session sur la branche
> `claude/clever-curie-ly6jcx`. Résume ce qui est **fait** (4 commits), ce qui
> est **planifié**, et la **prochaine tâche** (traiter le dataset Hugging Face
> `OpenSynth/D-GITT-RTE7000-2021`, bloqué jusqu'ici par l'allowlist réseau).

---

## 1. Ce qui est fait sur cette branche

### Commit `a26531a` — Couche pluggable des 3 phases de calcul (`manoeuvre/plugins/`)

Trois contrats substituables (PEP 544, doc `docs/manoeuvre_plugins.md`) :

- **Phase A** `IdentificateurTopologieDetaillee.identifier(poste, topo_cible)` :
  topologie **nodale** cible → topologie **détaillée** cible ;
- **Phase B** `SequenceurManoeuvres.sequencer(poste, cible)` : topologie
  détaillée cible → **séquence de manœuvres** ;
- **Phase C** `PlanificateurNodal.planifier(poste, topo_cible)` : nodale →
  séquence + détaillée, en une passe.

Pièces : type pivot sérialisable `CibleDetaillee` (`{switch_id: ouvert?}`,
`from_graph`/`from_manoeuvres`/`to_graph`/`diff`) ; registre par phase
(`register`/`get`/`disponibles` + entry points
`expert_op4grid_recommender.manoeuvre`, nom `<phase>.<nom>`) ; façade
`PlanificateurTopologie` (compose les phases manquantes : A+B⇒C, C⇒A par
rejeu) ; **vérification indépendante** `verifier_sequence` (partition, écarts
détaillés, règle du sectionneur, alertes R10ter — un plugin mensonger est
démasqué) ; adaptateurs natifs « libtopo » (défaut des 3 phases). Tests :
`tests/manoeuvre/test_plugins_interface.py` (13 cas) ; verrou de surface
publique mis à jour.

### Commit `57ade3e` — IHM migrée sur la façade + sélection d'algos

`scripts/manoeuvre_ihm.py` : `/api/nodale_to_detaillee` → phase A,
`/api/sequence` → phase B, avec l'algorithme **sélectionné par phase**
(`Session.algos`, `GET/POST /api/algos`). Front : sélecteurs « Algo » (volet
nodal = A, panneau Séquence = B), badge `algo <nom>`. Tests :
`tests/manoeuvre/test_ihm_algo_selection.py` (7 cas, dont plugin menteur
démasqué dans la payload).

### Commit `109950d` — Plan de travail publication (`docs/plan_dataset_rte7000.md`)

Plan en 6 phases (audit → extraction → blocs → tagging → dataset de test /
benchmark → publication), avec logique, risques/parades, jalons. À lire en
premier.

### Commit `7e5a8c7` — Pipeline dataset historique (`manoeuvre/dataset/`)

- `timeline.py` : chronologies d'états d'organes par poste, `topologie_id`
  canonique, états stables (`min_stabilite`), **blocs de transition**
  (départ → cible détaillées + états transitoires + **manœuvres observées**
  ordonnées = séquence réelle approchée), oscillations A→bruit→A repliées,
  réversibilité ;
- `tagging.py` : tags multi-label (`consignation_ouvrage`,
  `remise_en_service`, `scission_noeud`, `fusion_noeuds`,
  `reaiguillage_departs`, `sectionnement_barre`, `reconfiguration_durable`,
  `inclasse`) — régime structurel (fixture du poste) ou repli par nommage RTE ;
- `extraction.py` : blocs → scénarios (format `tests/manoeuvre/scenarios`) +
  séquences observées (format `sequences`) + `stats_blocs` ;
- `structure.py` : `PosteTopologique` depuis fixture JSON, sans pypowsybl ;
- `dgitt.py` : **adaptateur du dataset HF** — lecture parquet/CSV format long
  `(timestamp, voltage_level, switch, état)`, auto-détection de colonnes
  (`_ALIASES`), erreurs explicites listant les colonnes trouvées. **Écrit
  défensivement : le schéma réel n'a pas pu être inspecté** (réseau bloqué) ;
- `scripts/build_rte7000_blocks.py` : CLI bout-en-bout (`--input`,
  `--fixtures`, `--vl`, `--min-stabilite`, `--seuil-durable`) + mode
  `--demo` (chronologies reconstruites depuis les 18 séquences réelles du
  dépôt → **19 blocs / 11 postes**, tags validés : CARRIP3 1 nœud →
  `fusion_noeuds`, PALUNP3 expert → 2 blocs `consignation` puis
  `remise_en_service`).

Tests : `tests/manoeuvre/test_dataset_timeline.py` (13 cas, dont reproduction
exacte d'une séquence réelle de 20 manœuvres). Suite `tests/manoeuvre/`
complète : **918 passed**.

---

## 2. Le point bloquant (résolu pour les nouvelles sessions)

`huggingface.co` était hors allowlist réseau de l'environnement → impossible
de télécharger `OpenSynth/D-GITT-RTE7000-2021` dans la session précédente.
L'utilisateur a **ajouté `huggingface.co` + `cdn-lfs.huggingface.co` à
l'allowlist** ; le changement s'applique aux **nouvelles sessions** (rebuild
du cache d'environnement). Si le téléchargement échoue encore : vérifier
l'environnement modifié, et ajouter aussi `cas-bridge.xethub.hf.co` /
`transfer.xethub.hf.co` (backend de stockage Xet de HF).

## 3. Prochaine tâche (nouvelle session)

1. **Vérifier l'accès** :
   `curl -s https://huggingface.co/api/datasets/OpenSynth/D-GITT-RTE7000-2021`.
2. **Télécharger** (taille à vérifier d'abord via l'API `/tree/main`) :
   ```bash
   pip install -U huggingface_hub
   hf download OpenSynth/D-GITT-RTE7000-2021 --repo-type dataset \
       --local-dir data/dgitt_rte7000_2021
   ```
   Ne **pas** commiter les données brutes dans le dépôt.
3. **Inspecter le schéma réel** (fichiers, colonnes, pas temporel, période,
   les états SA sont-ils présents ?) et **ajuster
   `manoeuvre/dataset/dgitt.py`** (alias de colonnes, format long vs large,
   conversion des états). Documenter le schéma constaté dans
   `docs/plan_dataset_rte7000.md` (section « État d'avancement »).
4. **Pilote** sur les postes ayant une fixture (tagging structurel) :
   ```bash
   python scripts/build_rte7000_blocks.py --input data/dgitt_rte7000_2021 \
       --fixtures tests/manoeuvre/fixtures --output out_pilote \
       --vl CARRIP3 --vl MORBRP6 --vl TAVELP7 --vl ROMAIP6 --vl MUHLBP7
   ```
   Vérifier la cohérence des blocs/tags (au besoin via l'IHM).
5. **Passe complète** (tous postes ; tagging par nommage là où la structure
   manque), produire `blocs.jsonl`, `scenarios/`, `sequences/`, `stats.json`.
   Commiter les **stats + un échantillon** de scénarios (pas les données
   brutes), + les ajustements de `dgitt.py` avec tests.
6. **Benchmark (phase 4 du plan)** : sur les blocs à structure connue, lancer
   le séquenceur via la façade (`PlanificateurTopologie().sequencer`, modes
   smooth/aggressive) vers chaque cible et comparer aux **séquences
   observées** (nb manœuvres, verdicts `is_verified_detaillee`, écarts,
   alertes). Script suggéré : `scripts/run_benchmark.py`.

## 4. Repères

| Quoi | Où |
|---|---|
| Plan publication 6 phases | `docs/plan_dataset_rte7000.md` |
| Doc couche pluggable | `docs/manoeuvre_plugins.md` |
| Doc module manoeuvre (conventions, invariants) | `expert_op4grid_recommender/manoeuvre/CLAUDE.md` |
| Pipeline dataset | `expert_op4grid_recommender/manoeuvre/dataset/` |
| CLI pipeline | `scripts/build_rte7000_blocks.py` (`--demo` marche sans données) |
| Tests | `pytest tests/manoeuvre/` (918 ✓ ; deps : `pip install pytest networkx pandas pypowsybl flask`) |
| Formats scénario/séquence | `tests/manoeuvre/scenarios/`, `tests/manoeuvre/sequences/` |
| Fixtures structurelles de postes | `tests/manoeuvre/fixtures/` |

Conventions : développer sur `claude/clever-curie-ly6jcx`, commits descriptifs
en français (style des commits existants), ne jamais muter `poste.graph`,
suite `tests/manoeuvre/` verte avant chaque push.
