# Plan de travail — Dataset « Topologies historiques RTE 7000 » & benchmark de l'algorithme de séquences de manœuvres

> **But** : associer ce dépôt au dataset RTE 7000 (historique de topologies du
> réseau de transport français sur plusieurs années) pour produire **une
> publication** (dataset + benchmark) : extraire les postes et les topologies
> détaillées rencontrées, découper l'historique en **blocs temporels de
> transition** (topologie détaillée de départ → topologie détaillée cible),
> **taguer le type d'intervention** de chaque bloc, puis en dériver un **dataset
> de test/benchmark** pour l'algorithme de séquences de manœuvres
> (`manoeuvre` + façade pluggable).

---

## 0. La logique d'ensemble (pourquoi ce découpage)

Quatre constats structurent le plan :

1. **Le pivot existe déjà.** Une topologie détaillée de poste s'exprime
   nativement dans le dépôt : `CibleDetaillee` = `{switch_id: ouvert ?}`,
   sérialisable, hashable, comparable (`diff`), convertible en partition
   nodale (`topologie_nodale`). Le dataset peut donc être **directement
   exprimé dans les structures du dépôt** — aucune représentation ad hoc.

2. **Un bloc de transition EST un cas de test.** Le couple (état détaillé
   stable avant, état détaillé stable après) d'un poste est **exactement
   l'entrée de la phase B** du séquenceur (`sequencer(poste, cible)`).
   L'historique réel devient mécaniquement un banc d'essai : chaque
   reconfiguration réellement opérée par les dispatchers est un scénario,
   au format déjà consommé par les tests et l'IHM
   (`tests/manoeuvre/scenarios/*.json`).

3. **Si le pas temporel est fin, l'historique contient la séquence réelle.**
   Entre deux états stables, les snapshots intermédiaires capturent les
   manœuvres successives de l'opérateur → une **vérité terrain de
   séquencement** (ordre, nombre de manœuvres, respect de la règle du
   sectionneur). C'est l'argument scientifique le plus fort de la
   publication : comparer la séquence calculée à la séquence **réellement
   exécutée**, pas seulement à une borne basse.

4. **La façade pluggable est le cadre d'évaluation.** La vérification
   indépendante (`verifier_sequence` : partition atteinte, écarts détaillés,
   règle du sectionneur, alertes « 1 ouvrage à la fois ») donne des verdicts
   homogènes quel que soit l'algorithme branché → le benchmark est
   multi-algorithmes par construction (libtopo smooth/aggressive aujourd'hui,
   plugins tiers demain), ce qui positionne la publication comme un
   **benchmark communautaire ouvert**.

Chaîne de production :

```
RTE 7000 (snapshots t0…tN, ~7000 postes, plusieurs années)
   │ Phase 1 : extraction + canonicalisation par poste
   ▼
catalogue {postes, topologies détaillées distinctes, topologies nodales}
   │ Phase 2 : segmentation temporelle par poste
   ▼
blocs de transition {poste, t, topo_depart → topo_cible, états transitoires}
   │ Phase 3 : tagging du type d'intervention (règles + vérités partielles)
   ▼
blocs tagués
   │ Phase 4 : conversion scénarios + stratification + exécution façade
   ▼
dataset de test + résultats de benchmark (smooth/aggressive/plugins, vs réel)
   │ Phase 5
   ▼
publication (dataset paper + benchmark) + dépôt de données (Zenodo/HF)
```

---

## Phase 0 — Audit des données & cadrage (décisions structurantes)

**Objectif** : sécuriser les hypothèses avant d'industrialiser.

- **Inventaire du dataset RTE 7000** : format des instantanés (XIIDM ?
  observations grid2op `.gz` comme celles parcourues par
  `utils/load_training_data.list_all_obs_files` ? export SCADA ?), pas de
  temps (5 min ? horaire ?), période couverte, complétude des états
  d'organes (tous les DJ **et** SA en NODE_BREAKER ? ou états nodaux
  seulement ?), stabilité des identifiants d'organes sur plusieurs années.
- **Décision A — niveau de détail disponible.** Si certains postes ne sont
  connus qu'au niveau nodal (ou BUS_BREAKER), la « topologie détaillée » des
  bornes de bloc devra être **reconstruite** via la phase A de la façade
  (`identifier_topologie_detaillee`) et le bloc tagué `detaillee_reconstruite`
  (à séparer dans les analyses : ce n'est plus une observation).
- **Décision B — évolution structurelle.** Sur plusieurs années, des départs
  apparaissent/disparaissent (cf. `DELETED_LINE_NAME` dans le dépôt), des
  postes sont recâblés. On versionne la **structure** de chaque poste
  (hash du graphe node/breaker hors états) ; un bloc n'est valide que si la
  structure est identique à ses deux bornes ; un changement de version =
  frontière de segmentation, taguée `evolution_structurelle`.
- **Décision C — pas d'échantillonnage et stabilité.** Choisir la durée
  minimale de stabilité définissant un « état stable » (p. ex. ≥ 2 pas
  consécutifs, parametrable) — c'est le filtre anti-bruit de
  télésignalisation et anti-oscillation.
- **Pilote** : calibrer tout le pipeline sur 3–5 postes bien connus du dépôt
  (CARRIP3, MORBRP6, TAVELP7, ROMAIP6, MUHLBP7) × 1–2 mois, en confrontant
  visuellement les blocs trouvés à l'IHM (chargement départ/cible comme
  scénario).

**Livrables** : note de cadrage data (formats, volumétrie, choix A/B/C),
pipeline pilote validé visuellement.

## Phase 1 — Extraction des postes et des topologies rencontrées

**Objectif** : catalogue canonique, dédoublonné, des topologies par poste.

- **Extraction** : généraliser `scripts/extract_test_fixtures.py` en
  `scripts/extract_topology_history.py` — pour chaque snapshot et chaque VL
  NODE_BREAKER : structure (une fois par version) + état détaillé
  `{switch_id: open}` (via `build_vl_graph` ou lecture directe
  `get_switches`).
- **Canonicalisation** : `topologie_id` = hash stable de l'état détaillé
  (paires `(switch_id, open)` triées) ; `nodale_id` = hash de la **partition**
  nodale (`TopologieNodale.partition()`, isomorphe aux noms de nœuds près).
  Deux niveaux car plusieurs topologies détaillées réalisent la même nodale
  (faisceaux équivalents) — distinction déjà au cœur du module
  (`is_verified` vs `is_verified_detaillee`).
- **Volumétrie** : ~7000 postes × N snapshots → traitement **incrémental**
  (ne retraiter un VL que si l'un de ses organes a changé depuis t-1),
  parallélisation par poste, stockage **parquet**.
- **Schéma de sortie** :
  - `postes.parquet` : vl_id, version_structure, nb JdB, nb SJB, nb départs,
    typologie (réutiliser les catégories de `POSTES_CATALOG` : omnibus,
    faisceau partagé, ≥ 3 JdB…), période de validité de la version ;
  - `topologies.parquet` : vl_id, version, topologie_id, nodale_id, états
    d'organes, nb nœuds, première/dernière occurrence, nb d'occurrences,
    durée cumulée.

**Livrables** : les deux tables + statistiques descriptives (nb de topologies
distinctes par poste, distribution par typologie — déjà du contenu de papier).

## Phase 2 — Blocs temporels de transition

**Objectif** : pour chaque poste, les intervalles où la topologie a changé,
bornés par deux états stables.

- **Segmentation** : la série `topologie_id(t)` d'un poste → segments stables
  (≥ durée minimale) ; un **bloc** = (segment stable k, transition, segment
  stable k+1) :
  `{vl_id, version, t_fin_stable_avant, t_debut_stable_apres, topologie_depart_id,
  topologie_cible_id, duree_transition, etats_transitoires:[topologie_id…]}`.
- **États transitoires** : conservés ordonnés — c'est la **séquence réelle**
  approchée (à la résolution du pas). En dériver, quand le pas le permet, la
  liste de manœuvres observées (diff organe à organe entre snapshots
  consécutifs) avec horodatage.
- **Attributs calculés par bloc** (alimentent le tagging et la stratification) :
  diff d'organes par genre (`SwitchKind` : DJ/SA, et
  sectionnement vs couplage via `troncons.py`), diff nodal (nb nœuds avant →
  après), départs isolés avant/après (`nœud 0-barre`), **réversibilité**
  (retour ultérieur à `topologie_depart_id` ? délai ?), durées de stabilité
  avant/après, heure/jour/saison.
- **Filtres qualité** : oscillations (A→B→A en < seuil) repliées et taguées
  `transitoire_bref` ; blocs à structure changée exclus du benchmark
  (gardés au catalogue, tagués).

**Livrable** : `blocs.parquet` + visualisation IHM d'un échantillon (frise des
blocs d'un poste).

## Phase 3 — Tagging du type d'intervention

**Objectif** : à chaque bloc, un ou plusieurs tags interprétables métier.

- **Taxonomie proposée** (multi-label, à valider avec le métier) :

  | Tag | Signature principale |
  |---|---|
  | `consignation_ouvrage` | départ isolé (DJ+SA ouverts, nœud 0-barre) durablement ; retour ultérieur |
  | `remise_en_service` | inverse du précédent |
  | `scission_noeud` | nb nœuds ↑ (ouverture de couplage) — parade topologique / sécurité N-1 |
  | `fusion_noeuds` | nb nœuds ↓ (fermeture de couplage) |
  | `reaiguillage_departs` | barre câblée de ≥ 1 départ change, partition quasi constante |
  | `sectionnement_barre` | ouverture/fermeture de sectionnement (travaux JdB, demi-rame) |
  | `reconfiguration_durable` | nouvel état stable longue durée / récurrence calendaire (schéma d'exploitation, saisonnier) |
  | `transitoire_bref` | A→B→A court, DJ seul (déclenchement / essai) |
  | `evolution_structurelle` | structure du poste changée (hors benchmark) |
  | `inclasse` | reste |

- **Logique de tagging** : règles déterministes sur les attributs de la
  phase 2 (diff par genre d'organe, diff nodal, durée, réversibilité,
  calendrier). Les primitives existent : `CibleDetaillee.diff`,
  `TopologieNodale.meme_topologie`/`partition`, distinction
  sectionnement/couplage, détection des nœuds 0-barre.
- **Vérités partielles pour calibrer** : les CSV de périodes déjà présents
  dans le dépôt (`consignation_manoeuvres_suav_periods.csv` : labels
  `consignation`, `suav_manoeuvre` par ouvrage et intervalle) → croiser les
  blocs avec ces périodes quand elles existent ; mesurer précision/rappel des
  règles sur ce sous-ensemble. Si d'autres journaux d'exploitation sont
  accessibles côté RTE, même usage.
- **Validation humaine** : revue d'un échantillon stratifié (par tag ×
  typologie de poste) dans l'IHM ; figer un **jeu d'or de tags** (~200–500
  blocs annotés) servant de référence dans le papier.

**Livrables** : `blocs_tagues.parquet`, module `manoeuvre/dataset/tagging.py`
(règles testées unitairement), rapport précision/rappel vs vérités partielles
et jeu d'or.

## Phase 4 — Dataset de test & benchmark du séquenceur

**Objectif** : transformer les blocs en suite de tests exécutable et en
résultats de benchmark.

- **Conversion** : chaque bloc retenu → scénario au format existant
  `{voltage_level_id, name, depart{sw:open}, cible{sw:open}, depart_nodale,
  cible_nodale}` + bloc `meta` (tags, durées, séquence réelle si disponible,
  provenance anonymisée). Fixtures structurelles par (poste, version) pour
  exécuter **sans pypowsybl ni XIIDM** en CI (mécanisme
  `fixture_loader` existant).
- **Critères d'inclusion** : structure identique aux bornes, diff ≥ 1 organe,
  hors `transitoire_bref`/`evolution_structurelle` ; **cap par poste** pour ne
  pas sur-représenter les postes très actifs.
- **Stratification** : typologie de poste (2 JdB / 3+ JdB / omnibus /
  faisceaux partagés…) × type d'intervention × taille de diff. Splits
  publics : `calibration` (réglage d'algos) / `test` (résultats du papier),
  avec seed et listes figées.
- **Exécution via la façade** (`PlanificateurTopologie`) pour chaque scénario
  et chaque algo (libtopo smooth, libtopo aggressive, plugins candidats) :
  - verdicts : `is_verified`, `is_verified_detaillee`, écarts, violations
    sectionneur, alertes « 1 ouvrage à la fois » ;
  - coûts : nb manœuvres vs **borne basse** (nb d'organes différents) vs
    **séquence réelle observée** (quand pas fin) ; temps de calcul ;
  - cas de dégradation gracieuse (`noeuds_non_realisables`) documentés.
- **Intégration CI** : sous-ensemble représentatif gelé en goldens
  (mécanisme `test_golden_sequences` + `UPDATE_GOLDENS`), le reste en suite
  longue optionnelle (marqueur pytest `dataset`).

**Livrables** : `scripts/build_manoeuvre_dataset.py` (blocs → scénarios +
fixtures), dataset versionné, `scripts/run_benchmark.py` + tables/figures de
résultats, tests CI.

## Phase 5 — Publication

- **Artefact données** : dépôt Zenodo (DOI) ou HuggingFace Datasets —
  `postes/topologies/blocs_tagues` (parquet) + scénarios JSON + fixtures +
  datasheet (composition, collecte, limites, usages prévus) + scripts de
  reconstruction.
- **Papier** (dataset & benchmark ; cibles possibles : PSCC, IEEE PES GM /
  Transactions, NeurIPS Datasets & Benchmarks, PowerTech) :
  1. motivation : planification de manœuvres & assistant dispatcher, absence
     de benchmark public sur séquences de manœuvres réelles ;
  2. construction du dataset (phases 1–3) + statistiques descriptives
     (fréquence de reconfiguration par typologie, distribution des
     interventions, saisonnalité) ;
  3. l'algorithme (portage libTOPO : règles R1–R14, modes, replis
     transactionnels) et le **cadre pluggable** comme protocole d'évaluation
     ouvert (contrats A/B/C, vérification indépendante) ;
  4. résultats : taux de cibles atteintes exactement, manœuvres vs borne
     basse vs séquences réelles, analyse des échecs par typologie ;
  5. appel à contributions (plugins via entry points).
- **Conformité / anonymisation** (décision RTE en amont, conditionne tout le
  reste) : publiabilité des noms de postes/organes et horodatages exacts ;
  sinon pseudonymisation **cohérente par poste** (les ids restent stables
  intra-poste pour que les scénarios restent exécutables) et dates dégradées
  (semaine/saison). Licences : code MPL-2.0 (existant), données CC-BY-4.0
  (à valider RTE).
- **Reproductibilité** : tag de version du dépôt, seeds, environnement figé,
  notebook des figures.

---

## Risques & parades

| Risque | Parade |
|---|---|
| Dataset seulement nodal (pas d'états SA) sur tout ou partie | Reconstruire la détaillée via phase A, taguer `detaillee_reconstruite`, analyser séparément |
| Bruit de télésignalisation / oscillations | Durée minimale de stabilité, repli A→B→A, tag `transitoire_bref` |
| Renommages / recâblages sur plusieurs années | Versionnage structurel par poste, table de correspondance d'ids par période |
| Volumétrie (7000 postes × années) | Extraction incrémentale (diff d'organes), parallélisation par poste, parquet |
| Sur-représentation de quelques postes très actifs | Cap par poste + stratification |
| Tags non fiables | Calibration sur CSV de consignation existants + jeu d'or annoté via IHM, précision/rappel publiés |
| Confidentialité | Décision RTE en phase 0 ; pseudonymisation cohérente ; datasheet explicite |
| Le séquenceur échoue sur des familles de postes (regroupements arbitraires, barres de réserve — limites connues documentées) | C'est un **résultat** du papier (analyse d'échecs), pas un bloqueur ; alimente la feuille de route plugins |

## Jalons indicatifs

| Jalon | Contenu | Effort indicatif |
|---|---|---|
| J1 | Phase 0 : audit + pilote 5 postes validé IHM | 1–2 sem. |
| J2 | Phase 1 : catalogue postes+topologies complet | 2–3 sem. |
| J3 | Phase 2 : blocs de transition + attributs | 1–2 sem. |
| J4 | Phase 3 : tagging + validation (jeu d'or) | 2–3 sem. |
| J5 | Phase 4 : dataset de test + benchmark exécuté | 2–3 sem. |
| J6 | Phase 5 : dépôt de données + draft de papier | 3–4 sem. |

## Questions ouvertes (à trancher pour démarrer la phase 0)

1. **Format et accès** du dataset RTE 7000 : XIIDM par snapshot ? observations
   grid2op `.gz` (comme `load_training_data`) ? autre export ? Où est-il
   stocké et quelle volumétrie ?
2. **Pas temporel** réel et période couverte (conditionne la capture des
   séquences réelles — l'atout n°1 du papier).
3. Les **états des sectionneurs (SA)** sont-ils présents partout, ou seulement
   les DJ / la topologie nodale ?
4. **Publiabilité** : noms réels des postes/organes et horodatages exacts
   autorisés ? (sinon : pseudonymisation cohérente, prévue au plan).
5. **Venue visée** pour la publication (papier dataset vs papier méthode) —
   cela arbitre l'équilibre phases 3/4 vs 5.
