# Module manœuvre — spécification des règles implémentées

> Traçabilité des **règles métier** du séquencement de manœuvres
> (topologie nodale cible → séquence détaillée d'organes de coupure).
> Chaque règle référence sa fonction et son test.

Code : `expert_op4grid_recommender/manoeuvre/` — Tests : `tests/manoeuvre/`.

Point d'entrée unique : `algo.determiner_topo_complete_cible(poste, topo_cible)`.

---

## Vocabulaire

| Terme | Définition |
|-------|------------|
| **SJB** | Section de jeu de barres (busbar section). |
| **Barre** (jeu de barres) | Rail électrique courant sur la longueur du poste ; peut être sectionné en plusieurs sections longitudinales. |
| **Sectionnement (de barre)** | Sectionneur (SA / DISCONNECTOR) reliant *directement* deux SJB d'une **même** barre. Manœuvrable **hors charge** uniquement. |
| **Couplage** | Travée à disjoncteur (DJ / BREAKER) reliant deux **barres** différentes. Le DJ coupe le courant nominal. |
| **DJ de cellule** | Disjoncteur d'ensemble d'une cellule de départ, côté sélecteurs de barre ; met le départ hors tension. |
| **SA d'aiguillage** | Sectionneur sélecteur reliant une cellule de départ à une barre donnée. |
| **Boucle courte** | Ré-aiguillage sans coupure : le départ reste sous tension (un chemin parallèle existe via le couplage fermé). |
| **Boucle longue** | Ré-aiguillage avec mise hors tension du départ (ouverture du DJ de cellule). |
| **Tronçon** | Composante structurelle de SJB ; `nb_jeux_barres` = nombre de barres distinctes du tronçon. |
| **Nœud électrique** | Bus : ensemble d'équipements au même potentiel. |

---

## Règles

### R1 — Faisabilité / nettoyage des ouvrages
Tout départ de la topologie cible doit exister physiquement dans le poste ;
sinon la cible est déclarée infaisable. (Portage de `connectAndDeconnectOuvrageHS`.)
- **Code** : `determiner_topo_complete_cible` (contrôle des départs manquants).
- **Test** : `test_algo.py` (cibles standard), `test_carrip3_manoeuvre.py`.

### R2 — Court-circuit d'identité
Si la topologie nodale courante satisfait déjà la cible (même partition), on
ne génère **aucune** manœuvre.
- **Code** : `determiner_topo_complete_cible` (test `meme_topologie` initial).
- **Test** : `test_algo.py::test_noop_aucune_manoeuvre`.

### R3 — Distinction sectionnement vs couplage
Une liaison inter-SJB est un **sectionnement** si elle ne comporte que des
sectionneurs (pas de DJ) — les deux SJB sont alors sur la même barre ; sinon
c'est un **couplage** (présence d'un DJ).
Détection des barres : nommage RTE (entier de tête du `busbar_section_id`) en
primaire, repli structurel par connectivité de sectionnement (chemins sans DJ).
- **Code** : `troncons._detecter_barres`, `algo._inter_sjb_couplers`
  (`_InterSjbCoupler.is_sectionnement`).
- **Test** : `test_troncons.py` (nb de barres par poste), `test_carrip3_3noeuds.py`.

### R4 — Tronçonnement structurel
Le tronçonnement regroupe les SJB indépendamment de l'état courant des organes
(structure du poste). `nb_jeux_barres` du tronçon = nombre de barres distinctes.
- **Code** : `troncons.construire_tronconnement`.
- **Test** : `test_troncons.py` (CARRIP3 : 1 tronçon, 2 barres, DJ de couplage).

### R5 — Placement nœud → sections de barres (modèle par segments)
À chaque nœud cible on attribue un **groupe connexe de SJB** (dans le graphe des
couplers). Un nœud peut donc **occuper plusieurs barres** : les couplers internes
au groupe sont fermés (un seul potentiel), ceux entre groupes ouverts.
- un **départ reste sur sa barre courante** si elle appartient au groupe de son
  nœud (ré-aiguillage évité) ;
- contrainte de connexité (un groupe = une partie connexe du graphe des
  couplers) et de faisabilité (chaque départ atteint une SJB de son groupe) ;
- on retient l'affectation de **coût minimal** :
  `5 × ré-aiguillages + manœuvres de couplers + 4 × ouvertures de sectionnement`.
  Les ré-aiguillages étant fortement pénalisés, **fermer un couplage** est
  privilégié quand cela suffit (cf. R6).
- une cible demandant plus de nœuds que physiquement réalisable (p.ex. nœuds
  mixtes > barres) n'a aucune affectation valide → **infaisable**.
- **Code** : `algo._placement_automatique`.
- **Test** : `test_carrip3_3noeuds.py::test_point_entree_unique_cree_3eme_noeud_automatiquement`,
  `::test_infaisable_trois_noeuds_mixtes`, `test_algo.py::test_trois_noeuds_sur_deux_barres_infaisable`.

### R6 — Évaluation de l'état de couplage (utiliser les barres disponibles)
Lorsqu'il y a **moins de nœuds que de barres**, on **utilise les barres
disponibles** : un nœud s'étend sur plusieurs barres en gardant le **couplage
fermé**, plutôt que de ramener tous les départs sur une seule barre. Concrètement
(découle de R5) : un couplage est **fermé** entre SJB d'un même nœud, **ouvert**
entre SJB de nœuds différents. Les départs restent sur leurs barres, sans
ré-aiguillage, dans la mesure du possible.
- **Code** : `algo._placement_automatique` (groupes multi-barres),
  `determiner_manoeuvres_avec_sections` (`to_open` / `to_close`).
- **Test** : `test_algo.py::test_cible_un_noeud_referme_couplage_sans_reaiguiller`,
  `::test_split_ouvre_le_couplage`.

### R6bis — Fermeture **sûre** des couplers (règle du sectionneur)
La fermeture d'un coupler respecte l'invariant des sectionneurs :
- un **DJ de couplage** (BREAKER) peut relier deux barres de **potentiels
  différents** : c'est l'organe de couplage, fermé en premier — il
  équipotentialise alors ses barres ;
- un **sectionneur** (DISCONNECTOR entre deux SJB) n'est fermé que si ses deux
  côtés sont **déjà équipotentiels** (reliés par ailleurs, p.ex. par un DJ
  fermé juste avant) **ou** si l'un des côtés est **hors tension** ;
- si un sectionneur à fermer reste non sûr (deux potentiels vifs, sans pont DJ
  possible), on **dé-énergise d'abord le côté « stub »** (le moins chargé) en
  ré-aiguillant ses départs (boucle longue) vers une SJB du même nœud déjà
  équipotentielle au côté conservé, **puis** on ferme le sectionneur (section
  morte). C'est une **manœuvre préalable** au sens demandé.

Ordre de fermeture : DJ d'abord, puis sectionneurs devenus sûrs, enfin
dé-énergisation + fermeture des sectionneurs « stub ».
- **Code** : `determiner_manoeuvres_avec_sections` (Phase 0 : `_equipotentiel`,
  `_departs_cables`, dé-énergisation des stubs).
- **Test** : `test_algo.py::test_fusion_un_noeud_respecte_regle_sectionneur`
  (rejoue la séquence et vérifie chaque manœuvre de sectionneur).

### R7 — Ré-aiguillage : DJ d'ensemble de cellule uniquement
Pour ré-aiguiller un départ en **boucle longue**, on ouvre **uniquement le DJ
d'ensemble de la cellule** (côté sélecteurs de barre), ce qui met la cellule
hors tension et suffit pour basculer ensuite les sectionneurs. On n'ouvre
**pas** les disjoncteurs propres aux équipements situés en aval (cas omnibus :
un DJ de cellule commun alimente plusieurs équipements — inutile et à éviter
d'ouvrir un DJ « le long de la charge »).
- **Code** : `algo._own_breakers_to_sjb` (parcours depuis le côté barre, arrêt
  au premier nœud de branchement omnibus).
- **Test** : `test_carrip3_3noeuds.py::test_boucle_longue_ouvre_seulement_le_dj_de_cellule`.

### Invariant de sécurité des sectionneurs
Un sectionneur (SA d'aiguillage ou sectionnement de barre) ne se manœuvre **hors
charge** : il ne doit jamais relier ni séparer deux points de **potentiels
différents**. Concrètement :

- **Fermeture** d'un SA reliant deux barres : autorisée seulement si les deux
  barres sont **déjà au même potentiel** (déjà le même nœud électrique), ou si
  l'un des côtés est **hors tension**. Sinon → court-circuit.
- **Ouverture** d'un SA (alors que les deux barres sont reliées par lui) :
  autorisée seulement si les deux barres **restent au même potentiel** après
  (encore reliées par ailleurs → aucun courant coupé), ou si l'un des côtés est
  **hors tension**.

Conséquence pratique sur le ré-aiguillage : le choix **boucle courte vs longue
est déduit automatiquement** de ce critère (et non d'une heuristique de phase) —
courte si la barre actuelle et la barre cible sont déjà le même nœud électrique
(hors cellule), longue sinon.
- **Code** : `algo._meme_noeud_hors_cellule` (test d'équipotentialité hors
  cellule), `algo._reaiguiller_vers_sjb` (décision automatique).

Cet invariant gouverne l'ordre des manœuvres en R8, R9 et R10.

### R8 — Boucle courte (couplage fermé)
Quand un chemin parallèle existe (couplage encore fermé → les deux barres sont
au **même potentiel**), le ré-aiguillage se fait **sous tension** : fermer le SA
vers la barre cible **puis** ouvrir le SA vers l'ancienne barre. Le pont
temporaire entre les deux SA est sûr (même potentiel) ; aucun DJ n'est ouvert.
- **Code** : `algo._reaiguiller_vers_sjb` (branche `COURTE`).
- **Test** : `test_algo.py::test_split_boucle_courte`,
  `test_carrip3_3noeuds.py::test_boucle_courte_avant_sectionnement_longue_apres`.

### R9 — Boucle longue (pas de chemin parallèle)
Les deux barres étant à des potentiels **différents** (couplage ouvert), l'ordre
est impératif pour respecter l'invariant ci-dessus :
1. **ouvrir le DJ de cellule** (R7) → départ hors tension, jonction des SA morte ;
2. **ouvrir le SA vers l'ancienne barre** (la jonction reste morte) ;
3. **fermer le SA vers la barre cible** (jamais simultané avec l'ancien SA) ;
4. **refermer le DJ de cellule** → départ ré-alimenté sur la barre cible.
On ne ferme **jamais** le SA cible tant que l'ancien SA est fermé (ce serait un
pont entre deux barres de tensions différentes = court-circuit).
- **Code** : `algo._reaiguiller_vers_sjb` (branche `LONGUE`).
- **Test** : `test_carrip3_3noeuds.py::test_departs_du_3eme_noeud_en_boucle_longue`,
  `::test_boucle_longue_ouvre_ancien_sa_avant_fermer_nouveau`,
  `::test_boucle_longue_jamais_deux_sa_fermes_simultanement`.

### R10 — Règle du sectionneur de barre (dé-énergisation)
Un sectionneur de barre ne se manœuvre que **hors charge**. Pour scinder une
barre en deux nœuds (créer un nœud au-delà du nombre de barres) :
1. ré-aiguiller (boucle courte) tous les départs de la section à isoler sur
   l'autre barre, afin de la laisser **hors tension** ;
2. **ouvrir le sectionnement** (sûr, section morte) ;
3. ré-aiguiller (boucle longue) les départs du nouveau nœud sur la section
   désormais isolée.
La section « hors tension » est jugée sur le **câblage des SA** (et non sur la
connectivité électrique globale, qui voit tout relié tant que le couplage est
fermé).
- **Code** : `algo.determiner_manoeuvres_avec_sections` (phases ré-aiguillage /
  parking / ouverture sectionnement), `_wired_sjbs`.
- **Test** : `test_carrip3_3noeuds.py::test_sectionnement_ouvert_hors_tension`,
  `::test_3noeuds_atteint_et_verifie`.

### R10bis — Isoler **par les disjoncteurs d'abord** (minimiser les ré-aiguillages)
Avant de dé-énergiser une section par ré-aiguillage (parking, coûteux), on
**isole d'abord** cette section autant que possible par **ouverture des
disjoncteurs de couplage et de tronçonnement environnants** — organes qui
**coupent la charge** et n'imposent donc aucune contrainte « hors tension ».
Le ré-aiguillage n'intervient qu'**en dernier recours**, sur le **résidu**.

Conséquences sur l'algorithme :
1. Une section isolable de la référence par **simple ouverture d'un couplage
   (DJ)** — ou par un sectionnement **déjà ouvert** — **ne nécessite aucun
   parking** : ses départs restent en place. Seules les sections **incidentes à
   un sectionnement fermé destiné à s'ouvrir** sont conservées dans
   `sjb_isoles` (filtre `sect_isol_sjbs`).
2. Avant d'ouvrir un **sectionnement** (phase C), on **ouvre d'abord les
   couplages destinés à s'ouvrir qui touchent la section à isoler** (frontière
   ou interne). Cela réduit la section à mettre hors tension à son **résidu**
   minimal, puis :
   - **0 ouvrage** restant → ouverture directe (section morte) ;
   - **n ≥ 1 ouvrages** restants → ouverture momentanée de leur **DJ d'ouvrage**
     (mise hors tension), ouverture du sectionneur, puis refermeture des DJ.
3. Exemple **PALUNP3 → 4 nœuds** : section 1.2 reliée à 2.2 par `COUPL.2` (DJ).
   En ouvrant `COUPL.2` **avant** `SS.1.12`, la section 1.2 se réduit à son seul
   résidu au lieu d'entraîner la dé-énergisation des 5 départs de 2.2 — la
   séquence passe de **40 à 12 manœuvres** (cf. séquence experte ≈ 9).
- **Code** : `algo.determiner_manoeuvres_avec_sections` — filtre
  `sect_isol_sjbs` de `sjb_isoles` ; phase C, ouverture des couplages incidents
  à `side_isol` avant le sectionneur (`_live_graph_sans`,
  `_ouvrages_energises_sur`).
- **Test** : `test_palunp3_isolation_disjoncteurs.py`.

### R10ter — Modes de dé-énergisation : **smooth** vs **agressif**
Le séquenceur détaillé (`determiner_manoeuvres_cible_detaillee(..., mode=)`)
propose deux stratégies pour ouvrir un sectionnement de barre :

- **smooth** (défaut) — **un seul ouvrage hors tension à la fois**. Pour mettre
  une section hors tension, chaque ouvrage est **garé** (ré-aiguillé), **un par
  un**, sur une **section de parking** : une SJB atteignable distincte, **hors
  section isolée** de préférence, et **équipotentielle** si possible — auquel cas
  le ré-aiguillage est en **boucle courte** et l'ouvrage n'est **pas** déconnecté
  du tout (`parking_sjb`). Les ouvrages garés sont **ramenés** ensuite (boucle
  longue) sur la section isolée. **Exception** : si aucun parking n'existe pour
  un ouvrage, il est dé-énergisé **en place** (les ouvrages sans parking peuvent
  alors être hors tension simultanément). `target_sjb` est amorcé avec la barre
  cible exacte (`cible_busbar`) pour éviter les déplacements inutiles.
- **agressif** (`_sequence_detaillee_aggressive`) — dé-énergiser **en lot** :
  ouvrir en une fois les DJ de tous les ouvrages concernés (côté le plus petit
  de chaque sectionnement + ouvrages dont un SA change), commuter
  couplages/sectionnements et SA **hors tension**, puis ré-alimenter **une seule
  fois**. Bien moins de manœuvres, au prix de **plusieurs ouvrages momentanément
  hors tension** simultanément.

Les deux modes atteignent la **même** topologie détaillée cible, sectionneurs
ouverts hors tension (`_verifier_securite_sectionneurs`). Ordres de grandeur :
SSAVOP3 → 6 nœuds : smooth **62**, agressif **30** ; CZTRYP6 → 3 nœuds : smooth
**20**, agressif **8**.
- **Code** : `algo.determiner_manoeuvres_cible_detaillee` (param `mode`),
  `determiner_manoeuvres_avec_sections` (parking `parking_sjb`, param
  `cible_busbar`), `_sequence_detaillee_aggressive`.
- **Vérification** : `_verifier_un_seul_hors_tension` (smooth) rejoue la séquence
  et signale tout chevauchement de ré-aiguillages (> 1 ouvrage hors tension par
  parking) — intégré aux `ecarts` du mode smooth.
- **Test** : `test_ssavop3_modes.py`, `test_cztryp6_3noeuds.py`.
- **IHM** : sélecteur « Mode : Smooth / Agressif » avant le calcul.

### R11 — Ordonnancement de la séquence (`listeDordre`)
Ordre imposé pour minimiser les risques :
1. **fermer** les couplages nécessaires (préparation) ;
2. ré-aiguillages **boucle courte** ;
3. **ouvrir les sectionnements** (sections hors tension) ;
4. **ouvrir les couplages** (DJ) ;
5. ré-aiguillages **boucle longue** vers les sections isolées.

**Exception (R10bis)** : un couplage (DJ) touchant la **section à isoler** est
ouvert **dès la phase C, avant** le sectionnement de cette section, afin de la
réduire à son résidu (isolement par les disjoncteurs d'abord). Cela ne
contrevient pas à la sûreté (le DJ coupe la charge) ; l'ordre 3-avant-4 reste la
règle pour les couplages **non incidents** à une section en cours d'isolement
(p.ex. CARRIP3, dont la section isolée est une feuille sans couplage adjacent).
- **Code** : `algo.determiner_manoeuvres_avec_sections` (phases 0 → E ; phase C
  ouvre les couplages incidents avant le sectionneur).
- **Test** : `test_carrip3_3noeuds.py::test_ordre_sectionnement_avant_couplage`,
  `test_palunp3_isolation_disjoncteurs.py`.

### R12 — Contrôle de court-circuit avant fermeture de couplage
On ne ferme un couplage que si les deux SJB visent le **même nœud cible**
(même potentiel attendu) ; sinon la fermeture est refusée (risque de
court-circuit) et signalée.
- **Code** : `algo.determiner_manoeuvres_avec_sections` (phase 0, garde).
- **Statut** : simplifié (égalité de nœud cible, pas de calcul de potentiel fin).

### R13 — Suppression des manœuvres sans effet
La séquence est rejouée depuis l'état initial ; toute manœuvre plaçant un OC
dans l'état où il se trouve déjà est retirée. Les bascules réelles (ex.
ouverture/fermeture d'un DJ en boucle longue) sont conservées.
- **Code** : `algo._optimiser_sequence`.
- **Test** : couvert indirectement (cohérence des séquences vérifiées).

### R14 — Vérification post-manœuvre (nodale)
Après application de la séquence, la topologie nodale est recalculée
(`TopologieNodale.from_graph`) et comparée à la cible par isomorphisme de
partition (`meme_topologie`). Le résultat porte `is_verified`.
- **Code** : `algo.determiner_topo_complete_cible` / `determiner_manoeuvres_avec_sections`.
- **Test** : tous les tests `test_carrip3_*` et `test_algo.py` (assertion `is_verified`).

### R15 — Topologie détaillée imposée (barre exacte + vérification détaillée)
Quand une **topologie détaillée cible** est imposée (état précis de chaque
organe, donc la barre exacte de chaque départ — plus spécifique que la seule
partition nodale), on vise cet état exact :
1. atteindre la **topologie nodale** cible de façon sûre (R1-R14) ;
2. **raffiner** : ramener chaque départ sur sa **barre imposée** (ré-aiguillage
   boucle courte, sûr car les barres d'un même nœud sont équipotentielles) — par
   défaut les départs reviennent sur leur barre d'origine, au prix de
   **manœuvres supplémentaires** (« requinçonçage ») ;
3. **vérifier la topologie détaillée** (barre de chaque départ + état de chaque
   coupler) ; les **écarts** résiduels sont consignés (`ecarts`,
   `is_verified_detaillee`).
- **Code** : `algo.determiner_manoeuvres_cible_detaillee`, `_ecarts_detailles`.
- **Test** : `test_algo.py::test_cible_detaillee_atteinte_avec_barres_exactes`,
  `::test_cible_detaillee_signale_les_ecarts`,
  `test_scenarios_sauvegardes.py::test_carrip3_1noeud_requinconcage`.
- **IHM** : la cible éditée étant détaillée, l'IHM appelle ce mode et affiche
  « DÉTAILLÉE VÉRIFIÉE » ou « NODALE OK · N écart(s) ».

### R16 — Dégradation gracieuse
Si une étape n'est pas réalisable en sécurité (pas de SJB tampon pour isoler une
section, départ inatteignable…), l'algorithme **ne s'interrompt pas** : il
consigne l'**écart** (`res.ecarts`) et poursuit ; `topo_obtenue` est toujours
renseignée. L'IHM affiche alors « NODALE OK · N écart(s) » au lieu d'un échec
opaque.
- **Code** : `determiner_manoeuvres_avec_sections` (collecte des écarts au lieu
  de `return` anticipé), `determiner_manoeuvres_cible_detaillee`.
- **Test** : `test_scenarios_sauvegardes.py` (postes réels rejoués).

---

## Postes multi-sections

Les postes à plusieurs **sections par barre** (ex. **CARRIP6** : 2 barres × 3
sections = 6 SJB ; chaîne de couplers
`1.1–1.2–1.3–(DJ)–2.3–2.2–2.1`) sont gérés : pour ouvrir un sectionnement, les
départs de la section à isoler sont garés temporairement sur une SJB
**équipotentielle** accessible (même destinée à être isolée ensuite), puis
ré-aiguillés en boucle longue/courte. Validé par
`test_scenarios_sauvegardes.py` (CARRIP6 → 5 nœuds, topologie détaillée
vérifiée).

## Limites connues (documentées)

| Cas | Statut |
|-----|--------|
| Ré-aiguillage d'omnibus complexes (scission d'un groupe sur deux barres) | partiel |
| Contrôle de court-circuit fin (potentiel / déphasage) | simplifié (R12) |
| Topologies de couplers non chaînées (≥ 3 barres en anneau) | partiel |
| Nœuds mêlant départs connectés et déconnectés | partiel |

---

## Correspondance avec le C++ `libTOPO`

| Règle | Fonction C++ (`TOPOPoste.cc`) |
|-------|-------------------------------|
| R1 | `connectAndDeconnectOuvrageHS` |
| R4 | `buildTronconnement`, `CelluleBarresTopo::tronconneGraph` |
| R5 | `identifySuperTronconnement`, `getTronconnementBesoinReaiguillage2barres` |
| R6 | `evalueEtatCouplage` |
| R7–R9 | `reaiguillage2barres`, `CelluleDepartTopo::reaiguillageBarres` |
| R10 | `tronconnerSJB`, `isolateDepartFromBarre` |
| R11 | `listeDordre` |
| R14 | fin de `determineTopoCompleteCible` (`memeTopologie`) |
