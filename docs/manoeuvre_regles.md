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

### R5 — Placement nœud → sections de barres (faisabilité de partition)
Chaque départ atteint une **classe de position** (les SJB qu'il peut rejoindre,
une par barre). Un nœud occupe, pour chacune de ses positions, **une seule
barre** (ses SJB restent connectées via les sectionnements internes). Contraintes :
- nombre de nœuds **mixtes** (≥ 2 positions) ≤ nombre de barres ;
- pour chaque position, nombre de nœuds la requérant ≤ nombre de barres ;
- les **départs fixes** (atteignant une seule barre) contraignent l'affectation.

On retient l'affectation barre↔nœud de **coût minimal**
(`ré-aiguillages + 10 × ouvertures de sectionnement`). Une cible demandant plus
de nœuds que physiquement réalisable est signalée **infaisable**.
- **Code** : `algo._placement_automatique`.
- **Test** : `test_carrip3_3noeuds.py::test_point_entree_unique_cree_3eme_noeud_automatiquement`,
  `::test_infaisable_trois_noeuds_mixtes`, `test_algo.py::test_trois_noeuds_sur_deux_barres_infaisable`.

### R6 — Évaluation de l'état de couplage
Pour un tronçon : si `nbNoeuds < nbBarres` le couplage doit être **fermé**
(barres en parallèle, départs d'un même nœud répartis sur les barres) ; si
`nbNoeuds ≥ nbBarres` il doit être **ouvert** (nœuds distincts par barre).
Implémenté implicitement par le placement R5 : un couplage est ouvert entre SJB
de nœuds différents, fermé entre SJB d'un même nœud.
- **Code** : `algo.determiner_manoeuvres_avec_sections` (calcul `to_open` / `to_close`).
- **Test** : `test_algo.py::test_split_ouvre_le_couplage`, `test_carrip3_manoeuvre.py`.

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

### R8 — Boucle courte (couplage fermé)
Quand un chemin parallèle existe (couplage / sectionnement encore fermé entre
l'ancienne et la nouvelle barre), le ré-aiguillage se fait **sous tension** :
fermer le SA vers la barre cible, ouvrir le SA vers l'ancienne barre. Aucun DJ
n'est ouvert.
- **Code** : `algo._reaiguiller_vers_sjb` (branche `COURTE`).
- **Test** : `test_algo.py::test_split_boucle_courte`,
  `test_carrip3_3noeuds.py::test_boucle_courte_avant_sectionnement_longue_apres`.

### R9 — Boucle longue (pas de chemin parallèle)
Sinon : ouvrir le DJ de cellule (R7) → ouvrir le SA vers l'ancienne barre →
fermer le SA vers la barre cible → refermer le DJ de cellule.
- **Code** : `algo._reaiguiller_vers_sjb` (branche `LONGUE`).
- **Test** : `test_carrip3_3noeuds.py::test_departs_du_3eme_noeud_en_boucle_longue`.

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

### R11 — Ordonnancement de la séquence (`listeDordre`)
Ordre imposé pour minimiser les risques :
1. **fermer** les couplages nécessaires (préparation) ;
2. ré-aiguillages **boucle courte** ;
3. **ouvrir les sectionnements** (sections hors tension) ;
4. **ouvrir les couplages** (DJ) ;
5. ré-aiguillages **boucle longue** vers les sections isolées.
- **Code** : `algo.determiner_manoeuvres_avec_sections` (phases 0 → E).
- **Test** : `test_carrip3_3noeuds.py::test_ordre_sectionnement_avant_couplage`.

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

### R14 — Vérification post-manœuvre
Après application de la séquence, la topologie nodale est recalculée
(`TopologieNodale.from_graph`) et comparée à la cible par isomorphisme de
partition (`meme_topologie`). Le résultat porte `is_verified`.
- **Code** : `algo.determiner_topo_complete_cible` / `determiner_manoeuvres_avec_sections`.
- **Test** : tous les tests `test_carrip3_*` et `test_algo.py` (assertion `is_verified`).

---

## Limites connues (documentées)

| Cas | Statut |
|-----|--------|
| Ré-aiguillage d'omnibus complexes (scission d'un groupe sur deux barres) | partiel |
| Contrôle de court-circuit fin (calcul de potentiel) | simplifié (R12) |
| Postes ≥ 3 barres physiques / topologies multi-tronçons non chaînées | partiel |
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
