---
title: Expert Op4Grid — IHM Manœuvre
emoji: 🔌
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mpl-2.0
---

# Expert Op4Grid — IHM Manœuvre (dataset RTE 7000)

Interface web pour **identifier et séquencer les manœuvres** de reconfiguration
d'un poste électrique (topologie NODE_BREAKER), bâtie sur le module `manoeuvre`
d'[Expert Op4Grid Recommender](https://github.com/marota/Expert_op4grid_recommender).

Ce Space est **branché directement sur le dataset RTE 7000**
([`OpenSynth/D-GITT-RTE7000-2021`](https://huggingface.co/datasets/OpenSynth/D-GITT-RTE7000-2021)) :
les situations réseau (instantanés XIIDM du réseau France, un toutes les 5 min)
sont **téléchargées à la demande** au moment où vous choisissez une date/heure —
rien n'est embarqué dans l'image.

## Déroulé

1. **📅 Dataset RTE7000** — choisissez une **date** (un accès rapide propose des
   journées documentées) puis une **heure** (instantané, pas de 5 min ; **midi
   présélectionné**) et cliquez **Charger la situation**. Le réseau France de
   cette date/heure est téléchargé puis chargé.
2. **Poste** — la liste de **tous les voltage levels** (postes NODE_BREAKER) de
   la situation se peuple ; sélectionnez-en un (menu épinglé, exploration par
   typologie, ou recherche). Sa **topologie détaillée** (SLD) s'affiche.
3. **Éditer & séquencer** — modifiez la topologie cible (clic sur les organes ou
   volet nodal), **validez la cible**, puis **calculez la séquence de manœuvres**
   (modes smooth / agressif, algorithmes pluggables) et **animez-la** étape par
   étape. Changez de date/heure à tout moment : le poste courant est rechargé à
   la nouvelle situation.

## Un utilisateur par instance

L'IHM garde un **unique état réseau** en mémoire (singleton mono-utilisateur,
requêtes sérialisées), donc un Space en cours sert **un utilisateur à la fois**.
Pour plusieurs personnes, utilisez **Duplicate this Space** (chaque copie est
isolée).

## Ressources & notes

- Pile scientifique légère (`pypowsybl` + `networkx` + `pandas` + `flask`). Le
  réseau France complet (~7000 postes) se charge en quelques secondes au premier
  choix d'une date ; les changements de poste à date constante sont instantanés.
- **Internet sortant requis** : le Space récupère les instantanés depuis
  `huggingface.co`. Un secret **`HF_TOKEN`** (jeton de lecture) est optionnel —
  il desserre le rate-limit anonyme du CDN.
- **Stockage éphémère** : le cache des instantanés et les scénarios/séquences
  sauvegardés ne survivent pas à un redémarrage du Space.
