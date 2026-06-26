# Expert Op4Grid — IHM Manœuvre : image mono-conteneur pour un HuggingFace
# Docker Space.
#
# Un seul process Flask sert l'IHM de manœuvre (HTML statique + API) sur le port
# 7860, en **mode dataset** : les situations réseau (instantanés XIIDM du dataset
# RTE 7000) sont téléchargées **à la demande** depuis HuggingFace au moment où
# l'utilisateur choisit une date/heure — rien n'est embarqué dans l'image.
#
# Build context = racine du dépôt :  docker build -t eo4g-ihm .
# Test local              :  docker run --rm -p 7860:7860 eo4g-ihm
#                            puis ouvrir http://localhost:7860

FROM python:3.11-slim-bookworm

# libgomp1 : runtime OpenMP lié par certaines wheels scientifiques (numpy).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Les Spaces HuggingFace exécutent le conteneur en uid 1000 ("user"). On garde
# l'app sous son HOME pour que les écritures runtime (cache des instantanés
# téléchargés) tombent sur un chemin inscriptible.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dépendances (couche propre pour le cache). L'IHM de manœuvre ne dépend que de
# flask + pypowsybl + networkx + pandas : le module ``manoeuvre`` n'a aucune
# autre dépendance externe. On évite ainsi tout le tas scientifique du
# recommandeur (grid2op, expertop4grid, scipy, matplotlib…) — image légère.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        flask \
        "pypowsybl>=1.13.0" \
        networkx \
        pandas

# Code applicatif : le package (importé via sys.path par le script) + l'IHM et
# ses assets HTML. Pas de données embarquées : la source est le dataset distant.
COPY --chown=user expert_op4grid_recommender/ ./expert_op4grid_recommender/
COPY --chown=user scripts/ ./scripts/

# Source des données = dataset RTE 7000, téléchargé à la demande dans ce cache
# (éphémère sur un Space). HF_TOKEN (optionnel, secret du Space) desserre le
# rate-limit anonyme du CDN HuggingFace.
#
# « Explorer la journée » (carte des postes) : coordonnées du **plan de masse RTE
# committé** (manoeuvre/dataset/grid_layout_rte.json, ~98 %, hors-ligne) → **rien à
# configurer, aucun accès réseau**. Repli **OSM/Overpass** (ref:FR:RTE) pour les
# postes manquants, persisté **dans le cache** (DGITT_CACHE_DIR). Pour faire
# survivre le cache aux redémarrages : monter le **stockage persistant HF** sur
# /data et définir DGITT_CACHE_DIR=/data/dgitt (couvre les instantanés XIIDM **et**
# les coordonnées). MANOEUVRE_ENABLE_OSM=0 désactive le repli OSM.
# Deux espaces distincts :
#  - DGITT_CACHE_DIR : cache des **instantanés XIIDM téléchargés** (volumineux,
#    régénérables) → **éphémère** par défaut (ne PAS le mettre sur /data).
#  - MANOEUVRE_DATA_DIR : **données à conserver** (base partagée de scénarios /
#    séquences + coordonnées résolues) → à pointer sur le **stockage persistant**
#    (Settings → Variable `MANOEUVRE_DATA_DIR=/data` + Persistent storage sur /data).
#    Non défini ici : la persistance est **opt-in** (cf. deploy/huggingface/SETUP.md).
ENV DGITT_REPO=OpenSynth/D-GITT-RTE7000-2021 \
    DGITT_CACHE_DIR=/home/user/app/.cache/dgitt \
    DGITT_DEFAULT_DATE=2021-01-03 \
    MANOEUVRE_IHM_HOSTED=1 \
    PORT=7860

EXPOSE 7860
CMD ["python", "scripts/manoeuvre_ihm.py", "--dataset", "--host", "0.0.0.0", "--port", "7860"]
