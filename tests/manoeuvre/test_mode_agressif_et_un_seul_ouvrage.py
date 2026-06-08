"""
tests/manoeuvre/test_mode_agressif_et_un_seul_ouvrage.py
--------------------------------------------------------
Deux exigences sur le séquenceur détaillé :

1. **Mode AGRESSIF — boucle courte sans déconnexion.** Quand deux barres sont au
   **même potentiel** (couplage fermé), un départ doit pouvoir être switché d'une
   barre à l'autre en **boucle courte** (fermer le SA cible, ouvrir l'ancien)
   **sans ouvrir son disjoncteur**. Réf. ``CPNIEP6`` : la séquence experte
   (``CPNIEP6_cible_2noeuds_agressif_expert.json``) tient en ~5 manœuvres, là où
   l'ancien mode agressif dé-énergisait tous les ouvrages (≈ 17, *_wrong*).

2. **Mode SMOOTH — un seul ouvrage hors tension à la fois (bonne pratique).** Le
   ré-aiguillage (déconnexion → manip SA → reconnexion) doit privilégier de n'avoir
   qu'**un** ouvrage temporairement hors tension à la fois (hors ouvrages déjà
   déconnectés). Le vérificateur ``ouvrages_simultanement_hors_tension`` le contrôle
   et l'alerte (non bloquante) est surfacée dans ``res.alertes``. Réf. ``ROMAIP6``.

Sans dépendance pypowsybl (fixtures + scénarios sauvegardés).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    determiner_manoeuvres_cible_detaillee,
    ouvrages_simultanement_hors_tension,
)
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _set_switch

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

_SEQ = pathlib.Path(__file__).parent / "sequences"
_CPNIE = "CPNIEP6_cible_2noeuds_agressif_expert.json"
_ROMAI = "ROMAIP6_cible_3noeuds.json"


def _load(name: str, stem: str):
    path = _SEQ / name
    if not path.exists() or stem not in list_available_fixtures():
        pytest.skip(f"Scénario/fixture absent : {name}")
    d = json.loads(path.read_text())

    def g(states: dict):
        G = build_graph_from_fixture(stem)
        for sid, op in states.items():
            _set_switch(G, sid, op)
        return G

    poste = PosteTopologique.from_graph(g(d["depart"]), d["voltage_level_id"])
    return poste, d, g


# --------------------------------------------------------------------------
# 1. Mode agressif : boucle courte (sans déconnexion) quand équipotentiel
# --------------------------------------------------------------------------

def test_agressif_boucle_courte_sans_ouvrir_les_dj():
    """CPNIEP6 (barres couplées) : le mode agressif switche les départs en boucle
    courte **sans ouvrir leur DJ** ; aucune « dé-énergisation groupée »."""
    poste, d, g = _load(_CPNIE, "CPNIEP6")
    r = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="aggressive")

    assert r.is_verified_detaillee, r.ecarts
    assert r.nb_manoeuvres <= 6, (
        f"{r.nb_manoeuvres} manœuvres — le mode agressif ne doit pas dé-énergiser "
        "tous les ouvrages (régression boucle courte)")
    # Aucun DJ d'ouvrage (hors couplage) n'est ouvert : tout se fait sous tension.
    dj_ouvrage_ouverts = [
        m.switch_id for m in r.manoeuvres
        if m.action == "OPEN" and " DJ" in m.switch_id and "COUPL" not in m.switch_id]
    assert not dj_ouvrage_ouverts, (
        f"DJ d'ouvrage ouverts (devrait être boucle courte) : {dj_ouvrage_ouverts}")
    # Des bascules de SA en boucle courte ont bien eu lieu.
    assert any(m.type_boucle == "COURTE" for m in r.manoeuvres)
    # Aucune alerte « un seul ouvrage » : rien n'est dé-énergisé.
    assert r.alertes == []


def test_agressif_jamais_plus_long_que_smooth():
    """Le mode agressif ne doit jamais être plus verbeux que le smooth (CPNIEP6)."""
    poste, d, g = _load(_CPNIE, "CPNIEP6")
    agg = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="aggressive")
    smo = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="smooth")
    assert agg.is_verified_detaillee and smo.is_verified_detaillee
    assert agg.nb_manoeuvres <= smo.nb_manoeuvres


def test_agressif_repli_sur_de_energisation_si_necessaire():
    """ROMAIP6 (re-sectionnement) : la boucle courte ne suffit pas (une section doit
    être morte pour ouvrir un sectionnement) → le mode agressif **retombe** sur la
    dé-énergisation groupée, mais reste **exact** (aucun écart) — pas de régression
    de sûreté/exactitude."""
    poste, d, g = _load(_ROMAI, "ROMAIP6")
    r = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="aggressive")
    assert r.is_verified_detaillee, r.ecarts
    assert r.ecarts == []


# --------------------------------------------------------------------------
# 2. Vérificateur « un seul ouvrage hors tension à la fois » (smooth)
# --------------------------------------------------------------------------

def test_verif_un_seul_ouvrage_propre_si_boucle_courte():
    """CPNIEP6 smooth = boucle courte (aucun ouvrage hors tension) → 0 alerte."""
    poste, d, g = _load(_CPNIE, "CPNIEP6")
    r = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="smooth")
    assert ouvrages_simultanement_hors_tension(poste, r.manoeuvres) == []
    assert r.alertes == []


def test_smooth_romaip6_un_ouvrage_a_la_fois():
    """ROMAIP6 smooth atteint désormais le **« un seul ouvrage hors tension à la
    fois »** (0 alerte) : la réduction de section morte (ouverture temporaire du
    couplage ``COUPL.1`` pour couper le cross-feed) ramène les coupures à un ouvrage
    à la fois — séquence ≈ experte (``ROMAIP6_cible_3noeuds_1ouvrageDeconnecteAlaFois``)."""
    poste, d, g = _load(_ROMAI, "ROMAIP6")
    r = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="smooth")
    assert r.alertes == [], f"coupures simultanées résiduelles : {r.alertes}"


def test_mode_agressif_exempt_de_l_alerte_un_seul():
    """Le mode agressif batch-dé-énergise **volontairement** → exempté de l'alerte
    « un seul ouvrage à la fois » (``res.alertes`` vide)."""
    poste, d, g = _load(_ROMAI, "ROMAIP6")
    r = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="aggressive")
    assert r.alertes == []


# --------------------------------------------------------------------------
# 3. Vérificateur vs séquences de référence + amélioration du parking smooth
# --------------------------------------------------------------------------

def _replay_manoeuvres(seqfile: str, stem: str):
    """Charge (poste de départ, manœuvres) depuis une séquence sauvegardée."""
    from expert_op4grid_recommender.manoeuvre.algo.results import Manoeuvre
    path = _SEQ / seqfile
    if not path.exists() or stem not in list_available_fixtures():
        pytest.skip(f"Séquence/fixture absente : {seqfile}")
    d = json.loads(path.read_text())
    G = build_graph_from_fixture(stem)
    for sid, op in d["depart"].items():
        _set_switch(G, sid, op)
    poste = PosteTopologique.from_graph(G, d["voltage_level_id"])
    manos = [Manoeuvre(m["switch_id"], m["action"], m.get("raison", ""))
             for m in d["manoeuvres"]]
    return poste, manos


def test_verificateur_reconnait_la_sequence_experte_un_a_la_fois():
    """Le vérificateur distingue correctement la **séquence experte** « 1 ouvrage
    déconnecté à la fois » (→ **0** alerte) de l'ancienne séquence smooth **batch**
    (→ alertes), réglant le défaut signalé (l'ancien vérificateur ne détectait pas
    les dé-énergisations de section)."""
    poste_x, expert = _replay_manoeuvres(
        "ROMAIP6_cible_3noeuds_1ouvrageDeconnecteAlaFois.json", "ROMAIP6")
    assert ouvrages_simultanement_hors_tension(poste_x, expert) == []

    poste_b, batch = _replay_manoeuvres(
        "ROMAIP6_cible_3noeuds_smooths.json", "ROMAIP6")
    assert ouvrages_simultanement_hors_tension(poste_b, batch), (
        "l'ancienne séquence batch déconnecte plusieurs ouvrages à la fois")


def test_smooth_reduction_section_morte_par_cross_feed():
    """ROMAIP6 smooth : pour vider la section adjacente d'un nœud couplé, un
    couplage DJ frontière est ouvert **temporairement** puis **refermé** (réduction
    de section morte). On vérifie la présence de la paire ouverture/refermeture ET
    que l'état **final** du couplage est conforme à la cible (rétabli)."""
    poste, d, g = _load(_ROMAI, "ROMAIP6")
    cible = g(d["cible"])
    r = determiner_manoeuvres_cible_detaillee(poste, cible, mode="smooth")
    temp_open = [m for m in r.manoeuvres
                 if m.action == "OPEN" and "temporaire couplage" in m.raison]
    restore = [m for m in r.manoeuvres
               if m.action == "CLOSE" and "rétabli" in m.raison]
    assert temp_open and restore, "ouverture temporaire + refermeture attendues"
    # État final = cible pour chaque couplage ouvert temporairement (rétabli).
    from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _is_open, _set_switch
    G = build_graph_from_fixture("ROMAIP6")
    for sid, op in d["depart"].items():
        _set_switch(G, sid, op)
    for m in r.manoeuvres:
        _set_switch(G, m.switch_id, m.action == "OPEN")
    for m in temp_open:
        assert _is_open(G, m.switch_id) == _is_open(cible, m.switch_id), (
            f"{m.switch_id} non rétabli à l'état cible")


def test_smooth_parking_un_par_un_reduit_les_coupures_simultanees():
    """Le mode smooth gare désormais les départs **un par un** sur le côté
    survivant (boucle courte) au lieu de dé-énergiser une section en bloc : sur
    ROMAIP6 le nombre de moments à > 1 ouvrage hors tension est **strictement
    inférieur** à celui de l'ancienne séquence batch."""
    poste_b, batch = _replay_manoeuvres(
        "ROMAIP6_cible_3noeuds_smooths.json", "ROMAIP6")
    n_batch = len(ouvrages_simultanement_hors_tension(poste_b, batch))

    poste, d, g = _load(_ROMAI, "ROMAIP6")
    r = determiner_manoeuvres_cible_detaillee(poste, g(d["cible"]), mode="smooth")
    assert len(r.alertes) < n_batch, (
        f"smooth actuel {len(r.alertes)} alerte(s) vs ancienne batch {n_batch} : "
        "le parking un-par-un devrait réduire les coupures simultanées")
