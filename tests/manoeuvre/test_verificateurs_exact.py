"""
tests/manoeuvre/test_verificateurs_exact.py
---------------------------------------------
**Filet de sécurité ciblé pour le refactor « passe de rejeu unique »** (#3).

Les vérificateurs de règles (``_sectionneurs_sous_charge_par_manoeuvre``,
``_verifier_sectionneurs_hors_charge``) rejouent aujourd'hui la séquence
chacun de leur côté. Un refactor qui les fusionne en **une seule passe** doit
préserver, exactement :

1. l'**alignement** de la liste « par manœuvre » sur la séquence (même longueur) ;
2. les **positions exactes** des infractions (et leur identité d'organe) ;
3. la **liste d'écarts dédupliquée** (mêmes organes, même ordre, sans doublon) ;
4. l'**idempotence** (rejouer deux fois donne le même résultat) ;
5. la **non-mutation** du graphe du poste (les vérificateurs travaillent sur une
   copie ; ils ne doivent pas altérer ``poste.graph``).

Contrairement au golden (qui fige la sortie de bout en bout), ces tests
attaquent **directement** les fonctions que #3 va réécrire, avec des assertions
d'égalité de liste **structurelles** (identité d'organe + violé/non-violé),
robustes au libellé exact des messages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    Manoeuvre,
    _sectionneurs_sous_charge_par_manoeuvre,
    _verifier_sectionneurs_hors_charge,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

SEQ_DIR = Path(__file__).parent / "sequences"


def _graph_from_states(vl: str, states: dict):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


def _switch_state(G) -> dict:
    return {d["switch_id"]: d.get("open", False)
            for _u, _v, d in G.edges(data=True) if d.get("switch_id")}


def _load(name: str):
    path = SEQ_DIR / name
    if not path.exists():
        pytest.skip(f"Séquence absente : {name}")
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    if vl not in list_available_fixtures():
        pytest.skip(f"Fixture {vl} absente")
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    manos = [Manoeuvre(m["switch_id"], m["action"], m.get("raison", ""))
             for m in d["manoeuvres"]]
    return poste, manos


def _offending_switches(ecarts: list[str]) -> list[str]:
    """Identifiant d'organe en tête de chaque écart « <sid> : <message> »."""
    return [e.split(" : ", 1)[0] for e in ecarts]


# Vérité-terrain (cf. fixtures de séquences « fautives ») : une seule infraction,
# en dernière position, sur un sélecteur de barre de départ.
BAD_SEQUENCES = [
    ("MORBRP6_cible_4noeuds_mauvaise_manoeuvre.json",
     [3], "MORBRP6_MORBR6AT761 SA.1_OC"),
    ("MORBRP6_cible_4noeuds_wrong_last_step.json",
     [7], "MORBRP6_MORBR6ARRIG.1 SA.1_OC"),
]


@pytest.mark.parametrize("name,viol_idx,sid", BAD_SEQUENCES,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_par_manoeuvre_alignement_et_indices_exacts(name, viol_idx, sid):
    """``par_man`` est **aligné** sur la séquence et les infractions tombent aux
    **positions exactes** attendues (égalité de liste structurelle)."""
    poste, manos = _load(name)
    par_man = _sectionneurs_sous_charge_par_manoeuvre(poste, manos)

    assert len(par_man) == len(manos), "liste non alignée sur la séquence"
    assert [i for i, v in enumerate(par_man) if v is not None] == viol_idx
    # L'organe fautif est bien celui de la manœuvre incriminée.
    for i in viol_idx:
        assert manos[i].switch_id == sid
        assert "disjoncteur" in par_man[i]  # parade « par son disjoncteur »


@pytest.mark.parametrize("name,viol_idx,sid", BAD_SEQUENCES,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_ecarts_hors_charge_liste_exacte(name, viol_idx, sid):
    """``_verifier_sectionneurs_hors_charge`` renvoie **exactement** la liste des
    organes fautifs (identité + nombre), dédupliquée."""
    poste, manos = _load(name)
    ecarts = _verifier_sectionneurs_hors_charge(poste, manos)

    assert _offending_switches(ecarts) == [sid]
    assert len(ecarts) == len(set(ecarts)), "écarts non dédupliqués"


@pytest.mark.parametrize("name,viol_idx,sid", BAD_SEQUENCES,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_verificateurs_idempotents(name, viol_idx, sid):
    """Rejouer deux fois les vérificateurs sur le même couple (poste, séquence)
    donne un résultat **identique** (pas d'état partagé entre passes — invariant
    à préserver si #3 fusionne les rejeux)."""
    poste, manos = _load(name)
    a1 = _sectionneurs_sous_charge_par_manoeuvre(poste, manos)
    a2 = _sectionneurs_sous_charge_par_manoeuvre(poste, manos)
    assert a1 == a2
    b1 = _verifier_sectionneurs_hors_charge(poste, manos)
    b2 = _verifier_sectionneurs_hors_charge(poste, manos)
    assert b1 == b2


@pytest.mark.parametrize("name,viol_idx,sid", BAD_SEQUENCES,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_verificateurs_ne_mutent_pas_le_graphe(name, viol_idx, sid):
    """Les vérificateurs travaillent sur une **copie** : ``poste.graph`` doit
    rester intact (une passe fusionnée ne doit pas muter l'état partagé)."""
    poste, manos = _load(name)
    avant = _switch_state(poste.graph)
    _sectionneurs_sous_charge_par_manoeuvre(poste, manos)
    _verifier_sectionneurs_hors_charge(poste, manos)
    assert _switch_state(poste.graph) == avant


def test_message_parade_selon_type_de_sectionneur():
    """La parade dépend du type d'organe : **sectionnement de barre** → « section
    de barre » ; **sélecteur de barre de départ** → « par son disjoncteur ».
    Invariant métier que #3 ne doit pas brouiller en fusionnant les passes."""
    # Sélecteur de barre de départ (séquence fautive MORBRP6).
    poste, manos = _load("MORBRP6_cible_4noeuds_mauvaise_manoeuvre.json")
    msg_depart = next(m for m in _sectionneurs_sous_charge_par_manoeuvre(poste, manos)
                      if m is not None)
    assert "disjoncteur" in msg_depart and "section de barre" not in msg_depart

    # Sectionnement de barre (CZBEVP3, sectionneur de section SS.1.12).
    VL = "CZBEVP3"
    if VL not in list_available_fixtures():
        pytest.skip("Fixture CZBEVP3 absente")
    G = build_graph_from_fixture(VL)
    sid = "CZBEVP3_CZBEV3CBO.1 SS.1.12_OC"
    if not any(d.get("switch_id") == sid for _u, _v, d in G.edges(data=True)):
        pytest.skip("Sectionneur de section CZBEVP3 absent de la fixture")
    poste = PosteTopologique.from_graph(G, VL)
    msgs = _sectionneurs_sous_charge_par_manoeuvre(
        poste, [Manoeuvre(sid, "OPEN", "manœuvre manuelle (expert)")])
    assert msgs[0] is not None and "section de barre" in msgs[0]
    assert "disjoncteur" not in msgs[0]
