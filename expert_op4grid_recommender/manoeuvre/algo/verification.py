"""
manoeuvre/algo/verification.py — Vérificateurs de règles du sectionneur (passe de rejeu unique) et optimisation de séquence.
"""
from __future__ import annotations

from typing import Optional
import networkx as nx

from ..models import SwitchKind
from ..topologie import PosteTopologique
from .results import Manoeuvre
from .graph_ops import _inter_sjb_couplers, _is_open, _live_graph_sans, _ouvrages_energises_sur, _set_switch


def _rejeu_securite(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> tuple[list[str], list[Optional[str]], list[str]]:
    """**Passe de rejeu unique** des règles du sectionneur (au lieu de trois
    rejeux séparés). Rejoue la séquence une seule fois depuis l'état initial et
    calcule en parallèle les trois diagnostics, en partageant le graphe « live »
    construit au plus une fois par étape.

    Returns
    -------
    (securite, sous_charge_par_man, un_seul)
        - ``securite`` : sectionnements de barre ouverts **sous tension** (deux
          côtés énergisés) — règle stricte du sectionneur de barre ;
        - ``sous_charge_par_man`` : liste **alignée** sur ``manoeuvres`` ; message
          d'infraction si la manœuvre ouvre un sectionneur (DISCONNECTOR) sous
          charge, sinon ``None`` ;
        - ``un_seul`` : moments où **plus d'un** ouvrage est hors tension par
          ré-aiguillage (boucle longue) simultanément (R10ter, mode smooth).
    """
    couplers = _inter_sjb_couplers(poste)
    cells = poste.cellules
    all_sjb = set(poste.tronconnement.barre_par_busbar)

    # Sectionnements de barre : sid -> (SJB a, SJB b) (règle stricte).
    sect_pairs: dict[str, tuple[int, int]] = {}
    for cp in couplers:
        if cp.is_sectionnement:
            for sid in cp.switch_ids:
                sect_pairs[sid] = (cp.sjb_a, cp.sjb_b)
    sect_id_set = set(sect_pairs)

    # Tous les sectionneurs (DISCONNECTOR) : sid -> arête (« sous charge »).
    disc: dict[str, tuple[int, int]] = {}
    for u, v, d in poste.graph.edges(data=True):
        sid = d.get("switch_id")
        if sid and d.get("kind") == SwitchKind.DISCONNECTOR:
            disc[sid] = (u, v)
    equip = {n for n, d in poste.graph.nodes(data=True) if d.get("equipment_id")}

    # DJ de départ (hors couplage) : sid -> équipement (« un seul HS »).
    coupling_sids = {s for cp in couplers for s in cp.switch_ids}
    dj_eq: dict[str, str] = {}
    for c in cells.cellules_depart:
        for b in c.breakers:
            if b.switch_id not in coupling_sids:
                dj_eq[b.switch_id] = c.equipment_id

    securite: list[str] = []
    sous_charge: list[Optional[str]] = []
    un_seul: list[str] = []
    temp_parking: set[str] = set()

    G = poste.graph.copy()
    for m in manoeuvres:
        opening = (m.action == "OPEN" and not _is_open(G, m.switch_id))
        H = None  # graphe « live » privé du switch, construit au plus une fois

        # --- sectionneur sous charge (tout DISCONNECTOR), aligné sur la séquence
        msg: Optional[str] = None
        if opening and m.switch_id in disc:
            H = _live_graph_sans(G, [m.switch_id])
            u, v = disc[m.switch_id]
            if not (u in H and v in H and nx.has_path(H, u, v)):
                cu = nx.node_connected_component(H, u) if u in H else {u}
                cv = nx.node_connected_component(H, v) if v in H else {v}
                if (cu & equip) and (cv & equip):
                    if m.switch_id in sect_id_set:
                        msg = ("sectionneur de barre manœuvré sous charge — mettre "
                               "la section de barre hors tension (ré-aiguiller ses "
                               "départs sur l'autre section) avant d'ouvrir ce "
                               "sectionnement")
                    else:
                        msg = ("sectionneur manœuvré sous charge — dé-énergiser la "
                               "branche par son disjoncteur avant d'ouvrir ce "
                               "sectionneur")
        sous_charge.append(msg)

        # --- sectionnement de barre ouvert sous tension (règle stricte) --------
        if opening and m.switch_id in sect_pairs:
            if H is None:
                H = _live_graph_sans(G, [m.switch_id])
            a, b = sect_pairs[m.switch_id]
            if not (a in H and b in H and nx.has_path(H, a, b)):
                side_a = (nx.node_connected_component(H, a)
                          if a in H else {a}) & all_sjb
                side_b = (nx.node_connected_component(H, b)
                          if b in H else {b}) & all_sjb
                if (_ouvrages_energises_sur(G, cells, side_a, H)
                        and _ouvrages_energises_sur(G, cells, side_b, H)):
                    securite.append(
                        f"sectionneur {m.switch_id} ouvert sous tension "
                        "(deux côtés énergisés)")

        # --- un seul ouvrage hors tension par ré-aiguillage (boucle longue) ----
        eq = dj_eq.get(m.switch_id)
        if eq is not None:
            longue = (m.type_boucle == "LONGUE"
                      or "boucle longue" in (m.raison or ""))
            if m.action == "OPEN" and longue:
                temp_parking.add(eq)
            elif m.action == "CLOSE":
                temp_parking.discard(eq)
            if len(temp_parking) > 1:
                un_seul.append(
                    "plus d'un ouvrage hors tension simultanément (ré-aiguillage) : "
                    + ", ".join(sorted(temp_parking)))

        _set_switch(G, m.switch_id, m.action == "OPEN")

    return securite, sous_charge, list(dict.fromkeys(un_seul))


def _verifier_securite_sectionneurs(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Sectionnements de barre ouverts sous tension (cf. ``_rejeu_securite``)."""
    return _rejeu_securite(poste, manoeuvres)[0]


def _verifier_un_seul_hors_tension(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Règle R10ter : pas plus d'**un** ouvrage hors tension par ré-aiguillage
    (boucle longue) simultanément (cf. ``_rejeu_securite``)."""
    return _rejeu_securite(poste, manoeuvres)[2]


def sectionneurs_sous_charge_par_manoeuvre(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[Optional[str]]:
    """Analyse, **manœuvre par manœuvre**, la règle du sectionneur (hors charge).

    Retourne une liste **alignée sur ``manoeuvres``** : pour chaque manœuvre, un
    message d'infraction si elle ouvre un **sectionneur** (DISCONNECTOR) **sous
    charge** — i.e. elle déconnecte, sans chemin parallèle, deux parties **toutes
    deux porteuses d'un ouvrage** énergisé — sinon ``None``.

    Un sectionneur ne se manœuvre que hors charge : la parade est de dé-énergiser
    la branche par son **disjoncteur** avant d'ouvrir le sectionneur.

    Point d'entrée **public** : c'est le vérificateur de règle exposé aux clients
    du module (IHM, notebooks) qui souhaitent valider une séquence éditée à la
    main sans réimporter de symbole privé.
    """
    return _rejeu_securite(poste, manoeuvres)[1]


def _verifier_regles(
    poste: PosteTopologique,
    manoeuvres: list[Manoeuvre],
    un_seul: bool = True,
) -> list[str]:
    """Agrège, en **une seule passe de rejeu** (``_rejeu_securite``), les écarts
    des règles du sectionneur dans l'ordre historique : sécurité des
    sectionnements, [un seul ouvrage hors tension,] sectionneurs hors charge.

    ``un_seul`` : inclure la règle R10ter (modes smooth / multi-barres) ; les
    orchestrations « batch » (agressif, séquenceur à sections) l'omettent.
    """
    securite, sous_charge, un = _rejeu_securite(poste, manoeuvres)
    hors_charge = list(dict.fromkeys(
        f"{m.switch_id} : {msg}"
        for m, msg in zip(manoeuvres, sous_charge) if msg))
    out = list(securite)
    if un_seul:
        out += un
    out += hors_charge
    return out


def _verifier_sectionneurs_hors_charge(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[str]:
    """Règle générale du sectionneur : un **sectionneur** (DISCONNECTOR) ne se
    manœuvre que **hors charge** (cf. ``sectionneurs_sous_charge_par_manoeuvre``).
    Retourne les écarts agrégés (sectionneurs manœuvrés sous charge), pour
    **tous** les sectionneurs (sélecteurs de barre comme sectionnements)."""
    par_man = sectionneurs_sous_charge_par_manoeuvre(poste, manoeuvres)
    ecarts = [f"{m.switch_id} : {msg}"
              for m, msg in zip(manoeuvres, par_man) if msg]
    return list(dict.fromkeys(ecarts))


def _optimiser_sequence(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[Manoeuvre]:
    """
    Supprime les manœuvres sans effet en rejouant la séquence depuis l'état
    initial : une manœuvre qui place un OC dans l'état où il se trouve déjà est
    redondante et retirée. Les bascules réelles (ex. ouverture/fermeture d'un DJ
    en boucle longue) sont conservées.
    """
    G = poste.graph.copy()
    out: list[Manoeuvre] = []
    for m in manoeuvres:
        want_open = (m.action == "OPEN")
        if _is_open(G, m.switch_id) == want_open:
            continue
        _set_switch(G, m.switch_id, want_open)
        out.append(m)
    return out


# Alias privé historique (compat. tests/imports antérieurs).
_sectionneurs_sous_charge_par_manoeuvre = sectionneurs_sous_charge_par_manoeuvre
