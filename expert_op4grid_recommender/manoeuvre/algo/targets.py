"""
manoeuvre/algo/targets.py — Points d'entrée : topologie nodale cible et cible détaillée (modes smooth / aggressive / multi-barres).
"""
from __future__ import annotations

from typing import Optional
import networkx as nx

from ..topologie import TopologieNodale, PosteTopologique
from .results import Manoeuvre, ResultatManoeuvres
from .graph_ops import _edges_of_switches, _inter_sjb_couplers, _is_open, _live_graph_sans, _organes_internes_2bornes, _ouvrages_energises_sur, _set_switch, _wired_busbar
from .placement import _departure_dj_changes, _placement_automatique, _placement_avec_reconnexions
from .verification import _optimiser_sequence, _verifier_regles
from .sequencing import _appliquer_changements_dj, _consigner_non_realisables, _isoler_depart_hors_barre, _reaiguiller_vers_sjb, determiner_manoeuvres_avec_sections, determiner_manoeuvres_par_connectivite


def _ecarts_detailles(
    poste: PosteTopologique,
    G: nx.Graph,
    cible_graph: nx.Graph,
    cible_busbar: dict[str, int],
) -> list[str]:
    """Liste des écarts entre l'état détaillé obtenu ``G`` et la cible."""
    cells = poste.cellules
    sjb_id = {n: G.nodes[n].get("busbar_section_id")
              for n in poste.tronconnement.barre_par_busbar}
    ecarts: list[str] = []
    # Barre de chaque départ
    for eq, target in cible_busbar.items():
        cell = cells.get_cellule_depart(eq)
        cur = _wired_busbar(cell, G)
        if cur != target:
            ecarts.append(
                f"'{eq}' sur {sjb_id.get(cur, cur)} au lieu de {sjb_id.get(target, target)}")
    # État des couplers inter-SJB
    coupling_sids: set[str] = set()
    for cp in _inter_sjb_couplers(poste):
        for sid in cp.switch_ids:
            coupling_sids.add(sid)
            cur = _is_open(G, sid)
            tgt = any(cible_graph.edges[u, v].get("open", False)
                      for u, v in _edges_of_switches(cible_graph, [sid]))
            if cur != tgt:
                ecarts.append(f"organe {sid} {'ouvert' if cur else 'fermé'} "
                              f"au lieu de {'ouvert' if tgt else 'fermé'}")
    # État des DJ de départ (hors couplage)
    for c in cells.cellules_depart:
        for sw in c.breakers:
            if sw.switch_id in coupling_sids:
                continue
            cur = _is_open(G, sw.switch_id)
            tgt = _is_open(cible_graph, sw.switch_id)
            if cur != tgt:
                ecarts.append(
                    f"DJ {sw.switch_id} {'ouvert' if cur else 'fermé'} "
                    f"au lieu de {'ouvert' if tgt else 'fermé'}")
    return ecarts


def _sequence_detaillee_aggressive(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    cible_busbar: dict[str, int],
) -> ResultatManoeuvres:
    """
    Mode **agressif** : atteint la topologie détaillée cible par une
    orchestration « batch » minimisant les bascules de disjoncteurs.

    Au lieu de dé-énergiser/ré-aiguiller **un ouvrage à la fois** (mode smooth,
    boucle longue, ré-alimentation immédiate), on **cumule** les
    dé-énergisations : on ouvre en une fois les DJ de tous les ouvrages
    concernés, on commute couplages/sectionnements et SA **hors tension**, puis
    on **ré-alimente une seule fois**. Bien moins de manœuvres, au prix de
    plusieurs ouvrages momentanément hors tension simultanément.

    Ordre (règle du sectionneur respectée : sections mortes avant ouverture) :
    1. ouvrir les DJ du **lot** (SA à changer ∪ ouvrages énergisés sur une
       section incidente à un sectionnement fermé à ouvrir) ;
    2. ouvrir les couplages (DJ) destinés à s'ouvrir ;
    3. ouvrir les sectionnements destinés à s'ouvrir (sections désormais mortes) ;
    4. fermer les couplages puis sectionnements destinés à se fermer ;
    5. positionner tous les SA de départ sur leur état cible (hors tension) ;
    6. refermer les DJ du lot / appliquer reconnexions-déconnexions (cible).
    """
    vl = poste.voltage_level_id
    cells = poste.cellules
    G = poste.graph.copy()
    couplers = _inter_sjb_couplers(poste)
    coupling_sids = {s for cp in couplers for s in cp.switch_ids}
    manoeuvres: list[Manoeuvre] = []

    res = ResultatManoeuvres(
        voltage_level_id=vl,
        topo_initiale=poste.topologie_nodale,
        topo_cible=TopologieNodale.from_graph(cible_graph, vl),
    )

    def setsw(sid: str, open_: bool, raison: str) -> None:
        if _is_open(G, sid) != open_:
            _set_switch(G, sid, open_)
            manoeuvres.append(Manoeuvre(
                sid, "OPEN" if open_ else "CLOSE", raison))

    all_sjb = set(poste.tronconnement.barre_par_busbar)

    # Sectionnements fermés destinés à s'ouvrir.
    sect_to_open = [cp for cp in couplers
                    if cp.is_sectionnement
                    and any(not _is_open(G, s) for s in cp.switch_ids)
                    and any(_is_open(cible_graph, s) for s in cp.switch_ids)]

    # --- Lot à dé-énergiser ------------------------------------------------
    # (a) tout ouvrage dont un SA change (ré-aiguillage hors tension) ;
    # (b) tout ouvrage **énergisé** (connectivité électrique réelle) sur une
    #     section isolée par un sectionnement à ouvrir — y compris les ouvrages
    #     raccordés directement par DJ sans SA (ex. transformateurs).
    batch: dict[str, list[str]] = {}
    for c in cells.cellules_depart:
        eq = c.equipment_id
        brk = [b.switch_id for b in c.breakers]
        if not brk or any(_is_open(G, b) for b in brk):
            continue  # pas de DJ propre, ou déjà hors tension
        if any(_is_open(G, s.switch_id) != _is_open(cible_graph, s.switch_id)
               for s in c.disconnectors if s.switch_id not in coupling_sids):
            batch[eq] = brk

    if sect_to_open:
        # Graphe avec les couplages destinés à s'ouvrir déjà ouverts, pour
        # délimiter correctement les sections à isoler.
        G_cut = G.copy()
        for cp in couplers:
            if cp.is_sectionnement:
                continue
            for sid in cp.breaker_ids:
                if _is_open(cible_graph, sid):
                    _set_switch(G_cut, sid, True)
        sect_sids = [s for cp in sect_to_open for s in cp.switch_ids]
        H = _live_graph_sans(G_cut, sect_sids)
        # Pour chaque sectionnement, ne dé-énergiser que **le plus petit côté**
        # (un seul côté mort suffit pour ouvrir le sectionneur) — minimise le lot.
        for cp in sect_to_open:
            sa = ((nx.node_connected_component(H, cp.sjb_a)
                   if cp.sjb_a in H else {cp.sjb_a}) & all_sjb)
            sb = ((nx.node_connected_component(H, cp.sjb_b)
                   if cp.sjb_b in H else {cp.sjb_b}) & all_sjb)
            liv_a = _ouvrages_energises_sur(G_cut, cells, sa, H)
            liv_b = _ouvrages_energises_sur(G_cut, cells, sb, H)
            petit = liv_a if len(liv_a) <= len(liv_b) else liv_b
            for eq, brk in petit:
                if brk and eq not in batch:
                    batch[eq] = brk

    # 1. Dé-énergiser le lot (ouverture des DJ d'ouvrage, une fois chacun).
    for eq, brk in batch.items():
        for sid in brk:
            setsw(sid, True, f"mise hors tension '{eq}' (dé-énergisation groupée)")

    # 2. Ouvrir les couplages (DJ) destinés à s'ouvrir.
    for cp in couplers:
        if cp.is_sectionnement:
            continue
        for sid in cp.breaker_ids:
            if _is_open(cible_graph, sid):
                setsw(sid, True, "ouverture couplage de barres")

    # 3. Ouvrir les sectionnements destinés à s'ouvrir (sections mortes).
    for cp in couplers:
        if not cp.is_sectionnement:
            continue
        for sid in cp.switch_ids:
            if _is_open(cible_graph, sid):
                setsw(sid, True,
                      "ouverture sectionnement de barre (section hors tension)")

    # 4. Fermer les couplages (DJ d'abord) puis sectionnements destinés à fermer.
    for cp in couplers:
        if cp.is_sectionnement:
            continue
        for sid in cp.switch_ids:
            if not _is_open(cible_graph, sid):
                setsw(sid, False, "fermeture couplage de barres")
    for cp in couplers:
        if not cp.is_sectionnement:
            continue
        for sid in cp.switch_ids:
            if not _is_open(cible_graph, sid):
                setsw(sid, False,
                      "fermeture sectionnement de barre (barres équipotentielles)")

    # 5. Positionner tous les SA de départ sur leur état cible (hors tension).
    for c in cells.cellules_depart:
        for s in c.disconnectors:
            if s.switch_id in coupling_sids:
                continue
            setsw(s.switch_id, _is_open(cible_graph, s.switch_id),
                  f"ré-aiguillage '{c.equipment_id}' (hors tension)")

    # 6. Ramener tous les DJ de départ à leur état cible (ré-alimentation /
    #    mise en service / mise hors service), une fois chacun.
    for c in cells.cellules_depart:
        for b in c.breakers:
            if b.switch_id in coupling_sids:
                continue
            setsw(b.switch_id, _is_open(cible_graph, b.switch_id),
                  f"remise en service '{c.equipment_id}'")

    manoeuvres = _optimiser_sequence(poste, manoeuvres)
    res.manoeuvres = manoeuvres
    res.is_changed = bool(manoeuvres)

    # --- Vérification (nodale + détaillée + sûreté des sectionneurs) --------
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = res.topo_cible.meme_topologie(res.topo_obtenue)
    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_regles(poste, manoeuvres, un_seul=False))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if not res.is_verified:
        res.message = (
            "Topologie nodale cible non atteinte (mode agressif) : obtenu "
            f"{res.topo_obtenue.nb_noeuds} nœud(s), visé {res.topo_cible.nb_noeuds}.")
    elif res.is_verified_detaillee:
        res.message = "Topologie détaillée cible atteinte et vérifiée (mode agressif)."
    else:
        res.message = (f"Topologie nodale atteinte ; {len(res.ecarts)} écart(s) : "
                       + " ; ".join(res.ecarts[:6]))
    return res


def _aligner_couplers_sur_cible(poste, G, cible_graph, topo_cible):
    """Aligne l'état des **faisceaux de couplage** sur la cible **détaillée**, sans
    changer la partition nodale (mute ``G`` ; retourne les manœuvres d'alignement).

    Le séquenceur/réalisateur atteint la bonne **partition** mais peut coupler les
    barres via un **faisceau équivalent** différent de celui encodé par la cible
    (un poste triple-barre a plusieurs faisceaux — COUPL.A/COUPL.B/LIAIS — donnant
    la même partition) → écarts détaillés « cosmétiques ».

    On réaligne **faisceau par faisceau** (organe-diff sûr) : pour chaque faisceau
    dont un organe diffère de la cible, on **dé-énergise** (ouverture du DJ), on
    positionne les **SA hors charge** sur l'état cible, puis on **ramène le DJ** à
    l'état cible. Les faisceaux **actifs dans la cible** (DJ fermé) sont traités en
    premier pour maintenir une liaison. Garde **transactionnelle** finale : si
    l'alignement ne reproduit pas exactement la partition cible, tout est annulé.
    """
    vl = poste.voltage_level_id
    couplers = _inter_sjb_couplers(poste)

    def _cible_open(sid):
        for u, v in _edges_of_switches(cible_graph, [sid]):
            return bool(cible_graph.edges[u, v].get("open", False))
        return None

    # Faisceaux groupés par DJ partagé (un faisceau = un DJ + ses SA).
    bays: dict = {}
    for cp in couplers:
        key = frozenset(cp.breaker_ids)
        bay = bays.setdefault(key, {"dj": set(cp.breaker_ids), "sa": set()})
        bay["sa"].update(s for s in cp.switch_ids if s not in cp.breaker_ids)

    out: list[Manoeuvre] = []

    def _set(sid, want_open, raison):
        if _is_open(G, sid) != want_open:
            _set_switch(G, sid, want_open)
            out.append(Manoeuvre(sid, "OPEN" if want_open else "CLOSE", raison))

    def _bay_actif_cible(bay):
        return any(_cible_open(dj) is False for dj in bay["dj"])

    # Faisceaux actifs (DJ cible fermé) d'abord → établir les liaisons cibles
    # avant d'ouvrir les faisceaux redondants.
    for bay in sorted(bays.values(), key=lambda b: not _bay_actif_cible(b)):
        organs = bay["dj"] | bay["sa"]
        if all(_cible_open(s) is None or _is_open(G, s) == _cible_open(s)
               for s in organs):
            continue  # faisceau déjà aligné
        for dj in sorted(bay["dj"]):                       # 1) dé-énergiser
            _set(dj, True, "dé-énergisation faisceau (alignement cible)")
        for sa in sorted(bay["sa"]):                       # 2) SA hors charge
            co = _cible_open(sa)
            if co is not None:
                _set(sa, co, "alignement SA de faisceau sur la cible")
        for dj in sorted(bay["dj"]):                       # 3) DJ -> cible
            co = _cible_open(dj)
            if co is not None:
                _set(dj, co, "alignement DJ de faisceau sur la cible")

    # Garde transactionnelle : ne conserver l'alignement que s'il **préserve**
    # exactement la partition cible.
    if not topo_cible.meme_topologie(TopologieNodale.from_graph(G, vl)):
        for m in reversed(out):
            _set_switch(G, m.switch_id, m.action != "OPEN")
        return []
    return out


def _sequence_detaillee_multibarres(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    topo_cible: TopologieNodale,
) -> ResultatManoeuvres:
    """
    Séquence pour un poste à **> 2 jeux de barres**.

    Le placement nodal classique (recherche combinatoire) ne couvre pas ces
    postes (nœuds à 0 barre, organes internes à 2 bornes, barres multiples). On
    dérive ici le placement **directement des composantes connexes du graphe
    cible** (chaque nœud = ses équipements + les sections de barre qu'il occupe),
    ce qui donne les groupes exacts, y compris :
    - les **nœuds à 0 barre** (départ isolé sur son DJ, SA ouverts) → isolés ;
    - les **organes internes à 2 bornes** (self/réactance) → laissés en place.

    Les départs/sections que l'algorithme ne sait pas réaliser sont consignés en
    écart (dégradation gracieuse) pour complétion manuelle.
    """
    vl = poste.voltage_level_id
    cells = poste.cellules
    G0 = poste.graph
    organes_fixes = _organes_internes_2bornes(poste)

    # Barre cible exacte de chaque départ (hors organes 2 bornes, ambigus).
    cible_busbar: dict[str, int] = {}
    for c in cells.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            if eq in organes_fixes:
                continue
            bb = _wired_busbar(c, cible_graph)
            if bb is not None:
                cible_busbar[eq] = bb

    dep_eqs = {c.equipment_id for c in cells.cellules_depart}
    for c in cells.cellules_depart:
        dep_eqs |= set(c.shared_equipment_ids)
    sjb_nodes = set(poste.tronconnement.barre_par_busbar)
    sjb_id = {n: G0.nodes[n].get("busbar_section_id") for n in sjb_nodes}

    # Placement depuis les composantes connexes (switches fermés) du graphe cible.
    closed = nx.Graph()
    closed.add_nodes_from(cible_graph.nodes(data=True))
    for u, v, d in cible_graph.edges(data=True):
        if not d.get("open", False):
            closed.add_edge(u, v)

    placement: list[tuple[set[str], set[str]]] = []
    noeuds_isoles: list[set[str]] = []
    for comp in nx.connected_components(closed):
        eqs = {cible_graph.nodes[n].get("equipment_id") for n in comp} & dep_eqs
        if not eqs:
            continue
        sjbs = {sjb_id[n] for n in comp if n in sjb_nodes}
        if sjbs:
            placement.append((eqs, sjbs))
        else:
            noeuds_isoles.append(eqs)

    res = determiner_manoeuvres_avec_sections(
        poste, placement, cible_busbar, organes_fixes=organes_fixes)
    res.topo_initiale = poste.topologie_nodale
    res.topo_cible = topo_cible

    # État détaillé après la séquence de placement.
    G = poste.graph.copy()
    for m in res.manoeuvres:
        _set_switch(G, m.switch_id, m.action == "OPEN")

    extra: list[Manoeuvre] = []
    # Isolation des nœuds à 0 barre : détacher le départ de ses barres en
    # respectant la **règle du sectionneur** (dé-énergiser par le DJ d'abord).
    for eqs in noeuds_isoles:
        for eq in sorted(eqs):
            if eq in organes_fixes:
                continue
            cell = cells.get_cellule_depart(eq)
            if cell is None:
                continue
            extra += _isoler_depart_hors_barre(G, cell, cible_graph)

    # Changements de DJ de départ (mise en service / hors service).
    reconnections, disconnections = _departure_dj_changes(poste, cible_graph)
    extra += _appliquer_changements_dj(G, reconnections, disconnections)

    res.manoeuvres = res.manoeuvres + extra

    # Vérification nodale + détaillée.
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = topo_cible.meme_topologie(res.topo_obtenue)
    res.is_changed = bool(res.manoeuvres)

    # --- Repli connectivité-based (faisceaux de couplage partagés) ----------
    # Si la voie multi-barres n'atteint pas la cible **nodale** (typiquement des
    # postes en triangle à faisceaux partagés où la séparation par couplers
    # échoue), on tente le **réalisateur connectivité-based** sur le placement
    # dérivé du graphe cible. **Transactionnel / only-on-failure** : on ne le
    # retient que s'il vérifie la cible nodale → les postes déjà réalisés (ex.
    # MORBRP6) ne sont pas touchés (goldens inchangés).
    if not res.is_verified:
        alt = determiner_manoeuvres_par_connectivite(poste, placement, topo_cible)
        if alt.is_verified:
            alt.topo_initiale = poste.topologie_nodale
            alt.topo_cible = topo_cible
            Galt = poste.graph.copy()
            for m in alt.manoeuvres:
                _set_switch(Galt, m.switch_id, m.action == "OPEN")
            # Aligne les faisceaux de couplage sur la cible détaillée (cosmétique,
            # partition préservée) pour viser is_verified_detaillee.
            alt.manoeuvres = alt.manoeuvres + _aligner_couplers_sur_cible(
                poste, Galt, cible_graph, topo_cible)
            alt.topo_obtenue = TopologieNodale.from_graph(Galt, vl)
            alt.ecarts = (_ecarts_detailles(poste, Galt, cible_graph, cible_busbar)
                          + _verifier_regles(poste, alt.manoeuvres, un_seul=True))
            alt.is_verified_detaillee = not alt.ecarts
            alt.message = (
                "Topologie détaillée cible atteinte et vérifiée "
                "(réalisateur connectivité, poste multi-barres)."
                if alt.is_verified_detaillee else
                f"Topologie nodale atteinte (réalisateur connectivité, poste "
                f"multi-barres) ; {len(alt.ecarts)} écart(s) détaillé(s) résiduel(s) : "
                + " ; ".join(alt.ecarts[:6]))
            return alt

    # Nodale atteinte par la voie multi-barres : aligner les faisceaux de couplage
    # sur la cible détaillée (faisceau équivalent → écarts cosmétiques) pour viser
    # is_verified_detaillee, sans changer la partition.
    if res.is_verified:
        res.manoeuvres = res.manoeuvres + _aligner_couplers_sur_cible(
            poste, G, cible_graph, topo_cible)
        res.topo_obtenue = TopologieNodale.from_graph(G, vl)

    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_regles(poste, res.manoeuvres, un_seul=True))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if res.is_verified_detaillee:
        res.message = ("Topologie détaillée cible atteinte et vérifiée "
                       "(poste multi-barres).")
    elif res.is_verified:
        res.message = (f"Topologie nodale atteinte (poste multi-barres) ; "
                       f"{len(res.ecarts)} écart(s) détaillé(s) résiduel(s) : "
                       + " ; ".join(res.ecarts[:6]))
    else:
        # Dégradation gracieuse : nœuds cibles non réalisés (typiquement ceux qui
        # exigent des manœuvres sur les niveaux de barres supplémentaires — self/
        # réactance des JdB 3/4 — hors de portée de l'algorithme).
        obtenu = {frozenset(n.equipment_ids)
                  for n in res.topo_obtenue.noeuds.values()}
        non_realises: list[list[str]] = []
        seen: set[frozenset] = set()
        for n in topo_cible.noeuds.values():
            g = frozenset(n.equipment_ids)
            if g not in obtenu and g not in seen:
                seen.add(g)
                non_realises.append(sorted(g))
        _consigner_non_realisables(res, non_realises)
        res.message = (
            "Topologie cible partiellement atteinte (poste multi-barres) : "
            f"{res.topo_obtenue.nb_noeuds}/{topo_cible.nb_noeuds} nœuds atteints ; "
            f"{len(non_realises)} nœud(s) à compléter manuellement — manœuvres sur "
            "les niveaux de barres supplémentaires (self/réactance) non gérées.")
    return res


def determiner_topo_complete_cible(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    cible_busbar: Optional[dict[str, int]] = None,
) -> ResultatManoeuvres:
    """
    Calcule la séquence de manœuvres pour atteindre ``topo_cible`` depuis
    l'état détaillé courant de ``poste``.

    Parameters
    ----------
    poste :
        Vue complète du poste (cellules, tronçonnement, topologie nodale
        courante), construite par ``PosteTopologique.from_graph``.
    topo_cible :
        Topologie nodale visée.
    cible_busbar :
        Optionnel — barre cible exacte de chaque départ (transmis au séquenceur
        pour placer chaque départ directement sur sa barre finale).

    Returns
    -------
    ResultatManoeuvres
    """
    res = ResultatManoeuvres(
        voltage_level_id=poste.voltage_level_id,
        topo_initiale=poste.topologie_nodale,
        topo_cible=topo_cible,
    )

    # On travaille sur une copie du graphe pour simuler les manœuvres.
    G = poste.graph.copy() if poste.graph is not None else None
    if G is None:
        res.message = "Graphe absent : impossible de calculer les manœuvres."
        return res

    # --- Court-circuit : la topologie courante satisfait déjà la cible ----
    if poste.topologie_nodale.meme_topologie(topo_cible):
        res.is_changed = False
        res.is_verified = True
        res.topo_obtenue = poste.topologie_nodale
        res.message = "La topologie courante satisfait déjà la cible (aucune manœuvre)."
        return res

    # --- Phase 2.1 : faisabilité (départs cibles présents) ----------------
    departs_poste = {c.equipment_id for c in poste.cellules.cellules_depart}
    for c in poste.cellules.cellules_depart:
        departs_poste |= set(c.shared_equipment_ids)
    manquants = set(topo_cible.noeud_par_depart) - departs_poste
    if manquants:
        res.topo_obtenue = poste.topologie_nodale
        res.message = f"Départs cibles absents du poste : {sorted(manquants)}"
        return res

    # --- Phases 2.2-2.4 : placement automatique nœud -> sections de barres -
    placement, faisable, msg, non_places = _placement_automatique(poste, topo_cible)
    if not placement:
        # Rien de plaçable, même en best-effort : aucune manœuvre possible.
        res.topo_obtenue = poste.topologie_nodale
        res.message = msg
        res.noeuds_non_realisables = non_places
        return res

    # --- Délégation au séquenceur général (couplage + sectionnement) -------
    core = determiner_manoeuvres_avec_sections(poste, placement, cible_busbar)
    core.topo_initiale = poste.topologie_nodale
    core.topo_cible = topo_cible
    core.is_verified = bool(
        core.topo_obtenue and topo_cible.meme_topologie(core.topo_obtenue)
    )
    core.is_changed = bool(core.manoeuvres)

    # --- Repli connectivité-based (postes > 2 barres à faisceaux partagés) --
    # Si le séquenceur général n'atteint pas la cible sur un poste > 2 barres
    # (faisceaux de couplage partagés mal décomposés), on tente le réalisateur
    # connectivité-based. **Transactionnel** : on ne le retient que s'il vérifie
    # exactement la cible → ne peut jamais dégrader un résultat déjà correct.
    if (faisable and not core.is_verified
            and len(set(poste.tronconnement.barre_par_busbar.values())) > 2):
        alt = determiner_manoeuvres_par_connectivite(poste, placement, topo_cible)
        if alt.is_verified:
            alt.topo_initiale = poste.topologie_nodale
            alt.topo_cible = topo_cible
            return alt

    if faisable:
        core.message = (
            "Topologie cible atteinte et vérifiée." if core.is_verified
            else "La topologie obtenue ne correspond pas à la cible "
                 f"(obtenu {core.topo_obtenue.nb_noeuds if core.topo_obtenue else 0} "
                 f"nœud(s), visé {topo_cible.nb_noeuds})."
        )
    else:
        # Dégradation gracieuse (option 4) : placement partiel + diagnostic.
        core.noeuds_non_realisables = non_places
        core.message = msg
    return core


def determiner_manoeuvres_cible_detaillee(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    mode: str = "smooth",
) -> ResultatManoeuvres:
    """
    Atteint une **topologie détaillée cible imposée** (état précis de chaque
    organe, donc de la barre de chaque départ), plus spécifique que la seule
    topologie nodale.

    ``mode`` :
    - ``"smooth"`` (défaut) : dé-énergise **un ouvrage à la fois** (boucle longue,
      ré-alimentation immédiate) ; chaque départ est placé directement sur sa
      barre cible (pas de double-déplacement).
    - ``"aggressive"`` : orchestration **batch** — dé-énergise en une fois tous
      les ouvrages concernés, commute les SA hors tension, puis ré-alimente une
      seule fois (bien moins de manœuvres, plus d'ouvrages momentanément hors
      tension). Voir ``_sequence_detaillee_aggressive``.

    Démarche (mode smooth) :
    1. **séquence nodale** sûre. Si des **DJ de départ changent d'état**
       (reconnexions / déconnexions), le placement nœud→SJB est calculé sur un
       poste virtuel (reconnexions appliquées) mais le **séquenceur tourne sur
       le poste réel** (DJ encore ouverts) afin que les sections cibles restent
       hors tension pendant les manœuvres de sectionnement (règle du
       sectionneur). Sinon, on délègue à ``determiner_topo_complete_cible`` ;
    2. **raffiner** : ramener chaque départ sur sa barre exacte imposée par la
       cible (ré-aiguillage en boucle courte, équipotentiel) ;
    3. **changements de DJ de départ** : fermer les DJ des reconnexions (mise en
       service, la barre cible est désormais au bon potentiel) et ouvrir ceux
       des déconnexions (mise hors service) ;
    4. **vérifier** topologie nodale + détaillée ; consigner les **écarts**.
    """
    vl = poste.voltage_level_id
    cells = poste.cellules

    reconnections, disconnections = _departure_dj_changes(poste, cible_graph)

    # Barre cible (imposée) de chaque départ
    cible_busbar: dict[str, int] = {}
    for c in cells.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            bb = _wired_busbar(c, cible_graph)
            if bb is not None:
                cible_busbar[eq] = bb

    topo_cible = TopologieNodale.from_graph(cible_graph, vl)

    # Dégradation gracieuse : diagnostic + nœuds non réalisés (placement partiel).
    degradation: Optional[str] = None
    non_places: list[list[str]] = []

    # --- Mode agressif : orchestration batch (dé-énergiser une fois) --------
    if mode == "aggressive":
        return _sequence_detaillee_aggressive(poste, cible_graph, cible_busbar)

    # --- Poste à > 2 jeux de barres : placement par composantes ------------
    if len(set(poste.tronconnement.barre_par_busbar.values())) > 2:
        return _sequence_detaillee_multibarres(poste, cible_graph, topo_cible)

    # --- Phase 1 : séquence nodale sûre ------------------------------------
    if reconnections:
        # Faisabilité : départs cibles présents
        departs_poste = {c.equipment_id for c in cells.cellules_depart}
        for c in cells.cellules_depart:
            departs_poste |= set(c.shared_equipment_ids)
        manquants = set(topo_cible.noeud_par_depart) - departs_poste

        res = ResultatManoeuvres(
            voltage_level_id=vl,
            topo_initiale=poste.topologie_nodale,
            topo_cible=topo_cible,
        )
        if manquants:
            res.topo_obtenue = poste.topologie_nodale
            res.message = ("Topologie nodale cible non atteinte : départs cibles "
                           f"absents du poste : {sorted(manquants)}")
            return res

        placement, faisable, msg, np_ = _placement_avec_reconnexions(
            poste, cible_graph, topo_cible, reconnections)
        if not placement:
            res.topo_obtenue = poste.topologie_nodale
            res.message = "Topologie nodale cible non atteinte : " + msg
            res.noeuds_non_realisables = np_
            return res

        # Séquenceur sur le poste RÉEL (DJ reconnectés encore ouverts).
        res = determiner_manoeuvres_avec_sections(poste, placement, cible_busbar)
        res.topo_initiale = poste.topologie_nodale
        res.topo_cible = topo_cible
        if not faisable:
            degradation, non_places = msg, np_
    else:
        res = determiner_topo_complete_cible(poste, topo_cible, cible_busbar)
        if not res.is_verified and not res.noeuds_non_realisables:
            res.message = "Topologie nodale cible non atteinte : " + res.message
            return res
        if res.noeuds_non_realisables:
            degradation, non_places = res.message, res.noeuds_non_realisables

    # État détaillé atteint après la séquence nodale
    G = poste.graph.copy()
    for m in res.manoeuvres:
        _set_switch(G, m.switch_id, m.action == "OPEN")

    # 2. Raffinement : ramener chaque départ sur sa barre cible (boucle courte,
    #    équipotentielle puisque le nœud est déjà constitué). Pour un départ
    #    reconnecté, le DJ est encore ouvert : le ré-aiguillage SA reste sûr.
    extra: list[Manoeuvre] = []
    # Départs des nœuds non réalisés : laissés strictement en place (l'opérateur
    # complètera la séquence) — on ne les raffine ni ne touche leurs DJ.
    non_places_eqs = {eq for grp in non_places for eq in grp}
    for eq, target in sorted(cible_busbar.items()):
        if eq in non_places_eqs:
            continue
        cell = cells.get_cellule_depart(eq)
        cur = _wired_busbar(cell, G)
        if cur == target:
            continue
        if cur is None:
            # Départ non câblé : on ne « garage » son SA sur la barre cible que
            # s'il est hors tension (DJ propre ouvert) — manœuvre sûre et
            # conforme à la cible détaillée (préparation de section).
            if not any(_is_open(G, b.switch_id) for b in cell.breakers):
                continue
        if _reaiguiller_vers_sjb(G, cells, eq, target, extra):
            res.departs_reaiguilles.add(eq)

    # 3. Changements de DJ de départ (la barre cible est désormais au bon
    #    potentiel : fermeture/ouverture sûres ; nœuds non réalisés ignorés).
    post_manoeuvres = _appliquer_changements_dj(
        G, reconnections, disconnections, skip=non_places_eqs)

    res.manoeuvres = res.manoeuvres + extra + post_manoeuvres

    # 4. Vérification nodale + détaillée + écarts
    res.topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.is_verified = topo_cible.meme_topologie(res.topo_obtenue)
    res.is_changed = bool(res.manoeuvres)
    res.ecarts = (_ecarts_detailles(poste, G, cible_graph, cible_busbar)
                  + _verifier_regles(poste, res.manoeuvres, un_seul=True))
    res.is_verified_detaillee = res.is_verified and not res.ecarts
    if degradation:
        # Dégradation gracieuse (option 4) : cible partiellement atteinte.
        _consigner_non_realisables(res, non_places)
        res.message = (
            degradation
            + f" Placement partiel : {len(non_places)} nœud(s) non réalisé(s) "
            f"laissé(s) à l'opérateur ; {res.nb_manoeuvres} manœuvre(s) "
            "partielle(s) générée(s) — complétez la séquence manuellement."
        )
    elif not res.is_verified:
        res.message = (
            "Topologie nodale cible non atteinte : la topologie obtenue ne "
            f"correspond pas à la cible (obtenu {res.topo_obtenue.nb_noeuds} "
            f"nœud(s), visé {topo_cible.nb_noeuds})."
        )
    elif res.is_verified_detaillee:
        res.message = "Topologie détaillée cible atteinte et vérifiée."
    else:
        res.message = (
            f"Topologie nodale atteinte ; {len(res.ecarts)} écart(s) détaillé(s) "
            "résiduel(s) : " + " ; ".join(res.ecarts[:6])
        )
    return res
