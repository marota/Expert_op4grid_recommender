"""
tests/manoeuvre/test_dataset_combinaisons.py
---------------------------------------------
Catalogue des topologies rencontrées (``TimelinePoste.catalogue``) et
**scénarios combinés** (``generer_combinaisons``) : toute paire ordonnée de
topologies stables d'un même poste devient un scénario départ → cible
réaliste, potentiellement jamais observé et plus dur que les blocs réels.

Couverture :

- catalogue : dédoublonnage, occurrences (snapshots/épisodes), première/
  dernière vue, marquage ``stable``, tri ;
- combinaisons : paires dans les deux sens, tri par diff décroissant +
  plafond par poste, filtre ``min_organes``, exclusion des topologies
  instables, garde « même ensemble d'organes » (structure changée),
  dédoublonnage multi-catalogues (même ``topologie_id`` vu deux journées),
  format compatible ``blocs.jsonl`` (méta source/diff, pas de séquence
  observée).
"""
from __future__ import annotations

from expert_op4grid_recommender.manoeuvre.dataset import (
    Snapshot,
    TimelinePoste,
    TopologieRencontree,
    generer_combinaisons,
)


def _snap(t: int, etats: dict) -> Snapshot:
    return Snapshot(timestamp=f"2021-01-03T{t:04d}", etats=dict(etats))


def _tl_aba() -> TimelinePoste:
    """A A | B B B | A A : 2 topologies stables, A en 2 épisodes."""
    a = {"s1": False, "s2": False}
    b = {"s1": True, "s2": False}
    snaps = [_snap(0, a), _snap(1, a),
             _snap(2, b), _snap(3, b), _snap(4, b),
             _snap(5, a), _snap(6, a)]
    return TimelinePoste("POSTE1", snaps)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def test_catalogue_dedoublonne_et_compte():
    cat = _tl_aba().catalogue(min_stabilite=2)
    assert len(cat) == 2
    par_id = {e.etats["s1"]: e for e in cat}
    a, b = par_id[False], par_id[True]
    assert (a.nb_snapshots, a.nb_episodes, a.stable) == (4, 2, True)
    assert (b.nb_snapshots, b.nb_episodes, b.stable) == (3, 1, True)
    assert (a.premiere, a.derniere) == ("2021-01-03T0000", "2021-01-03T0006")
    # tri par occurrences décroissantes
    assert cat[0] is a


def test_catalogue_marque_instable():
    a = {"s1": False}
    c = {"s1": True}
    snaps = [_snap(0, a), _snap(1, a), _snap(2, c), _snap(3, a), _snap(4, a)]
    cat = TimelinePoste("P", snaps).catalogue(min_stabilite=2)
    etats = {e.etats["s1"]: e for e in cat}
    assert etats[False].stable is True
    assert etats[True].stable is False      # plateau d'1 snapshot seulement


# ---------------------------------------------------------------------------
# Combinaisons
# ---------------------------------------------------------------------------

def _entree(vl, tid, etats, stable=True):
    return TopologieRencontree(
        voltage_level_id=vl, topologie_id=tid, etats=etats,
        premiere="t0", derniere="t1", nb_snapshots=2, nb_episodes=1,
        stable=stable)


def test_combinaisons_paires_ordonnee_et_format():
    cat = [_entree("P", "ta", {"s1": False, "s2": False}),
           _entree("P", "tb", {"s1": True, "s2": True})]
    scs = generer_combinaisons(cat, min_organes=2)
    assert len(scs) == 2                     # A→B et B→A
    sc = next(s for s in scs if s["meta"]["topologie_depart_id"] == "ta")
    assert sc["voltage_level_id"] == "P"
    assert sc["depart"] == {"s1": False, "s2": False}
    assert sc["cible"] == {"s1": True, "s2": True}
    assert sc["meta"]["source"] == "combinaison"
    assert sc["meta"]["nb_organes_changes"] == 2
    assert sc["meta"]["nb_manoeuvres_observees"] is None
    assert "combinaison" in sc["name"]


def test_combinaisons_tri_par_diff_et_plafond():
    cat = [_entree("P", "t0", {"s1": False, "s2": False, "s3": False}),
           _entree("P", "t1", {"s1": True, "s2": False, "s3": False}),
           _entree("P", "t3", {"s1": True, "s2": True, "s3": True})]
    scs = generer_combinaisons(cat, max_par_poste=2, min_organes=1)
    assert len(scs) == 2
    # les plus durs d'abord : t0↔t3 (diff 3) dans les deux sens
    assert all(s["meta"]["nb_organes_changes"] == 3 for s in scs)


def test_combinaisons_min_organes_et_instables():
    cat = [_entree("P", "ta", {"s1": False, "s2": False}),
           _entree("P", "tb", {"s1": True, "s2": False}),          # diff 1
           _entree("P", "tc", {"s1": True, "s2": True}, stable=False)]
    assert generer_combinaisons(cat, min_organes=2) == []
    assert len(generer_combinaisons(cat, min_organes=1)) == 2     # ta↔tb


def test_combinaisons_structures_differentes_ecartees():
    cat = [_entree("P", "ta", {"s1": False, "s2": False}),
           _entree("P", "tb", {"s1": True, "s3": True})]   # autres organes
    assert generer_combinaisons(cat, min_organes=1) == []


def test_combinaisons_dedoublonnage_multi_catalogues():
    a = {"s1": False, "s2": False}
    b = {"s1": True, "s2": True}
    deux_jours = [_entree("P", "ta", a), _entree("P", "tb", b),
                  _entree("P", "ta", a), _entree("P", "tb", b)]   # jour 2
    assert len(generer_combinaisons(deux_jours, min_organes=1)) == 2
