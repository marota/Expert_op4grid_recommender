"""
tests/manoeuvre/test_ihm_cache_and_api.py
-----------------------------------------
Filets de sécurité pour la revue qualité de l'IHM (``scripts/manoeuvre_ihm.py``) :

- **#4** — le vérificateur de règle « sectionneur sous charge » est désormais
  **public** (``manoeuvre.sectionneurs_sous_charge_par_manoeuvre``), l'alias privé
  historique étant conservé pour compatibilité ;
- **#5** — ``Session.applied(state)`` restaure l'état d'affichage courant en
  sortie de bloc, **y compris sur exception** ;
- **#6** — graphe / topologie / load flow d'un état sont **mémoïsés par état** et
  invalidés au chargement d'un poste, **sans changer les sorties**.

Nécessitent ``flask`` et ``pypowsybl`` (sinon test ignoré).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

_IHM_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "manoeuvre_ihm.py")
VL = "S1VL2"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_cache_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


@pytest.fixture()
def session():
    net = pp.network.create_four_substations_node_breaker_network()
    s = ihm.Session(net)
    s.load(VL)
    return s


# --------------------------------------------------------------------------
# #4 — vérificateur de règle public
# --------------------------------------------------------------------------

def test_public_checker_exported_from_package():
    from expert_op4grid_recommender import manoeuvre
    assert hasattr(manoeuvre, "sectionneurs_sous_charge_par_manoeuvre")
    assert "sectionneurs_sous_charge_par_manoeuvre" in manoeuvre.__all__


def test_public_checker_is_private_alias():
    from expert_op4grid_recommender.manoeuvre.algo import (
        sectionneurs_sous_charge_par_manoeuvre as pub,
        _sectionneurs_sous_charge_par_manoeuvre as priv,
    )
    assert pub is priv


def test_ihm_uses_public_checker():
    # L'IHM ne franchit plus la frontière privée : elle référence le symbole public.
    assert hasattr(ihm, "sectionneurs_sous_charge_par_manoeuvre")
    assert not hasattr(ihm, "_sectionneurs_sous_charge_par_manoeuvre")


# --------------------------------------------------------------------------
# #5 — gestionnaire de contexte applied(state)
# --------------------------------------------------------------------------

def test_applied_restores_current_on_success(session):
    before = dict(session.current)
    with session.applied(session.initial):
        pass
    # Réseau restauré sur l'affichage courant.
    g = ihm.build_vl_graph(session.net, VL)
    restored = ihm.TopologieNodale.from_graph(g, VL)
    cur = session._topo(session.current)
    assert restored.meme_topologie(cur)
    assert session.current == before


def test_applied_restores_current_on_exception(session):
    before = dict(session.current)
    with pytest.raises(RuntimeError):
        with session.applied(session.initial):
            raise RuntimeError("boom")
    assert session.current == before
    # L'état réseau effectif correspond bien à ``current`` (et non à ``initial``).
    g = ihm.build_vl_graph(session.net, VL)
    assert ihm.TopologieNodale.from_graph(g, VL).meme_topologie(
        session._topo(session.current))


# --------------------------------------------------------------------------
# #6 — mémoïsation par état (graphe / topo / flux)
# --------------------------------------------------------------------------

def test_caches_start_empty_then_fill(session):
    assert session._graph_cache == {}
    assert session._topo_cache == {}
    assert session._flow_cache == {}
    session.nodale_payload(session.initial)
    assert session._graph_cache
    assert session._topo_cache
    assert session._flow_cache


def test_repeated_payload_is_cache_hit_and_identical(session):
    first = session.nodale_payload(session.initial)
    n_graph = len(session._graph_cache)
    n_topo = len(session._topo_cache)
    second = session.nodale_payload(session.initial)
    # Même état → pas de nouvelle entrée de cache, sortie identique.
    assert len(session._graph_cache) == n_graph
    assert len(session._topo_cache) == n_topo
    assert first == second


def test_load_clears_caches(session):
    session.nodale_payload(session.initial)
    assert session._graph_cache and session._topo_cache and session._flow_cache
    session.load(VL)
    assert session._graph_cache == {}
    assert session._topo_cache == {}
    assert session._flow_cache == {}


def test_cache_keyed_by_state_distinguishes_topologies(session):
    # Tout ouvrir produit un état distinct → clé de cache distincte, sortie distincte.
    base = session.nodale_state(session.initial)
    n_keys = len(session._graph_cache)
    for sid in list(session.switches_df(VL).index):
        session.current[sid] = True
    opened = session.nodale_state(session.current)
    assert len(session._graph_cache) > n_keys          # nouvelle entrée
    assert opened["isolated"]                            # tout déconnecté
    assert base["isolated"] == []


def test_memoized_topo_matches_fresh_build(session):
    # La topo mémoïsée doit être équivalente à une reconstruction directe.
    session.apply(session.initial)
    fresh = ihm.TopologieNodale.from_graph(
        ihm.build_vl_graph(session.net, VL), VL)
    cached = session._topo(session.initial)
    assert cached.meme_topologie(fresh)


# --------------------------------------------------------------------------
# Sélecteur de fichier natif (/api/pick_grid_file) — onglet « Local »
# --------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_pick_grid_file_returns_selected_path(monkeypatch):
    monkeypatch.setattr(ihm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ihm.subprocess, "run",
                        lambda *a, **k: _FakeProc(0, stdout="/data/grid.xiidm\n"))
    r = ihm.app.test_client().get("/api/pick_grid_file")
    assert r.status_code == 200
    assert r.get_json() == {"path": "/data/grid.xiidm"}


def test_pick_grid_file_degrades_without_display(monkeypatch):
    # Sans afficheur / tkinter (Space headless) : le sous-processus échoue ;
    # l'endpoint renvoie une erreur exploitable (jamais de 500).
    monkeypatch.setattr(ihm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ihm.subprocess, "run",
                        lambda *a, **k: _FakeProc(1, stderr="No module named tkinter"))
    d = ihm.app.test_client().get("/api/pick_grid_file").get_json()
    assert d["path"] == ""
    assert "tkinter" in d["error"]


def test_pick_grid_file_timeout_is_graceful(monkeypatch):
    def _boom(*a, **k):
        raise ihm.subprocess.TimeoutExpired(cmd="picker", timeout=300)
    monkeypatch.setattr(ihm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ihm.subprocess, "run", _boom)
    d = ihm.app.test_client().get("/api/pick_grid_file").get_json()
    assert d["path"] == ""
    assert "expiré" in d["error"]


# --------------------------------------------------------------------------
# promote_cible : promouvoir la cible courante en nouvel état de départ
# --------------------------------------------------------------------------

def test_promote_cible_makes_target_the_new_departure(session):
    sid = next(iter(session.current))
    session.toggle(sid)                       # édite la cible
    edited = dict(session.current)
    session.seq_manoeuvres = [
        {"switch_id": sid, "action": "OPEN", "raison": "x", "boucle": None}]
    session.promote_cible()
    assert session.initial == edited          # départ = ancienne cible
    assert session.current == edited          # cible repart = nouveau départ
    assert session.seq_manoeuvres == []       # séquence réinitialisée
    assert session.scenario_name is None


def test_api_promote_cible_returns_both_panes(session, monkeypatch):
    monkeypatch.setattr(ihm, "SESSION", session)
    d = ihm.app.test_client().post("/api/promote_cible", json={}).get_json()
    assert {"initial_svg", "svg", "switches", "nb_noeuds",
            "nodale_depart", "nodale_cible", "vl"} <= set(d)


# --------------------------------------------------------------------------
# Sauvegarde : contenu renvoyé (téléchargement local) + nœuds vides ignorés
# --------------------------------------------------------------------------

def test_api_save_returns_content_for_download(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SESSION", session)
    monkeypatch.setattr(ihm, "SCEN_DIR", tmp_path)
    d = ihm.app.test_client().post("/api/save", json={"name": "t"}).get_json()
    assert d["name"] == "t"
    assert d["content"] and '"voltage_level_id"' in d["content"]
    assert (tmp_path / "t.json").exists()


def test_api_save_dedup_and_autoindex(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SESSION", session)
    monkeypatch.setattr(ihm, "SCEN_DIR", tmp_path)
    c = ihm.app.test_client()
    r1 = c.post("/api/save", json={"name": "s"}).get_json()
    assert r1["name"] == "s" and (tmp_path / "s.json").exists()
    # ré-enregistrer un scénario identique (départ + cible) → non écrasé.
    r2 = c.post("/api/save", json={"name": "s"}).get_json()
    assert r2.get("already_exists") and r2["name"] == "s"
    assert not (tmp_path / "s_0.json").exists()
    # modifier la cible → nom unique indexé _0.
    sid = next(iter(session.current))
    session.current[sid] = not session.current[sid]
    r3 = c.post("/api/save", json={"name": "s"}).get_json()
    assert r3["name"] == "s_0" and (tmp_path / "s_0.json").exists()
    # ré-enregistrer ce même état modifié → déjà existant (s_0), pas de _1.
    r4 = c.post("/api/save", json={"name": "s"}).get_json()
    assert r4.get("already_exists") and r4["name"] == "s_0"
    assert not (tmp_path / "s_1.json").exists()


def test_api_scenarios_metadata(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SESSION", session)
    monkeypatch.setattr(ihm, "SCEN_DIR", tmp_path)
    c = ihm.app.test_client()
    c.post("/api/save", json={"name": "m"})
    scens = c.get("/api/scenarios").get_json()["scenarios"]
    assert scens and isinstance(scens[0], dict)
    meta = next(x for x in scens if x["name"] == "m")
    for k in ("vl", "nominal_v", "nb_barres", "n_dj", "n_sa", "n_int",
              "n_nodal", "nb_depart", "nb_cible"):
        assert k in meta


def test_api_save_sequence_returns_content_for_download(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SESSION", session)
    monkeypatch.setattr(ihm, "SEQ_DIR", tmp_path)
    d = ihm.app.test_client().post("/api/save_sequence", json={"name": "sq"}).get_json()
    assert d["name"] == "sq"
    assert d["content"] and '"manoeuvres"' in d["content"]
    assert (tmp_path / "sq.json").exists()


def test_normalize_groups_ignores_empty_nodes():
    # Un groupe vide (nœud « ＋ Nœud » resté vide) est ignoré ; l'orphelin va
    # dans un nœud dédié → aucun nœud vide dans le résultat.
    out = ihm._normalize_groups(["a", "b", "c"], [["a"], [], ["b"]])
    assert [] not in out
    assert ["a"] in out and ["b"] in out and ["c"] in out
