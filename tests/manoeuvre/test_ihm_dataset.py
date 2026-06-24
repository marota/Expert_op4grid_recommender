"""
tests/manoeuvre/test_ihm_dataset.py
-----------------------------------
Endpoints de la **source dataset** de l'IHM de manœuvre
(``/api/dataset/config``, ``/api/dataset/timestamps``, ``/api/dataset/load``) et
garde-fous « aucune situation chargée » (``SESSION is None``) du mode dataset.

La frontière réseau (``manoeuvre.dataset.source``) est monkeypatchée : **aucun
appel HuggingFace**. ``/api/dataset/load`` est servi sur le réseau de test
pypowsybl.
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


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_dataset_mod",
                                                  _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


@pytest.fixture()
def dataset_mode(monkeypatch):
    """Mode dataset activé, aucune situation chargée."""
    monkeypatch.setattr(ihm, "SESSION", None)
    monkeypatch.setitem(ihm.DATASET, "enabled", True)
    monkeypatch.setitem(ihm.DATASET, "repo", "OpenSynth/D-GITT-RTE7000-2021")
    monkeypatch.setitem(ihm.DATASET, "default_date", "2021-01-03")
    monkeypatch.setitem(ihm.DATASET, "default_time", "12:00")
    monkeypatch.setitem(ihm.DATASET, "sample_dates", ["2021-01-03", "2021-01-05"])
    return ihm.app.test_client()


# --- /api/dataset/config ---------------------------------------------------

def test_config_enabled(dataset_mode):
    d = dataset_mode.get("/api/dataset/config").get_json()
    assert d["enabled"] is True
    assert d["repo"].endswith("D-GITT-RTE7000-2021")
    assert d["default_date"] == "2021-01-03"
    assert d["default_time"] == "12:00"
    assert d["sample_dates"] == ["2021-01-03", "2021-01-05"]


def test_config_disabled(monkeypatch):
    monkeypatch.setitem(ihm.DATASET, "enabled", False)
    d = ihm.app.test_client().get("/api/dataset/config").get_json()
    assert d["enabled"] is False


# --- garde-fous SESSION is None --------------------------------------------

def test_postes_needs_date(dataset_mode):
    d = dataset_mode.get("/api/postes").get_json()
    assert d["needs_date"] is True
    assert d["postes"] == [] and d["all"] == [] and d["catalog"] == []


def test_algos_defaults_sans_session(dataset_mode):
    d = dataset_mode.get("/api/algos").get_json()
    assert set(d["selection"]) == {"identificateur", "sequenceur", "planificateur"}
    assert d["selection"]["sequenceur"] == "libtopo"
    assert "libtopo" in d["disponibles"]["sequenceur"]


def test_scenarios_vide_sans_session(dataset_mode):
    assert dataset_mode.get("/api/scenarios").get_json()["scenarios"] == []


# --- /api/dataset/timestamps -----------------------------------------------

def test_timestamps_ok(dataset_mode, monkeypatch):
    monkeypatch.setattr(ihm.dataset_source, "lister_instantanes",
                        lambda repo, date, token=None: [
                            {"ts": "00:00", "iso": date + "T00:00", "path": "a"},
                            {"ts": "12:00", "iso": date + "T12:00", "path": "b"},
                            {"ts": "23:55", "iso": date + "T23:55", "path": "c"}])
    d = dataset_mode.get("/api/dataset/timestamps?date=2021-01-03").get_json()
    assert d["ok"] is True
    assert [t["ts"] for t in d["timestamps"]] == ["00:00", "12:00", "23:55"]
    assert d["default"] == "12:00"


def test_timestamps_date_invalide(dataset_mode):
    # prefixe_jour (réel) lève ValueError AVANT tout accès réseau => 400.
    r = dataset_mode.get("/api/dataset/timestamps?date=notadate")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


# --- /api/dataset/load -----------------------------------------------------

def test_load_situation(dataset_mode, monkeypatch):
    net = pp.network.create_four_substations_node_breaker_network()
    monkeypatch.setattr(
        ihm.dataset_source, "charger_situation",
        lambda repo, date, cache, heure="12:00", token=None: (
            net, {"date": date, "ts": heure, "iso": date + "T" + heure,
                  "path": "p", "local": "/tmp/p"}))
    d = dataset_mode.post("/api/dataset/load",
                          json={"date": "2021-01-03", "time": "12:00"}).get_json()
    assert d["ok"] is True
    assert d["date"] == "2021-01-03" and d["time"] == "12:00"
    assert "S1VL2" in d["all"]                       # VL du réseau de test
    assert ihm.SESSION is not None                   # session reconstruite
    # La topologie d'un poste s'extrait ensuite via l'endpoint existant /api/load.
    v = dataset_mode.post("/api/load", json={"vl": "S1VL2"}).get_json()
    assert {"svg", "switches", "nb_noeuds", "nodale_depart"} <= set(v)


def test_load_situation_absente(dataset_mode, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("Aucun instantané pour 2099-01-01")
    monkeypatch.setattr(ihm.dataset_source, "charger_situation", boom)
    r = dataset_mode.post("/api/dataset/load",
                          json={"date": "2099-01-01", "time": "12:00"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


# --- repo_pour_date : sélection du dataset par année (2021/2022/2023) -------

def test_repo_pour_date_swaps_year():
    f = ihm.dataset_source.repo_pour_date
    base = "OpenSynth/D-GITT-RTE7000-2021"
    assert f(base, "2022-06-15") == "OpenSynth/D-GITT-RTE7000-2022"
    assert f(base, "2023-02-08") == "OpenSynth/D-GITT-RTE7000-2023"
    assert f(base, "2021-01-03") == "OpenSynth/D-GITT-RTE7000-2021"
    assert f("me/custom-grid", "2022-06-15") == "me/custom-grid"   # pas d'année
    assert f(base, "notadate") == base                              # date invalide


def test_timestamps_uses_year_derived_repo(dataset_mode, monkeypatch):
    seen = {}

    def fake(repo, date, token=None):
        seen["repo"] = repo
        return [{"ts": "12:00", "iso": date + "T12:00", "path": "p"}]

    monkeypatch.setattr(ihm.dataset_source, "lister_instantanes", fake)
    dataset_mode.get("/api/dataset/timestamps?date=2022-06-15")
    assert seen["repo"].endswith("D-GITT-RTE7000-2022")
