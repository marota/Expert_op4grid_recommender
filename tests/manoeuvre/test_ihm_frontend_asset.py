"""
tests/manoeuvre/test_ihm_frontend_asset.py
------------------------------------------
Vérifie l'**externalisation du front-end** de l'IHM (#8) : le HTML/CSS/JS vit
désormais dans ``scripts/manoeuvre_ihm_assets/index.html`` et est servi tel quel
par la route ``/``. Garde-fou contre un asset manquant / un chargement cassé.

Nécessite ``flask`` et ``pypowsybl`` (sinon test ignoré).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_IHM_PATH = _ROOT / "scripts" / "manoeuvre_ihm.py"
_ASSET = _ROOT / "scripts" / "manoeuvre_ihm_assets" / "index.html"
VL = "S1VL2"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_front_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


def test_asset_file_exists_and_is_html():
    assert _ASSET.is_file(), "asset front-end absent"
    txt = _ASSET.read_text(encoding="utf-8")
    assert txt.startswith("<!DOCTYPE html>")
    assert "</html>" in txt


def test_page_constant_loaded_from_asset():
    # PAGE est chargé depuis le fichier (pas de HTML embarqué dans le .py).
    assert ihm.PAGE == _ASSET.read_text(encoding="utf-8")


def test_index_route_serves_asset():
    ihm.SESSION = ihm.Session(pp.network.create_four_substations_node_breaker_network())
    ihm.SESSION.load(VL)
    client = ihm.app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert r.mimetype == "text/html"
    assert r.get_data(as_text=True) == _ASSET.read_text(encoding="utf-8")


def test_python_source_no_longer_embeds_full_page():
    # Le .py ne doit plus contenir le gros bloc HTML (uniquement le loader).
    src = _IHM_PATH.read_text(encoding="utf-8")
    assert 'PAGE = r"""' not in src
    assert "manoeuvre_ihm_assets" in src
