"""
tests/manoeuvre/test_ihm_algo_selection.py
-------------------------------------------
Migration de l'IHM vers la **façade pluggable** (``manoeuvre.plugins``) et
sélection des algorithmes par phase :

- sélection par défaut « libtopo » pour les trois phases ; ``GET /api/algos``
  expose disponibles + sélection ;
- ``Session.set_algos`` / ``POST /api/algos`` : choix valide appliqué, phase ou
  nom inconnus ignorés ;
- un **plugin tiers** enregistré apparaît dans les disponibles, est utilisable
  pour le séquencement, et ses verdicts sont **recalculés indépendamment**
  (un plugin mensonger est démasqué dans la payload IHM) ;
- les payloads ``sequence`` / ``nodale_to_detaillee`` portent l'``algo`` utilisé.

Réseau de référence : ``create_four_substations_node_breaker_network`` (S1VL2).
Nécessite ``flask`` + ``pypowsybl``.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

from expert_op4grid_recommender.manoeuvre.algo.results import (  # noqa: E402
    ResultatManoeuvres,
)
from expert_op4grid_recommender.manoeuvre.plugins import register  # noqa: E402
from expert_op4grid_recommender.manoeuvre.plugins import (  # noqa: E402
    registry as registry_mod,
)

_IHM_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "manoeuvre_ihm.py")
VL = "S1VL2"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_algos_mod",
                                                  _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


def _session():
    s = ihm.Session(pp.network.create_four_substations_node_breaker_network())
    s.load(VL)
    return s


def _session_avec_cible_scindee():
    """Session avec une cible scindée en 2 nœuds (exige des manœuvres)."""
    s = _session()
    flat = sorted(eq for g in s.groups_of(s.initial) for eq in g)
    half = len(flat) // 2
    s.nodale_to_detaillee([flat[:half], flat[half:]])
    return s


class SequenceurMenteurIHM:
    """Plugin tiers : séquence vide, verdicts mensongers."""
    nom = "menteur_ihm"

    def sequencer(self, poste, cible, **options) -> ResultatManoeuvres:
        res = ResultatManoeuvres(
            voltage_level_id=poste.voltage_level_id,
            topo_initiale=poste.topologie_nodale,
            topo_cible=cible.topologie_nodale(poste),
        )
        res.is_verified = True            # mensonge
        res.is_verified_detaillee = True  # mensonge
        return res


@pytest.fixture()
def plugin_menteur():
    register("sequenceur", "menteur_ihm", SequenceurMenteurIHM)
    try:
        yield "menteur_ihm"
    finally:
        registry_mod._registres["sequenceur"].pop("menteur_ihm", None)


# ---------------------------------------------------------------------------
# Sélection d'algorithmes (Session + API)
# ---------------------------------------------------------------------------

def test_selection_par_defaut_libtopo():
    s = _session()
    assert s.algos == {"identificateur": "libtopo", "sequenceur": "libtopo",
                       "planificateur": "libtopo"}


def test_set_algos_ignore_les_choix_invalides():
    s = _session()
    sel = s.set_algos({"sequenceur": "inexistant", "phase_bidon": "libtopo"})
    assert sel["sequenceur"] == "libtopo"
    assert "phase_bidon" not in sel


def test_set_algos_accepte_un_plugin_enregistre(plugin_menteur):
    s = _session()
    sel = s.set_algos({"sequenceur": plugin_menteur})
    assert sel["sequenceur"] == plugin_menteur
    # Les autres phases restent inchangées.
    assert sel["identificateur"] == "libtopo"


def test_api_algos_get_et_post(monkeypatch, plugin_menteur):
    s = _session()
    monkeypatch.setattr(ihm, "SESSION", s)
    client = ihm.app.test_client()
    r = client.get("/api/algos").get_json()
    for phase in ("identificateur", "sequenceur", "planificateur"):
        assert "libtopo" in r["disponibles"][phase]
        assert r["selection"][phase] == "libtopo"
    assert plugin_menteur in r["disponibles"]["sequenceur"]
    r2 = client.post("/api/algos",
                     json={"sequenceur": plugin_menteur}).get_json()
    assert r2["selection"]["sequenceur"] == plugin_menteur
    # Nom inconnu : ignoré, la sélection courante est renvoyée.
    r3 = client.post("/api/algos", json={"sequenceur": "bidon"}).get_json()
    assert r3["selection"]["sequenceur"] == plugin_menteur


# ---------------------------------------------------------------------------
# Calculs via la façade : algo utilisé + vérification indépendante
# ---------------------------------------------------------------------------

def test_sequence_libtopo_via_facade():
    s = _session_avec_cible_scindee()
    payload = s.sequence("smooth")
    assert payload["algo"] == "libtopo"
    assert payload["verified"] is True
    assert payload["nb_manoeuvres"] > 0
    assert payload["matches_cible"] is True


def test_sequence_plugin_menteur_demasque(plugin_menteur):
    """Le plugin prétend tout vérifier avec une séquence vide : la payload IHM
    (issue de la vérification indépendante de la façade) le démasque."""
    s = _session_avec_cible_scindee()
    s.set_algos({"sequenceur": plugin_menteur})
    payload = s.sequence("smooth")
    assert payload["algo"] == plugin_menteur
    assert payload["verified"] is False
    assert payload["verified_detaillee"] is False
    assert payload["nb_manoeuvres"] == 0
    assert payload["matches_cible"] is False
    assert payload["ecarts"], "écarts détaillés vs la cible attendus"


def test_nodale_to_detaillee_porte_l_algo_et_reste_verifiee():
    s = _session()
    p = s.nodale_payload(s.initial)
    res = s.nodale_to_detaillee(p["groups"], p["isolated"])
    assert res["algo"] == "libtopo"
    assert res["is_verified"] is True
