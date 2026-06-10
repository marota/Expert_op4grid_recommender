"""
tests/manoeuvre/test_dataset_dgitt_xiidm.py
--------------------------------------------
Adaptateur ``manoeuvre.dataset.dgitt`` sur le **format réel** du dataset
``OpenSynth/D-GITT-RTE7000-2021`` : des instantanés **XIIDM** (un par pas de
temps), nommés ``recollement-auto-YYYYMMDD-HHMM-enrichi.xiidm.bz2``.

Couverture :

- horodatage extrait du nom de fichier (ISO, ordre lexical = ordre temporel) ;
- découverte des instantanés (``.md5`` jumeaux exclus) ;
- **bout en bout** : une petite série d'instantanés XIIDM réels (réseau
  pypowsybl de référence, avec un organe basculé) est écrite en ``.xiidm.bz2``,
  relue par l'adaptateur, et la chronologie reconstruite retrouve la transition
  (bloc unique, manœuvre observée) — sans aucun accès réseau ;
- ``vl_filter``, sous-échantillonnage, auto-détection du format, gestion
  d'erreur.

Le chemin XIIDM dépend de pypowsybl (``importorskip``) ; les tests d'horodatage
et de découverte sont en Python pur.
"""
from __future__ import annotations

import bz2
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre.dataset.dgitt import (
    _a_des_xiidm,
    _est_xiidm,
    _fichiers_xiidm,
    charger_timelines,
    charger_timelines_xiidm,
    horodatage_depuis_nom,
)

# Organe basculé pour fabriquer une transition (DJ fermé au départ).
_SWITCH_BASCULE = "S1VL1_LD1_BREAKER"
_VLS_RESEAU = {"S1VL1", "S1VL2", "S2VL1", "S3VL1", "S4VL1"}


# ---------------------------------------------------------------------------
# Horodatage depuis le nom de fichier (Python pur)
# ---------------------------------------------------------------------------

def test_horodatage_depuis_nom_format_rte():
    assert (horodatage_depuis_nom(
        "recollement-auto-20210103-0005-enrichi.xiidm.bz2")
        == "2021-01-03T00:05")
    assert (horodatage_depuis_nom(
        "recollement-auto-20211231-2355-enrichi.xiidm.bz2")
        == "2021-12-31T23:55")


def test_horodatage_ordre_lexical_est_temporel():
    noms = [
        "recollement-auto-20210103-0010-enrichi.xiidm.bz2",
        "recollement-auto-20210103-0000-enrichi.xiidm.bz2",
        "recollement-auto-20210102-2355-enrichi.xiidm.bz2",
    ]
    assert sorted(horodatage_depuis_nom(n) for n in noms) == [
        "2021-01-02T23:55", "2021-01-03T00:00", "2021-01-03T00:10"]


def test_horodatage_nom_sans_date_leve():
    with pytest.raises(ValueError, match="non horodaté"):
        horodatage_depuis_nom("sans_date-enrichi.xiidm.bz2")


def test_horodatage_date_invalide_leve():
    # mois 13 : motif reconnu mais date invalide
    with pytest.raises(ValueError, match="invalide"):
        horodatage_depuis_nom("recollement-auto-20211345-0000-enrichi.xiidm")


# ---------------------------------------------------------------------------
# Découverte des instantanés
# ---------------------------------------------------------------------------

def test_est_xiidm_exclut_md5(tmp_path):
    assert _est_xiidm(tmp_path / "x.xiidm.bz2")
    assert _est_xiidm(tmp_path / "x.xiidm")
    assert _est_xiidm(tmp_path / "x.iidm.gz")
    assert not _est_xiidm(tmp_path / "x.xiidm.bz2.md5")
    assert not _est_xiidm(tmp_path / "x.csv")
    # artefacts du cache `hf download` (constatés sur le dataset réel)
    assert not _est_xiidm(tmp_path / "x.xiidm.bz2.lock")
    assert not _est_xiidm(tmp_path / "x.xiidm.bz2.metadata")
    assert not _est_xiidm(tmp_path / "h=.aaaa.bbbb.incomplete")


def test_fichiers_xiidm_recursif_et_md5_ignores(tmp_path):
    d = tmp_path / "2021" / "01" / "03"
    d.mkdir(parents=True)
    (d / "recollement-auto-20210103-0000-enrichi.xiidm.bz2").write_bytes(b"x")
    (d / "recollement-auto-20210103-0000-enrichi.xiidm.bz2.md5").write_text("h")
    (tmp_path / "README.md").write_text("doc")
    fic = _fichiers_xiidm(tmp_path)
    assert [p.name for p in fic] == [
        "recollement-auto-20210103-0000-enrichi.xiidm.bz2"]
    assert _a_des_xiidm(tmp_path) is True
    assert _a_des_xiidm(tmp_path / "2021" / "01") is True


# ---------------------------------------------------------------------------
# Bout en bout : instantanés XIIDM réels (réseau pypowsybl) → chronologie
# ---------------------------------------------------------------------------

def _ecrire_instantane(net, dossier: pathlib.Path, horodatage_compact: str):
    """Écrit l'état courant de ``net`` en ``.xiidm.bz2`` nommé à la RTE
    (``recollement-auto-YYYYMMDD-HHMM-enrichi.xiidm.bz2``) + un ``.md5`` jumeau
    (pour vérifier qu'il est ignoré)."""
    dossier.mkdir(parents=True, exist_ok=True)
    base = f"recollement-auto-{horodatage_compact}-enrichi.xiidm"
    plat = dossier / base
    net.save(str(plat), format="XIIDM")
    brut = plat.read_bytes()
    plat.unlink()
    (dossier / (base + ".bz2")).write_bytes(bz2.compress(brut))
    (dossier / (base + ".bz2.md5")).write_text("00000000000000000000000000000000\n")


def _ecrire_serie(tmp_path) -> pathlib.Path:
    """Série d'instantanés : état A (0000, 0005), puis un organe ouvert →
    état B (0010, 0015). Retourne le dossier racine du dataset factice."""
    pp = pytest.importorskip("pypowsybl")
    racine = tmp_path / "dataset"
    jour = racine / "2021" / "01" / "03"
    net = pp.network.create_four_substations_node_breaker_network()
    _ecrire_instantane(net, jour, "20210103-0000")
    _ecrire_instantane(net, jour, "20210103-0005")
    net.update_switches(id=_SWITCH_BASCULE, open=True)
    _ecrire_instantane(net, jour, "20210103-0010")
    _ecrire_instantane(net, jour, "20210103-0015")
    return racine


def test_charger_timelines_xiidm_bout_en_bout(tmp_path):
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)

    tls = {tl.voltage_level_id: tl
           for tl in charger_timelines_xiidm(racine)}

    # Tous les postes du réseau, 4 instantanés chacun, horodatés et triés.
    assert set(tls) == _VLS_RESEAU
    assert all(len(tl.snapshots) == 4 for tl in tls.values())
    assert [s.timestamp for s in tls["S1VL1"].snapshots] == [
        "2021-01-03T00:00", "2021-01-03T00:05",
        "2021-01-03T00:10", "2021-01-03T00:15"]

    # Seul S1VL1 (organe basculé) présente une transition.
    blocs, osc = tls["S1VL1"].detecter_blocs(min_stabilite=2)
    assert len(blocs) == 1 and osc == []
    b = blocs[0]
    assert b.diff().get(_SWITCH_BASCULE) == (False, True)
    assert [(m["switch_id"], m["action"]) for m in b.manoeuvres_observees] == [
        (_SWITCH_BASCULE, "OPEN")]
    assert b.t_depart == "2021-01-03T00:05"
    assert b.t_cible == "2021-01-03T00:10"

    # Les autres postes sont stables : aucun bloc.
    for vl in _VLS_RESEAU - {"S1VL1"}:
        assert tls[vl].detecter_blocs(min_stabilite=2)[0] == []


def test_charger_timelines_xiidm_vl_filter(tmp_path):
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)
    tls = list(charger_timelines_xiidm(racine, vl_filter={"S1VL1"}))
    assert len(tls) == 1 and tls[0].voltage_level_id == "S1VL1"


def test_charger_timelines_xiidm_sous_echantillon(tmp_path):
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)
    tls = {tl.voltage_level_id: tl
           for tl in charger_timelines_xiidm(racine, sous_echantillon=2)}
    # 4 instantanés → un sur deux (indices 0 et 2).
    assert [s.timestamp for s in tls["S1VL1"].snapshots] == [
        "2021-01-03T00:00", "2021-01-03T00:10"]


def test_charger_timelines_auto_detecte_xiidm(tmp_path):
    """``charger_timelines`` emprunte le chemin XIIDM dès qu'un instantané est
    présent, même si un CSV parasite traîne (sinon il lèverait « Colonnes »)."""
    pytest.importorskip("pypowsybl")
    racine = _ecrire_serie(tmp_path)
    (racine / "parasite.csv").write_text("foo,bar\n1,2\n")
    tls = {tl.voltage_level_id: tl for tl in charger_timelines(racine)}
    assert "S1VL1" in tls


# ---------------------------------------------------------------------------
# Robustesse
# ---------------------------------------------------------------------------

def test_charger_timelines_aucun_fichier_message_clair(tmp_path):
    with pytest.raises(FileNotFoundError, match="Aucun instantané reconnu"):
        list(charger_timelines(tmp_path))


def test_instantane_illisible_ignore_ou_leve(tmp_path):
    pytest.importorskip("pypowsybl")
    jour = tmp_path / "2021" / "01" / "03"
    jour.mkdir(parents=True)
    (jour / "recollement-auto-20210103-0000-enrichi.xiidm.bz2").write_bytes(
        b"ceci n'est pas du bzip2")
    # Par défaut, l'erreur de lecture remonte.
    with pytest.raises(Exception):
        list(charger_timelines_xiidm(tmp_path))
    # En mode « ignorer », l'instantané fautif est sauté → aucune chronologie.
    assert list(charger_timelines_xiidm(tmp_path, sur_erreur="ignorer")) == []
