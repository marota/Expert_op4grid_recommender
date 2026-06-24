"""
tests/manoeuvre/test_dataset_source.py
--------------------------------------
Couche « source instantané par date » du dataset RTE 7000
(``manoeuvre/dataset/source.py``) : résolution date/heure → instantané,
téléchargement (reprise + md5) — **sans aucun appel réseau** (la frontière HTTP
est monkeypatchée). Branche l'IHM de manœuvre sur le dataset HuggingFace.
"""

from __future__ import annotations

import hashlib

import pytest

from expert_op4grid_recommender.manoeuvre.dataset import source


# --- prefixe_jour / _minutes -----------------------------------------------

def test_prefixe_jour_ok():
    assert source.prefixe_jour("2021-01-03") == "2021/01/03"
    assert source.prefixe_jour("  2022-12-31 ") == "2022/12/31"


@pytest.mark.parametrize("bad", ["notadate", "2021/01/03", "2021-1-3", ""])
def test_prefixe_jour_invalide(bad):
    with pytest.raises(ValueError):
        source.prefixe_jour(bad)


def test_minutes_parsing():
    assert source._minutes("12:00") == 720
    assert source._minutes("09h05") == 545
    assert source._minutes("0000") == 0
    assert source._minutes("bof") == 12 * 60   # repli midi


# --- choisir_instantane (le plus proche de l'heure) ------------------------

def test_choisir_instantane_proche_midi():
    insts = [{"ts": "00:05"}, {"ts": "11:55"}, {"ts": "12:10"}, {"ts": "23:55"}]
    assert source.choisir_instantane(insts, "12:00")["ts"] == "11:55"  # 5 min < 10 min


def test_choisir_instantane_heure_visee():
    insts = [{"ts": "08:00"}, {"ts": "18:30"}]
    assert source.choisir_instantane(insts, "18:00")["ts"] == "18:30"


def test_choisir_instantane_vide():
    assert source.choisir_instantane([]) is None


# --- lister_instantanes (parse les noms, trie, ignore l'inhorodatable) -----

def test_lister_instantanes(monkeypatch):
    paths = [
        "2021/01/03/recollement-auto-20210103-1200-enrichi.xiidm.bz2",
        "2021/01/03/recollement-auto-20210103-0005-enrichi.xiidm.bz2",
        "2021/01/03/sans-horodatage.xiidm.bz2",   # ignoré (ValueError)
    ]
    monkeypatch.setattr(source, "lister_fichiers",
                        lambda repo, prefix, token=None: paths)
    insts = source.lister_instantanes("repo", "2021-01-03")
    assert [d["ts"] for d in insts] == ["00:05", "12:00"]        # trié, junk écarté
    assert insts[0]["iso"] == "2021-01-03T00:05"
    assert insts[1]["path"].endswith("1200-enrichi.xiidm.bz2")


def test_lister_instantanes_date_invalide(monkeypatch):
    # prefixe_jour lève AVANT tout accès réseau.
    monkeypatch.setattr(source, "lister_fichiers",
                        lambda *a, **k: pytest.fail("ne doit pas lister"))
    with pytest.raises(ValueError):
        source.lister_instantanes("repo", "notadate")


# --- resoudre_et_telecharger -----------------------------------------------

def test_resoudre_et_telecharger(monkeypatch, tmp_path):
    monkeypatch.setattr(source, "lister_instantanes",
                        lambda repo, date, token=None: [
                            {"ts": "12:00", "iso": date + "T12:00", "path": "P12"},
                            {"ts": "00:00", "iso": date + "T00:00", "path": "P00"}])
    capt = {}

    def fake_dl(repo, hf_path, cache, token=None):
        capt["path"] = hf_path
        return tmp_path / hf_path

    monkeypatch.setattr(source, "telecharger_instantane", fake_dl)
    local, meta = source.resoudre_et_telecharger(
        "repo", "2021-01-03", tmp_path, heure="12:00")
    assert capt["path"] == "P12"                  # midi choisi
    assert meta["ts"] == "12:00" and meta["date"] == "2021-01-03"
    assert local == tmp_path / "P12"


def test_resoudre_aucun_instantane(monkeypatch, tmp_path):
    monkeypatch.setattr(source, "lister_instantanes",
                        lambda repo, date, token=None: [])
    with pytest.raises(FileNotFoundError):
        source.resoudre_et_telecharger("repo", "2021-01-03", tmp_path)


# --- telecharger_un (reprise / téléchargement / md5) -----------------------

def test_telecharger_un_reprise(monkeypatch, tmp_path):
    """Fichier déjà présent + md5 concordant => aucun téléchargement."""
    data = b"hello"
    dest = tmp_path / "2021/01/03/x.xiidm.bz2"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(data)
    monkeypatch.setattr(source, "_md5_attendu",
                        lambda repo, path, token=None: hashlib.md5(data).hexdigest())
    monkeypatch.setattr(source, "_http_get",
                        lambda *a, **k: pytest.fail("ne doit pas télécharger"))
    out = source.telecharger_un("repo", "2021/01/03/x.xiidm.bz2", tmp_path)
    assert out == dest


def test_telecharger_un_download(monkeypatch, tmp_path):
    data = b"world"
    monkeypatch.setattr(source, "_md5_attendu",
                        lambda repo, path, token=None: hashlib.md5(data).hexdigest())
    monkeypatch.setattr(source, "_http_get", lambda url, token=None, **k: data)
    out = source.telecharger_un("repo", "2021/01/03/y.xiidm.bz2", tmp_path)
    assert out.read_bytes() == data


def test_telecharger_un_md5_invalide(monkeypatch, tmp_path):
    monkeypatch.setattr(source, "_md5_attendu",
                        lambda repo, path, token=None: "0" * 32)
    monkeypatch.setattr(source, "_http_get", lambda url, token=None, **k: b"data")
    with pytest.raises(IOError):
        source.telecharger_un("repo", "2021/01/03/z.xiidm.bz2", tmp_path)
