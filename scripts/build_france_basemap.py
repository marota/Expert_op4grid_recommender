#!/usr/bin/env python3
"""
scripts/build_france_basemap.py
-------------------------------
Régénère le **fond de carte** committé
``expert_op4grid_recommender/manoeuvre/dataset/france_basemap.json`` (frontières
des départements France + pays voisins) **projeté dans le repère du plan de masse
RTE** (``grid_layout_rte.json``) — pour qu'il s'aligne sur les disques de la carte
« Explorer la journée » sans reprojection au runtime.

Méthode (le plan de masse RTE n'est pas une projection géographique standard
documentée — on la **calibre** empiriquement) :

1. centroïdes des **départements** (geojson, lon/lat) ;
2. ODRE ``postes-electriques-rte`` (``exports/json``) → ``code_poste`` → département ;
3. réseau RTE 7000 (instantané XIIDM) → ``substation_id`` = ``code_poste`` →
   coordonnée du plan de masse ; appariés au centroïde du département du poste ;
4. **ajustement affine** ``(lon, lat) → (x, y)`` du plan de masse (moindres carrés ;
   le résidu est dominé par l'étalement intra-département, pas l'erreur de projection
   — la plage lon/lat recouvrée recouvre bien la France) ;
5. transformation des frontières (départements + voisins) dans le repère du plan
   de masse + décimation, écriture du JSON.

Réseau sortant requis : ODRE + les geojson GitHub (peut être bloqué au build de
l'assistant — lancer ailleurs et committer le JSON produit).

Usage
-----
    python scripts/build_france_basemap.py --xiidm instantane.xiidm.bz2
    python scripts/build_france_basemap.py --date 2021-01-03
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys
import unicodedata
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEPTS_URL = ("https://raw.githubusercontent.com/gregoiredavid/france-geojson/"
             "master/departements-version-simplifiee.geojson")
WORLD_URL = ("https://raw.githubusercontent.com/johan/world.geo.json/master/"
             "countries.geo.json")
ODRE_JSON = ("https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
             "postes-electriques-rte/exports/json")
NEIGHBORS = ("BEL", "LUX", "DEU", "CHE", "ITA", "ESP", "AND", "GBR", "NLD")
_OUT = ("expert_op4grid_recommender/manoeuvre/dataset/france_basemap.json")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "eo4g-basemap/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return "".join(c for c in s if c.isalnum())


def _rings(geom: dict) -> list[list]:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [ring for poly in polys for ring in poly]


def _substations(xiidm, date, cache):
    from expert_op4grid_recommender.manoeuvre.dataset.dgitt import _charger_reseau
    from expert_op4grid_recommender.manoeuvre.dataset import source as src
    if xiidm:
        net = _charger_reseau(pathlib.Path(xiidm))
    elif date:
        net, _ = src.charger_situation(src.repo_pour_date(src.REPO_DEFAUT, date), date, cache)
    else:
        raise SystemExit("Préciser --xiidm ou --date.")
    vlt = net.get_voltage_levels(all_attributes=True)
    vlt = vlt[vlt["topology_kind"] == "NODE_BREAKER"]
    return dict(zip(vlt.index.astype(str), vlt["substation_id"].astype(str)))


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xiidm")
    ap.add_argument("--date")
    ap.add_argument("--cache-dir", default=".cache/dgitt")
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--step", type=int, default=2, help="décimation des anneaux.")
    args = ap.parse_args()

    from expert_op4grid_recommender.manoeuvre.dataset import geographie

    print("→ départements + monde (geojson)…")
    dj = json.loads(_get(DEPTS_URL))
    wj = json.loads(_get(WORLD_URL))
    cent = {}
    for f in dj["features"]:
        pts = [p for r in _rings(f["geometry"]) for p in r]
        cent[_norm(f["properties"]["nom"])] = (sum(p[0] for p in pts) / len(pts),
                                               sum(p[1] for p in pts) / len(pts))

    print("→ ODRE (code_poste → département)…")
    recs = json.loads(_get(ODRE_JSON))
    code2dept = {r.get("code_poste"): r.get("departement") for r in recs
                 if r.get("code_poste")}

    print("→ substations + plan de masse…")
    vl2sub = _substations(args.xiidm, args.date, args.cache_dir)
    layout = geographie.charger_layout()

    X, Y, LON, LAT, seen = [], [], [], [], set()
    for vl, sub in vl2sub.items():
        if vl in layout and sub not in seen and _norm(code2dept.get(sub, "")) in cent:
            seen.add(sub)
            X.append(layout[vl][0]); Y.append(layout[vl][1])
            lo, la = cent[_norm(code2dept[sub])]; LON.append(lo); LAT.append(la)
    n = len(X)
    print(f"  {n} paires de calibration")
    M = np.column_stack([LON, LAT, np.ones(n)])
    cx = np.linalg.lstsq(M, np.array(X), rcond=None)[0]
    cy = np.linalg.lstsq(M, np.array(Y), rcond=None)[0]

    def to_xy(lon, lat):
        return [round(cx[0] * lon + cx[1] * lat + cx[2], 1),
                round(cy[0] * lon + cy[1] * lat + cy[2], 1)]

    def conv(features, want=None):
        out = []
        for f in features:
            if want and f.get("id") not in want:
                continue
            for ring in _rings(f["geometry"]):
                if len(ring) < 4:
                    continue
                r = ring[::args.step] + [ring[-1]]
                out.append([to_xy(p[0], p[1]) for p in r])
        return out

    basemap = {"depts": conv(dj["features"]),
               "neighbors": conv(wj["features"], set(NEIGHBORS))}
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(basemap, separators=(",", ":")), encoding="utf-8")
    print(f"✓ {out} — {len(basemap['depts'])} anneaux départements, "
          f"{len(basemap['neighbors'])} anneaux voisins "
          f"({out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
