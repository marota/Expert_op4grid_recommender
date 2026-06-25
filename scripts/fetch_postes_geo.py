#!/usr/bin/env python3
"""
scripts/fetch_postes_geo.py
---------------------------
Génère l'instantané committé des **coordonnées des postes** RTE
(``data/postes_rte_geo.json``) consommé par l'IHM « explorer la journée »
(carte des postes localisés).

À lancer **là où ODRE est joignable** (poste de dev, Space HuggingFace…) ;
l'environnement de build de l'assistant peut avoir ``odre.opendatasoft.com``
bloqué par la politique de sortie — dans ce cas, lancez ce script ailleurs et
committez le JSON produit.

Ce que fait le script :

1. télécharge le dataset ODRE ``postes-electriques-rte`` (coordonnées + tension) ;
2. charge un **instantané XIIDM** du dataset RTE 7000 pour relever les
   ``substation_id`` réels ;
3. **apparie** ODRE ↔ ``substation_id`` (mnémoniques normalisés) et **mesure**
   le taux d'appariement ;
4. écrit ``data/postes_rte_geo.json`` **indexé par ``substation_id``** — la
   résolution runtime devient un simple lookup (cf.
   ``manoeuvre/dataset/geographie.charger_snapshot``).

Usage
-----
    # à partir d'un instantané local (.xiidm[.bz2]) :
    python scripts/fetch_postes_geo.py --xiidm chemin/vers/instantane.xiidm.bz2

    # ou en téléchargeant un instantané du dataset par date :
    python scripts/fetch_postes_geo.py --date 2021-01-03

    # sortie / cache / clé API ODRE (optionnelle) personnalisables :
    python scripts/fetch_postes_geo.py --xiidm grid.xiidm \
        --out data/postes_rte_geo.json --odre-token $ODRE_TOKEN
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from expert_op4grid_recommender.manoeuvre.dataset import geographie
from expert_op4grid_recommender.manoeuvre.dataset import source as dataset_source


def _substation_ids(xiidm: str | None, date: str | None, cache_dir: str) -> list[str]:
    """``substation_id`` des VL NODE_BREAKER d'un instantané (fichier local ou
    téléchargé par date depuis HuggingFace)."""
    from expert_op4grid_recommender.manoeuvre.dataset.dgitt import _charger_reseau
    if xiidm:
        net = _charger_reseau(pathlib.Path(xiidm))
    elif date:
        repo = dataset_source.repo_pour_date(dataset_source.REPO_DEFAUT, date)
        net, _ = dataset_source.charger_situation(repo, date, cache_dir)
    else:
        raise SystemExit("Préciser --xiidm ou --date.")
    vlt = net.get_voltage_levels(all_attributes=True)
    if "topology_kind" in vlt.columns:
        vlt = vlt[vlt["topology_kind"] == "NODE_BREAKER"]
    subs = vlt["substation_id"].dropna().astype(str)
    return sorted(set(subs[subs != ""]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xiidm", help="Instantané .xiidm[.bz2] local (pour les substation_id).")
    ap.add_argument("--date", help="Date YYYY-MM-DD (télécharge un instantané du dataset).")
    ap.add_argument("--out", default=geographie.SNAPSHOT_DEFAUT,
                    help="Fichier de sortie (défaut : %(default)s).")
    ap.add_argument("--cache-dir", default=".cache/dgitt", help="Cache des instantanés.")
    ap.add_argument("--odre-cache-dir", default=".cache/odre",
                    help="Cache du dump ODRE brut.")
    ap.add_argument("--odre-token", default=None, help="Clé API ODRE (optionnelle).")
    ap.add_argument("--force", action="store_true", help="Forcer le re-téléchargement ODRE.")
    ap.add_argument("--no-prefix", action="store_true",
                    help="Désactiver le repli d'appariement par préfixe.")
    args = ap.parse_args()

    print("→ Téléchargement ODRE (postes-electriques-rte)…")
    records = geographie.fetch_odre_records(
        args.odre_cache_dir, token=args.odre_token, force=args.force)
    print(f"  {len(records)} enregistrements ODRE.")

    print("→ Lecture des substation_id de l'instantané…")
    sub_ids = _substation_ids(args.xiidm, args.date, args.cache_dir)
    print(f"  {len(sub_ids)} postes (substations NODE_BREAKER).")

    print("→ Appariement…")
    positions, stats = geographie.apparier_odre(
        records, sub_ids, prefix_fallback=not args.no_prefix)
    print(f"  appariés : {stats['n_apparies']}/{stats['n_substations']} "
          f"(taux {stats['taux']:.1%} ; exacts {stats['n_exact']}, "
          f"préfixe {stats['n_prefixe']}).")
    if stats["taux"] < 0.5:
        print("  ⚠ Taux faible : les codes ODRE ne recoupent peut-être pas les "
              "mnémoniques du dataset. Inspecter un enregistrement ODRE "
              "(clés de code) et ajuster geographie._CLES_CODE / l'appariement.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(positions, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"✓ Écrit {out} ({len(positions)} postes localisés).")


if __name__ == "__main__":
    main()
