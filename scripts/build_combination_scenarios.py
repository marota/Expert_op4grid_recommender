#!/usr/bin/env python3
"""
scripts/build_combination_scenarios.py
---------------------------------------
Scénarios **combinés** depuis le(s) catalogue(s) de topologies
(``topologies.jsonl`` produits par ``scripts/build_rte7000_blocks.py``) :
toute paire ordonnée de topologies détaillées **stables** d'un même poste
(départ ≠ cible) est un scénario réaliste — les deux états ont réellement été
occupés par le poste — potentiellement **jamais observé** et plus dur (diff
plus grand) que les blocs réels.

Plusieurs catalogues (journées/saisons/années différentes) peuvent être
concaténés : les combinaisons **inter-journées** (p. ex. topologie d'hiver →
topologie d'été d'un même poste) sont les plus longues. Les paires dont
l'ensemble d'organes diffère (structure changée entre les deux dates) sont
écartées.

Sortie : un ``blocs_combinaisons.jsonl`` au même format que ``blocs.jsonl``
(sans séquence observée), directement consommable par
``scripts/run_benchmark.py --blocs``.

Exemple ::

    python scripts/build_combination_scenarios.py \
        --catalogue out_20210103/topologies.jsonl \
        --catalogue out_20210715/topologies.jsonl \
        --max-par-poste 4 --min-organes 3 \
        --output out_combinaisons/blocs_combinaisons.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from expert_op4grid_recommender.manoeuvre.dataset import (   # noqa: E402
    TopologieRencontree,
    generer_combinaisons,
)


def _charger_catalogue(path: pathlib.Path) -> list[TopologieRencontree]:
    entrees = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        entrees.append(TopologieRencontree(
            voltage_level_id=d["voltage_level_id"],
            topologie_id=d["topologie_id"],
            etats={k: bool(v) for k, v in d["etats"].items()},
            premiere=d["premiere"], derniere=d["derniere"],
            nb_snapshots=d.get("nb_snapshots", 0),
            nb_episodes=d.get("nb_episodes", 0),
            stable=bool(d.get("stable", False)),
        ))
    return entrees


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", action="append", required=True,
                    type=pathlib.Path,
                    help="topologies.jsonl à fusionner (répétable)")
    ap.add_argument("--max-par-poste", type=int, default=6,
                    help="Plafond de combinaisons par poste, les plus grands "
                         "diffs d'abord (défaut : %(default)s)")
    ap.add_argument("--min-organes", type=int, default=2,
                    help="Diff minimal pour émettre une paire (défaut : "
                         "%(default)s)")
    ap.add_argument("--vl", action="append", default=None,
                    help="Limiter à ce(s) poste(s) (répétable)")
    ap.add_argument("--output", type=pathlib.Path, required=True,
                    help="Fichier blocs_combinaisons.jsonl à écrire")
    args = ap.parse_args()

    catalogue: list[TopologieRencontree] = []
    for p in args.catalogue:
        entrees = _charger_catalogue(p)
        catalogue.extend(entrees)
        print(f"{p} : {len(entrees)} topologie(s)")
    if args.vl:
        voulus = set(args.vl)
        catalogue = [e for e in catalogue if e.voltage_level_id in voulus]

    scenarios = generer_combinaisons(
        catalogue, max_par_poste=args.max_par_poste,
        min_organes=args.min_organes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for sc in scenarios:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    vls = {sc["voltage_level_id"] for sc in scenarios}
    diffs = [sc["meta"]["nb_organes_changes"] for sc in scenarios]
    print(f"\n→ {len(scenarios)} combinaison(s) sur {len(vls)} poste(s)"
          + (f", diff organe(s) min/médian/max = {min(diffs)}/"
             f"{sorted(diffs)[len(diffs) // 2]}/{max(diffs)}" if diffs else ""))
    print(f"→ écrit dans {args.output} (format blocs.jsonl — benchmark via "
          f"scripts/run_benchmark.py --blocs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
