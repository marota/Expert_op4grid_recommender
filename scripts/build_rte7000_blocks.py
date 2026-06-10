#!/usr/bin/env python3
"""
scripts/build_rte7000_blocks.py
--------------------------------
Pipeline « dataset topologies historiques » (cf. docs/plan_dataset_rte7000.md,
phases 1-4) : depuis un historique d'états d'organes par poste, ce script

1. reconstitue les chronologies de topologies détaillées par poste ;
2. détecte les **blocs de transition** (topologie détaillée de départ →
   topologie détaillée cible) avec l'évolution observée (états transitoires +
   manœuvres ordonnées) ;
3. **tague le type d'intervention** de chaque bloc (consignation, scission,
   fusion, ré-aiguillage, sectionnement…) — structurel si une fixture du
   poste est fournie, sinon repli par nommage des organes ;
4. écrit le dataset : ``blocs.jsonl``, scénarios (format
   ``tests/manoeuvre/scenarios``), séquences observées (format
   ``tests/manoeuvre/sequences``) et ``stats.json``.

Sur le dataset Hugging Face (à télécharger au préalable, cf.
``manoeuvre/dataset/dgitt.py``) ::

    hf download OpenSynth/D-GITT-RTE7000-2021 --repo-type dataset \
        --local-dir data/dgitt_rte7000_2021
    python scripts/build_rte7000_blocks.py \
        --input data/dgitt_rte7000_2021 \
        --fixtures tests/manoeuvre/fixtures \
        --output out_rte7000 [--vl CARRIP3 --vl MORBRP6] [--min-stabilite 2]

Mode **démo** (sans le dataset ni le réseau) : des chronologies sont
reconstruites depuis les séquences réelles du dépôt
(``tests/manoeuvre/sequences/*.json`` rejouées sur les fixtures) et le
pipeline complet tourne dessus ::

    python scripts/build_rte7000_blocks.py --demo --output out_demo
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from expert_op4grid_recommender.manoeuvre.dataset import (   # noqa: E402
    Snapshot,
    TimelinePoste,
    ecrire_dataset,
    poste_from_fixture_json,
    stats_blocs,
    taguer_blocs,
)
from expert_op4grid_recommender.manoeuvre.dataset.extraction import (  # noqa: E402
    bloc_to_scenario,
)

REPO = pathlib.Path(__file__).resolve().parent.parent


def _charger_postes(fixtures_dir: pathlib.Path, vls: set[str]) -> dict:
    """Structures de postes disponibles (fixture JSON par VL, nommage
    point/espace → underscore toléré)."""
    postes = {}
    if not fixtures_dir or not fixtures_dir.exists():
        return postes
    for vl in vls:
        for cand in (fixtures_dir / f"{vl}.json",
                     fixtures_dir / f"{vl.replace('.', '_').replace(' ', '_')}.json"):
            if cand.exists() and cand.stem != "index":
                try:
                    postes[vl] = poste_from_fixture_json(cand, vl)
                except Exception as exc:   # fixture incompatible : tag par nommage
                    print(f"  ! structure {vl} illisible ({exc}) — repli nommage")
                break
    return postes


def _timelines_demo():
    """Chronologies de démonstration : les séquences réelles du dépôt rejouées
    depuis leur état de départ (3 snapshots stables, puis un snapshot par
    manœuvre, puis 3 snapshots stables sur la cible)."""
    seq_dir = REPO / "tests" / "manoeuvre" / "sequences"
    for path in sorted(seq_dir.glob("*.json")):
        data = json.loads(path.read_text())
        vl = data["voltage_level_id"]
        depart = {k: bool(v) for k, v in data["depart"].items()}
        manos = sorted(data.get("manoeuvres", []),
                       key=lambda m: m.get("ordre", 0))
        if not manos:
            continue
        snaps, etat, t = [], dict(depart), 0

        def push(e):
            nonlocal t
            snaps.append(Snapshot(timestamp=f"2021-01-01T{t:04d}", etats=dict(e)))
            t += 1

        for _ in range(3):
            push(etat)
        for m in manos:
            etat[m["switch_id"]] = (m["action"] == "OPEN")
            push(etat)
        for _ in range(2):
            push(etat)
        yield path.stem, TimelinePoste(vl, snaps)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=pathlib.Path,
                    help="Dossier du dataset téléchargé : instantanés XIIDM "
                         "(*.xiidm.bz2, format réel RTE 7000) ou tabulaire "
                         "parquet/CSV (format auto-détecté)")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--fixtures", type=pathlib.Path,
                    default=REPO / "tests" / "manoeuvre" / "fixtures",
                    help="Dossier des structures de postes (fixtures JSON) "
                         "pour le tagging structurel")
    ap.add_argument("--vl", action="append", default=None,
                    help="Limiter à ce(s) poste(s) (répétable)")
    ap.add_argument("--min-stabilite", type=int, default=2,
                    help="Nb min de snapshots consécutifs d'un état stable")
    ap.add_argument("--sous-echantillon", type=int, default=None,
                    help="Ne garder qu'un instantané XIIDM sur N (allègement "
                         "mémoire/CPU ; perd la résolution fine des séquences)")
    ap.add_argument("--seuil-durable", type=int, default=None,
                    help="Plateau cible ≥ N snapshots → tag "
                         "reconfiguration_durable")
    ap.add_argument("--demo", action="store_true",
                    help="Chronologies reconstruites depuis les séquences du "
                         "dépôt (aucune donnée externe requise)")
    args = ap.parse_args()

    if args.demo:
        timelines = [(nom, tl) for nom, tl in _timelines_demo()]
    else:
        if not args.input:
            ap.error("--input requis (ou --demo)")
        from expert_op4grid_recommender.manoeuvre.dataset.dgitt import (
            charger_timelines,
        )
        vl_filter = set(args.vl) if args.vl else None
        timelines = [(tl.voltage_level_id, tl)
                     for tl in charger_timelines(
                         args.input, vl_filter,
                         sous_echantillon=args.sous_echantillon)]

    vls = {tl.voltage_level_id for _, tl in timelines}
    postes = _charger_postes(args.fixtures, vls)
    print(f"{len(timelines)} chronologie(s), {len(postes)} structure(s) de "
          f"poste pour le tagging structurel")

    tous_blocs, toutes_osc = [], []
    for nom, tl in timelines:
        blocs, osc = tl.detecter_blocs(min_stabilite=args.min_stabilite)
        taguer_blocs(blocs, postes, seuil_durable=args.seuil_durable)
        tous_blocs.extend(blocs)
        toutes_osc.extend(osc)
        for b in blocs:
            print(f"  [{nom}] {b.resume()}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "blocs.jsonl").open("w") as f:
        for b in tous_blocs:
            f.write(json.dumps(bloc_to_scenario(
                b, postes.get(b.voltage_level_id)), ensure_ascii=False) + "\n")
    ecrire_dataset(tous_blocs, out, postes)
    stats = stats_blocs(tous_blocs)
    stats["oscillations_repliees"] = len(toutes_osc)
    (out / "stats.json").write_text(json.dumps(stats, indent=2,
                                               ensure_ascii=False))
    print(f"\n→ {stats['nb_blocs']} bloc(s) ({stats['nb_postes']} poste(s)), "
          f"{stats['blocs_avec_sequence_observee']} avec séquence observée, "
          f"{len(toutes_osc)} oscillation(s) repliée(s)")
    print(f"→ dataset écrit dans {out} (blocs.jsonl, scenarios/, sequences/, "
          f"stats.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
