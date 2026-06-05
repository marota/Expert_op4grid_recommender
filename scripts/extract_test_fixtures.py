#!/usr/bin/env python3
"""
extract_test_fixtures.py
========================
Extrait les topologies node/breaker des postes ciblés depuis un fichier XIIDM,
et les sérialise en fixtures JSON pour les tests unitaires du module Manoeuvre.

Usage (local, Python avec pypowsybl installé) :
    python scripts/extract_test_fixtures.py \
        --xiidm /path/to/grid.xiidm \
        --output tests/manoeuvre/fixtures/

Les fixtures produites ne contiennent que la structure topologique abstraite
(nœuds, switches, busbar sections, équipements, internal connections) sans
aucune donnée de transit, seuils, ou paramètres sensibles.

Postes ciblés (issus de la documentation algo Apogée) :
- Postes standards  : CARRIP3, CARRIP6, CZTRYP6, COMPIP3, BXTO5, CZBEVP3,
                      PALUNP3, NOVIOP3, SSAVOP3, VIELMP6
- Départs multiples : CORNIP3, CNIEP6, GUARBP6, MORBRP6, RAN.PP6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pypowsybl as pp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Postes ciblés par la documentation algo Apogée
# ────────────────────────────────────────────────────────────────────
POSTES_STANDARDS = [
    "CARRIP3", "CARRIP6", "CZTRYP6", "COMPIP3", "BXTO5",
    "CZBEVP3", "PALUNP3", "NOVIOP3", "SSAVOP3", "VIELMP6",
]
POSTES_DEPARTS_MULTIPLES = [
    "CORNIP3", "CNIEP6", "GUARBP6", "MORBRP6", "RAN.PP6",
]
ALL_POSTES = POSTES_STANDARDS + POSTES_DEPARTS_MULTIPLES


def find_voltage_levels_for_poste(
    network: pp.network.Network,
    poste_prefix: str,
) -> list[str]:
    """
    Trouve les voltage levels dont l'ID de substation ou l'ID du VL
    commence par le nom de poste donné.

    Convention RTE : un poste "CARRIP3" peut correspondre aux VL
    "CARRIP3", "CARRIP3_0", "CARRI_P3", etc.
    On tente d'abord une correspondance exacte de substation, puis par préfixe.
    """
    vls = network.get_voltage_levels()
    subs = network.get_substations()

    # Stratégie 1 : chercher une substation avec ce nom
    matching_subs = [s for s in subs.index if s.startswith(poste_prefix)]

    matching_vls = []
    if matching_subs:
        for sub_id in matching_subs:
            matching_vls.extend(
                vls[vls["substation_id"] == sub_id].index.tolist()
            )

    # Stratégie 2 : chercher directement par préfixe de VL
    if not matching_vls:
        matching_vls = [
            vl_id for vl_id in vls.index
            if vl_id.startswith(poste_prefix)
        ]

    # Stratégie 3 : recherche partielle (le nom contient le poste)
    if not matching_vls:
        matching_vls = [
            vl_id for vl_id in vls.index
            if poste_prefix in vl_id
        ]

    return matching_vls


def extract_vl_topology(
    network: pp.network.Network,
    vl_id: str,
) -> dict[str, Any] | None:
    """
    Extrait la topologie node/breaker d'un voltage level en dict sérialisable.

    Retourne None si le VL n'est pas en topologie NODE_BREAKER.
    """
    # Certaines versions de pypowsybl ne retournent pas topology_kind par défaut.
    # On essaie all_attributes=True, puis on détecte via get_busbar_sections().
    vls = network.get_voltage_levels(all_attributes=True)
    if vl_id not in vls.index:
        logger.warning("VL '%s' introuvable.", vl_id)
        return None

    # Détection du topology_kind
    if "topology_kind" in vls.columns:
        topo_kind = vls.loc[vl_id, "topology_kind"]
    else:
        # Fallback : si le VL a des busbar_sections avec colonne 'node', c'est NODE_BREAKER
        bbs = network.get_busbar_sections(all_attributes=True)
        vl_bbs = bbs[bbs["voltage_level_id"] == vl_id] if "voltage_level_id" in bbs.columns else bbs.iloc[0:0]
        if not vl_bbs.empty and "node" in vl_bbs.columns:
            topo_kind = "NODE_BREAKER"
        else:
            topo_kind = "BUS_BREAKER"
        logger.debug(
            "VL '%s' : colonne topology_kind absente, détecté '%s' via BBS.",
            vl_id, topo_kind,
        )

    if topo_kind != "NODE_BREAKER":
        logger.info("VL '%s' en topologie '%s' (ignoré).", vl_id, topo_kind)
        return None

    row = vls.loc[vl_id]
    result: dict[str, Any] = {
        "voltage_level_id": vl_id,
        "substation_id": str(row.get("substation_id", "")),
        "nominal_v": float(row.get("nominal_v", row.get("nominalV", 0.0))),
        "topology_kind": topo_kind,
    }

    # ── Switches ──────────────────────────────────────────────────────
    switches = network.get_switches(all_attributes=True)
    vl_sw = switches[switches["voltage_level_id"] == vl_id]
    result["switches"] = [
        {
            "id": str(sw_id),
            "kind": str(row["kind"]),
            "node1": int(row["node1"]),
            "node2": int(row["node2"]),
            "open": bool(row["open"]),
        }
        for sw_id, row in vl_sw.iterrows()
    ]

    # ── Busbar sections ───────────────────────────────────────────────
    bbs = network.get_busbar_sections(all_attributes=True)
    vl_bbs = bbs[bbs["voltage_level_id"] == vl_id]
    result["busbar_sections"] = [
        {
            "id": str(bbs_id),
            "node": int(row["node"]),
        }
        for bbs_id, row in vl_bbs.iterrows()
    ]

    # ── Internal connections ──────────────────────────────────────────
    try:
        nbt = network.get_node_breaker_topology(vl_id)
        ic = nbt.internal_connections
        if ic is not None and not ic.empty:
            result["internal_connections"] = [
                {"node1": int(row["node1"]), "node2": int(row["node2"])}
                for _, row in ic.iterrows()
            ]
        else:
            result["internal_connections"] = []
    except Exception as exc:
        logger.debug("Pas d'internal connections pour '%s': %s", vl_id, exc)
        result["internal_connections"] = []

    # ── Équipements connectés ─────────────────────────────────────────
    equipment = []

    # Injections (1 seul nœud de connexion)
    # ⚠  all_attributes=True est OBLIGATOIRE pour obtenir la colonne "node"
    #    dans les réseaux en topologie NODE_BREAKER.
    for getter, eq_type, vl_col, node_col in [
        (network.get_loads,                   "LOAD",                   "voltage_level_id", "node"),
        (network.get_generators,              "GENERATOR",              "voltage_level_id", "node"),
        (network.get_shunt_compensators,      "SHUNT_COMPENSATOR",      "voltage_level_id", "node"),
        (network.get_static_var_compensators, "STATIC_VAR_COMPENSATOR", "voltage_level_id", "node"),
        (network.get_batteries,               "BATTERY",                "voltage_level_id", "node"),
        (network.get_dangling_lines,          "DANGLING_LINE",          "voltage_level_id", "node"),
    ]:
        try:
            df = getter(all_attributes=True)
            if vl_col not in df.columns or node_col not in df.columns:
                logger.debug(
                    "VL '%s' — %s : colonnes manquantes (%s ou %s). "
                    "Équipements de ce type non extraits.",
                    vl_id, eq_type, vl_col, node_col,
                )
                continue
            vl_df = df[df[vl_col] == vl_id]
            for eq_id, row in vl_df.iterrows():
                equipment.append({
                    "id": str(eq_id),
                    "type": eq_type,
                    "node": int(row[node_col]),
                })
        except Exception as exc:
            logger.debug("Erreur extraction %s pour VL '%s' : %s", eq_type, vl_id, exc)

    # Branches (2 nœuds : côté 1 et côté 2)
    # ⚠  all_attributes=True obligatoire pour node1/node2 en NODE_BREAKER.
    for getter, eq_type_s1, eq_type_s2, vl_col1, node_col1, vl_col2, node_col2 in [
        (network.get_lines,
         "LINE_SIDE1", "LINE_SIDE2",
         "voltage_level1_id", "node1", "voltage_level2_id", "node2"),
        (network.get_2_windings_transformers,
         "TRANSFORMER_SIDE1", "TRANSFORMER_SIDE2",
         "voltage_level1_id", "node1", "voltage_level2_id", "node2"),
    ]:
        try:
            df = getter(all_attributes=True)
        except Exception as exc:
            logger.debug("Erreur extraction branches %s/%s : %s", eq_type_s1, eq_type_s2, exc)
            continue

        # Côté 1
        if vl_col1 in df.columns and node_col1 in df.columns:
            side1 = df[df[vl_col1] == vl_id]
            for eq_id, row in side1.iterrows():
                equipment.append({
                    "id": str(eq_id),
                    "type": eq_type_s1,
                    "node": int(row[node_col1]),
                })

        # Côté 2
        if vl_col2 in df.columns and node_col2 in df.columns:
            side2 = df[df[vl_col2] == vl_id]
            for eq_id, row in side2.iterrows():
                equipment.append({
                    "id": str(eq_id),
                    "type": eq_type_s2,
                    "node": int(row[node_col2]),
                })

    result["equipment"] = equipment

    # ── Statistiques ──────────────────────────────────────────────────
    result["stats"] = {
        "nb_switches": len(result["switches"]),
        "nb_busbar_sections": len(result["busbar_sections"]),
        "nb_internal_connections": len(result["internal_connections"]),
        "nb_equipment": len(equipment),
        "nb_breakers": sum(1 for s in result["switches"] if s["kind"] == "BREAKER"),
        "nb_disconnectors": sum(1 for s in result["switches"] if s["kind"] == "DISCONNECTOR"),
    }

    return result


def extract_all_postes(
    network: pp.network.Network,
    poste_names: list[str],
) -> dict[str, dict]:
    """
    Extrait les topologies de tous les postes ciblés.

    Retourne un dict { poste_name: { vl_id: topology_dict, ... }, ... }
    """
    results: dict[str, dict] = {}

    for poste in poste_names:
        vl_ids = find_voltage_levels_for_poste(network, poste)
        if not vl_ids:
            logger.warning("Poste '%s' : aucun voltage level trouvé.", poste)
            results[poste] = {}
            continue

        logger.info("Poste '%s' : %d VL trouvé(s) : %s", poste, len(vl_ids), vl_ids)
        poste_topos = {}
        for vl_id in vl_ids:
            topo = extract_vl_topology(network, vl_id)
            if topo is not None:
                poste_topos[vl_id] = topo

        results[poste] = poste_topos

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extrait les fixtures de topologie node/breaker depuis un fichier XIIDM."
    )
    parser.add_argument(
        "--xiidm", required=True,
        help="Chemin vers le fichier réseau .xiidm",
    )
    parser.add_argument(
        "--output", default="tests/manoeuvre/fixtures",
        help="Répertoire de sortie pour les fixtures JSON (default: tests/manoeuvre/fixtures/)",
    )
    parser.add_argument(
        "--postes", nargs="*", default=None,
        help="Liste de postes à extraire (default: tous les postes ciblés)",
    )
    args = parser.parse_args()

    xiidm_path = Path(args.xiidm)
    if not xiidm_path.exists():
        logger.error("Fichier introuvable : %s", xiidm_path)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le réseau
    logger.info("Chargement du réseau depuis '%s'...", xiidm_path)
    network = pp.network.load(str(xiidm_path))
    logger.info(
        "Réseau chargé : %d substations, %d voltage levels",
        len(network.get_substations()),
        len(network.get_voltage_levels()),
    )

    # Postes à extraire
    postes = args.postes or ALL_POSTES

    # Extraction
    all_results = extract_all_postes(network, postes)

    # Sérialisation
    for poste_name, vl_topos in all_results.items():
        if not vl_topos:
            continue

        for vl_id, topo_dict in vl_topos.items():
            safe_name = vl_id.replace(".", "_").replace("/", "_")
            fixture_path = output_dir / f"{safe_name}.json"
            with open(fixture_path, "w", encoding="utf-8") as f:
                json.dump(topo_dict, f, indent=2, ensure_ascii=False)
            logger.info(
                "Fixture '%s' : %d switches, %d BBS, %d éq., %d IC",
                fixture_path.name,
                topo_dict["stats"]["nb_switches"],
                topo_dict["stats"]["nb_busbar_sections"],
                topo_dict["stats"]["nb_equipment"],
                topo_dict["stats"]["nb_internal_connections"],
            )

    # Index global (utile pour les tests paramétrés)
    index = {
        "source": str(xiidm_path.name),
        "postes_standards": POSTES_STANDARDS,
        "postes_departs_multiples": POSTES_DEPARTS_MULTIPLES,
        "extracted": {
            poste: list(vl_topos.keys())
            for poste, vl_topos in all_results.items()
            if vl_topos
        },
        "not_found": [
            poste for poste, vl_topos in all_results.items()
            if not vl_topos
        ],
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logger.info("Index écrit : %s", index_path)

    # Résumé
    found = sum(1 for v in all_results.values() if v)
    not_found = sum(1 for v in all_results.values() if not v)
    total_vl = sum(len(v) for v in all_results.values())
    logger.info(
        "Résumé : %d/%d postes trouvés, %d voltage levels extraits, "
        "%d postes non trouvés : %s",
        found, len(postes), total_vl, not_found,
        [p for p, v in all_results.items() if not v],
    )


if __name__ == "__main__":
    main()
