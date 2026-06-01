#!/usr/bin/env python3
"""
scripts/render_carrip3_sld.py
-------------------------------
Rend les schémas unifilaires (SLD) du poste CARRIP3, **avant** et **après**
application de la séquence de manœuvres calculée par le module ``manoeuvre``
(cible 3 nœuds via ouverture de sectionnement de barre).

La **coloration est entièrement automatique** : c'est le module SLD de pypowsybl
(``topological_coloring=True``) qui colore barres et conducteurs par nœud
électrique, à partir des positions de switches. On n'applique **aucune** palette
maison : la couleur de base est celle du niveau de tension (≈ violet pour le
63 kV), déclinée en teintes distinctes par nœud lorsqu'il y en a plusieurs.

Usage
-----
    python scripts/render_carrip3_sld.py \
        --grid /chemin/vers/grid.xiidm \
        --out  docs/manoeuvre/sld

Sorties (dans --out) :
    CARRIP3_avant.svg / .png   (1 nœud : tout couplé, couleur 63 kV)
    CARRIP3_apres.svg / .png   (3 nœuds : barre 1 / 2.1 / 2.2)

Conversion SVG → PNG
--------------------
pypowsybl encode ses couleurs via des **variables CSS** (``var(--sld-vl-color)``).
``librsvg`` (rsvg-convert) ne les résout pas (rendu noir). On utilise donc un
moteur compatible variables CSS : **Chrome/Chromium headless**. Le SVG reste la
sortie de référence (couleurs natives pypowsybl) ; le PNG n'est qu'un aperçu.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess

import pypowsybl as pp
import pypowsybl.network as ppn

from expert_op4grid_recommender.manoeuvre.graph import build_vl_graph
from expert_op4grid_recommender.manoeuvre.topologie import (
    PosteTopologique,
    TopologieNodale,
)
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_manoeuvres_avec_sections,
    _set_switch,
)

VL = "CARRIP3"

# Classes d'aiguillage (cf. test_carrip3_3noeuds.py)
CLASSE_A = ["BERT L31CARRI", "CARRIL31VALES", "CARRI3T314", "BARR6L31CARRI",
            "BRENOL31CARRI", "CARRIL31U.MON", "CARRIY631", "CARRI3T313"]
CLASSE_B = ["CARRIY632", "CARRIL31RANTI", "CARRIL31V.PAU", "CARRIL31PERSA",
            "CARRIL32U.MON", "CARRIY633", "CARRI3T312"]
NODE_1 = {"BERT L31CARRI", "BARR6L31CARRI"}        # -> 2.1
NODE_2 = {"CARRIL31RANTI", "CARRIL31PERSA"}        # -> 2.2 (section isolée)
NODE_0 = set(CLASSE_A + CLASSE_B) - NODE_1 - NODE_2  # -> barre 1

# Emplacements usuels du binaire Chrome/Chromium
_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]


def _placement():
    return [
        (NODE_0, {"CARRIP3_1.1", "CARRIP3_1.2"}),
        (NODE_1, {"CARRIP3_2.1"}),
        (NODE_2, {"CARRIP3_2.2"}),
    ]


def _find_chrome() -> str | None:
    for c in _CHROME_CANDIDATES:
        if pathlib.Path(c).exists():
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def _svg_to_png(svg_path: pathlib.Path, scale: int = 2) -> None:
    """
    Convertit le SVG en PNG en **préservant les couleurs natives** pypowsybl.

    Utilise Chrome/Chromium headless (résout les variables CSS). À défaut,
    tente rsvg-convert en avertissant que les couleurs ne seront pas rendues.
    """
    svg = svg_path.read_text()
    m = re.search(r'<svg[^>]*\bwidth="([0-9.]+)"[^>]*\bheight="([0-9.]+)"', svg)
    w = int(float(m.group(1))) if m else 1080
    h = int(float(m.group(2))) if m else 557
    png = svg_path.with_suffix(".png")

    chrome = _find_chrome()
    if chrome:
        subprocess.run([
            chrome, "--headless", "--disable-gpu", "--no-sandbox",
            "--hide-scrollbars", "--default-background-color=FFFFFFFF",
            f"--force-device-scale-factor={scale}",
            f"--window-size={w},{h}",
            f"--screenshot={png}",
            svg_path.resolve().as_uri(),
        ], check=True, capture_output=True)
        print(f"  -> {png}  (couleurs natives pypowsybl, via Chrome headless)")
    elif shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-z", str(scale), "-b", "white",
                        str(svg_path), "-o", str(png)], check=True)
        print(f"  -> {png}  (ATTENTION: rsvg-convert ne résout pas les "
              "variables CSS → couleurs non rendues ; voir le .svg)")
    else:
        print("  (aucun convertisseur PNG ; le .svg conserve les couleurs natives)")


def _legend(poste: PosteTopologique, G_state) -> None:
    """Affiche les nœuds électriques recalculés par le module (information)."""
    topo = TopologieNodale.from_graph(G_state, poste.voltage_level_id)
    print(f"  Nœuds recalculés par le module : {topo.nb_noeuds}")
    for nom in sorted(topo.noeuds):
        ids = sorted(topo.noeuds[nom].equipment_ids)
        print(f"    {nom}: {', '.join(ids)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, help="Chemin du réseau .xiidm")
    ap.add_argument("--out", default="docs/manoeuvre/sld", help="Dossier de sortie")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = pp.network.load(args.grid)
    G0 = build_vl_graph(n, VL)
    poste = PosteTopologique.from_graph(G0, VL)
    res = determiner_manoeuvres_avec_sections(poste, _placement())
    print(f"Séquence : {res.nb_manoeuvres} manœuvres, vérifié={res.is_verified}")

    # pypowsybl colore automatiquement par nœud (topological_coloring).
    par = ppn.SldParameters(topological_coloring=True)

    print("\nAVANT :")
    _legend(poste, G0)
    avant = out / "CARRIP3_avant.svg"
    n.write_single_line_diagram_svg(VL, str(avant), parameters=par)
    _svg_to_png(avant)

    # Application des manœuvres au réseau pypowsybl ET au graphe module.
    G_apres = G0.copy()
    for m in res.manoeuvres:
        n.update_switches(id=m.switch_id, open=(m.action == "OPEN"))
        _set_switch(G_apres, m.switch_id, m.action == "OPEN")

    print("\nAPRÈS :")
    _legend(poste, G_apres)
    apres = out / "CARRIP3_apres.svg"
    n.write_single_line_diagram_svg(VL, str(apres), parameters=par)
    _svg_to_png(apres)


if __name__ == "__main__":
    main()
