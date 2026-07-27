"""
tests/manoeuvre/test_ihm_i18n.py
--------------------------------
Garde-fou de la **spécification systématique bilingue FR/EN** de l'IHM manœuvre
(docs/manoeuvre/ihm.md § 2bis) :

1. le **commutateur FR/EN** (haut à droite) et la **couche i18n** sont présents
   dans l'asset (dictionnaire exact ``I18N_EN``, motifs ``I18N_PATTERNS``,
   helpers ``t``/``tp``/``translateDom``/``setLang``/``applyLang``, persistance
   ``localStorage["manoeuvre_lang"]``, surcharge CSS ``html[lang="en"]``) ;
2. **couverture systématique** : chaque ``title="…"`` / ``placeholder="…"``
   du markup (hors ``<script>``) contenant du texte doit avoir une entrée
   ``I18N_EN`` — ajouter une info-bulle sans sa traduction casse la CI ;
3. chaque **nœud texte** statique du markup contenant des lettres doit avoir
   une entrée ``I18N_EN`` (ou figurer dans l'allowlist explicite) ;
4. un échantillon de **chaînes dynamiques critiques** (messages JS, libellés
   serveur/module traduits à l'affichage) est couvert (entrée exacte ou motif).

Test **pur texte** : aucune dépendance Flask / pypowsybl.
"""

from __future__ import annotations

import pathlib
import re
from html.parser import HTMLParser

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ASSET = _ROOT / "scripts" / "manoeuvre_ihm_assets" / "index.html"


def _asset() -> str:
    return _ASSET.read_text(encoding="utf-8")


def _markup(txt: str) -> str:
    """Partie markup de l'asset (avant le bloc <script>)."""
    return txt.split("<script>", 1)[0]


def _norm(s: str) -> str:
    """Normalisation blanche identique à celle de la couche i18n (trEn)."""
    return re.sub(r"\s+", " ", s).strip()


def _i18n_keys(txt: str) -> set[str]:
    """Clés (françaises) du dictionnaire exact I18N_EN de l'asset."""
    assert "const I18N_EN" in txt, "dictionnaire I18N_EN absent de l'asset"
    block = txt.split("const I18N_EN", 1)[1]
    block = block.split("};", 1)[0]
    keys = set()
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"\s*:', block):
        keys.add(m.group(1).replace('\\"', '"').replace("\\\\", "\\"))
    return keys


# ---------------------------------------------------------------------------
# 1. Présence de la couche i18n + commutateur haut-droite
# ---------------------------------------------------------------------------

REQUIRED_I18N_MARKERS = [
    # commutateur FR/EN (haut à droite, position fixe)
    'id="langSwitch"', 'id="langFr"', 'id="langEn"',
    "setLang('fr')", "setLang('en')",
    # couche i18n
    "const I18N_EN", "const I18N_PATTERNS",
    "function setLang", "function applyLang", "function translateDom",
    "function trEn(", "function tp(",
    # persistance du choix de langue
    "manoeuvre_lang",
    # surcharge CSS du texte injecté (« topologie cible atteinte »)
    'html[lang="en"]',
]


def test_i18n_layer_and_switcher_present():
    txt = _asset()
    missing = [m for m in REQUIRED_I18N_MARKERS if m not in txt]
    assert not missing, f"marqueurs i18n absents : {missing}"


def test_lang_switch_is_top_right_fixed():
    txt = _asset()
    m = re.search(r"#langSwitch\{([^}]*)\}", txt)
    assert m, "règle CSS #langSwitch absente"
    css = m.group(1)
    assert "position:fixed" in css.replace(" ", "")
    assert "right" in css and "top" in css, "le commutateur doit être en haut à droite"


# ---------------------------------------------------------------------------
# 2. Couverture systématique des attributs title= / placeholder= du markup
# ---------------------------------------------------------------------------

#: valeurs d'attributs délibérément non traduites (identiques FR/EN ou vides)
ATTR_ALLOWLIST = {
    "",  # title vide (rempli dynamiquement, ex. dsRepoInfo)
}

_HAS_LETTER = re.compile(r"[A-Za-zÀ-ÿœŒ]")


def test_every_title_and_placeholder_translated():
    txt = _asset()
    keys = _i18n_keys(txt)
    markup = _markup(txt)
    missing = []
    for attr in ("title", "placeholder"):
        for m in re.finditer(rf'{attr}="([^"]*)"', markup):
            val = _norm(m.group(1))
            if val in ATTR_ALLOWLIST or not _HAS_LETTER.search(val):
                continue
            if val not in keys:
                missing.append(f'{attr}="{val}"')
    assert not missing, (
        "Attributs sans entrée I18N_EN (spécification bilingue systématique — "
        f"docs/manoeuvre/ihm.md § 2bis) : {missing}"
    )


# ---------------------------------------------------------------------------
# 3. Couverture systématique des nœuds texte statiques du markup
# ---------------------------------------------------------------------------

#: nœuds texte délibérément non traduits (symboles, marques, identiques FR/EN)
TEXT_ALLOWLIST = {
    "FR", "EN",       # boutons du commutateur de langue lui-même
    "RTE 7000 ⓘ",     # marque du dataset (bulle remplie dynamiquement)
}


class _TextCollector(HTMLParser):
    """Collecte les nœuds texte du markup, hors <script>/<style>."""

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.texts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.texts.append(data)


def test_every_static_text_node_translated():
    txt = _asset()
    keys = _i18n_keys(txt)
    parser = _TextCollector()
    parser.feed(_markup(txt))
    missing = []
    for raw in parser.texts:
        val = _norm(raw)
        if not val or not _HAS_LETTER.search(val):
            continue
        if val in TEXT_ALLOWLIST or val in ATTR_ALLOWLIST:
            continue
        if val not in keys:
            missing.append(val)
    assert not missing, (
        "Nœuds texte du markup sans entrée I18N_EN (spécification bilingue "
        f"systématique — docs/manoeuvre/ihm.md § 2bis) : {missing}"
    )


# ---------------------------------------------------------------------------
# 4. Échantillon de chaînes dynamiques critiques (JS + serveur/module)
# ---------------------------------------------------------------------------

#: chaînes construites en JS ou reçues du serveur, à couvrir en exact
CRITICAL_EXACT = [
    "Choisissez un poste…",
    "Cible (éditable)",
    "Chargement du réseau…",
    "État de départ",                      # seq_labels[0] côté serveur
    "manœuvre manuelle (expert)",          # raison des manœuvres manuelles
    "DÉTAILLÉE VÉRIFIÉE",
    "NON VÉRIFIÉE",
    "Calcul en cours…",
    "ouverture couplage de barres",        # raison module (constante)
]

#: fragments de motifs attendus dans I18N_PATTERNS (chaînes paramétrées)
CRITICAL_PATTERN_HINTS = [
    "ré-aiguillage",          # raison "ré-aiguillage '<eq>' vers <barre>"
    "quitte",                 # raison "'<eq>' quitte <barre>"
    "mise hors tension",      # raison boucle longue
    "remise sous tension",
    "sous charge",            # messages de la règle du sectionneur
    "Topologie nodale atteinte",
    "Topologie détaillée cible atteinte",
    # diagnostics nominatifs v0.3.2 (message de /api/nodale_to_detaillee)
    "non regroupés comme visé",
    "impossible\\(s\\) à isoler",
]


def test_critical_dynamic_strings_covered():
    txt = _asset()
    keys = _i18n_keys(txt)
    missing = [s for s in CRITICAL_EXACT if s not in keys]
    assert not missing, f"chaînes dynamiques critiques sans entrée exacte : {missing}"
    assert "const I18N_PATTERNS" in txt
    patterns_block = txt.split("const I18N_PATTERNS", 1)[1].split("];", 1)[0]
    absent = [h for h in CRITICAL_PATTERN_HINTS if h not in patterns_block]
    assert not absent, f"motifs de traduction absents de I18N_PATTERNS : {absent}"
