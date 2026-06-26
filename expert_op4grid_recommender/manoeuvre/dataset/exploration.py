"""
manoeuvre/dataset/exploration.py — Exploration de l'**intérêt d'une journée**.

Pour une date donnée, on charge **trois situations** du réseau France (par
défaut **minuit**, **midi** et **23 h**) et on estime, **par poste**, les
**changements d'organes de coupure (OC)** observés au cours de la journée :
le **nombre d'OC dont l'état diffère** entre les trois instantanés, **ventilé
par type d'OC** (``BREAKER`` = disjoncteur, ``DISCONNECTOR`` = sectionneur).

S'y ajoutent les **re-groupements de nœuds** (scissions / fusions) : le nombre
**minimal d'ouvrages** ayant été séparés dans un nouveau nœud ou ayant rejoint un
nœud au cours de la journée (``extraire_structure_topo`` +
``changements_nodaux_par_vl``) — ce sont aussi des configurations intéressantes,
comptabilisées dans le total.

Les postes les plus « actifs » (le plus de changements) sont mis en évidence :
ce sont les journées/postes intéressants à inspecter dans l'IHM de manœuvre.

Découpage :

- **extraction** (``extraire_etats_kinds``, ``structure_reseau``) : dépend de
  pypowsybl — lit un réseau chargé ;
- **cœur d'agrégation** (``changements_par_vl``, ``agreger_par_poste``,
  ``classer_postes``) : **Python pur** (dicts/listes), testable sans pypowsybl.

Un « poste » géographique = une **substation** pypowsybl (regroupe les voltage
levels de ses différents niveaux de tension). L'IHM affiche un disque par
substation ; le détail par voltage level reste disponible (un VL = un « poste »
au sens de la vue topologique historique).
"""
from __future__ import annotations

from typing import Optional

#: heures par défaut visées pour caractériser une journée (instantané le plus
#: proche choisi dans le dataset, pas de 5 min).
HEURES_DEFAUT: tuple[str, ...] = ("00:00", "12:00", "23:00")

#: types d'OC ventilés (alignés sur ``get_switches().kind`` de pypowsybl) :
#: ``BREAKER`` = disjoncteur (DJ), ``DISCONNECTOR`` = sectionneur (SA/SL/SS),
#: ``LOAD_BREAK_SWITCH`` = interrupteur.
TYPES_OC: tuple[str, ...] = ("BREAKER", "DISCONNECTOR", "LOAD_BREAK_SWITCH")


# ===========================================================================
# Extraction (pypowsybl)
# ===========================================================================

def extraire_etats_kinds(
    net, vl_filter: Optional[set[str]] = None,
) -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    """``(etats_par_vl, kinds)`` d'un réseau chargé.

    - ``etats_par_vl`` : ``{voltage_level_id: {switch_id: ouvert ?}}`` ;
    - ``kinds``        : ``{switch_id: 'BREAKER' | 'DISCONNECTOR' | …}``.

    Une seule passe ``get_switches(all_attributes=True)`` (colonnes
    ``voltage_level_id``, ``open``, ``kind``)."""
    sw = net.get_switches(all_attributes=True)
    for col in ("voltage_level_id", "open", "kind"):
        if col not in sw.columns:
            raise ValueError(
                f"Colonne '{col}' absente de get_switches() — version de "
                f"pypowsybl ? Colonnes : {sorted(map(str, sw.columns))}")
    if vl_filter:
        sw = sw[sw["voltage_level_id"].isin(vl_filter)]
    etats: dict[str, dict[str, bool]] = {}
    kinds: dict[str, str] = {}
    vls = sw["voltage_level_id"].tolist()
    opens = sw["open"].tolist()
    knds = sw["kind"].tolist()
    ids = sw.index.tolist()
    for sid, vl, op, kd in zip(ids, vls, opens, knds):
        sid = str(sid)
        etats.setdefault(str(vl), {})[sid] = bool(op)
        kinds[sid] = str(kd)
    return etats, kinds


def structure_reseau(net) -> tuple[dict[str, dict], dict[str, str]]:
    """Structure (invariante dans la journée) du réseau :

    - ``vl_meta`` : ``{vl: {'substation', 'nominal_v', 'name'}}`` (VL
      NODE_BREAKER uniquement — ceux inspectables en topologie détaillée) ;
    - ``sub_name`` : ``{substation_id: nom_affiché}``.
    """
    vlt = net.get_voltage_levels(all_attributes=True)
    if "topology_kind" in vlt.columns:
        vlt = vlt[vlt["topology_kind"] == "NODE_BREAKER"]
    vl_meta: dict[str, dict] = {}
    subs = vlt["substation_id"].tolist() if "substation_id" in vlt else []
    nvs = vlt["nominal_v"].tolist() if "nominal_v" in vlt else []
    names = vlt["name"].tolist() if "name" in vlt else []
    ids = vlt.index.tolist()
    for i, vl in enumerate(ids):
        vl_meta[str(vl)] = {
            "substation": str(subs[i]) if i < len(subs) and subs[i] else "",
            "nominal_v": float(nvs[i]) if i < len(nvs) and nvs[i] == nvs[i] else 0.0,
            "name": str(names[i]) if i < len(names) and names[i] else str(vl),
        }
    sub_name: dict[str, str] = {}
    try:
        st = net.get_substations(all_attributes=True)
        snames = st["name"].tolist() if "name" in st else []
        for i, sid in enumerate(st.index.tolist()):
            nm = str(snames[i]) if i < len(snames) and snames[i] else str(sid)
            sub_name[str(sid)] = nm
    except Exception:
        pass
    return vl_meta, sub_name


def extraire_structure_topo(net) -> dict[str, dict]:
    """Structure topologique **invariante** (par VL) servant à quantifier les
    **re-groupements de nœuds** (scissions / fusions) au cours de la journée :

        ``{vl: {'edges': {switch_id: (node1, node2)}, 'poids': {node: n_ouvrages}}}``

    - ``edges`` : arêtes du graphe nœud-disjoncteur (un switch relie ``node1`` et
      ``node2``) — seul l'état *ouvert/fermé* varie dans la journée, pas la
      structure ;
    - ``poids`` : nombre d'**ouvrages** (connectables : jeux de barres + départs)
      raccordés à chaque nœud — pour pondérer « combien d'ouvrages » bougent.

    Lue **une fois** sur le réseau de référence (les nœuds n'existent que pour les
    VL NODE_BREAKER ; les autres getters renvoient ``NaN`` → ignorés). pypowsybl."""
    struct: dict[str, dict] = {}

    def _vl(vl) -> dict:
        return struct.setdefault(str(vl),
                                 {"edges": {}, "poids": {}, "barres": set()})

    def _ajouter_ouvrage(vl, node, est_barre: bool = False) -> None:
        try:
            nd = int(node)
        except (TypeError, ValueError):
            return
        if nd != nd:  # NaN
            return
        s = _vl(vl)
        s["poids"][nd] = s["poids"].get(nd, 0) + 1
        if est_barre:
            s["barres"].add(nd)   # ancres des nœuds électriques (jeux de barres)

    # arêtes node1–node2 par VL (mêmes colonnes que extraire_etats_kinds).
    sw = net.get_switches(all_attributes=True)
    if {"voltage_level_id", "node1", "node2"} <= set(sw.columns):
        for sid, vl, n1, n2 in zip(sw.index.tolist(),
                                   sw["voltage_level_id"].tolist(),
                                   sw["node1"].tolist(), sw["node2"].tolist()):
            try:
                _vl(vl)["edges"][str(sid)] = (int(n1), int(n2))
            except (TypeError, ValueError):
                pass

    # connectables mono-VL (un (voltage_level_id, node)).
    mono = ("get_busbar_sections", "get_loads", "get_generators", "get_batteries",
            "get_shunt_compensators", "get_static_var_compensators",
            "get_dangling_lines", "get_lcc_converter_stations",
            "get_vsc_converter_stations")
    for getter in mono:
        try:
            df = getattr(net, getter)(all_attributes=True)
        except Exception:
            continue
        if "voltage_level_id" not in df.columns or "node" not in df.columns:
            continue
        est_barre = (getter == "get_busbar_sections")
        for vl, nd in zip(df["voltage_level_id"].tolist(), df["node"].tolist()):
            _ajouter_ouvrage(vl, nd, est_barre)

    # connectables multi-VL : lignes et TD (chaque extrémité = un ouvrage côté VL).
    multi = {"get_lines": (1, 2), "get_2_windings_transformers": (1, 2),
             "get_3_windings_transformers": (1, 2, 3)}
    for getter, cotes in multi.items():
        try:
            df = getattr(net, getter)(all_attributes=True)
        except Exception:
            continue
        for k in cotes:
            vlc, ndc = f"voltage_level{k}_id", f"node{k}"
            if vlc not in df.columns or ndc not in df.columns:
                continue
            for vl, nd in zip(df[vlc].tolist(), df[ndc].tolist()):
                _ajouter_ouvrage(vl, nd)
    return struct


def extraire_connexions(net, vl_meta: dict[str, dict]) -> list[dict]:
    """Connexions **inter-postes** (lignes électriques) pour les tracer sur la
    carte : ``[{'s1', 's2', 'nv'}, …]`` (substations reliées + tension nominale du
    palier), **dédupliquées** par couple de postes + tension. Les transformateurs
    (intra-poste) sont ignorés ; on ne garde que les lignes reliant **deux postes
    distincts**. ``vl_meta`` = sortie de ``structure_reseau`` (substation +
    nominal_v par VL). pypowsybl ; best-effort (réseau quelconque : local ou
    dataset)."""
    out: dict[tuple, dict] = {}

    def _ajouter(vl1, vl2) -> None:
        m1, m2 = vl_meta.get(str(vl1)), vl_meta.get(str(vl2))
        if not m1 or not m2:
            return
        s1, s2 = m1.get("substation"), m2.get("substation")
        if not s1 or not s2 or s1 == s2:
            return
        nv = m1.get("nominal_v") or m2.get("nominal_v") or 0.0
        key = (min(s1, s2), max(s1, s2), round(float(nv)))
        out.setdefault(key, {"s1": s1, "s2": s2, "nv": float(nv)})

    # Lignes AC (+ liaisons HVDC) : chaque extrémité porte un voltage_level_id.
    for getter in ("get_lines", "get_tie_lines", "get_hvdc_lines"):
        try:
            df = getattr(net, getter)(all_attributes=True)
        except Exception:
            continue
        cols = set(df.columns)
        if not {"voltage_level1_id", "voltage_level2_id"} <= cols:
            continue
        for vl1, vl2 in zip(df["voltage_level1_id"].tolist(),
                            df["voltage_level2_id"].tolist()):
            _ajouter(vl1, vl2)
    return list(out.values())


# ===========================================================================
# Cœur d'agrégation (Python pur — testable sans pypowsybl)
# ===========================================================================

def _a_change(etats: list[Optional[bool]]) -> bool:
    """Un organe a-t-il changé d'état dans la journée ? (≥ 2 valeurs distinctes
    parmi les états **renseignés** — l'absence d'un instantané = pas d'info,
    pas un changement)."""
    vus = {e for e in etats if e is not None}
    return len(vus) > 1


def changements_par_vl(
    situations: list[dict[str, dict[str, bool]]],
    kinds: dict[str, str],
) -> dict[str, dict]:
    """Changements d'OC **par voltage level** sur la journée.

    ``situations`` : un ``{vl: {switch_id: ouvert ?}}`` par instantané (ordre
    chronologique). Pour chaque VL, on compte les organes dont l'état n'est pas
    constant sur la journée, **ventilés par type d'OC**.

    Retourne ``{vl: {'total': int, 'BREAKER': int, 'DISCONNECTOR': int,
    'n_oc': int, 'changed': [switch_id, …]}}``. Fonction pure."""
    vls: list[str] = []
    vus: set[str] = set()
    for situ in situations:
        for vl in situ:
            if vl not in vus:
                vus.add(vl)
                vls.append(vl)

    out: dict[str, dict] = {}
    for vl in vls:
        sw_ids: list[str] = []
        seen: set[str] = set()
        for situ in situations:
            for sid in situ.get(vl, {}):
                if sid not in seen:
                    seen.add(sid)
                    sw_ids.append(sid)
        par_type: dict[str, int] = {t: 0 for t in TYPES_OC}
        changed: list[str] = []
        for sid in sw_ids:
            etats = [situ.get(vl, {}).get(sid) for situ in situations]
            if _a_change(etats):
                changed.append(sid)
                kd = kinds.get(sid, "AUTRE")
                par_type[kd] = par_type.get(kd, 0) + 1
        out[vl] = {
            "total": len(changed),
            "n_oc": len(sw_ids),
            "changed": changed,
            **{t: par_type.get(t, 0) for t in TYPES_OC},
            **{k: v for k, v in par_type.items() if k not in TYPES_OC},
        }
    return out


# --- Re-groupements de nœuds (scissions / fusions) — Python pur --------------

def partition_ouvrages(
    etats_vl: dict[str, bool],
    edges: dict[str, tuple[int, int]],
    poids: dict[int, int],
    barres: Optional[set] = None,
) -> dict[int, int]:
    """Partition des **nœuds porteurs d'ouvrage** (clés de ``poids``) en **nœuds
    électriques** : ``{node: id_composante}``. Deux nœuds sont dans la même
    composante s'ils sont reliés par un chemin de switches **fermés**
    (``etats_vl[sid]`` faux = fermé). ``edges`` = ``{switch_id: (n1, n2)}``.

    Si ``barres`` (nœuds jeux de barres) est fourni, les **ouvrages isolés** —
    ceux dont la composante ne contient **aucune barre** (équipement déconnecté /
    hors service) — sont **exclus** : un ouvrage isolé n'est pas un nœud électrique
    (il ne doit pas gonfler le décompte). Union-find. Fonction pure."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for node in poids:
        find(node)
    for sid, (n1, n2) in edges.items():
        find(n1)
        find(n2)
        if not etats_vl.get(sid, False):   # fermé → fusionne les deux nœuds
            ra, rb = find(n1), find(n2)
            if ra != rb:
                parent[ra] = rb
    part = {node: find(node) for node in poids}
    if barres:
        # composantes reliées à au moins une barre = nœuds électriques réels ;
        # les autres (ouvrages déconnectés) sont ignorées.
        racines_en_service = {find(b) for b in barres if b in parent}
        part = {n: c for n, c in part.items() if c in racines_en_service}
    return part


def _blocs(partition: dict[int, int]) -> list[set]:
    """Liste des blocs (ensembles de nœuds) d'une partition ``{node: comp}``."""
    blocs: dict[int, set] = {}
    for node, comp in partition.items():
        blocs.setdefault(comp, set()).add(node)
    return list(blocs.values())


def _alignement_max(blocs_a: list[set], blocs_b: list[set],
                    poids: dict[int, int]) -> list[tuple[int, int]]:
    """Appariement bloc_a ↔ bloc_b **maximisant le poids des ouvrages conservés**
    (intersection). Exact pour de petits effectifs (DP sur masque de bits des
    blocs B) ; repli glouton au-delà de 12 blocs. Renvoie les paires d'indices."""
    na, nb = len(blocs_a), len(blocs_b)
    if not na or not nb:
        return []
    ov = [[sum(poids.get(n, 0) for n in (blocs_a[i] & blocs_b[j]))
           for j in range(nb)] for i in range(na)]
    if na > 12 or nb > 12:   # repli glouton (rare)
        cand = sorted(((ov[i][j], i, j) for i in range(na) for j in range(nb)),
                      reverse=True)
        ua, ub, pairs = set(), set(), []
        for w, i, j in cand:
            if w <= 0:
                break
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            pairs.append((i, j))
        return pairs
    memo: dict[tuple[int, int], tuple[int, list]] = {}

    def rec(i: int, mask: int) -> tuple[int, list]:
        if i == na:
            return 0, []
        key = (i, mask)
        if key in memo:
            return memo[key]
        best_w, best_p = rec(i + 1, mask)   # bloc A_i non apparié
        for j in range(nb):
            if not (mask >> j) & 1 and ov[i][j] > 0:
                w1, p1 = rec(i + 1, mask | (1 << j))
                if w1 + ov[i][j] > best_w:
                    best_w, best_p = w1 + ov[i][j], [(i, j)] + p1
        memo[key] = (best_w, best_p)
        return memo[key]

    return rec(0, 0)[1]


def noeuds_deplaces(part_a: dict[int, int], part_b: dict[int, int],
                    poids: dict[int, int]) -> set:
    """Ensemble **minimal** de nœuds porteurs d'ouvrage ayant changé de nœud
    électrique entre deux partitions (ceux hors de l'intersection de leurs blocs
    appariés au mieux). Sur une scission ou une fusion simple = le **plus petit**
    groupe séparé / ayant rejoint. Fonction pure."""
    blocs_a, blocs_b = _blocs(part_a), _blocs(part_b)
    conserves: set = set()
    for ia, jb in _alignement_max(blocs_a, blocs_b, poids):
        conserves |= (blocs_a[ia] & blocs_b[jb])
    # ensemble de référence = ouvrages **en service dans les deux** situations
    # (présents dans les deux partitions) : une (dé)connexion d'ouvrage isolé
    # n'est pas un re-groupement.
    return (set(part_a) & set(part_b)) - conserves


def changements_nodaux_par_vl(
    situations: list[dict[str, dict[str, bool]]],
    struct: dict[str, dict],
) -> dict[str, int]:
    """Nombre d'**ouvrages re-groupés** (scission / fusion de nœuds) **par VL**
    sur la journée : pour chaque VL, on compare les partitions nodales des
    situations **consécutives** et on retient le **plus grand** nombre (pondéré
    par ``poids``) d'ouvrages minimalement déplacés en une transition — le « plus
    petit » groupe séparé / ayant rejoint, lors de la reconfiguration la plus
    marquée de la journée. ``struct`` = sortie de ``extraire_structure_topo``.

    On prend le **max** (et non la somme/union) sur les transitions : stable et
    sans double-comptage d'une scission suivie d'une fusion (l'ancrage du bloc
    « conservé » est ambigu sur une scission symétrique). Fonction pure."""
    out: dict[str, int] = {}
    for vl, s in struct.items():
        edges, poids = s.get("edges", {}), s.get("poids", {})
        barres = s.get("barres") or None
        if not poids:
            out[vl] = 0
            continue
        parts = [partition_ouvrages(situ[vl], edges, poids, barres)
                 for situ in situations if vl in situ]
        if len(parts) < 2:
            out[vl] = 0
            continue
        out[vl] = max(
            (sum(poids.get(n, 0) for n in noeuds_deplaces(a, b, poids))
             for a, b in zip(parts, parts[1:])),
            default=0)
    return out


def fusionner_nodaux(changes: dict[str, dict],
                     nodaux: dict[str, int]) -> dict[str, dict]:
    """Intègre les re-groupements de nœuds (``nodaux``) aux changements par VL :
    ajoute le champ ``nodal`` et **l'additionne au ``total``** (une scission /
    fusion est une configuration intéressante même si peu d'OC ont bougé).
    Modifie ``changes`` en place. Fonction pure."""
    for vl, ch in changes.items():
        n = int(nodaux.get(vl, 0))
        ch["nodal"] = n
        ch["total"] = ch.get("total", 0) + n
    return changes


def agreger_par_poste(
    changes_par_vl: dict[str, dict],
    vl_meta: dict[str, dict],
    sub_name: Optional[dict[str, str]] = None,
) -> dict[str, dict]:
    """Agrège les changements **par poste (substation)**.

    Retourne ``{substation: {'name', 'nominal_v_max', 'total', 'BREAKER',
    'DISCONNECTOR', 'n_oc', 'vls': [{'vl', 'nominal_v', 'total', 'BREAKER',
    'DISCONNECTOR', 'n_oc'}, …]}}`` (les VL d'un poste triés par changements
    décroissants). Fonction pure."""
    sub_name = sub_name or {}
    postes: dict[str, dict] = {}
    for vl, ch in changes_par_vl.items():
        meta = vl_meta.get(vl, {})
        sub = meta.get("substation") or vl
        nv = meta.get("nominal_v", 0.0)
        p = postes.get(sub)
        if p is None:
            p = postes[sub] = {
                "substation": sub,
                "name": sub_name.get(sub) or sub,
                "nominal_v_max": 0.0,
                "total": 0, "n_oc": 0, "nodal": 0,
                **{t: 0 for t in TYPES_OC},
                "vls": [],
            }
        p["nominal_v_max"] = max(p["nominal_v_max"], nv)
        p["total"] += ch["total"]
        p["n_oc"] += ch["n_oc"]
        p["nodal"] += ch.get("nodal", 0)
        for t in TYPES_OC:
            p[t] += ch.get(t, 0)
        p["vls"].append({
            "vl": vl, "nominal_v": nv,
            "name": meta.get("name") or vl,
            "total": ch["total"], "n_oc": ch["n_oc"],
            "nodal": ch.get("nodal", 0),
            **{t: ch.get(t, 0) for t in TYPES_OC},
        })
    for p in postes.values():
        p["vls"].sort(key=lambda d: (-d["total"], -d["nominal_v"], d["vl"]))
    return postes


def classer_postes(postes: dict[str, dict], n_top: int = 10) -> list[str]:
    """Identifiants des ``n_top`` postes les plus actifs (changements totaux
    décroissants ; départage par tension max puis nom). Fonction pure."""
    actifs = [s for s, p in postes.items() if p["total"] > 0]
    actifs.sort(key=lambda s: (-postes[s]["total"],
                               -postes[s]["nominal_v_max"],
                               postes[s]["name"]))
    return actifs[:n_top]


def vl_le_plus_actif(poste: dict) -> Optional[str]:
    """Voltage level le plus changeant d'un poste (pour ouvrir la vue
    topologique la plus « intéressante » au double-clic). ``None`` si aucun."""
    vls = poste.get("vls") or []
    return vls[0]["vl"] if vls else None
