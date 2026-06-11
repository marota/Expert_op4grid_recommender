#!/bin/bash
# scripts/process_dgitt_days.sh
# ------------------------------
# Télécharge et traite N journées du dataset D-GITT RTE 7000 (Hugging Face) :
# pour chaque journée, téléchargement vérifié md5 puis pipeline « blocs de
# transition » (scripts/build_rte7000_blocks.py), avec au plus -j pipelines
# concurrents (2 par défaut — chaque pipeline crête à ~4 Go de RAM).
#
# Usage :
#   bash scripts/process_dgitt_days.sh [options] JJ1 [JJ2 ...]
#     JJn               journée au format YYYY/MM/DD (le dépôt HF est déduit
#                       de l'année : OpenSynth/D-GITT-RTE7000-<YYYY>)
#     -d DATA_ROOT      racine des données brutes   (défaut : data)
#     -o OUT_ROOT       racine des sorties          (défaut : out_rte7000)
#     -j MAX_PIPELINES  pipelines concurrents       (défaut : 2)
#     -s MIN_STABILITE  plateaux stables, snapshots (défaut : 2)
#
# Exemple — une semaine d'été 2022 :
#   bash scripts/process_dgitt_days.sh \
#       2022/07/11 2022/07/12 2022/07/13 2022/07/14 2022/07/15
#
# Chaque journée ≈ 430 Mo téléchargés (jamais à committer) et ≈ 25-35 min de
# traitement ; sorties dans OUT_ROOT/<YYYYMMDD>/ + journal OUT_ROOT/<YYYYMMDD>.log.
set -u

DATA_ROOT="data"
OUT_ROOT="out_rte7000"
MAX_PIPELINES=2
MIN_STABILITE=2
while getopts "d:o:j:s:h" opt; do
  case "$opt" in
    d) DATA_ROOT="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    j) MAX_PIPELINES="$OPTARG" ;;
    s) MIN_STABILITE="$OPTARG" ;;
    h|*) tail -n +2 "$0" | grep "^#" | sed 's/^# \{0,1\}//' | head -24; exit 0 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "usage : $0 [-d data] [-o out] [-j 2] [-s 2] YYYY/MM/DD [...]" >&2; exit 2; }

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT_ROOT"
pids=()

wait_slot() {
  while true; do
    alive=()
    for p in "${pids[@]:-}"; do
      [ -n "$p" ] && kill -0 "$p" 2>/dev/null && alive+=("$p")
    done
    pids=("${alive[@]:-}")
    [ "${#pids[@]}" -lt "$MAX_PIPELINES" ] && return
    sleep 15
  done
}

for day in "$@"; do
  case "$day" in
    [0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]) ;;
    *) echo "!! journée ignorée (format attendu YYYY/MM/DD) : $day" >&2; continue ;;
  esac
  year="${day%%/*}"
  tag="${day//\//}"
  data_dir="$DATA_ROOT/dgitt_rte7000_$year"

  echo "=== [$(date +%H:%M:%S)] TÉLÉCHARGEMENT $day (OpenSynth/D-GITT-RTE7000-$year)"
  ok=0
  for essai in 1 2 3; do
    if python "$REPO_DIR/scripts/download_dgitt_subset.py" \
        --repo "OpenSynth/D-GITT-RTE7000-$year" --prefix "$day" \
        --output "$data_dir" --jobs 3 >> "$OUT_ROOT/dl_$tag.log" 2>&1; then
      ok=1; break
    fi
    echo "    reprise $essai/3 (échecs partiels — relance idempotente)"
    sleep 10
  done
  n=$(ls "$data_dir/$day"/*.xiidm.bz2 2>/dev/null | wc -l)
  echo "    $n instantané(s) présent(s)"
  if [ "$ok" -ne 1 ] || [ "$n" -eq 0 ]; then
    echo "!! téléchargement incomplet pour $day — journée sautée (cf. $OUT_ROOT/dl_$tag.log)" >&2
    continue
  fi

  wait_slot
  echo "=== [$(date +%H:%M:%S)] PIPELINE $day → $OUT_ROOT/$tag"
  python "$REPO_DIR/scripts/build_rte7000_blocks.py" \
      --input "$data_dir/$day" --output "$OUT_ROOT/$tag" \
      --min-stabilite "$MIN_STABILITE" > "$OUT_ROOT/$tag.log" 2>&1 &
  pids+=("$!")
done

for p in "${pids[@]:-}"; do
  [ -n "$p" ] && wait "$p"
done

echo
echo "=== [$(date +%H:%M:%S)] RÉSUMÉ"
code=0
for day in "$@"; do
  tag="${day//\//}"
  if [ -f "$OUT_ROOT/$tag/stats.json" ]; then
    python - "$OUT_ROOT/$tag/stats.json" "$day" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); t = s["par_tag"]
struct = sum(t.get(k, 0) for k in ("fusion_noeuds", "scission_noeud",
                                   "reaiguillage_departs", "sectionnement_barre"))
print(f"  {sys.argv[2]} : {s['nb_blocs']} blocs / {s['nb_postes']} postes, "
      f"{struct} reconfigurations structurelles, "
      f"{s.get('catalogue_topologies', {}).get('topologies_stables', '?')} topologies stables")
PY
  else
    echo "  $day : ÉCHEC (cf. $OUT_ROOT/$tag.log)"; code=1
  fi
done
exit "$code"
