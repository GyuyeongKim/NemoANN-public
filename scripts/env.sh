ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${NEMO_DATA:-$ROOT/data}
RESULTS=${NEMO_RESULTS:-$ROOT/results}
BIN=$ROOT/build/tests
NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)

LS_100M="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"
LS_1B="20 30 40 50 70 100 150 200 300 400 500 700 1000"

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

die() {
  log "$*"
  exit 1
}

dtype() {
  case $1 in
    deep*) echo float ;;
    *) echo uint8 ;;
  esac
}

vec_ext() {
  case $1 in
    deep*) echo fbin ;;
    *) echo u8bin ;;
  esac
}

query_file() {
  echo "$DATA/$1/query.$(vec_ext "$1")"
}

base_file() {
  echo "$DATA/$1/base.$(vec_ext "$1")"
}

index_prefix() {
  echo "$DATA/$1/index/$1"
}

label_prefix() {
  echo "$DATA/$1/labels/sel$2"
}

require() {
  for f in "$@"; do
    [ -e "$f" ] || die "missing $f; run scripts/prepare.sh first"
  done
}

new_table() {
  mkdir -p "$(dirname "$1")"
  [ -s "$1" ] || printf 'system\tdataset\tsel\tthreads\tW\tmem_L\tR_max\tL\tQPS\tLat_us\tRecall\tIOs\n' > "$1"
}

run_disk() {
  local table=$1 ds=$2 sel=$3 sys=$4 threads=$5 w=$6 mem_l=$7 r_max=$8
  shift 8
  local lp=$(label_prefix "$ds" "$sel")
  local logs=$RESULTS/logs/$(basename "$table" .tsv)
  require "$(index_prefix "$ds")_disk.index" "${lp}_base.bin" "${lp}_query.bin" "${lp}_gt.bin"
  [ "$mem_l" = 0 ] || require "$(index_prefix "$ds")_mem.index"
  new_table "$table"
  mkdir -p "$logs"
  local rcol=$r_max
  [ "$sys" = nemoann ] || rcol=-
  for L in "$@"; do
    local out=$logs/${ds}_sel${sel}_${sys}_R${rcol}_T${threads}_L${L}.log
    log "$sys $ds sel=$sel T=$threads L=$L"
    "$BIN/search_disk_index" "$(dtype "$ds")" "$(index_prefix "$ds")" "$threads" "$w" "$(query_file "$ds")" \
      "${lp}_base.bin" "${lp}_query.bin" "${lp}_gt.bin" 10 "$sys" "$mem_l" "$r_max" "$L" > "$out" 2>&1 ||
      die "search failed, see $out"
    local pre
    pre=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s' "$sys" "$ds" "$sel" "$threads" "$w" "$mem_l" "$rcol")
    awk -v pre="$pre" '$1 ~ /^[0-9]+$/ && NF >= 5 { printf "%s\t%s\t%s\t%s\t%s\t%s\n", pre, $1, $2, $3, $4, $5 }' \
      "$out" >> "$table"
  done
}
