#!/bin/bash
# End-to-end: build index (bucket2), optionally search & evaluate recall
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BUCKET_DIR="$SCRIPT_DIR/.."
BIN="$BUILD_DIR/bucket2"
SEARCH_BIN="$SCRIPT_DIR/search"

# ---- Usage ----
usage() {
    echo "Usage: $0 -b <base_file> [options]"
    echo ""
    echo "Required:"
    echo "  -b, --base       base vectors file (.fbin/.ibin/.bin)"
    echo ""
    echo "Optional:"
    echo "  -o, --output     output directory (default: ./output/recall_test)"
    echo "  --neighbors-m    per-vector KNN M (default: 32)"
    echo "  --centroid-ratio centroid ratio (default: 0.01)"
    echo "  --sample-rate    sample rate (default: 1.0)"
    echo "  --knn-k          centroid KNN graph degree (default: 32)"
    echo "  --search-iters   graph search max iterations (default: 64)"
    echo "  --seed           random seed (default: 42)"
    echo "  --no-build       skip compilation step"
    echo ""
    echo "Search (optional, requires --search):"
    echo "  --search         enable search & recall evaluation after building"
    echo "  -q, --query      query vectors file (.fbin/.ibin/.bin)"
    echo "  -g, --gt         ground truth file (.ibin)"
    echo "  --topk           search topk (default: 10)"
    echo "  --ef             search ef (default: 200)"
    echo "  --num-entry      random entry points (default: 4)"
    exit 1
}

# ---- Default Parameters ----
NEIGHBORS_M=32
CENTROID_RATIO=0.01
SAMPLE_RATE=1.0
KNN_K=32
SEARCH_ITERS=64
TOPK=10
EF=200
NUM_ENTRY=4
SEED=42
OUTPUT_DIR="$SCRIPT_DIR/output/recall_test"
NO_BUILD=0
DO_SEARCH=0

BASE_FILE=""
QUERY_FILE=""
GT_FILE=""

# ---- Parse Arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--base)        BASE_FILE="$2";       shift 2 ;;
        -q|--query)       QUERY_FILE="$2";      shift 2 ;;
        -g|--gt)          GT_FILE="$2";         shift 2 ;;
        -o|--output)      OUTPUT_DIR="$2";      shift 2 ;;
        --neighbors-m)    NEIGHBORS_M="$2";     shift 2 ;;
        --centroid-ratio) CENTROID_RATIO="$2";  shift 2 ;;
        --sample-rate)    SAMPLE_RATE="$2";     shift 2 ;;
        --knn-k)          KNN_K="$2";           shift 2 ;;
        --search-iters)   SEARCH_ITERS="$2";   shift 2 ;;
        --topk)           TOPK="$2";            shift 2 ;;
        --ef)             EF="$2";              shift 2 ;;
        --num-entry)      NUM_ENTRY="$2";       shift 2 ;;
        --seed)           SEED="$2";            shift 2 ;;
        --no-build)       NO_BUILD=1;           shift ;;
        --search)         DO_SEARCH=1;          shift ;;
        -h|--help)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

# ---- Validate Required Arguments ----
if [[ -z "$BASE_FILE" ]]; then
    echo "Error: --base is required."
    echo ""
    usage
fi

if [[ ! -f "$BASE_FILE" ]]; then
    echo "Error: file not found: $BASE_FILE"
    exit 1
fi

if [[ "$DO_SEARCH" -eq 1 ]]; then
    if [[ -z "$QUERY_FILE" || -z "$GT_FILE" ]]; then
        echo "Error: --search requires --query and --gt."
        exit 1
    fi
    for f in "$QUERY_FILE" "$GT_FILE"; do
        if [[ ! -f "$f" ]]; then
            echo "Error: file not found: $f"
            exit 1
        fi
    done
fi

echo "=========================================="
echo "Build Index"
echo "=========================================="
echo "  base:  $BASE_FILE"
if [[ "$DO_SEARCH" -eq 1 ]]; then
    echo "  query: $QUERY_FILE"
    echo "  gt:    $GT_FILE"
fi

# ============================================================
# Step 1: Compile
# ============================================================
if [[ "$NO_BUILD" -eq 0 ]]; then
    echo ""
    echo "[Step 1] Compiling bucket2..."
    cd "$BUILD_DIR"
    make -j$(nproc) bucket2
    echo "  bucket2 compiled: $BIN"
    if [[ "$DO_SEARCH" -eq 1 ]]; then
        echo "  Compiling search..."
        cd "$SCRIPT_DIR"
        make -j$(nproc) search
        echo "  search  compiled: $SEARCH_BIN"
    fi
else
    echo ""
    echo "[Step 1] Skipped compilation (--no-build)"
fi

# ============================================================
# Step 2: Run bucket2 to build index
# ============================================================
echo ""
echo "[Step 2] Building index with bucket2..."
mkdir -p "$OUTPUT_DIR"

time $BIN \
    -i "$BASE_FILE" \
    -o "$OUTPUT_DIR" \
    --centroid-ratio "$CENTROID_RATIO" \
    --sample-rate "$SAMPLE_RATE" \
    --knn-k "$KNN_K" \
    --search-iters "$SEARCH_ITERS" \
    --neighbors-m "$NEIGHBORS_M" \
    --seed "$SEED"

echo ""
echo "Index output:"
ls -lh "$OUTPUT_DIR"

# ============================================================
# Step 3 (optional): Run search (C++) to evaluate recall
# ============================================================
if [[ "$DO_SEARCH" -eq 1 ]]; then
    echo ""
    echo "[Step 3] Evaluating recall with search (C++)..."

    $SEARCH_BIN search \
        --base   "$BASE_FILE" \
        --query  "$QUERY_FILE" \
        --index  "$OUTPUT_DIR/neighbors.npy" \
        --gt     "$GT_FILE" \
        --topk   "$TOPK" \
        --ef     "$EF" \
        --num-entry "$NUM_ENTRY"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
