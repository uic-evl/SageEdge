#!/bin/bash

# ---------------------------
# Privacy Filter Benchmarking
# Runs all filters sequentially on Thor
# ---------------------------

# Paths
DATA_DIR="/home/thorwaggle/Desktop/SageEdge/AI_Programs/privacy_benchmarking"
OUT_DIR="$DATA_DIR/output"
VIDEO="$DATA_DIR/walking_test.mp4"

IMAGE="privacy-lab:latest"

# Filters to run
FILTERS=("pixelate" "gaussian" "box" "median")

echo "Starting full benchmarking run..."
echo "Video source: $VIDEO"
echo "Output directory: $OUT_DIR"
echo "Docker image: $IMAGE"
echo "---------------------------------------"
echo ""

# Loop through each filter
for FILTER in "${FILTERS[@]}"; do
    CSV="$OUT_DIR/metrics_${FILTER}.csv"
    OUT_VIDEO="$OUT_DIR/${FILTER}_out.mp4"

    echo ""
    echo "======================================="
    echo "Running filter: $FILTER"
    echo "CSV output:    $CSV"
    echo "Video output:  $OUT_VIDEO"
    echo "======================================="
    echo ""

    docker run --gpus all --rm \
      -v "$DATA_DIR":/data \
      -v "$OUT_DIR":/output \
      "$IMAGE" \
      python main.py \
        --filter "$FILTER" \
        --source /data/walking_test.mp4 \
        --no-display \
        --csv "/output/metrics_${FILTER}.csv" \
        --save-video "/output/${FILTER}_out.mp4"

    # If docker run failed, stop script
    if [ $? -ne 0 ]; then
        echo "Error running filter: $FILTER"
        echo "Stopping script."
        exit 1
    fi

    echo "Completed: $FILTER"
done

echo ""
echo "---------------------------------------"
echo "ALL FILTERS COMPLETED SUCCESSFULLY"
echo "Output files saved to: $OUT_DIR"
echo "---------------------------------------"
