#!/bin/bash
mkdir "/path/to/results/$1"

python deep_sort_app.py \
    --sequence_dir=/path/to/MOT17/train/ \
    --output_file=$1 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=False \
    --max-age=30 \
    --temporal_noise=True \
    --only_filtering=False \
    --default_matching=True \
    --freespace_filtering=True \
    --ah_velocity=False \
    --velocity_weighting=False \
    --tune_temporal_noise=True \
    --obs_constant=600 \
    --obs_factor=1 \
    --proc_constant=450 \
    --proc_factor=2 \
    --occluded_factor=1.06 \
    --filtering_factor=1.14 \
    --motion-aware=True \
    --output-uncertainty=True \
    --only-extrapolate=False \
    --extrapolated-iou-match=False \
    --appearance-match=True \
    --bugfix=True

# python tools/convert_sort_output_to_topk_format.py \
#     --split=sort \
#     --interp=0 \
#     --input=/path/to/results/$1/*.txt \
#     --output-json=/path/to/results_json/$1.json \
#     --k 5 \
#     --truncate

# cd evaluation/cocoapi

# python amodaltopkevaluate.py \
#     /data/tkhurana/MOT17/train_json/train_FRCNN_MOT17train_27812ignored_annotations.json \
#     /path/to/results_json/$1.json \
#     --output-file /path/to/outputs/$1.txt \
#     --pickle-file /path/to/outputs/$1.pkl \
#     --eval-ious 1 \
#     --ious 0.5
