#!/bin/bash
export BASEDATADIR="/home/alexli/data/dust_analysis_random_1000"
python src/europa/dl/dataset/pack_perception/ml_pack_perception.py \
    --dataset-id '649ef51a15639b3bbe271a8f' --data-dir ${BASEDATADIR} \
    --csv-path $BASEDATADIR/annotations.csv \
    --calib-tracker-csv files/calibration/motec_calibration_tracker_2019.csv --upload-to-s3 \
    --master-output-csv $BASEDATADIR/master_annotations_0.csv \
    --output-dir $BASEDATADIR --multiprocess-workers 24 --batch-size 24 \
    --model-path random_weights.pth --without-hdr-tonemap --model-type full \
    --max-disp 384 --gpu cpu --pandarallel-workers 24 --cam-calibration-path files/calibration --image-only