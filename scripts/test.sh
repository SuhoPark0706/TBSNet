#!/bin/bash

FOLDNUM="2"
train_fold="2"
nshot=1
dataset="pascal"
test_param="pascal"
TESTFILE="test.py"


python ./$TESTFILE --datapath "../datasets" \
                 --benchmark $dataset \
                 --fold $FOLDNUM \
                 --bsz 1 \
                 --nworker 8 \
                 --backbone swin \
                 --feature_extractor_path "../backbones/swin_base_patch4_window12_384.pth" \
                 --logpath "./logs" \
                 --load "./trained_weights_${test_param}_fold${train_fold}/best_model.pt" \
                 --nshot $nshot \
                 --vispath "./vis_{test.py}_${FOLDNUM}_${nshot}shot_${dataset}_${test_param}/" \
                 --visualize
