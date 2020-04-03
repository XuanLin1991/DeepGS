#!/bin/bash

DATASET=davis
# DATASET=celegans
# DATASET=yourdata

# radius=1
radius=2
# radius=3

# ngram=2
ngram=3

dim=32
layer_gnn=3
side=5
window=$((2*side+1))
layer_cnn=3
layer_output=3
lr=1e-3
lr_decay=0.9
decay_interval=30
weight_decay=1e-5
iteration=100

setting=$DATASET--radius$radius--ngram$ngram--dim$dim--layer_gnn$layer_gnn--window$window--layer_cnn$layer_cnn--layer_output$layer_output--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
python run_DeepGS.py $DATASET $radius $ngram $dim $layer_gnn $window $layer_cnn $layer_output $lr $lr_decay $decay_interval $weight_decay $iteration $setting
