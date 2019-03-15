#!/bin/bash

train_test_data_dir='./data'
# tune_set='nist06'
data_dir='./data'
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/gq"

python3.6 ./preprocess.py -train_src $train_test_data_dir/task3a.train.src \
                       -train_tgt $train_test_data_dir/task3a.train.mt \
                       -valid_src $train_test_data_dir/task3a.dev.src  \
                       -valid_tgt $train_test_data_dir/task3a.dev.mt \
                       -save_data $data_prefix \
                       -src_vocab_size 30000  \
                       -tgt_vocab_size 30000 \
                       -src_seq_length 150 \
                       -tgt_seq_length 150
