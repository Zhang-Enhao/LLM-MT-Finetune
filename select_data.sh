#!/bin/bash

data_folder="./human_written_data"
output_folder="./selected_data"
language_pairs="csen deen isen ruen zhen"

for lang_pair in $language_pairs; do
    python select_data.py "$data_folder" "$output_folder" "$lang_pair"
done