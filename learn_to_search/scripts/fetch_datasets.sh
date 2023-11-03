#!/bin/bash

# Create datasets dir if not exists
if [ ! -d "./data" ]; then
    mkdir ./data
fi

# Define an array of language codes
declare -a codes=("test" "train" "validation")

# Iterate over each language code and download the parquet file
for code in "${codes[@]}"; do
    wget "https://huggingface.co/datasets/ms_marco/resolve/refs%2Fconvert%2Fparquet/v1.1/${code}/0000.parquet" -O "./data/ms_marco_${code}.parquet"
done

