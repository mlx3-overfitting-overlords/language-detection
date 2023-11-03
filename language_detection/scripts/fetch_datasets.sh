#!/bin/bash

# Create datasets dir if not exists
if [ ! -d "./datasets" ]; then
    mkdir ./datasets
fi

# Define an array of language codes
declare -a language_codes=("it-en" "fr-en" "es-en" "de-en")

# Iterate over each language code and download the parquet file
for code in "${language_codes[@]}"; do
    wget "https://huggingface.co/datasets/iix/Parquet_FIles/resolve/main/CL_${code}.parquet" -O "./datasets/CL_${code}.parquet"
done

wget https://huggingface.co/datasets/iix/Parquet_FIles/resolve/main/Flores7Lang.parquet -O "./datasets/Flores7Lang.parquet"

