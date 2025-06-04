#!/bin/bash
MODEL_PATH=$1
TEST=$2
OUTPUT=$3

echo "Creating clean test files"
python python_scripts/data_cleaning.py --parquet_path $TEST --output_csv data/test_clean.csv

echo "Running TF-IDF Extraction on test files"
python python_scripts/tfidf.py --clean_csv data/test_clean.csv --output_json data/test_summaries.json --output_txt data/test_summaries.txt

echo "Running abstractive model on test files"
python python_scripts/fine_tune_abstractive.py --model_path $MODEL_PATH --input_file data/test_summaries.txt --output_file $OUTPUT

echo "Abstractive Summaries complete and exported to $OUTPUT."

