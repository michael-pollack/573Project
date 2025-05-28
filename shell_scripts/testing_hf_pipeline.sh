#!/bin/bash
TEST=$1
OUTPUT=$2

echo "Creating clean test files"
python data_cleaning.py --parquet_path $TEST --output_csv data/test_clean.csv

echo "Running TF-IDF Extraction on test files"
python tfidf.py --clean_csv data/test_clean.csv --output_json data/test_summaries.json --output_txt data/test_summaries.txt

echo "Running abstractive model on test files"
python summarize_hf.py --input_file data/test_summaries.json --text_column article --output_file $OUTPUT

echo "Abstractive Summaries complete and exported to $OUTPUT."