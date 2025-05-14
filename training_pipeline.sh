#!/bin/bash
MODEL_NAME=$1
TRAIN=$2
VAL=$3

echo "Creating clean train files"
python data_cleaning.py --parquet_path $TRAIN --output_csv data/train_clean.csv

echo "Creating clean val files"
python data_cleaning.py --parquet_path $VAL --output_csv data/val_clean.csv

echo "Running TF-IDF Extraction on train"
python tfidf.py --clean_csv data/train_clean.csv --output_json data/train_summaries.json --output_txt data/train_summaries.txt

echo "Running TF-IDF Extraction on val"
python tfidf.py --clean_csv data/val_clean.csv --output_json data/val_summaries.json --output_txt data/val_summaries.txt

echo "Fine tuning abstractive model"
python fine_tune_abstractive.py --train_file data/train_summaries.json --val_file data/val_summaries.json --output_dir $MODEL_NAME

echo "Training complete! Please use testing_pipeline.sh to run test files."

