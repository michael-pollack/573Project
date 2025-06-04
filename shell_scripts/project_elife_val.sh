#!/bin/sh

# DATA CLEANING: Creates new dataframes with a "clean" column
# Run with Elife validation data

python python_scripts/data_cleaning.py --parquet_path "data/Elife/validation-00000-of-00001.parquet" --output_csv "data/df_Elife_validation_clean.csv"

#TF-IDF EXTRACTIVE SUMMARIES: Creates new dataframes with a "tfidf_summary" column
# Run with Elife validation data

python python_scripts/tfidf.py --clean_csv "data/df_Elife_validation_clean.csv" --output_json "data/Elife_validation_summaries.json" --output_txt "data/Elife_validation_summaries.txt"

