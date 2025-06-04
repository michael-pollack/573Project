#!/bin/sh

# DATA CLEANING: Creates new dataframes with a "clean" column
# Run with elife training data, which is the default args
python python_scripts/data_cleaning.py

# Run with Elife training data
python python_scripts/data_cleaning.py --parquet_path "data/Elife/train-00000-of-00001.parquet" --output_csv "data/df_Elife_train_clean_00000_of_00003.csv"

#TF-IDF EXTRACTIVE SUMMARIES: Creates new dataframes with a "tfidf_summary" column
# Run with elife training data, which is the default args
python python_scripts/tfidf.py

# Run with Elife training data
python python_scripts/tfidf.py --clean_csv "data/df_Elife_train_clean_00000_of_00001.csv" --output_json "data/Elife_train_summaries_00000_of_00003.json" --output_txt "data/Elife_summaries_00000_of_00003.txt"

