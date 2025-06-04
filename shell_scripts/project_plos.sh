#!/bin/sh

# DATA CLEANING: Creates new dataframes with a "clean" column
# Run with elife training data, which is the default args
python python_scripts/data_cleaning.py

# Run with 1/3 plos training data
python python_scripts/data_cleaning.py --parquet_path "data/PLOS/train-00000-of-00003.parquet" --output_csv "data/df_plos_train_clean_00000_of_00003.csv"

# Run with 2/3 plos training data
python python_scripts/data_cleaning.py --parquet_path "data/PLOS/train-00001-of-00003.parquet" --output_csv "data/df_plos_train_clean_00001_of_00003.csv"

# Run with 3/3 plos training data
python python_scripts/data_cleaning.py --parquet_path "data/PLOS/train-00002-of-00003.parquet" --output_csv "data/df_plos_train_clean_00002_of_00003.csv"

#TF-IDF EXTRACTIVE SUMMARIES: Creates new dataframes with a "tfidf_summary" column
# Run with elife training data, which is the default args
python python_scripts/tfidf.py

# Run with 1/3 plos training data
python python_scripts/tfidf.py --clean_csv "data/df_plos_train_clean_00000_of_00003.csv" --output_json "data/plos_train_summaries_00000_of_00003.json" --output_txt "data/plos_summaries_00000_of_00003.txt"

# Run with 2/3 plos training data
python python_scripts/tfidf.py --clean_csv "data/df_plos_train_clean_00001_of_00003.csv" --output_json "data/plos_train_summaries_00001_of_00003.json" --output_txt "data/plos_summaries_00001_of_000003.txt"

# Run with 3/3 plos training data
python python_scripts/tfidf.py --clean_csv "data/df_plos_train_clean_00002_of_00003.csv" --output_json "data/plos_train_summaries_00002_of_00003.json" --output_txt "data/plos_summaries_00002_of_000003.txt"