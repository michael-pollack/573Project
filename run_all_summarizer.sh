#!/bin/bash
# Wrapper to run the summarizer with Conda environment activated

# Load Conda environment
source ~/home/jen/miniconda3/bin/conda
conda init
conda activate 573  

# # Run your summarization script
python summarize_hf.py \
  --input_file data/PLOS/clean_staged/test-00000-of-00001_clean.json \
  --text_column article \
  --output_file plos_plain.txt \
  --plain_language

python summarize_hf.py \
--input_file data/PLOS/clean_staged/test-00000-of-00001_clean.json \
--text_column article \
--output_file plos_complex.txt

python summarize_hf.py \
--input_file data/Elife/cleaned/test-00000-of-00001_clean.json \
--text_column article \
--output_file elife_complex.txt
