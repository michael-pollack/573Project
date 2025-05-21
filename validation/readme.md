## 1. How to use this validation data

The summaries to use for validation are:

`validation/validation_elife_summaries.txt
`1.38k rows

`validation/validation_plos_summaries.txt
` 241 rows

If you would like to use a subset of the data for validation, use the pruning.py file and select the number of lines you want to use:

This is an example of using the first 10 lines of the elife data:

```python
python validation/pruning.py --input validation/validation_elife_summaries.txt --output validation/elife_val.txt --num_lines 10
```
This is an example of using the first 100 lines of the plos data:

```python
python validation/pruning.py --input validation/validation_plos_summaries.txt --output validation/valdiation_plos_summaries.txt --num_lines 100
```

* If `num_lines` is not declared, it will keep the entire dataset.
* The output name you give your file will be edited to include _number at the end to indicate the number of lines kept. If your output file argument is `val_plos_summary.txt` and `num_lines = 100`, your actual output file is `val_plos_summary_100.txt`

## 2. Data pre-processing

   Clean your data by running it through data_cleaning.py
```python
    python /home/jen/573Project-1/data_cleaning.py   \
    --parquet_path data/Elife/validation-00000-of-00001.parquet  \
    --output_csv validation/elife_clean.csv \
    --num_lines 100
```

* If you are using a subset of the data, make sure you select the same number for cleaning. And make sure you are using the same dataset!

* If `num_lines` isn't declared, it will run the entire dataset.
* The output name you give your file will be edited to include _number at the end to indicate the number of lines kept. If your output file argument is `val_plos_summary.txt` and `num_lines = 100`, your actual output file is `val_plos_summary_100.txt`

From now on, you do not need to specify the number of lines. 

Run it through tfidf.py. Double check that you are using the correct files.

Then you can run the summarization and evaluation.
