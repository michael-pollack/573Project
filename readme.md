# Environment setup

How to set up your environment:
conda env create -f environment.yml
conda activate 573

Python is specified as 3.11.12. All other packages will be installed with the latest version.

# Data files

Due to the size of the data files, they are not hosted on this repo. Save the data sets locally in the following directory structure:

```
.
├── data
│   ├── Elife
│   │   ├── test-00000-of-00001.parquet
│   │   ├── train-00000-of-00001.parquet
│   │   └── validation-00000-of-00001.parquet
│   └── PLOS
│       ├── test-00000-of-00001.parquet
│       ├── train-00000-of-00003.parquet
│       ├── train-00001-of-00003.parquet
│       ├── train-00002-of-00003.parquet
│       └── validation-00000-of-00001.parquet
```

Here are the links for the data on hugging face

https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-PLOS/tree/main/data

https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife/tree/main/data

# Important Files and Their Purpose

data_cleaning.py - cleans and tokenizes the data so that it can be processed by subsequent files.

tfidf.py - performs TF-IDF extraction on articles to create our base extractive summaries.

fine_tune_abstractive.py - uses extractive summaries paired with gold standard abstractive summaries to train a pegasus-large-model for abstractive summarization.

run_abstractive.py - runs the fine-tuned model on test summaries to generate abstractive summaries.

layterm_injection.py - performs lay-term injections on abstractive summaries to replace any remaining jargon.

evaluation.py - runs evaluation metrics on abstractive summaries.

summarize_hf.py - work-around for run_abstractive.py that uses a Hugging Face model instead of a locally trained one in order to conserve local computing resources.


# Training the model
We have included a script that runs our full training pipeline from start to finish. Please call training_pipeline.sh specifying your training files and your validation files in the following manner:

./training_pipeline.sh [NAME_OF_MODEL] [PATH_TO_TRAIN_PARQUET_FILES] [PATH_TO_VAL_PARQUET_FILES]

NOTE: This pipeline is configured to train our pegasus_large model. Though this model build correctly and is functional, complications with computer resources and access to testing data means that the evaluation metrics we have provided in our paper do not come from this model. We hope to have these metrics and data for our final report. 

If you wish to test with the pegasus-xsum model, please edit lines 55 and 56 of fine_tune_abstractive.py to read:
'''
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
'''


# Running the model
Once the model has been trained, it will appear locally as a folder NAME_OF_MODEL. You can then run the test files through the trained model with the following command:

./testing_pipeline.sh [NAME_OF_MODEL] [PATH_TO_TEST_FILES] [PATH_TO_OUTPUT_FILES]

# Running Hugging Face Alternative
If you wish to run the falconsai/medical_texts model instead of our fine_tuned model, you can do so with the following command:

./testing_hf_pipeline.sh [PATH_TO_TEST_FILES] [PATH_TO_OUTPUT_FILES]
