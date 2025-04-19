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
