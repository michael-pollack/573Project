Step 1:
Set up your environment using *requirements.txt*

Step 2:
all the imports you need are in *importables.py*

Due to the size of the data files, they are not hosted on this repo. The expectation is that you have the files locally saved in the following directory structure:

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

Here are the links for our data on hugging face

https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-PLOS/tree/main/data

https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife/tree/main/data
