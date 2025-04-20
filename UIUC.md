## Environment preparation
Install 573 environment file

    pip install torch, langchain, logger, datasets, accelerate>=0.26.0, bert-extractive-summarizer


## Download files and place in data folder

- [eLife](https://drive.google.com/drive/folders/156GqK28jpHpmFpsqLLAxHxgQThLqe8Is?usp=sharing): [DPR-train](https://drive.google.com/file/d/1RvLsuaBZ83SLdm9C4h8WVeYvnKzvCNVW/view?usp=drive_link), [DPR-val](https://drive.google.com/file/d/13GUIIbsxNdAk0LceUiOvBlIhvhjPdxIx/view?usp=drive_link), [wiki-definition-train](https://drive.google.com/file/d/1HohUnEVW2sZ5OO1EPASH0WfLp2aBLsuW/view?usp=sharing), [wiki-definition-val](https://drive.google.com/file/d/1x_eNeuRys1b8OsehCQU8Gr_VtxzxzE3q/view?usp=sharing), [extractive-summary-train](https://drive.google.com/file/d/18ZD8cL48yenK6nJmlCPXpbbOWTmX3H8y/view?usp=drive_link), [extractive-summary-val](https://drive.google.com/file/d/1CIqRyA50IpkqyivAnmhpbM4ety-OojA0/view?usp=drive_link).
- [PLOS](https://drive.google.com/drive/folders/1KftJ_LKVG-DCNacL2xuqA-OdlmRG89AG?usp=sharing): [DPR-train](https://drive.google.com/file/d/1pnZgSUTXyVpSxKPC2nlcl3eLpZMoZKG9/view?usp=sharing), [DPR-val](https://drive.google.com/file/d/1EYhs3snDcPJ2Ao-ztfIrF8EqwmS2iaOF/view?usp=sharing), [wiki-definition-train](https://drive.google.com/file/d/1_j52ZVbvPYhHAlXOwrS9IKE68Yi2LmVN/view?usp=drive_link), [wiki-definition-val](https://drive.google.com/file/d/1mhdop2dx7whygTOk27b_DY3D4SOWroTw/view?usp=drive_link), [extractive-summary-train](https://drive.google.com/file/d/1uVZl21eyZAGJPQiBXA9Prb9jIzYhxdn0/view?usp=drive_link), [extractive-summary-val](https://drive.google.com/file/d/10ww5h3Fk-tyJvdmOvQRjGWHdU1VimvZt/view?usp=drive_link)

## File preparation
Change the data path in utils.py line 37 to your data folder

### Extractive Summarization
#### 1. Constractive Dataset Creation
This script is designed for creating the datasets to fine-tune the extractive summarization model. 

```bash
# You may define these hyper-parameters on your own
python contrastive_dataset_creation.py \
        --device cuda:0 \
        --chunk-size 600 \
        --pos-threshold 0.9 \
        --neg-threshold 0.01
```
#### 2. Fine-tune the Extractive Summarizer
Use the above constractive datasets to fine-tune the extractive summarizer:
```bash
python fine_tune_extractive_summarizer.py \
    --train_data data/elife_train_sentence_level_positive_negative_pairs.csv \
    --val_data data/elife_val_sentence_level_positive_negative_pairs.csv \
    --output_path elife_trained_model
```
#### 3. Generate Extractive Summaries
After fine-tuning the extractive summarization model, run the following command to generate extractive summaries:

```bash
python run_extractive_summarization.py \
    --data_folder data \
    --dataset eLife \
    --split train \
    --labels_file data/Structured-Abstracts-Labels-102615.txt \
    --model_path elife_trained_model \
    --output_dir elife_extractive_summaries