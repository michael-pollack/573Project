
import argparse
import os
import random
import torch
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(train_path, val_path):
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
    })
    return dataset

def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=200):
    model_inputs = tokenizer(
        examples["article"], max_length=max_input_length, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["summary"], max_length=max_target_length, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./pegasus_large_model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

    raw_datasets = load_data(args.train_file, args.val_file)

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
       #evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir=f"{args.output_dir}/logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
