from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import logging
import json

# --------------------------
# Config
# --------------------------
model_name = "google/pegasus-large"
data_files = {
    "train": "data/elife_json/train-00000-of-00001.json",
    "validation": "data/elife_json/validation-00000-of-00001.json",
    "test": "data/elife_json/validation-00000-of-00001.json"
}
max_input_length = 512
max_target_length = 64
output_dir = "./lora_pegasus_elife"

# --------------------------
# Set up logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Load dataset and tokenizer
# --------------------------
logger.info("Loading JSON datasets...")
raw_datasets = load_dataset("json", data_files=data_files)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(example):
    inputs = tokenizer(
        example["article"],
        max_length=max_input_length,
        truncation=True,
        padding="longest"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text_target=example["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="longest",
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

# Apply preprocessing with progress bars
logger.info("Tokenizing datasets...")
tokenized_datasets = {}
for split in raw_datasets:
    logger.info(f"Tokenizing {split} set...")
    tokenized_datasets[split] = raw_datasets[split].map(
        preprocess_function,
        batched=True,
        desc=f"Tokenizing {split}"
    )

# --------------------------
# Load model and apply LoRA
# --------------------------
logger.info("Loading model and applying LoRA...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

for name, _ in model.named_modules():
    if "proj" in name:
        print(f'name: {name}')

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Pegasus uses q/v attention names
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------------
# Training setup
# --------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    predict_with_generate=True,
    fp16=True,  # set False if using CPU
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_total_limit=2
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# --------------------------
# Train and Evaluate
# --------------------------
logger.info("Starting training...")
trainer.train()

# --------------------------
# Predict on Test Set
# --------------------------
logger.info("Generating predictions on test set...")
results = trainer.predict(tokenized_datasets["test"])
preds = tokenizer.batch_decode(results.predictions, skip_special_tokens=True)

with open("test_predictions.txt", "w") as f:
    for line in tqdm(preds, desc="Saving predictions"):
        f.write(line.strip() + "\n")

# --------------------------
# Evaluate on Validation Set
# --------------------------
logger.info("Evaluating on validation set...")
val_results = trainer.evaluate(tokenized_datasets["validation"])
print("Validation results:", val_results)
with open("validation_results.json", "w") as f:
    json.dump(val_results, f, indent=2)

logger.info("Done.")
