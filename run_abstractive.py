
import argparse
import torch
import pandas as pd
import tqdm
from transformers import PegasusTokenizer, AutoModelForSeq2SeqLM

def generate_summaries(model_path, input_file, output_file, max_input_length=512, max_output_length=128):
    # Load model and tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")

    # Load input articles
    # df = pd.read_json(input_file, lines=True)
    with open(input_file, 'r') as input:
        inputs = input.readlines()
    
    # Generate summaries
    summaries = []
    for text in tqdm(inputs, "Processing Inputs"):
        inputs_tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding="max_length"
        )

        if torch.cuda.is_available():
            inputs_tokenized = {k: v.to("cuda") for k, v in inputs_tokenized.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs_tokenized,
                max_length=max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Save to output file
    with open(output_file, 'w') as output:
        for summ in summaries:
            output.write(summ)
            output.write("\n\n")
    # df["generated_summary"] = summaries
    # df.to_json(output_file, orient="records", lines=True)
    print(f"Summaries written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    generate_summaries(args.model_path, args.input_file, args.output_file)
