
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import PegasusTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def load_index_and_metadata(indices: str, metadata: str) -> tuple:
    index = faiss.read_index(indices)
    with open(metadata, 'r') as file:
        metadata = json.load(file)
    return index, metadata

def semantic_search(query, model, index, metadata, top_k=5) -> list:
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=top_k)
    results = []
    for idx in I[0]:
        entry = metadata[idx]
        results.append({
            "title": entry.get("title", "No title"),
            "text": entry.get("text", "")[:1000],  # limit output length
            "source_keyword": entry.get("source_keyword", "N/A")
        })
    return results



def generate_summaries(
        model_path: str, 
        input_file: str,
        output_file: str, 
        indices_file: str, 
        metadata_file: str, 
        max_input_length: int=512, 
        max_output_length: int=128) -> None:
    tokenizer = PegasusTokenizer.from_pretrained(model_path, use_fast=False)
    abs_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    abs_model.eval()

    index, metadata = load_index_and_metadata(indices_file, metadata_file)


    if torch.cuda.is_available():
        abs_model.to("cuda")

    with open(input_file, 'r') as input:
        inputs = input.readlines()
    
    summaries = []
    for text in tqdm(inputs, "Processing Inputs"):

        search_results = semantic_search(text, emb_model, index, metadata)
        search_results = [s["text"] for s in search_results]
        print(search_results)
        rag_results = "\n".join(search_results)
        full_text = f"{text}\n Related background information:\n{rag_results}"

        inputs_tokenized = tokenizer(
            full_text[:512], #capped at model capacity
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding="max_length"
        )

        if torch.cuda.is_available():
            inputs_tokenized = {k: v.to("cuda") for k, v in inputs_tokenized.items()}

        with torch.no_grad():
            output_ids = abs_model.generate(
                **inputs_tokenized,
                max_length=max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = trim(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        summaries.append(summary)

    with open(output_file, 'w') as output:
        for summ in tqdm(summaries, "Writing Summaries"):
            output.write(summ)
            output.write("\n")
    print(f"Summaries written to {output_file}")

def trim(summary: str) -> str:
    summ = list(summary)
    while summ != [] and (summ[-1] != '.' or summ[-2] == 'al'):
        summ.pop()
    return ''.join(summ)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--indices_file", type=str, default="data/wiki_embeddings.faiss")
    parser.add_argument("--metadata_file", type=str, default="data/mini_wiki_dataset.json")
    args = parser.parse_args()


    generate_summaries(args.model_path, args.input_file, args.output_file, args.indices_file, args.metadata_file)
