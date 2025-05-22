import pandas as pd
import argparse
from transformers import pipeline
import tqdm
import torch
import time

CHUNK_CHAR_LIMIT = 3000  # conservative limit for huggingface models
MODEL_NAME = "Falconsai/medical_summarization"

# Load summarization pipeline
device = 0 if torch.cuda.is_available() else -1
#summarizer = pipeline("summarization", model=MODEL_NAME, device=device)

summarizer = pipeline("summarization", model="Falconsai/medical_summarization", device=device, torch_dtype=torch.float16)


def chunk_text(text, max_words=1000):
    words = text.strip().split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]

def summarize_chunk(text, prompt=None):
    text = text.strip()
    word_count = len(text.split())

    # Optionally prepend a prompt to guide the summarizer
    if prompt:
        text = f"{prompt}\n{text}"

    # Skip junk input
    if word_count < 10 or not any(char.isalnum() for char in text):
        print(f"[SKIP] Chunk too short or non-alphanumeric: '{text[:40]}...'")
        return text

    try:
        # Use max based on word count but cap it to input length
        max_len = min(512, max(30, int(word_count * 0.7)))
        min_len = max(10, int(word_count * 0.3))
        if min_len >= max_len:
            min_len = int(max_len * 0.5)  # fallback

        result = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )

        return result[0]['summary_text'].replace("\n", " ").strip()

    except IndexError as e:
        print(f"[ERROR] IndexError on chunk: '{text[:50]}...' (words: {word_count})")
        return "[ERROR: IndexError]"

    except Exception as e:
        print(f"❌ Summarization failed: {e}\nChunk was: '{text[:50]}...'")
        return "[ERROR]"


def summarize_article(text, plain_language=False, batch_size=32, prompt=None): # batch_size=8
    chunks = chunk_text(text)
    summaries = []

    # Batch summarization for efficiency
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            # Use max/min length based on the first chunk in the batch
            word_count = len(batch[0].split())
            max_len = min(512, max(30, int(word_count * 0.7)))
            min_len = max(10, int(word_count * 0.3))
            if min_len >= max_len:
                min_len = int(max_len * 0.5)  # fallback

            # If a prompt is provided, prepend it to each chunk in the batch
            if prompt:
                batch = [f"{prompt}\n{chunk}" for chunk in batch]

            results = summarizer(
                batch,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            for result in results:
                chunk_summary = result['summary_text'].replace("\n", " ").strip()
                if "ERROR" in chunk_summary:
                    print("[WARN] Skipping bad chunk during summary.")
                    continue
                summaries.append(chunk_summary)
        except Exception as e:
            print(f"❌ Summarization failed for batch: {e}")
            for _ in batch:
                summaries.append("[ERROR]")

        #sleep(1)

    if not summaries:
        return "[ERROR: all chunks failed]"

    combined = " ".join(summaries)

    if plain_language:
        # Use a final summarizer pass — but check it's not empty
        final_prompt = f"Explain this in plain language: {combined}"
        plain = summarize_chunk(final_prompt)
        return plain if "ERROR" not in plain else combined
    else:
        return combined


def main():
    parser = argparse.ArgumentParser(description="Summarize medical articles from a Parquet file using a Hugging Face model.")
    parser.add_argument("--input_file", required=True, help="Path to the input Parquet file.")
    parser.add_argument("--text_column", required=True, help="Column containing the article text.")
    parser.add_argument("--output_file", required=True, help="Path to output text file.")
    parser.add_argument("--plain_language", action="store_true", help="If set, rewrites the final summary in plain language.")
    args = parser.parse_args()

    df = pd.read_json(args.input_file, lines=True)  # or lines=False depending on your file

    print(f"Loaded {len(df)} articles.")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, text in tqdm.tqdm(enumerate(df[args.text_column]), total=len(df), desc="Summarizing"):
            t0 = time.time()
            if not isinstance(text, str) or not text.strip():
                f.write("EMPTY\n")
                continue
            # set custom prompt
            #prompt = "Summarize the following scientific article:"
            prompt = "You are a medical writer. Your task is to write a plain-language summary of the following biomedical article, aimed at the general public. Avoid jargon, explain technical terms, and focus on the main findings and significance. Cover all the main points, using similar phrasing,"
            summary = summarize_article(text, plain_language=args.plain_language, prompt=prompt)
            f.write(summary + "\n")
            print(f"Article {i} took {time.time() - t0:.2f} seconds")


    print(f"\n✅ All summaries saved to: {args.output_file}")

if __name__ == "__main__":
    main()


"""
python summarize_hf.py \
  --input_file validation/elife_clean_tfidf_val_50.json \
  --text_column article \
  --output_file data/results/elife_generated_summaries_50.txt
    \
  --plain_language

 python summarize_hf.py \
  --input_file data/validation/plos_clean_val.json \
  --text_column article \
  --output_file data/results/plos.txt
    \
  --plain_language
  
"""