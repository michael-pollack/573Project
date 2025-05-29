from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import re
import requests
import time
import json
from sentence_transformers import SentenceTransformer
import faiss

def load_articles(file: str) -> list:
    with open(file, 'r') as file:
        df = pd.read_csv(file)
        clean_column = df.clean.tolist()
        return clean_column
    
def remove_between_parens(doc: str) -> str:
    doc = re.sub(r"\([^()]*\)|\[[^\]]*\]|\{[^}]*\}", "", doc)
    return doc

def get_keywords(articles: list[str]) -> list:
    vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range=(1,1))
    vector_store = vectorizer.fit_transform(articles)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = vector_store.toarray()
    global_scores = np.sum(tfidf_matrix, axis=0)
    ranked_terms = sorted(zip(feature_names, global_scores), key=lambda x: x[1], reverse=True)
    top_keywords = [term for term, score in ranked_terms[:500]]
    return top_keywords

def scrape_wikipedia(keywords: list) -> list:
    wikipedia_articles = []
    for keyword in tqdm(keywords, "Retrieving Articles"):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{keyword}"
        r = requests.get(url)

        if r.status_code == 200:
            data = r.json()
            if "extract" in data and len(data["extract"]) > 300:
                wikipedia_articles.append({
                    "title": data.get("title"),
                    "text": data.get("extract"),
                    "source_keyword": keyword
                })
        time.sleep(0.5)
    return wikipedia_articles

def embed(wikis: json, embedding_file: str) -> None:
    texts = [item["text"] for item in wikis]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, embedding_file)

def main(args):
    articles = []
    for file in args.files:
        subset = load_articles(file)
        for a in subset:
            articles.append(a)
    keywords = get_keywords(articles)
    wikis = scrape_wikipedia(keywords)
    with open(args.summaries, 'w') as file:
        json.dump(wikis, file)
    embed(wikis, args.embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--summaries', type=str, default="data/mini_wiki_dataset.json")
    parser.add_argument('--embeddings', type=str, default="data/wiki_embeddings.faiss")
    args = parser.parse_args()
    main(args)

