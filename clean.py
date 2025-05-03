# load a parquet file
# clean it up and remove stop words
# run tf-idf on it
# save it as a text file

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def load_parquet(file_path):
    df = pd.read_parquet(file_path)
    if 'article' in df.columns:
        df = df[['article']]
    return df

def clean_text(text):   

    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return filtered_text

def run_tfidf(texts):
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer.get_feature_names_out()

def save_as_text(tfidf_matrix, feature_names, output_file): 
    
    with open(output_file, 'w') as f:
        for i in range(tfidf_matrix.shape[0]):
            f.write(f"Document {i}:\n")
            for j in range(tfidf_matrix.shape[1]):
                if tfidf_matrix[i, j] > 0:
                    f.write(f"{feature_names[j]}: {tfidf_matrix[i, j]}\n")
            f.write("\n")

def main(input_file, output_file):  
    df = load_parquet(input_file)
    # Use the correct column name ('articles') instead of 'text'
    df['cleaned_text'] = df['article'].apply(clean_text)
    df['filtered_text'] = df['cleaned_text'].apply(remove_stopwords)
    tfidf_matrix, feature_names = run_tfidf(df['filtered_text'])
    save_as_text(tfidf_matrix, feature_names, output_file)
if __name__ == "__main__":      
    
    input_file = 'data/Elife/test-00000-of-00001.parquet'
    output_file = 'test_summaries.txt'
    
    main(input_file, output_file)