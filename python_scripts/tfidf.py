import argparse
import nltk
from tqdm import tqdm
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer #regexp_tokenize, word_tokenize, 
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer


def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def data_cleaner(doc):
    """A function to strip punctuation, strip stopwords, casefold, lemmatize,
    And part of speech tag words for clean data for modeling"""
    custom_stops = ['doi', 'figure', 'elife', 'et', 'al']
    sw = stopwords.words('english')
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in sw and word not in custom_stops]
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    return ' '.join(doc)

#initally removes things between parenthesis to keep sentence number stable in later process
def remove_between_parens(doc):
    doc = re.sub(r"\([^()]*\)|\[[^\]]*\]|\{[^}]*\}", "", doc)
    return doc

def sentence_value_creator(doc_sents: list, doc_vector, vectorizer) -> list: # List of sentences in a document -> list of tuples (index, sum)
    """ 
    A function to take in a single article at a time, split the article by sentences, clean those sentences, split each sentence by words,
    match each word with its vector, sum the vectors and returns a list of tuples (sentence index, vector sums)
        """
    sent_index_val_dict = [] # Stores tuples (sentence_index, sum)
    for i, sent in enumerate(doc_sents): # For every index and sentence in doc_sents
        clean_sent = data_cleaner(sent) # Clean the sentence
        c = 0.0 # The vector sum (starting at 0)
        sent_split = clean_sent.split() # Split sentence by words
        for word in sent_split: # For every word in the sentence
            if word in vectorizer.vocabulary_: # If the word is in the TF-IDF vocabulary
                vec_val = doc_vector[0, vectorizer.vocabulary_[word]] # Get the vector score from TF_IDF vectorizer
                c += vec_val # Add the vector score to the total sentence score
        score = float(c)/float(len(sent_split)) if len(sent_split) != 0 else 0# Normalize for sentence length = c/sent_length
        sent_index_val_dict.append((i,score)) # Append a tuple of (index, score) for the sentence 
    return sent_index_val_dict # Return the list of sentence (index, score) tuples

def main(args):
    df_clean = pd.read_csv(args.clean_csv) # Read the clean csv in as a pandas dataframe

    # Make the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range=(1,1))

    # Get the clean column from the dataframe
    clean_column = df_clean.clean.tolist()

    # Calculate the TF-IDF score for unigrams and bigrams using the clean data
    vector_scores = vectorizer.fit_transform(clean_column)

    summary_percent = args.summary_percent # Percentage of total document sentences to save as the summary
    doc_summaries = [] # List to store the document summaries
    doc_list = df_clean.article.tolist() # List of all the docs in the article column

    for doc_idx, doc in tqdm(enumerate(doc_list), "Summarizing Documents"): # For every document
        no_parens = remove_between_parens(doc) # Remove citations and other parentheses from the document
        doc_sents = nltk.sent_tokenize(no_parens) # Split the document into sentences
        doc_length = len(doc_sents) # Get the total number of sentences in the document
        top_num = int(summary_percent * doc_length) # Calculate the number of sentences to keep for the summary
        doc_vector = vector_scores[doc_idx]
        sent_scores = sentence_value_creator(doc_sents, doc_vector, vectorizer) # Pass vectorizer as argument and Get list of (index, score) pairs for all the document sentences
        sorted_scores = sorted(sent_scores, key=lambda x: x[1]) # Sort based on second tuple object; sort by score
        sorted_scores = sorted_scores[-top_num:] # Crop to just the top top_num sents
        sorted_sents = sorted(sorted_scores, key = lambda x: x[0]) # Sort the top sents by index so they are in the logical order
        doc_summary = "" # Save summary of the document as string
        for (index, score) in sorted_sents: # For every (index, score) pair
            sent = doc_sents[index] # Get the original sentence using the index
            doc_summary = doc_summary + sent + " " # Add the original sent to the full summary 
        doc_summaries.append(doc_summary) # Add the summary to the list of all summaries

    # Save the summaries as a new column in the elife train dataframe
    df_clean['tfidf_summary'] = doc_summaries

    # save the summaries to a csv file
    #df_clean.to_csv(args.output_csv, index=True)

    # save the summaries to a json file
    df_clean.to_json(args.output_json, orient="records", lines=True)

    # save the tfidf_summary column to a text file
    summary_list = df_clean.tfidf_summary.str.replace('\n', ' ', regex=False).tolist()

    with open(args.output_txt, 'w') as f:
        for summary in summary_list:
            f.write(summary + '\n')
    
    # with open('summaries.txt', 'w') as f:
    #     for summ in summary_list:
    #         f.write(summ.replace('\n','') + '\n')

if __name__ == "__main__":
    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_percent", type=float, default=0.4) # This is the percent of the document that calculates the summary length
    parser.add_argument(
        "--clean_csv", # This is the path to the csv with a clean column 
        type=str,
        default="data/df_elife_train_clean.csv",
    )
    parser.add_argument(
        "--output_json", # This is where the new dataframe with a summaries column will be saved
        type=str,
        default="data/elife_train_summaries.json"
    )
    parser.add_argument(
        "--output_txt", #This is where the summaries will be saved to as a txt document
        type=str,
        default="data/elife_train_summaries.txt"
    )
    args = parser.parse_args()

    main(args)


    """

python ./tfidf.py \
  --summary_percent .4 \
  --clean_csv validation/elife_clean_50.csv\
  --output_json validation/elife_clean_tfidf_val_50.json\
  --output_txt validation/elife_clean_tfidf_val_50.txt

  python ./tfidf.py \
  --summary_percent .4 \
  --clean_csv data/validation/plos_clean_10.csv \
  --output_json data/validation/plos_clean_val_10.json\
  --output_txt data/validation/plos_clean_10.txt

    """