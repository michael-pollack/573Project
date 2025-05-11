import argparse
import nltk
import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer #regexp_tokenize, word_tokenize, 

nltk.download('stopwords')
nltk.download('wordnet')


#initally removes things between parenthesis to keep sentence number stable in later process
def remove_between_parens(doc):
    doc = re.sub(r"\([^()]*\)|\[[^\]]*\]|\{[^}]*\}", "", doc)
    return doc

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


if __name__ == "__main__":
    # argparse logic

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_folder", # This is the path to the folder containing JSON data files
        type=str,
        default="data/Elife/",
    )
    parser.add_argument(
        "--output_folder", # This is where the clean dataframes will be saved
        type=str,
        default="data/cleaned/"
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    for filename in os.listdir(args.json_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(args.json_folder, filename)
            output_csv = os.path.join(args.output_folder, filename.replace(".json", "_clean.csv"))
            output_json = os.path.join(args.output_folder, filename.replace(".json", "_clean.json"))

            orig_df = pd.read_json(input_path, lines=True)

            # code to remove documents of more than 11000 words
            new_df = orig_df[orig_df['article'].str.len() < 110000 ].copy()

            print(f"{filename}: {new_df.shape}")
            articles_no_outliers = new_df.article.tolist()
            print(f"{filename}: {len(articles_no_outliers)}")

            # run data through data parens cleaning function
            no_parens_corpus = []
            no_parens_corpus.extend(
                remove_between_parens(doc) for doc in articles_no_outliers
            )
            print(f"{filename}: done cleaning parentheses")

            # runs data through data cleaning function
            clean_corpus = []
            clean_corpus.extend(data_cleaner(doc) for doc in no_parens_corpus)
            print(f"{filename}: done cleaning data")

            # runtime message:
            # A value is trying to be set on a copy of a slice from a DataFrame.
            #Try using .loc[row_indexer,col_indexer] = value instead
            new_df.loc[:, 'clean'] = clean_corpus

            new_df.to_csv(output_csv, index=True)
            new_df.to_json(output_json, orient="records", lines=True)
            print(f"{filename}: saved to {output_csv} and {output_json}")

    """
    python data_cleaning.py   \
    --json_folder data/PLOS_json   \
    --output_folder data/PLOS/cleaned/

    python data_cleaning.py   \
    --json_folder data/elife_json   \
    --output_folder data/Elife/cleaned/

    """