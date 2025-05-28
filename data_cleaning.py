import argparse
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer #regexp_tokenize, word_tokenize, 
import pandas as pd
import re
from tqdm import tqdm

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
        "--parquet_path", # This is the path to the parquet data file
        type=str,
        default="data/Elife/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--output_csv", # This is where the clean dataframe will be saved
        type=str,
        default="data/df_elife_train_clean.csv"
    )
    parser.add_argument(
        "--num_lines", # Number of lines to process (default: all)
        type=int,
        default=None
    )
    args = parser.parse_args()

    orig_df = pd.read_parquet(args.parquet_path)

    # If num_lines is set, select only the first num_lines rows
    if args.num_lines is not None:
        new_df = orig_df.head(args.num_lines).copy()
    else:
        new_df = orig_df.copy()

    # code to remove documents of more than 11000 words
    # new_df = orig_df[orig_df['article'].str.len() < 110000 ]
    # print(new_df.shape)
    # articles = new_df.article.tolist()
    # print(len(articles))

    articles = new_df['article'].tolist()

    # run data through data parens cleaning function
    no_parens_corpus = []
    for doc in tqdm(articles, "Cleaning parenthesis..."):
        no_parens_corpus.append(remove_between_parens(doc))
    print("done cleaning parentheses")

    # runs data through data cleaning function
    clean_corpus = []
    for doc in tqdm(no_parens_corpus, "Cleaning data..."):
        clean_corpus.append(data_cleaner(doc))
    print("done cleaning data")
    new_df['clean'] = clean_corpus

    # if '.' in args.output_csv:
    #     name, ext = args.output_csv.rsplit('.', 1)
    # else:
    #     name, ext = args.output_csv, ''

    # filename = f"{name}_{args.num_lines}.{ext}"

    # new_df.to_csv(filename, index=True)
    new_df.to_csv(args.output_csv, index=True)

    """
    python /home/jen/573Project-1/data_cleaning.py   \
    --parquet_path data/PLOS/validation-00000-of-00001.parquet  \
    --output_csv data/validation/plos_clean.csv \
    --num_lines 10

    python /home/jen/573Project-1/data_cleaning.py   \
    --parquet_path data/Elife/validation-00000-of-00001.parquet  \
    --output_csv validation/elife_clean.csv \
    --num_lines 100
    """