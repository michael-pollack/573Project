import argparse
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
import pandas as pd
import re
import string

# THIS FUNCTION MAKES TWO VERSIONS OF THE DOC: ONE WITH ALL LEMMAS AND ONE THE RAW VERSION
def make_lemma_doc(og_doc):
    doc = og_doc.split()
    doc = [word.lower() for word in doc]
    pos_doc = pos_tag(doc)
    lemmatizer = WordNetLemmatizer() 
    lemmas = [(lemmatizer.lemmatize(word[0]), word[1]) for word in pos_doc]
    return doc, lemmas

# THIS FUNCTION RETURNS A BOOLEAN. TRUE=SKIP THE CURRENT WORD
def skip(raw, index):
    skip = False # Set skip to false at first
    # Check after words

    after_phrase = [] # To save the following phrase
    for i in range(1, max_after_skip + 1): # Look at next words
        new_index = index + i # Index of the neighbor
        if new_index < len(raw): # Make sure inex in range
            after_phrase.append(raw[new_index])
            if after_phrase in after_skip_phrases: # If it's a skip phrase
                skip = True # Change skip to true
                break # Leave the loop
        else: # If the index is out of range then so will the next one
            break 
    
    if skip == True: # If the skip is true already then there's no point in continuing
        return skip
    
    prev_phrase = [] # TO save the previous phrase
    for i in range(1, max_prev_skip + 1): # Look at previous words
        new_index = index - i 
        if new_index > -1:
            prev_phrase = [raw[index-i]] + prev_phrase # Update the prev phrase
            if prev_phrase in prev_skip_phrases:
                skip = True
                break
        else:
            break

    return skip 

# THIS FUNCTION FINDS WORDS THAT NEED TO BE REPLACED AND REPLACES THEM IN THE RAW VERSION
def find_and_replace(raw, lemma):
    """function for matching the location in the raw doc with contents of the lemma doc at the parallel location"""
    for word_id in range(0, len(lemma)): # For every word in the document
        #if skip(raw, word_id) == False: # If not a skip word
        curr_lemma_pos = lemma[word_id] # Get the current lemma,POS pair
        curr_lemma = curr_lemma_pos[0] # Get the current word's lemma
        if curr_lemma in term_dict: # If the lemma is in the dictionary
            replace_options = term_dict[curr_lemma] # Get the list of replacement options
            for layterm_tag in replace_options: # For every layterm, POS pair in the options
                if curr_lemma_pos[1] == layterm_tag[1]: # If the current POS matches the layterm POS
                    raw[word_id] = layterm_tag[0] # Replace the word in the raw doc with the new replacement

# THIS FUNCTION CONVERTS A SENTENCE IN LIST FORM TO A WELL-FORMATTED STRING
def format_summary(og_summ):
    first_word = og_summ[0] # Get first word of the summary
    new_summ = first_word.capitalize() # Capitalize the first word
    for index in range(1, len(og_summ)): # For everything after the first word
        prev_word = og_summ[index - 1] # Get the previous word to test if it's a period
        curr_word = og_summ[index] # Get the current word

        if prev_word == ".": # If the last word is a period, capitalize the next word
            curr_word = curr_word.capitalize() 

        if curr_word in string.punctuation or prev_word == "-" or prev_word == "/": 
            # If the current word is actually punctuation or prev word is a dash or slash
            new_summ = new_summ + curr_word # Add without space
        else:
            new_summ = new_summ + " " + curr_word 
    return new_summ


if __name__ == "__main__":
    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries_json", # Path to the summaries json
        type=str
    )
    parser.add_argument(
        "--output_json", # This is the path to the output
        type=str
    )
    parser.add_argument(
        "--output_txt", # Path to the text file to save the final summaries to
        type=str
    )
    args = parser.parse_args()

    # Read the summaries, skip_phrases, and layterm_dictionary as dataframes
    #summary_df = pd.read_json(args.summaries_json)
    summary_df = pd.read_csv(args.summaries_json)
    skip_phrases_df = pd.read_csv('dictionary/skip_phrases.csv')

    after_skip_phrases = skip_phrases_df['Phrase_after_medword'] # List of phrases AFTER the word that makes it skipable
    after_skip_phrases = [str(phrase).split(" ") for phrase in after_skip_phrases if isinstance(phrase,str)] # Split every phrase into list
    prev_skip_phrases = skip_phrases_df['Phrase_before_medword'] # List of phrases BEFORE the word that makes it skipable
    prev_skip_phrases = [str(phrase).split(" ") for phrase in prev_skip_phrases if isinstance(phrase,str)] # Split every phrase into a list

    max_after_skip = len(max(after_skip_phrases, key=len)) # Get max length of after_phrase
    max_prev_skip = len(max(prev_skip_phrases, key=len)) # Get max length of prev_phrase

    # ADD CODE TO MAKE TERM_DICT FROM DATAFRAME
    lexicon_df = pd.read_csv('dictionary/lexicon.csv') #read in lexicon csv
    lexicon_df_clean = lexicon_df.drop(['Lemma', 'POS', 'bib_num'], axis = 1) #drop uneeded columns
    df_lexicon_clean = lexicon_df_clean.dropna() #drop extra rows
    med_terms = df_lexicon_clean['Forms'].tolist() #get list of med terms
    lemmatizer = WordNetLemmatizer() #make lemmatizer
    lemma_list = []
    for term in med_terms:
        lemma_list.append(lemmatizer.lemmatize(term)) #lemmatize all med words
    check = pd.Series(lemma_list) 
    df_lexicon_clean['lemma'] = check.values #add lemmas back to df
    tup_obj = list(zip(df_lexicon_clean['layman'], df_lexicon_clean['Tag'], df_lexicon_clean['lemma'])) #make a tuple object in the form of (layman term, POS tag, lemma)
    term_dict = {}
    for tup in tup_obj:#create dict
        if tup[2] not in term_dict:
            term_dict[tup[2]] = [(tup[0], tup[1])]
        else:
            term_dict[tup[2]].append((tup[0], tup[1]))
    
    
    og_summ_list = summary_df.summary.tolist() # Get just the summaries from the previous dataframe
    layterm_summaries = [] # Save the replaced summaries

    for summ in og_summ_list: # For every summary in the data (currently elife_train, can be changed above)
        raw, lemma = make_lemma_doc(summ) # Make raw and lemma versions of the summary
        find_and_replace(raw, lemma) # Do layterm injection
        layterm_summaries.append(raw) # Add the new summary to the list

    # Try converting the layterm summaries to better format
    final_summaries = [] # Collect final formatted summaries
    for summ in layterm_summaries: 
        new_summ = format_summary(summ)
        final_summaries.append(new_summ)

    summary_df['final_summary'] = final_summaries # Add final summaries as a column in the df
    summary_df.to_json(args.output_json, orient="records", lines=True) # Save df to the output_json?

    summary_list = summary_df.final_summary.str.replace('\n', ' ', regex=False).tolist()

    with open(args.output_txt, 'w') as f:
        for summary in summary_list:
            f.write(summary + '\n')