

# Make the TF-IDF vectorizer
vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range=(1,1))

# Get the clean column from the elife dataframe
elife_train_clean = df_elife_train_clean.clean.tolist()

# Calculate the TF-IDF score for unigrams and bigrams using the clean data
elife_t_c = vectorizer.fit_transform(elife_train_clean)

def sentence_value_creator_2(doc_sents: list, vect_obj) -> list: # List of sentences in a document -> list of tuples (index, sum)
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
                vec_val = vect_obj[i, vectorizer.vocabulary_[word]] # Get the vector score from TF_IDF vectorizer
                c += vec_val # Add the vector score to the total sentence score
        sent_index_val_dict.append((i,float(c))) # Append a tuple of (index, score) for the sentence 
    return sent_index_val_dict # Return the list of sentence (index, score) tuples

summary_percent = 0.4 # Percentage of total document sentences to save as the summary
doc_summaries = [] # List to store the document summaries
doc_list = df_elife_train_clean.article.tolist() # List of all the docs in elife train


for doc in doc_list: # For every document
    no_parens = remove_between_parens(doc) # Remove citations and other parentheses from the document
    doc_sents = nltk.sent_tokenize(no_parens) # Split the document into sentences
    doc_length = len(doc_sents) # Get the total number of sentences in the document
    top_num = int(summary_percent * doc_length) # Calculate the number of sentences to keep for the summary
    sent_scores = sentence_value_creator_2(doc_sents, elife_t_c) # Get list of (index, score) pairs for all the document sentences
    sorted_scores = sorted(sent_scores, key=lambda x: x[1]) # Sort based on second tuple object; sort by score
    sorted_scores = sorted_scores[-top_num:] # Crop to just the top top_num sents
    sorted_sents = sorted(sorted_scores, key = lambda x: x[0]) # Sort the top sents by index so they are in the logical order
    doc_summary = "" # Save summary of the document as string
    for (index, score) in sorted_sents: # For every (index, score) pair
        sent = doc_sents[index] # Get the original sentence using the index
        doc_summary = doc_summary + sent + " " # Add the original sent to the full summary 
    doc_summaries.append(doc_summary) # Add the summary to the list of all summaries

# Save the summaries as a new column in the elife train dataframe
df_elife_train_clean['tfidf_summary'] = doc_summaries

# save the summaries to a csv file
df_elife_train_clean.to_csv('data/elife_summaries.csv', index=True)

# save the tfidf_summary column to a text file
with open('data/elife_summaries.txt', 'w') as f:
    for summary in df_elife_train_clean['tfidf_summary']:
        f.write(summary + '\n')

with open('elife_summaries.txt', 'w') as f:
    for summ in summary_list:
        f.write(summ.replace('\n','') + '\n')