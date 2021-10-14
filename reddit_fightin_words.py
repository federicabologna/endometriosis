import os
from datetime import datetime
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer as CV
import string

exclude = set(string.punctuation)


def cleaning_docs(df, df_ids, df_docs):
    doc_ids = []
    clean_docs = []

    lemmatizer = WordNetLemmatizer()
    n_tokxdoc = []
    vocab = set()
    for index, row in df.iterrows():
        doc_id = row[df_ids]
        doc = row[df_docs]
        clean_doc = []
        lowercase = doc.lower()
        tokenized = nltk.word_tokenize(lowercase)  # list of tokens
        for token in tokenized:
            token = re.sub(r'[^\w\s\d]', '', token)  # remove punctuation from token
            #if token and token not in stopwords:  # if token is not empty and is not in stopwords
            token = lemmatizer.lemmatize(token)  # lemmatize token
            clean_doc.append(token)
            vocab.add(token)
        n_tokxdoc.append(len(clean_doc))
        clean_doc = ' '.join(clean_doc)
        if clean_doc:
            doc_ids.append(doc_id)
            clean_docs.append(clean_doc)

    print("Number of Documents: {}".format(len(clean_docs)))
    print("Mean Number of Words per Document: {}".format(np.mean(n_tokxdoc)))
    print("Vocabulary Size: {}".format(len(list(vocab))))

    return doc_ids, clean_docs


# Monroe's fightin' words calculation
# Based on Jack Hessel's and Xanda Schofield's fightin words implementations

def basic_sanitize(in_string):
    '''Returns a very roughly sanitized version of the input string.'''
    return_string = ''.join([ch for ch in in_string if ord(ch) < 128 and ch not in exclude]).lower()
    return_string = ' '.join(return_string.split())
    return return_string


def bayes_compare_language(l1, l2, output_path, ngram=1, prior=.01, cv=None, sig_val=2.573):
    '''
    Arguments:
    - l1, l2; a list of strings from each language sample
    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
    2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.
    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    if cv is None and type(prior) is not float:
        print("If using a non-uniform prior:")
        print("Please also pass a count vectorizer with the vocabulary parameter set.")
        quit()
    l1 = [basic_sanitize(l) for l in l1]
    l2 = [basic_sanitize(l) for l in l2]
    if cv is None:
        cv = CV(decode_error = 'ignore', min_df = 10, max_df = .5, ngram_range=(1,ngram),
                binary = False,
                max_features = 15000)
    counts_mat = cv.fit_transform(l1+l2).toarray()
    # Now sum over languages...
    vocab_size = len(cv.vocabulary_)
    print("Vocab size is {}".format(vocab_size))
    if type(prior) is float:
        priors = np.array([prior for i in range(vocab_size)])
    else:
        priors = prior
    z_scores = np.empty(priors.shape[0])
    count_matrix = np.empty([2, vocab_size], dtype=np.float32)
    count_matrix[0, :] = np.sum(counts_mat[:len(l1), :], axis = 0)
    count_matrix[1, :] = np.sum(counts_mat[len(l1):, :], axis = 0)
    a0 = np.sum(priors)
    n1 = 1.*np.sum(count_matrix[0,:])
    n2 = 1.*np.sum(count_matrix[1,:])
    print("Comparing language...")
    for i in range(vocab_size):
        #compute delta
        term1 = np.log((count_matrix[0,i] + priors[i])/(n1 + a0 - count_matrix[0,i] - priors[i]))
        term2 = np.log((count_matrix[1,i] + priors[i])/(n2 + a0 - count_matrix[1,i] - priors[i]))        
        delta = term1 - term2
        #compute variance on delta
        var = 1./(count_matrix[0,i] + priors[i]) + 1./(count_matrix[1,i] + priors[i])
        #store final score
        z_scores[i] = delta/np.sqrt(var)
    index_to_term = {v: k for k, v in cv.vocabulary_.items()}
    sorted_indices = np.argsort(z_scores)
    return_list = [(index_to_term[i], z_scores[i]) for i in sorted_indices]
    

    # plotting z scores and frequencies
    x_vals = count_matrix.sum(axis=0)
    y_vals = z_scores
    sizes = abs(z_scores) * 2
    neg_color, pos_color, insig_color = ('blue', 'purple', 'grey')
    colors = []
    annots = []
    for i, y in enumerate(y_vals):
        if y > sig_val:
            colors.append(pos_color)
            annots.append(index_to_term[i])
        elif y < -sig_val:
            colors.append(neg_color)
            annots.append(index_to_term[i])
        else:
            colors.append(insig_color)
            annots.append(None)

    plt.figure(figsize=(18,18))
    fig, ax = plt.subplots()
    ax.scatter(x_vals, y_vals, c=colors, linewidth=0, alpha = 0.3)

    for i, annot in enumerate(annots):
        if annot is not None:
            if np.abs(y_vals[i]) > 4:
                ax.annotate(annot, (x_vals[i], y_vals[i]), color='black', fontsize=6)

    ax.set_xscale('log')
    
    plt.savefig(os.path.join(output_path, 'fightin_words.pdf'))

    return return_list


def main(subreddit_1, subreddit_2):

    output_path = os.path.join(os.getcwd(), 'output', f'{subreddit_1}'+ '_vs_' + f'{subreddit_2}')

    # GET TWO CLASSES OF DOCUMENTS TO COMPARE
    document_1 = os.path.join(os.getcwd(), 'reddit', 'posts', f'{subreddit_1}.csv')
    document_2 = os.path.join(os.getcwd(), 'reddit', 'posts', f'{subreddit_2}.csv')

    df_1 = pd.read_csv(document_1, index_col=0)
    df_2 = pd.read_csv(document_2, index_col=0)

    #PRE PROCESS DATA
    start = datetime.now()
    post_ids_1, clean_posts_1 = cleaning_docs(df_1, 'id', 'selftext')
    post_ids_2, clean_posts_2 = cleaning_docs(df_2, 'id', 'selftext')
    print(str(datetime.now()) + f'Data cleaning time: ' + str(datetime.now() - start))

    list_1 = clean_posts_1
    list_2 = clean_posts_2

    # PERFORM FIGHTIN WORDS ON TWO CLASSES
    output_list = bayes_compare_language(list_1, list_2, output_path)

    z_scores_df = pd.DataFrame(output_list)
    z_scores_df.to_csv(os.path.join(output_path, 'z_scores_fightin_words.csv'))

    return output_list

# RUNNING MAIN WILL CLEAN TWO DATASETS, PERFORM FIGHTIN WORDS, AND OUTPUT A CSV WITH WORD-ZSCORE PAIR AND PLOTTED RESULTS

if __name__ == '__main__':
    main('endometriosis', 'pcos')


