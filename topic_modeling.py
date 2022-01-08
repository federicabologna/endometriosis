import os
import re
from datetime import datetime
import json
import pandas as pd
import tomotopy as tp
import spacy
import numpy as np


def cleaning_docs(df, docs_file):

    docs_d = {}  # dictionary of documents to perform topic modeling on, here documents are posts' clean sentences
    stopwords = set([line.strip() for line in open("stoplist_final.txt")])  # creating list of stop words
    nlp = spacy.load("en_core_web_sm")  # loading the spacy language model
    lemmatizer = nlp.get_pipe("lemmatizer")  # getting the spacy lemmatizer

    for index, row in df.iterrows():  # iterating over posts
        post_id = row['concat_id']  # id of the post, e.g. 'Endo_xyz'
        post = row['selftext']  # textual content of the post
        post_url = row['url']  # url of the post
        doc = nlp(post)  # processing the post: tokenizing and  lemmatizing
        sent_n = 0  # counter of sentences in the post
        for sent in doc.sents:
            sent_id = f'{post_id}_{sent_n}'  # creating an id for each post' sentence
            sent_n += 1
            clean_sent = []  # sentences represented as list of lemmatized tokens
            for token in sent:
                lemma = token.lemma_
                clean_lemma = re.sub(r'[^\w\s\d]', '', lemma)  # remove punctuation from tokens
                clean_lemma = re.sub(r'[\n+\s+]', '', clean_lemma)  # remove empty spaces and new lines
                if clean_lemma and clean_lemma not in stopwords:  # remove empty tokens/stopwords
                    clean_sent.append(clean_lemma)  # adding clean lemma to the clean sentence's list
            if len(clean_sent) > 4:  # exclude sentences that are less than 5 words
                # add sentence id and clean sentence, og sentence, url to the dictionary as a key,value pair
                # the clean - tokenized and lemmatized - sentences are our documents
                docs_d[sent_id] = [clean_sent, sent.text, post_url]

    with open(docs_file, 'w') as jsonfile:  # creating a file with the dict of documents to topic model
        json.dump(docs_d, jsonfile)

    return docs_d


def perform_tm(s_ids, corpus, n_topics, rm_top, topwords_file):

    # setting and loading the LDA model
    lda_model = tp.LDAModel(k=n_topics,  # number of topics in the model
                            min_df=3,  # remove words that occur in less than n documents
                            rm_top=rm_top)  # remove n most frequent words
    vocab = set()
    for doc in corpus:
        lda_model.add_doc(doc)  # adding document to the model
        vocab.update(doc)  # adding tokens in the document to the vocabulary
    print('Num docs:{}'.format(len(lda_model.docs)))
    print("Vocabulary Size: {}".format(len(list(vocab))))
    print('Removed Top words: ', lda_model.removed_top_words)

    iterations = 10
    for i in range(0, 100, iterations):  # train model 10 times with 10 iterations at each training = 100 iterations
        lda_model.train(iterations)
        print(f'Iteration: {i}\tLog-likelihood: {lda_model.ll_per_word}')

    #TOP WORDS
    num_top_words = 10  # number of top words to print for each topic
    with open(topwords_file, "w", encoding="utf-8") as file:
        file.write(f"\nTopics in LDA model: {n_topics} topics {rm_top} removed top words\n\n")  # write settings of the model in file
        topic_individual_words = []
        for topic_number in range(0, n_topics):  # for each topic number in the total number of topics
            topic_words = ' '.join(  # string of top words in the topic
                word for word, prob in lda_model.get_topic_words(topic_id=topic_number, top_n=num_top_words))  # get_topic_words is a tomotopy function that returns a dict of words and their probabilities
            topic_individual_words.append(topic_words.split(' '))  # append list of the topic's top words for later
            file.write(f"Topic {topic_number}\n{topic_words}\n\n")  # write topic number and top words in file
        print(topic_individual_words)

    #TOPIC DISTRIBUTIONS
    topic_distributions = [list(doc.get_topic_dist()) for doc in lda_model.docs]  # list of lists of topic distributions for each document, get_topic_dist() is a tomotopy function
    topic_results = []
    for topic_distribution in topic_distributions:  # list of dicts of documents' topic distributions to convert into pandas' dataframe
        topic_results.append({'topic_distribution': topic_distribution})
    df = pd.DataFrame(topic_results, index=s_ids)  # df where each row is the list of topic distributions of a document, s_ids are the ids of the sentences
    column_names = [f"Topic {number} {' '.join(topic[:4])}" for number, topic in enumerate(topic_individual_words)]  # create list of column names from topic numbers and top words
    df[column_names] = pd.DataFrame(df['topic_distribution'].tolist(), index=df.index)  # df where topic distributions are not in a list and match the list of column names
    df = df.drop('topic_distribution', axis='columns')  # drop old topic distributions' column
    dominant_topic = np.argmax(df.values, axis=1)  # get dominant topic for each document
    df['dominant_topic'] = dominant_topic  # add column for the dominant topic in the document

    return df


def main(subreddit):

    reddit_df = pd.read_csv(os.path.join('data', f'{subreddit}.csv'))  # path of csv with reddit data
    tomo_folder = os.path.join('output', 'topic_modeling')  # results' folder
    if not os.path.exists(tomo_folder):  # create folder if it doesn't exist
        os.makedirs(tomo_folder)

    clean_docs_file = os.path.join(tomo_folder, f'{subreddit}.json')  # file with clean documents - here, post sentences
    if not os.path.exists(clean_docs_file):  # if clean documents file doesn't exist, executes data cleaning
        start = datetime.now()
        print("Data Cleaning...")
        docs_dict = cleaning_docs(reddit_df, clean_docs_file)
        print(f'{str(datetime.now())}________________{str(datetime.now() - start)}\n')  # print timing of data cleaning
    else:
        with open(clean_docs_file) as json_file:
            docs_dict = json.load(json_file)
    doc_ids = [doc_id for doc_id in docs_dict.keys()]  # get list of document ids for later
    clean_docs = [sent_url[0] for sent_url in docs_dict.values()]  # get list of clean documents for later
    og_docs = [[sent_url[1]] for sent_url in docs_dict.values()]  # get list of original documents for later
    #doc_urls = [sent_url[2] for sent_url in docs_dict.values()]  # get list of document urls for later

    for num_topics in [7, 10, 15]:  # for number of topics - for loops allow to run multiple models with different settings with one execution
        for rm_frequent in [15]:  # for number of most frequent words to remove

            txt_topwords = os.path.join(tomo_folder, f'{subreddit}-{num_topics}_{rm_frequent}.txt')  # path for top words file
            csv_dtm = os.path.join(tomo_folder, f'{subreddit}-{num_topics}_{rm_frequent}.csv')  # path for doc-topic matrix file

            if not os.path.exists(txt_topwords) or not os.path.exists(csv_dtm):  # if result files don't exist, performs topic modeling
                start = datetime.now()
                print("Performing Topic Modeling...")
                lda_dtm = perform_tm(doc_ids, clean_docs, num_topics, rm_frequent, txt_topwords)
                lda_dtm['sent'] = og_docs  # add original sentences to doc-topic df
                #lda_dtm['post_url'] = doc_urls  # add urls of the posts of the sentences to matrix
                lda_dtm.to_csv(csv_dtm)  # convert doc-topic df in csv file
                print(f'{str(datetime.now())}____Topic modeling {num_topics}, {rm_frequent} time:____{str(datetime.now() - start)}\n')  # print timing of topic modeling


if __name__ == '__main__':
    main('endo+endometriosis')  # name of the subreddit file
