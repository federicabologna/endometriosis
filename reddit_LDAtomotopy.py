import os
import re
from datetime import datetime
import json
import pandas as pd
import tomotopy as tp
import string
import spacy
import numpy as np
import pyLDAvis


def cleaning_docs(df, docs_file):

    docs_d = {}
    stopwords = set([line.strip() for line in open("stoplist_final.txt")])
    nlp = spacy.load("en_core_web_sm")
    lemmatizer = nlp.get_pipe("lemmatizer")

    vocab = set()
    for index, row in df.iterrows():
        post_id = row['concat_id']
        post = row['selftext']
        post_url = row['url']
        doc = nlp(post)
        sent_n = 0
        for sent in doc.sents:
            sent_id = f'{post_id}_{sent_n}'
            sent_n += 1
            clean_sent = []
            for token in sent:
                lemma = token.lemma_
                clean_lemma = re.sub(r'[^\w\s\d]', '', lemma)
                clean_lemma = re.sub(r'[\n+\s+]', '', clean_lemma)
                if clean_lemma and clean_lemma not in stopwords:
                    clean_sent.append(clean_lemma)
            if clean_sent:
                vocab.update(clean_sent)
                docs_d[sent_id] = [clean_sent, post_url]

    with open(docs_file, 'w') as jsonfile:
        json.dump(docs_d, jsonfile)

    print("Number of Documents: {}".format(len(docs_d.keys())))
    print("Vocabulary Size: {}".format(len(list(vocab))))

    return docs_d


def perform_tm(s_ids, corpus, s_urls, n_topics, rm_top, topwords_file):

    lda_model = tp.LDAModel(k=n_topics, min_df=3, rm_top=rm_top)
    vocab = set()
    for doc in corpus:
        lda_model.add_doc(doc)
        vocab.update(doc)
    print('Num docs:{}'.format(len(lda_model.docs)))
    print("Vocabulary Size: {}".format(len(list(vocab))))
    print('Removed Top words: ', *lda_model.removed_top_words)

    iterations = 10
    for i in range(0, 100, iterations):
        lda_model.train(iterations)
        print(f'Iteration: {i}\tLog-likelihood: {lda_model.ll_per_word}')

    #TOP WORDS
    num_top_words = 10  # number of top words to print for each topic
    with open(topwords_file, "w", encoding="utf-8") as file:
        file.write(f"\nTopics in LDA model: {n_topics} topics {rm_top} removed top words\n\n")
        topics = []
        topic_individual_words = []
        for topic_number in range(0, n_topics):
            topic_words = ' '.join(
                word for word, prob in lda_model.get_topic_words(topic_id=topic_number, top_n=num_top_words))
            topics.append(topic_words)
            topic_individual_words.append(topic_words.split())
            file.write(f"Topic {topic_number}\n{topic_words}\n\n")

    #TOPIC DISTRIBUTIONS
    topic_distributions = [list(doc.get_topic_dist()) for doc in lda_model.docs]
    topic_results = []
    for s_id, topic_distribution in zip(s_ids, topic_distributions):
        topic_results.append({'topic_distribution': topic_distribution})
    df = pd.DataFrame(topic_results, index=s_ids)
    column_names = [f"Topic {number} {' '.join(topic[:4])}" for number, topic in enumerate(topic_individual_words)]
    df[column_names] = pd.DataFrame(df['topic_distribution'].tolist(), index=df.index)
    df = df.drop('topic_distribution', axis='columns')
    dominant_topic = np.argmax(df.values, axis=1)  # get dominant topic for each document
    df['dominant_topic'] = dominant_topic

    return df


def main(subreddit):

    reddit_data = os.path.join(os.getcwd(), 'reddit', 'posts', f'{subreddit}.csv')
    tomo_folder = os.path.join(os.getcwd(), 'topic_modeling', 'sentence_level')
    if not os.path.exists(tomo_folder):
        os.makedirs(tomo_folder)
    tomo_sub_folder = os.path.join(tomo_folder, f'{subreddit}')
    if not os.path.exists(tomo_sub_folder):
        os.makedirs(tomo_sub_folder)

    reddit_df = pd.read_csv(reddit_data, index_col=0)

    #PRE PROCESS DATA
    print("Data Cleaning...")
    start = datetime.now()
    reddit_docs_file = os.path.join(tomo_sub_folder, f'{subreddit}.json')
    if not os.path.exists(reddit_docs_file):
        docs_dict = cleaning_docs(reddit_df, reddit_docs_file)
    else:
        with open(reddit_docs_file) as json_file:
            docs_dict = json.load(json_file)
    doc_ids = [doc_id for doc_id in docs_dict.keys()]
    clean_docs = [sent_url[0] for sent_url in docs_dict.values()]
    doc_urls = [sent_url[1] for sent_url in docs_dict.values()]
    print(f'{str(datetime.now())}________________{str(datetime.now() - start)}\n')

    #PERFORMING TOPIC MODELING
    for num_topics in [5, 10, 15]:  # number of topics, use larger value for larger corpora
        for rm_frequent in [5, 10]:  # keep words that appear on no more than x fraction of pages, removes high-frequency words

            txt_topwords = os.path.join(tomo_sub_folder, f'{subreddit}-{num_topics}_{rm_frequent}.txt')
            #html_viz = os.path.join(tomo_sub_folder, f'{subreddit}-{num_topics}_{rm_frequent}.html')
            csv_dtm_small = os.path.join(tomo_sub_folder, f'{subreddit}-{num_topics}_{rm_frequent}.csv')
            csv_dtm_large = os.path.join(tomo_sub_folder, f'{subreddit}-{num_topics}_{rm_frequent}_large.csv')

            if not os.path.exists(txt_topwords) or not os.path.exists(csv_dtm_small): #or not os.path.exists(html_viz):
                start = datetime.now()

                # PERFORM TOPIC MODELING AND GET DOC-TOPIC MATRIX
                print("Performing Topic Modeling...")
                start = datetime.now()
                lda_dtm = perform_tm(doc_ids, clean_docs, doc_urls, num_topics, rm_frequent, txt_topwords)
                lda_dtm.to_csv(csv_dtm_small)
                lda_dtm['lemm_sent'] = clean_docs
                lda_dtm['post_url'] = doc_urls
                lda_dtm.to_csv(csv_dtm_large)
                print(f'{str(datetime.now())}____Topic modeling {num_topics}, {rm_frequent} time:____{str(datetime.now() - start)}\n')


if __name__ == '__main__':
    main('endo+endometriosis')
