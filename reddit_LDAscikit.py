import os
from datetime import datetime
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np


def cleaning_docs(df, df_ids, df_docs):
    doc_ids = []
    clean_docs = []

    lemmatizer = WordNetLemmatizer()
    stopwords = set([line.strip() for line in open("stoplist_final.txt")])

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
            if token and token not in stopwords:  # if token is not empty and is not in stopwords
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


def print_top_words(model, feature_names, filename):  # function to print the top words of each topic
    n_top_words = 10  # number of top words to print for each topic
    index_words = []

    with open(filename, "w", encoding="utf-8") as file:
        file.write("\nTopics in LDA model:\n==========")
        for topic_idx, topic in enumerate(model.components_):  # num of the topic and the topic itself
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]  # get word/feature from the vectorizer array
                                 # select ten last indeces that correspond to top words in the topic
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            file.write("\n"+message)
            index_words.append(message.split(' ')[2])

    return index_words


def perform_tm(vectorizer, lda, ids, corpus, topwords_file, viz_file):
    # Perform vectorization
    tf = vectorizer.fit_transform(corpus)
    # print(tf.shape)

    # Fit model
    lda.fit(tf)

    # Topic index and Top words
    tf_feature_names = vectorizer.get_feature_names()
    topic_index_words = print_top_words(lda, tf_feature_names, topwords_file)

    # Doc-Topic Matrix
    lda_output = lda.transform(tf)  # transform input vectors to topics
    topicnames = [f"t{str(i)} {topic_index_words[i]}" for i in range(lda.n_components)]  # column names
    docnames = ids  # index names
    dtm = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)  # create the dataframe
    dominant_topic = np.argmax(dtm.values, axis=1)  # get dominant topic for each document
    dtm['dominant_topic'] = dominant_topic

    # Interactive Visualization
    visdata = pyLDAvis.sklearn.prepare(lda, tf, vectorizer)
    pyLDAvis.save_html(visdata, viz_file)

    return dtm


def main(subreddit):

    reddit_data = os.path.join(os.getcwd(), 'reddit', 'posts', f'{subreddit}.csv')
    tomo_folder = os.path.join(os.getcwd(), 'topic_modeling', f'{subreddit}')
    if not os.path.exists(tomo_folder):
        os.makedirs(tomo_folder)

    df = pd.read_csv(reddit_data, index_col=0)

    #PRE PROCESS DATA
    start = datetime.now()
    post_ids, clean_posts = cleaning_docs(df, 'id', 'selftext')
    print(str(datetime.now()) + f'____Data cleaning time:____' + str(datetime.now() - start))

    for n_components in [10, 15, 20]:  # number of topics, use larger value for larger corpora
        for max_df in [0.3, 0.45]: # 0.5]:  # keep words that appear on no more than x fraction of pages, removes high-frequency words

            txt_topwords = os.path.join(tomo_folder, f'{subreddit}-{n_components}_{max_df}.txt')
            html_viz = os.path.join(tomo_folder, f'{subreddit}-{n_components}_{max_df}.html')
            csv_dtm = os.path.join(tomo_folder, f'{subreddit}-{n_components}_{max_df}.csv')

            if not os.path.exists(txt_topwords) or not os.path.exists(html_viz) or not os.path.exists(csv_dtm):
                start = datetime.now()

                # VECTORIZER
                min_df = 3  # keep words that appear on at least x total pages, removes rare words, larger value for larger corpora
                n_features = 1000  # max unique words to use, remove very low-frequency words, larger value for larger corpora
                count_vectorizer = CountVectorizer(encoding='utf-8', max_df=max_df, min_df=min_df, max_features=n_features)

                # LDA MODEL
                max_iter = 20  # number of LDA passes over corpus
                LDA = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, learning_method='online',
                                                learning_offset=50., random_state=0, n_jobs=1)

                # PERFORM TOPIC MODELING AND GET DOC-TOPIC MATRIX
                lda_dtm = perform_tm(count_vectorizer, LDA, post_ids, clean_posts, txt_topwords, html_viz)
                lda_dtm.to_csv(csv_dtm)

                print(str(datetime.now()) + f'____Topic modeling {n_components}, {max_df} time:____' + str(datetime.now() - start))


if __name__ == '__main__':
    main('endo+endometriosis')
