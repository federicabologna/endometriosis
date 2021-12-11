import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict, Counter

from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns



'''
Get often used file paths by label type.

Input:
    label_type: type of label, options are "relations", "intent", and "topics"

Output:
    labels_file_path: location of labels used in prodigy
    annotations_path: location of annotations outputted from prodigy
    output_csv_name: location of where formatted csv is or will be saved.

'''

def get_file_paths_for_label(label_type = "relations"):
    if label_type == 'relations':
        annotation_file_name = 'relation_annotations.json'
        labels_file_name = 'relationships.txt'
        output_csv_name = 'relationships.csv'
    elif label_type == 'topics':
        annotation_file_name = 'topics_annotations.jsonl'
        labels_file_name = 'topics.txt'
        output_csv_name = 'topics.csv'
    elif label_type == 'intent':
        annotation_file_name = 'intent_annotations.json'
        labels_file_name = 'intent.txt'
        output_csv_name = 'intent.csv'   

    input_path = os.getcwd()
    labels_file_path = os.path.join(input_path, 'labeling', 'prodigy', labels_file_name)
    annotations_path = os.path.join(input_path, 'labeling', 'prodigy', 'output', annotation_file_name)
    output_csv_path = os.path.join(input_path, 'labeling', 'prodigy', 'output', 'ordered_csvs', output_csv_name)

    return labels_file_path, annotations_path, output_csv_path



'''
Convert from raw json annotations output to dataframe for easy classification. 

Input:
    text: options are "relations" "topics" or "intent"

Output:
    - annotations_df: a dataframe of annotations, that can be easily used for classification. 
        annotations_df has a column for each label type; cells are filled with 1 or 0 based on label
    - prints labeled data stats

Note: expects files are located in following tree structure: input_path/labeling/prodigy/output for any annotated files;
    labels are stored in input_path/labeling/prodigy

'''

def format_raw_annotations(label_type = "relations"):
    labels_file_path, annotations_path, output_csv_path = get_file_paths_for_label(label_type)
    annotations = []
    for line in open(annotations_path, 'r'):
        annotations.append(json.loads(line))

    annotations_df = pd.DataFrame(annotations)


    # keep only relevant columns
    annotations_df = annotations_df.drop(['_input_hash', '_task_hash', 'options', '_session_id', '_view_id', 'config', 'answer'], axis = 1)

    # read labels from text file
    with open(labels_file_path) as file:
        lines = file.readlines()
        label_names = [line.rstrip() for line in lines]

    def check_if_value_present(df_cell, label = 'DOCTORS'):
        value = np.where(label in df_cell, 1, 0)
        return value

    for col_name in label_names:
        annotations_df[col_name] = annotations_df['accept'].apply(check_if_value_present, label = col_name)


    annotations_df.to_csv(output_csv_path)

    return annotations_df
 

    
'''
Print label stats from formatted dataframe. 

Input:
    label_type: type of label.

Output:
    easy print of number of categories.
'''

def get_label_stats(label_type = "relations"):
    labels_file_path, annotations_path, output_csv_path = get_file_paths_for_label(label_type)
    with open(labels_file_path) as file:
        lines = file.readlines()
        label_names = [line.rstrip() for line in lines]
    annotations_df = pd.read_csv(output_csv_path)
    for label in label_names:
        print('Balance of '+ str(label) + ' is: ' + str(Counter(annotations_df[label])))



   
'''
Load formatted annotations for determined label category. 

Input:
    label_type: type of label

Output:
    annotations_df: dataframe with all annotations + metadata
'''

def load_annotations(label_type = "relations"):
    if label_type == 'relations':
        csv_name = 'relationships.csv'
    elif label_type == 'topics':
        csv_name = 'topics.csv'
    elif label_type == 'intent':
        csv_name = 'intent.csv'   

    input_path = os.getcwd()
    csv_path = os.path.join(input_path, 'labeling', 'prodigy', 'output', 'ordered_csvs', csv_name)
    annotations_df = pd.read_csv(csv_path)

    return annotations_df
    


'''
Get binary logistic regression accuracy for a single label category, showing trends in accuracy with more data. 

Input:
    label_type: type of label

Output:
    - plots increase in accuracy 

Note: Loads formatted annotations csv. 
'''

def classification_accuracy_by_data_amount(label_type = "relations"):
    labels_file_path, annotations_path, output_csv_path = get_file_paths_for_label(label_type)
    annotations_df = pd.read_csv(output_csv_path)
    # read labels from text file
    with open(labels_file_path) as file:
        lines = file.readlines()
        label_names = [line.rstrip() for line in lines]

        
    def tokenizer(text):
        tokens = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    vectorizer = TfidfVectorizer(
        encoding='utf-8',
        tokenizer = tokenizer,
        max_df=.8,
        min_df = 2
    )

    vectorizer.fit(annotations_df.text)
    lr = LogisticRegression()

    sample_sizes = range(200, len(annotations_df.text), 10)
    for label in label_names:
        try:
            x_values = []
            y_values = []
            scores_l = []
            label_values = []
            for i in sample_sizes:
                sample_df = annotations_df.sample(n=i)
                y = sample_df[label]
                X = vectorizer.transform(sample_df.text)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)
                balanced_accuracy = metrics.f1_score(y_test, y_pred, zero_division = 0)
                x_values.append(len(y))
                y_values.append(balanced_accuracy)
                label_values.append(label)
            
            sns.lineplot(x = x_values, y = y_values)
        except ValueError:
                print("Not enough data about "+ str(label))
                continue



'''
Get binary logistic regression accuracy for a single label category, showing trends in accuracy with more data. 

Input:
    label_type: type of label

Output:
    accuracy_list: list of all accuracy scores for each label type. accuracy_list[0] is label name, 
        accuracy_list[1] is dictionary of cross val scores (precision, f1, and recall)

Note: Loads formatted annotations csv. 

'''
def get_accuracy_metrics(label_type = "relations"):
    labels_file_path, annotations_path, output_csv_path = get_file_paths_for_label(label_type)
    annotations_df = pd.read_csv(output_csv_path)
    # read labels from text file
    with open(labels_file_path) as file:
        lines = file.readlines()
        label_names = [line.rstrip() for line in lines]

        
    def tokenizer(text):
        tokens = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    vectorizer = TfidfVectorizer(
        encoding='utf-8',
        tokenizer = tokenizer,
        max_df=.8,
        min_df = 2
    )

    vectorizer.fit(annotations_df.text)
    lr = LogisticRegression()

    X = vectorizer.transform(annotations_df.text)
    accuracy_list = []
    for label in label_names:
        y = annotations_df[label]
        stored = cross_validate(lr, X, y, cv=5, scoring=['precision', 'f1', 'recall'])
        accuracy = [label, stored]
        accuracy_list.append(accuracy)
    return accuracy_list

