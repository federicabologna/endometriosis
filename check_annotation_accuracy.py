import numpy as np
from numpy.lib.shape_base import column_stack
from numpy.lib.type_check import _common_type_dispatcher
import pandas as pd
import os
import json
from collections import defaultdict
from pandas.core.frame import DataFrame
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

# change this to the name of the label type
label_type = 'relations'

if label_type == 'relations':
    annotation_file_name = 'relation_annotations.json'
    labels_file_name = 'relationships.txt'
    csv_name = 'relationships.csv'
elif label_type == 'topics':
    annotation_file_name = 'topics_annotation.jsonl'
    labels_file_name = 'topics.txt'
    csv_name = 'topics.csv'
elif label_type == 'intent':
    print("Still need files.")



input_path = os.getcwd()
labels_file_path = os.path.join(input_path, 'labeling', 'prodigy', labels_file_name)
annotations_path = os.path.join(input_path, 'labeling', 'output', annotation_file_name)
output_csv_path = os.path.join(input_path, 'labeling', 'output', 'ordered_csvs', csv_name)

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


for label in label_names:
    print('Balance of '+ str(label) + ' is: ' + str(Counter(annotations_df[label])))


annotations_df.to_csv(output_csv_path)

# use above to determine what to include in labels_to_ignore:
if label_type == 'relations':
    labels_to_ignore = ["THERAPIST", "FRIEND", "OTHER", "NON-ENDO-AUTHOR", "PARTNER", "FAMILY"]
elif label_type == 'topics':
    print("Add labels to ignore!")
elif label_type == 'intent':
    print("Add labels to ignore!")

#############################################################################################
# GET TF IDF STRUCTURE AND ASSESS ACCURACY
#############################################################################################

vectorizer = TfidfVectorizer(
    encoding='utf-8',
    max_df=.95,
    #stop_words = 'english',
)

vectorizer.fit(annotations_df.text)
lr = LogisticRegression()

sample_sizes = range(200, len(annotations_df.text), 10)
palette = sns.color_palette("crest")



for label in label_names:
    if label in labels_to_ignore:
        print("Not enough data about "+ str(label))
    else:
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
            balanced_accuracy = metrics.f1_score(y_test, y_pred)

            #scores = cross_validate(lr, X, y, cv=5, scoring=['accuracy', 'f1', 'f1_macro', 'f1_micro'])
            #scores_l.append(scores)

            x_values.append(len(y))
            # y_values.append(np.mean(scores.get("test_f1")))
            y_values.append(balanced_accuracy)
            label_values.append(label)
        
        #plt.plot(x_values, y_values)
        sns.set_theme(style="darkgrid")
        sns.color_palette("crest")
        sns.lineplot(x = x_values, y = y_values)



# Accuracy scores for all labeled data:
for label in label_names:
    if label in labels_to_ignore:
        print("Not enough data about " + str(label))
    else:
        X = vectorizer.transform(annotations_df.text)
        y = annotations_df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print("Classification scores for " + label + ":")
        print(metrics.classification_report(y_test, y_pred))