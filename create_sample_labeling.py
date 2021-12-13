import os
import re
from datetime import datetime
import json
import pandas as pd
import spacy
import numpy as np
import Levenshtein


def find_duplicates(df):

    prev_post = ''
    map_dict = {}
    dup = []
    for index, row in df.iterrows():
        author = row['author']
        post = row['selftext']
        if author != '[deleted]':
            if author in map_dict.keys():
                flag = 0
                idx = 0
                while idx < len(map_dict[author]):
                    lev = Levenshtein.ratio(post, map_dict[author][idx])
                    if lev > 0.99:
                        dup.append(index)
                        flag = 1
                        idx = len(map_dict[author])
                    idx += 1
                if flag == 0:
                    map_dict[author].append(post)
            else:
                map_dict[author] = [post]

        else:
            lev = Levenshtein.ratio(row['selftext'], prev_post)
            if lev > 0.99:
                dup.append(index)

        prev_post = post

    return dup


def split_paragraphs(df):

    par_d = {'id': [], 'text': [], 'yy': [], 'url': []}

    for index, row in df.iterrows():
        post_id = row['concat_id']
        post = row['selftext']
        post_url = row['url']

        par_n = 0
        for paragraph in post.split('\n\n'):
            if len(paragraph.split(' ')) > 5:
                par_id = f'{post_id}_{par_n}'
                par_d['id'].append(par_id)
                par_d['text'].append(paragraph)
                par_d['yy'].append(row['yy'])
                par_d['url'].append(row['url'])
                par_n += 1

    par_df = pd.DataFrame.from_dict(par_d)

    return par_df


def sampling(df, size):

    init = True
    contrast_ratio = size / len(df)  # Number of randomly sampled volumes per scifi volume'
    for year in df.yy.unique():
        print(year)
        print(len(df.loc[df['yy'] == year]))
        sam = df.loc[df['yy'] == year].sample(frac=contrast_ratio) #, random_state=1234)
        print(len(sam))
        if init:
            samp = sam
            init = False
        else:
            samp = samp.append(sam)

    if len(samp) < size:
        sam = df.sample(n=size - len(samp))
        samp = samp.append(sam)
    elif len(samp) > size:
        samp = samp[:size]

    samp['yy'] = samp['yy'].astype(np.int64)

    return samp


def main():

    #settings
    subreddit = 'endo+endometriosis'
    level = 'parags'  #posts, sentences
    sample_size = 5000

    #getting data and removing duplicates
    reddit_data = os.path.join(os.getcwd(), 'reddit', 'posts', f'{subreddit}.csv')
    reddit_df = pd.read_csv(reddit_data, index_col=0)
    print(f'Total number of posts: {len(reddit_df)}')
    dupes = find_duplicates(reddit_df)  # find duplicates
    print(f'Total number of duplicates: {len(dupes)}')
    reddit_df.drop(dupes, inplace=True)
    print(f'Total number of posts after removing duplicates: {len(reddit_df)}')

    if level == 'posts':
        df = reddit_df
    elif level == 'parags':  # splitting into paragraphs
        par_reddit_df = split_paragraphs(reddit_df)
        print(f'Total number of paragraphs: {len(par_reddit_df)}')
        df = par_reddit_df

    sample = sampling(df, sample_size)  # sampling
    sample.to_csv('sample.csv')
    print(len(sample))

    for index, row in sample.iterrows():
        post_d = {'id': row['id'], 'text': row['text'], 'url': row['url']}
        sample_json = os.path.join('labeling', 'setup', f'sample_{level}_{sample_size}.jsonl')
        with open(sample_json, 'a', encoding="utf-8") as jsonfile:
            json.dump(post_d, jsonfile)
            jsonfile.write('\n')


if __name__ == '__main__':
    main()
