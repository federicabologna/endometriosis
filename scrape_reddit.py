from datetime import datetime
import os
import time
import glob
import pandas as pd
from psaw import PushshiftAPI


def scrape_posts_from_subreddit(subreddit, api, year, month, end_date):

    start_time = datetime.now()
    script_name = os.path.basename(__file__)

    start_epoch=int(datetime(year, month, 1).timestamp())
    end_epoch=int(datetime(year, month, end_date).timestamp())

    gen = api.search_submissions(after=start_epoch,
                                 before=end_epoch,
                                 subreddit=subreddit,
                                 filter=['url', 'author', 'created_utc', 'title', 'subreddit', 'selftext', 'num_comments', 'score', 'link_flair_text', 'id'])
    
    max_response_cache = 100000
    scraped_posts = []
    for _post in gen:
        if 'selftext' in _post.d_ and _post.d_['selftext'].strip() and _post.d_['selftext'].strip() != '[removed]' and _post.d_['selftext'].strip() != '[deleted]':
            scraped_posts.append(_post)
        if len(scraped_posts) >= max_response_cache:
            break

    scraped_posts_df = pd.DataFrame([p.d_ for p in scraped_posts])

    return scraped_posts_df


def scrape_comments_from_subreddit(subreddit, api, year, month, end_date):

    start_time = datetime.now()
    script_name = os.path.basename(__file__)

    start_epoch=int(datetime(year, month, 1).timestamp())
    end_epoch=int(datetime(year, month, end_date).timestamp())

    gen = api.search_comments(after=start_epoch,
                              before=end_epoch,
                              subreddit=subreddit)
                            #   filter=['url', 'author', 'created_utc', 'title', 'subreddit', 'selftext', 'num_comments', 'score', 'link_flair_text', 'id'])

    max_response_cache = 100000
    scraped_comments = []
    for _comment in gen:
        # if 'selftext' in _post.d_ and _post.d_['selftext'].strip() and _post.d_['selftext'].strip() != '[removed]' and _post.d_['selftext'].strip() != '[deleted]':
        #     scraped_posts.append(_post)
        # if len(scraped_posts) >= max_response_cache:
        #     break
        if len(scraped_comments) < max_response_cache:
            scraped_comments.append(_comment)
        else:
            break
    scraped_comments_df = pd.DataFrame([p.d_ for p in scraped_comments])

    return scraped_comments_df


def main():

    start_time = datetime.now()
    script_name = os.path.basename(__file__)

    api = PushshiftAPI()

    print(api.metadata_.get('shards'))

    # base_path = '/Volumes/Passport-1/data/birth-control/reddit/scraped'
    base_path = os.path.join(os.getcwd(), 'reddit')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    target_subreddits = ['endo', 'endometriosis', 'pcos']
    # target_type = 'comments' 
    target_type = 'posts'

    years = [2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2010]
    for _year in years:
        if _year < 2021:
            months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            end_dates = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        elif _year == 2021:
            months = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            end_dates = [31, 28, 31, 30, 31, 30, 31, 31, 30]

        for _month, _end_date in zip(months, end_dates):
            for _subreddit in target_subreddits:

                _output_directory_path = os.path.join(base_path, target_type, _subreddit)  # + '/' + str(_year)
                # _output_directory_path = base_path + '/' + target_type + '/' + str(_year) + '-' + str(_month)
                print(_output_directory_path)
                if not os.path.exists(_output_directory_path):
                    os.makedirs(_output_directory_path)

                _file_name = _subreddit + '-' + str(_year) + '-' + str(_month) + '.csv'
                # _file_name = 'chronic_illness.csv'

                if _file_name not in os.listdir(_output_directory_path):

                    print(str(datetime.now()) + ' ' + script_name + ': Scraping r/' + _subreddit + ' ' + str(_year) + '-' + str(_month) + '...')

                    if target_type == 'posts':
                        _posts_df = scrape_posts_from_subreddit(_subreddit, api, _year, _month, _end_date)
                        if not _posts_df.empty:
                            _posts_df.to_csv(os.path.join(_output_directory_path, _file_name))

                    if target_type == 'comments':
                        _comments_df = scrape_comments_from_subreddit(_subreddit, api, _year, _month, _end_date)
                        _comments_df.to_csv(os.path.join(_output_directory_path, _file_name))

                    time.sleep(1)

    print(str(datetime.now()) + ' ' + script_name + ': Run Time = ' + str(datetime.now() - start_time))


def aggregating(subreddits):

    for subreddit in subreddits:
        base_path = os.path.join(os.getcwd(), 'reddit', 'posts', subreddit)
        print(base_path)
        all_files = glob.glob(os.path.join(base_path, "*.csv"))
        df = pd.concat((pd.read_csv(f, index_col=0, header=0) for f in all_files), axis=0, ignore_index=True)
        df.sort_values('created_utc', inplace=True, ignore_index=True)
        print(df.head())
        df.to_csv(os.path.join(os.getcwd(), 'reddit', 'posts', f'{subreddit}.csv'))
        print(len(df))


if __name__ == '__main__':
    #main()
    aggregating(['endo'])#['endo', 'endometriosis', 'pcos'])
# ['chronicillness', 'pots', 'dysautonomia', 'diabetes_t1', 'diabetes_t2', 'asthma', 'crohnsdisease', 'kidneydisease']
