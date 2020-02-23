# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:08:07 2017
Parse log file  
@author: wangbao
"""

import os
import threading
import time
from multiprocessing import Pool

import pandas as pd

PATH = '/Users/wangbao/data/user_consume_level/'
os.chdir(path=PATH)


def get_file(path):
    """
    get '.log' files
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # O(in) > O(len)        
            if len(name) == 10:
                file_list.append(os.path.join(root, name))
    return file_list


def parse_json(filepath):
    """
    parse log to df
    :param filepath:
    """
    # read file
    if 'live' in filepath:
        consume_type = 'live'
    else:
        consume_type = 'media'
    df = pd.read_csv(filepath)
    '''
    # change dim
    data = df.values
    data_len = data.shape[0]
    data = data.reshape(data_len, )

    # json 2 python 
    data = list(map(lambda string: json.loads(string), data))

    # python 2 json
    data = json.dumps(data)

    # json 2 dataframe
    df = pd.read_json(data, typ='frame')

    # preprocess
    date = lambda string: pd.to_datetime(string).date()
    df['time'] = df['time'].apply(date)
    df['type'] = consume_type
    columns = ['time', 'uid', 'gift_bean', 'name', 'receive_uid', 'type']
    return df[columns]
    '''


def time_format(df):
    date = lambda string: pd.to_datetime(string).date()
    df['time'] = df['time'].apply(date)
    columns = ['time', 'uid', 'gift_bean', 'name', 'receive_uid', 'type']
    return df[columns]


def main(path):
    """
    get file list ,then parse log to df, finally concat df 
    """
    filelist = get_file(path)[:10]

    start = time.time()
    with Pool(6) as pool:
        dfs = pool.map(parse_json, filelist)
    end = time.time()
    print('process %.2f' % (end - start))


def main_(path):
    filelist = get_file(path)[:10]
    a = []
    start = time.time()
    for file in filelist:
        t = threading.Thread(target=parse_json, args=(file,))
        a.append(t)

    for i in a:
        i.start()
    for i in a:
        i.join()
    end = time.time()
    print('thread %.2f' % (end - start))


def new_old_concat(consume_file, new_file):
    new = pd.read_pickle(new_file)
    consume = pd.read_pickle(consume_file)
    earliest_date = consume['time'].min()
    consume = consume[consume['time'] != earliest_date]
    new_consume = pd.concat(new, consume)
    new_consume.to_pickle('new_consume.pickle')


if __name__ == '__main__':
    df = main(PATH)
    main_(PATH)
