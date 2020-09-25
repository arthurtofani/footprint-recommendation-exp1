# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import argparse
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lfm1b')
# Get the arguments
args = parser.parse_args()
#args.cuda = torch.cuda.is_available()

def get_data(dataset):

    if dataset == "lfm1b": 
        PATH_TO_ORIGINAL_DATA = '../data/lfm1b/december_2013/'
        PATH_TO_PROCESSED_DATA = 'data/'
        
        data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'pcounts_december_2013_sessions.csv')
        data.columns = ['user','artist','album','track','time','session']
        data = data.filter(items=['track','time','session'])
        data['time'] =  pd.to_datetime(data['time'], unit='s')
        data = data.reindex(columns=['session','track','time'])
        data = data.rename(columns={'session': 'SessionId', 'time': 'Time', 'track':'ItemId'})
        return data, PATH_TO_PROCESSED_DATA

def main():

    data, PATH_TO_PROCESSED_DATA = get_data(args.dataset)
    data.to_csv(args.dataset+'.csv',index=False)

    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]
    
    session_full = data.filter(['SessionId']).drop_duplicates()
    print(session_full)
  
    # five folds
    fold1 = session_full.sample(n=int(0.2*session_full.shape[0]), random_state=1)
    fold1.to_csv("fold1.csv", index=False)
    curr_sessions = list(fold1.SessionId.values)
    fold2 = session_full.loc[~session_full.SessionId.isin(curr_sessions)].sample(n=int(0.2*session_full.shape[0]), random_state=1)
    fold2.to_csv("fold2.csv", index=False)
    curr_sessions.append(list(fold2.SessionId.values))
    fold3 = session_full.loc[~session_full.SessionId.isin(curr_sessions)].sample(n=int(0.2*session_full.shape[0]), random_state=1)
    fold3.to_csv("fold3.csv", index=False)
    curr_sessions.append(list(fold3.SessionId.values))
    fold4 = session_full.loc[~session_full.SessionId.isin(curr_sessions)].sample(n=int(0.2*session_full.shape[0]), random_state=1)
    fold4.to_csv("fold4.csv", index=False)
    curr_sessions.append(list(fold4.SessionId.values))
    fold5 = session_full.loc[~session_full.SessionId.isin(curr_sessions)].sample(n=int(0.2*session_full.shape[0]), random_state=1)
    fold5.to_csv("fold5.csv", index=False)


if __name__ == '__main__':
    main()

