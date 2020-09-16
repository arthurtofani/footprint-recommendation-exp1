import time
import math
import numpy as np
import pandas as pd
import datetime
import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str)
args = parser.parse_args()

def gen_seq(data_folder):
    data = pd.read_csv(data_folder,sep=',',header=0)
    
    diff = 0
    puser = data.iloc[0,:]['user']
    ptime = datetime.datetime.fromtimestamp(data.iloc[0,:]['time'])
    time_lag = 30
    seq_id = 0

    for idx, item in data.iterrows():
       date = datetime.datetime.fromtimestamp(int(item['time'])) 
       if ptime: diff = date - ptime
       if abs(int(diff.total_seconds() / 60.0)) < time_lag and int(item['user']) == puser:
           print(", ".join((str(item.user), str(item.artist), str(item.album), str(item.track), str(item.time), str(seq_id))))
       else:
           seq_id+=1
           print(", ".join((str(item.user), str(item.artist), str(item.album), str(item.track), str(item.time), str(seq_id))))
       ptime = datetime.datetime.fromtimestamp(int(item['time'])) 
       puser = int(item['user'])

def main():
    gen_seq(args.data_folder)

if __name__ == '__main__':
    main()
