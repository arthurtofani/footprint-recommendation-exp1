#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from scipy import sparse
import pandas as pd
import pickle
from collections import Counter

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def get_user_by_mean(data):
    df1 = data.groupby('userId').size()
    quant1 = np.quantile(df1.values,1/3)
    quant2 = np.quantile(df1.values,2/3)
    print(quant1,quant2)

    user1 = data.loc[data['userId'].isin(df1[df1 < quant1].index.values)]
    l1 = list(df1[df1 >= quant1].index.values)
    l2 = list(df1[df1 < quant2].index.values)
    user2 = data.loc[data['userId'].isin(np.intersect1d(l1,l2))]
    user3 = data.loc[data['userId'].isin(df1[df1 >= quant2].index.values)]
    print(len(set(user1['userId'])),len(set(user2['userId'])),len(set(user3['userId'])))

    return user1, user2, user3

def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'trackId')
        tp = tp[tp['trackId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'trackId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.1):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list, raw_list = list(), list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        group = group.sort_values(["timestamp"], ascending = True)
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            #select last 0.2 items ordered by timestamp
            idx[np.arange(n_items_u)[-int(test_prop * n_items_u):].astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
            raw_list.append(group)
        else:
            tr_list.append(group)
            raw_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    data_raw = pd.concat(raw_list)

    return data_tr, data_te, data_raw

def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['trackId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


parser = argparse.ArgumentParser("Data creation for VAE based collaborative filtering")
parser.add_argument('--dataset_name', type=str, default='lfm1b_dec', help='dataset name', choices=['lfm1b_dec'])
parser.add_argument('--out_data_dir', type=str, default='./data/pro_sg/', help='output data directory')
parser.add_argument('--rating_file', type=str, default='../data/lfm1b/pcounts_december_2013_sessions.csv', help='rating file')
args = parser.parse_args()
print(args)

pro_dir = args.out_data_dir

os.makedirs(args.out_data_dir, exist_ok=True)
raw_data_orig = pd.read_csv(args.rating_file, sep=',', header=0)
raw_data_orig.columns = ['userId','artistId','albumId','trackId','timestamp','sessionId']
#raw_data_orig = pd.read_csv(args.rating_file, sep=',', header=None, columns=['userId','artistId','albumId','movieId','timestamp'])

max_seq_len = 1000
min_seq_len = 10

# Remove users with greater than $max_seq_len number of watched movies
raw_data_orig = raw_data_orig.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)
raw_data_orig = raw_data_orig.groupby(["userId"]).filter(lambda x: len(x) >= min_seq_len)

# Only keep items that are clicked on by at least 5 users

raw_data, user_activity, item_popularity = filter_triplets(raw_data_orig, min_uc=200, min_sc=10)

raw_data = raw_data.sort_values(by=['userId','trackId'])
raw_data = raw_data.reset_index(drop=True)
#_, _, _ = get_user_by_mean(raw_data)

sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index
print("UNIQUE USERS: {}".format(len(unique_uid)))

# shuffle user index
np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

# create train/validation/test users
n_users = unique_uid.size
n_heldout_users = 1000

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
unique_sid = pd.unique(train_plays['trackId'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
item_mapper = pd.DataFrame.from_dict(show2id, orient='index', columns=['new_trackId'])
item_mapper['trackId'] = item_mapper.index
item_mapper.to_csv(os.path.join(pro_dir, 'item_mapper.csv'), index=False)

profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
user_mapper = pd.DataFrame.from_dict(profile2id, orient='index', columns=['new_userId'])
user_mapper['userId'] = user_mapper.index
user_mapper.to_csv(os.path.join(pro_dir, 'user_mapper.csv'), index=False)

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['trackId'].isin(unique_sid)]
vad_plays_tr, vad_plays_te, vad_plays_raw = split_train_test_proportion(vad_plays)
print("VAL USERS: {}".format(len(vad_plays.userId.unique())))
print("VAL USERS TRAIN: {}".format(len(vad_plays_tr.userId.unique())))
print("VAL USERS TEST: {}".format(len(vad_plays_te.userId.unique())))

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['trackId'].isin(unique_sid)]
test_plays_tr, test_plays_te, test_plays_raw = split_train_test_proportion(test_plays)
print("TE USERS: {}".format(len(test_plays.userId.unique())))
print("TE USERS TRAIN: {}".format(len(test_plays_tr.userId.unique())))
print("TE USERS TEST: {}".format(len(test_plays_te.userId.unique())))
#user1, user2, user3 = get_user_by_mean(test_plays_raw)

train_data = numerize(train_plays, profile2id, show2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
vad_data_te = numerize(vad_plays_te, profile2id, show2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
vad_data = numerize(vad_plays_raw, profile2id, show2id)
vad_data.to_csv(os.path.join(pro_dir, 'validation.csv'), index=False)
test_data_tr = numerize(test_plays_tr, profile2id, show2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
test_data_te = numerize(test_plays_te, profile2id, show2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
test_data = numerize(test_plays_raw, profile2id, show2id)
test_data.to_csv(os.path.join(pro_dir, 'test.csv'), index=False)


def load_matrix_df(ratings, users, items):
    num_users = users.shape[0]
    num_items = items.shape[0]
    #num_users = ratings.user.unique().shape[0]
    items_list = ratings['sid'].unique()
    users_list = ratings['uid'].unique()
    t0 = time.time()
    counts = sparse.dok_matrix((num_users, num_items), dtype=int)
    total = 0.0
    num_zeros = num_users * num_items
    songs_list = []
    for i, line in ratings.iterrows():
        #row = line.replace('\n','').split('\t')
        user = line['uid']
        item = line['sid']
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if counts[user, item] == 0: num_zeros -= 1
        counts[user, item] = counts[user, item] + 1
        total += 1
        if total % 1000000 == 0:
            print('loaded %i counts...' % total)
    alpha = num_zeros / total
    print('alpha %.2f' % alpha)
    #counts *= alpha
    counts = counts.tocsr()
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    #return counts, songs_list
    return counts

###################
rating_matrix = load_matrix_df(pd.concat([train_data, vad_data, test_data_tr], axis=0),user_mapper, item_mapper)
print(rating_matrix.shape)
filehandler = open(os.path.join(pro_dir,"rating_matrix.obj"),"wb")
pickle.dump(rating_matrix,filehandler)
filehandler.close()
####################
