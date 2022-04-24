# based on https://github.com/dawenl/vae_cf

import os
import sys

import numpy as np
from scipy import sparse
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--threshold', type=float)
parser.add_argument('--min_items_per_user', type=int, default=5)
parser.add_argument('--min_users_per_item', type=int, default=0)
parser.add_argument('--heldout_users', type=int,default= 3000)

args = parser.parse_args()

dataset = args.dataset
output_dir = args.output_dir
threshold = args.threshold
min_uc = args.min_items_per_user
min_sc = args.min_users_per_item
n_heldout_users = args.heldout_users

raw_data = pd.read_csv(dataset, header=0)
raw_data.head()
leaderboarddf=pd.read_csv('/opt/ml/RecsysChal/data/test_leaderboard_sessions.csv')

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


usercount, itemcount = get_count(raw_data, 'session_id'), get_count(raw_data, 'item_id') 
raw_data, user_activity, item_popularity = raw_data, usercount, itemcount

leaderboardusercount,leaderboarditemcount=get_count(leaderboarddf,'session_id'), get_count(raw_data, 'item_id') 

sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("There are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index
unique_uid_leaderboard = leaderboardusercount.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
leaderboardidx_perm = np.random.permutation(unique_uid_leaderboard.size)
unique_uid = unique_uid[idx_perm]
unique_uid_leaderboard=unique_uid_leaderboard[leaderboardidx_perm]

n_testusers = unique_uid_leaderboard.size
n_users = unique_uid.size

# tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
# vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
# te_users = unique_uid[(n_users - n_heldout_users):]
tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid_leaderboard.copy()

train_plays = raw_data.loc[raw_data['session_id'].isin(tr_users)]

unique_sid = pd.unique(train_plays['item_id'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
        
with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('session_id')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d sessions sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te


vad_plays = raw_data.loc[raw_data['session_id'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['item_id'].isin(unique_sid)]

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['session_id'].isin(te_users)]
test_plays = test_plays.loc[test_plays['item_id'].isin(unique_sid)]

test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['session_id']))
    sid = list(map(lambda x: show2id[x], tp['item_id']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


train_data = numerize(train_plays)
train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(output_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(output_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)

