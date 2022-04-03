import numpy as np

import torch
from torch import optim

import random
from copy import deepcopy

from utils import get_data, ndcg, recall
from model import VAE
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
parser.add_argument('--model', type=str)
args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0")

data = get_data(args.dataset)
train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)




model=torch.load(args.model)
model.eval()
total_batch=[]
first=True

for batch in generate(batch_size=500,
                              device=device,
                              data_in=test_in_data,
                              data_out=test_out_data,
                              samples_perc_per_epoch=1
                             ):
        ratings_in = batch.get_ratings_to_dev()
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
        if first:
            total_batch=ratings_pred
            first = False
        else:
              total_batch=np.concatenate([total_batch,ratings_pred],axis=0)

print(total_batch.shape)

temp=pd.DataFrame(total_batch)

unique_sid=pd.read_csv('/home/hojin/MultiVAE/data/train/pro_sg/unique_sid.txt',sep=" ",header=None)

unique_uid=pd.read_csv('/home/hojin/MultiVAE/data/train/pro_sg/unique_uid.txt',sep=" ",header=None)

id2show = dict((i, sid) for (i, sid) in enumerate(unique_sid.squeeze()))
id2profile = dict((i, pid) for (i, pid) in enumerate(unique_uid.squeeze()))

column=list(temp.columns)
origin_mid=[id2show[x] for x in column]

row=list(temp.index)
origin_uid=[id2profile[x] for x in row]

temp.columns=origin_mid
temp.index=origin_uid
raw_data=pd.read_csv('/home/hojin/RECVAE/data/train/train_ratings.csv')

watchedm=raw_data.groupby('user')['item'].apply(list)


from tqdm import tqdm
sumbission=dict()
sumbission={'user': [],'item': []}
sumbission

for row in tqdm(temp.iterrows(),total=31360):
    userid=row[0]
    movies=row[1]
    watchedmovies=watchedm[userid]

    for _ in range(10):
        sumbission['user'].append(userid)
    
    itemp=[]
    for movie in reversed(list(movies.sort_values().index)):
        if len(itemp)==10:
            break
        else:
            if movie not in watchedmovies:
                itemp.append(movie)
    
    sumbission['item']+=itemp

sumbission=pd.DataFrame(sumbission)
sumbission.sort_values('user')
sumbission.to_csv('sumbission.csv')