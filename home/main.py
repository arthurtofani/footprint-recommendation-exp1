import time
from multiprocessing import Pool
from itertools import chain, combinations, product
from footprint.models import Project
import os
import logging
import numpy as np
import pandas as pd
from exp1 import db
from exp1 import tokenizer
from exp1.evaluator import Evaluator
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)



logging.info("Starting evaluation")
db_index_name = 'recommendation_exp1'
dataset = pd.read_csv('/dataset/sessions.md.csv')
field_name = 'track'

ds_counts = dataset.groupby(['session']).count()
cond = (ds_counts['track'] >= 9) & (ds_counts['track'] <= 15)
cond = (ds_counts['track'] >= 2)
dataset = dataset[dataset.session.isin(ds_counts[cond].index)]

msk = np.random.rand(len(dataset.session.unique())) < 0.8
train_dataset = dataset[dataset.session.isin(dataset.session.unique()[msk])]
query_dataset_full = dataset[~dataset.session.isin(dataset.session.unique()[msk])]

#train_dataset = dataset[dataset.session.isin(dataset.session.unique()[msk])]
#query_dataset_full = dataset[~dataset.session.isin(dataset.session.unique()[msk])]

recs_to_remove = query_dataset_full.groupby(['session']).apply(lambda x: x.sample(int(len(x)/2)))[field_name]
ds_counts = dataset.groupby(['session']).count()

# create a new dataset by removing 1 track (randomly) from each session
query_dataset = query_dataset_full[~query_dataset_full[field_name].isin(recs_to_remove)].copy().reset_index()
#query_dataset = query_dataset[~query_dataset[field_name].isin(tracks_to_remove)].copy().reset_index()



def get_terms(document):
  tokens = dataset[dataset.session==int(document.filename)][field_name].astype(str)
  return ' '.join(tokens)

def get_terms_with_seq(document):
  r = dataset[dataset.session==int(document.filename)][field_name].astype(str)
  r2 = list(r[1:]) + ['-']
  tokens = list(r + ':' + r2)
  return ' '.join(tokens)


# Initialize project, db instance and evaluator
project = Project()
db.connect_to_elasticsearch(project, db_index_name, True)
project.client.set_scope(db_index_name, [field_name], 'tokens_by_spaces')
evaluator = Evaluator(project)

project.tokenize(field_name, get_terms_with_seq)
#project.tokenize('tracks', session_artists)

evaluator.build(train_dataset)
evaluator.match(field_name, query_dataset)
evaluator.evaluate(field_name, query_dataset_full)
print('Train dataset: ', len(train_dataset.session.unique()), 'sessions')
print('Query dataset: ', len(query_dataset.session.unique()), 'sessions')
print(evaluator.results.mean())


#import code; code.interact(local=dict(globals(), **locals()))

#p.process_feature('track_ids', features.session_tracks)

#max_processors = 3

#evaluator = evaluators.CSI(p)


#keys = [k for k in p.tokenization_methods.keys() if k not in already_processed_keys]

import code; code.interact(local=dict(globals(), **locals()))
