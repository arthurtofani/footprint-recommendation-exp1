import sys
from multiprocessing import Pool
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
import execution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Usage: python3 preprocess_dataset.py /cache /dataset/sessions.md.csv 3

def preprocess_dataset(dataset_file, cache_folder, document_field, term_field, query_terms_ratio, factor=0.8, min_seq_len=5, max_seq_len=100, timestamp_field='timestamp'):

  dataset_cache_folder = cache_folder + dataset_file
  os.makedirs(dataset_cache_folder, exist_ok=True)
  filename = '%s_%s_%s' % (document_field, term_field, query_terms_ratio)
  query_dataset_file = dataset_cache_folder + '/' + filename + '_queries.csv'
  full_query_dataset_file = dataset_cache_folder + '/' + filename + '_fullqueries.csv'
  train_dataset_file = dataset_cache_folder + '/' + filename + '_train.csv'
  if os.path.exists(query_dataset_file):
    return

  logging.info('loading dataset: %s' % dataset_file)
  dataset = pd.read_csv(dataset_file)
  logging.info('filtering')
  dataset = dataset.groupby([document_field]).filter(lambda x: len(x) <= max_seq_len)
  dataset = dataset.groupby([document_field]).filter(lambda x: len(x) > min_seq_len)

  logging.info('splitting test/train')
  documents = dataset.sort_values([timestamp_field])[document_field].unique()
  train_documents = documents[0:int(len(documents) * factor)]
  test_documents = documents[int(len(documents) * factor):]

  logging.info('creating query documents list')
  query_dataset_full = dataset[dataset[document_field].isin(test_documents)]
  recs_to_remove = query_dataset_full.groupby([document_field]).apply(lambda x: x.sample(int(len(x)*query_terms_ratio)))[term_field]
  #ds_counts = dataset.groupby([document_field]).count()

  # create a new dataset by removing 1 track (randomly) from each session
  query_dataset = query_dataset_full[~query_dataset_full[term_field].isin(recs_to_remove)].copy().reset_index()
  query_dataset[term_field] = query_dataset[term_field].astype(str)
  query_dataset = query_dataset.groupby([document_field])[term_field].apply(' '.join)
  query_dataset = query_dataset.reset_index()
  query_dataset.columns = ['document', 'terms']

  query_dataset_full[term_field] = query_dataset_full[term_field].astype(str)
  query_dataset_full = query_dataset_full.groupby([document_field])[term_field].apply(' '.join)
  query_dataset_full = query_dataset_full.reset_index()
  query_dataset_full.columns = ['document', 'terms']


  logging.info('creating db documents list')
  train_dataset = dataset[dataset[document_field].isin(train_documents)]
  train_dataset[term_field] = train_dataset[term_field].astype(str)
  train_dataset = train_dataset.groupby([document_field])[term_field].apply(' '.join)
  train_dataset = train_dataset.reset_index()
  train_dataset.columns = ['document', 'terms']


  logging.info('saving...')
  query_dataset.to_csv(query_dataset_file, index=False)
  logging.info('created %s' % query_dataset_file)

  query_dataset_full.to_csv(full_query_dataset_file, index=False)
  logging.info('created %s' % full_query_dataset_file)

  train_dataset.to_csv(train_dataset_file, index=False)
  logging.info('created %s' % train_dataset_file)

  logging.info('Total queries: %s' % len(query_dataset))
  logging.info('Total load docs: %s' % len(train_dataset))



def preprocess_eval_instance(arg):
  dataset_file, cache_folder, itr, term_field, document_field, query_terms_ratio, memory_size = arg
  logging.info('preprocessing %s, %s, %s, %s' % (dataset_file, term_field, document_field, query_terms_ratio))
  preprocess_dataset(dataset_file, cache_folder, document_field, term_field, query_terms_ratio)


try:
  num_processors = int(sys.argv[3])
except:
  num_processors = None

pool = Pool(num_processors)
args = [[sys.argv[2], sys.argv[1], *inst] for inst in execution.evaluation_instances()]
pool.map(preprocess_eval_instance, args)
pool.close()
pool.join()
#preprocess_eval_instance(args[0])

#import code; code.interact(local=dict(globals(), **locals()))

