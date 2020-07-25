import time
import random
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


def already_done(output_file, itr, term_field, document_field, query_terms_ratio, memory_size):
  d = load_output_dataframe(output_file)
  recs = d[(d.iter==itr) &
    (d.term_field==term_field) &
    (d.document_field==document_field) &
    (d.strgy==memory_size) &
    (d.query_terms_ratio==query_terms_ratio)
    ]
  return (len(recs)>0)
#import code; code.interact(local=dict(globals(), **locals()))

def get_terms(document):
  tokens = dataset[dataset[document_field]==int(document.filename)][term_field].astype(str)
  return ' '.join(tokens)

def terms_ngram(n, train_ds, query_ds):
  all_rows = pd.concat([train_ds, query_ds])
  def tokenizer(document):
    r = list(all_rows[all_rows.document==int(document.filename)].terms.astype(str))[0]
    if n==0:
      return r
    else:
      r = r.split(' ')
      grams = [':'.join(r[i:i+n]) for i in range(len(r)-n+1)]
      return ' '.join(grams)
  return tokenizer

def load_output_dataframe(output_file):
  try:
    tmp_df = pd.read_csv(output_file)
  except FileNotFoundError:
    tmp_df = pd.DataFrame(columns=['iter', 'num_query_terms', 'document_field', 'term_field', 'strgy',
                                  'top1_acc', 'top5_acc', 'top10_acc',
                                  'num_train_documents', 'num_test_documents', 'query_terms_ratio'])
  return tmp_df


def exec_process(dataset_file, cache_folder, output_file, itr, term_field, document_field, query_terms_ratio, memory_size):
  db_index_name = 'recm_%s_%s_%s_%s_%s' % (itr, term_field, document_field, query_terms_ratio, memory_size)
  # Initialize project, db instance and evaluator
  project = Project()
  db.connect_to_elasticsearch(project, db_index_name, True)
  project.client.set_scope(db_index_name, [term_field], 'tokens_by_spaces')
  evaluator = Evaluator(project)

  #project.tokenize('tracks', session_artists)

  dataset_cache_folder = cache_folder + dataset_file
  filename = '%s_%s_%s' % (document_field, term_field, query_terms_ratio)
  query_dataset_file = dataset_cache_folder + '/' + filename + '_queries.csv'
  fullquery_dataset_file = dataset_cache_folder + '/' + filename + '_fullqueries.csv'
  train_dataset_file = dataset_cache_folder + '/' + filename + '_train.csv'


  train_dataset = pd.read_csv(train_dataset_file)
  query_dataset = pd.read_csv(query_dataset_file)
  query_dataset_full = pd.read_csv(fullquery_dataset_file)
  project.tokenize(term_field, terms_ngram(memory_size, train_dataset, query_dataset))

  evaluator.build(train_dataset)
  evaluator.match(query_dataset)
  evaluator.evaluate(query_dataset_full)
  print('Train dataset: ', len(train_dataset.document.unique()), 'documents')
  print('Query dataset: ', len(query_dataset.document.unique()), 'documents')
  print(evaluator.results.mean())

  complete_row = [itr, 0, document_field, term_field,
    memory_size, evaluator.results.mean().top1_match, evaluator.results.mean().top_5_match,
    evaluator.results.mean().top_10_match, len(train_dataset),
    len(query_dataset_full), query_terms_ratio]

  final_df = load_output_dataframe(output_file)
  try:
    final_df.loc[len(final_df)] = complete_row
  except:
    import code; code.interact(local=dict(globals(), **locals()))

  for i, r in evaluator.results.groupby(['terms_in_document']).mean().iterrows():
    #import code; code.interact(local=dict(globals(), **locals()))
    row = [itr, i, document_field, term_field,
    memory_size, r.top1_match, r.top_5_match,
    r.top_10_match, len(train_dataset),
    len(query_dataset_full), query_terms_ratio]
    final_df.loc[len(final_df)] = row
  safe_sleep = random.random() * 10
  time.sleep(safe_sleep)
  final_df.to_csv(output_file, index=False)


logging.info("Starting evaluation")

output_file = '/notebook/all_results.md.csv'

#dataset_file = '/dataset/pcounts_filtered_december_2013.csv'
dataset_file = '/dataset/sessions.md.csv'
cache_folder = '/cache'

#preprocess_dataset(dataset_file, '/cache', 'session', 'track', 0.8)

#dataset = pd.read_csv('/dataset/pcounts_filtered_december_2013.csv')



num_processors = 3
pool = Pool(num_processors)
for evaluation_instance in execution.evaluation_instances():
  if not already_done(output_file, *evaluation_instance):
    pool.apply_async(exec_process, args=(dataset_file, cache_folder, output_file, *evaluation_instance))
    #exec_process(dataset_file, cache_folder, output_file, *evaluation_instance)
    #print('skipping', *evaluation_instance)
  #d = load_output_dataframe(output_file)
#pool.map(preprocess_eval_instance, args)
pool.close()
pool.join()



#if already_done(final_df, itr, term_field, document_field, query_terms_ratio, memory_size):
  #print('skipping')
  #continue




print("Done!")
