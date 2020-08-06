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

def create_summary_df():
  return pd.DataFrame(columns=['itr', 'build_time', 'match_time', 'amnt_docs', 'amnt_queries', 'term_field', 'document_field', 'query_terms_ratio', 'memory_size', 'index_size_kb'])

def create_summary():
  create_summary_df().to_csv(summary_filename, index=False)

def add_summary_row(itr, build_time, match_time, amnt_docs, amnt_queries, term_field, document_field, query_terms_ratio, memory_size, index_size_kb):
  df = create_summary_df()
  df.loc[len(df)] = [itr, build_time, match_time, amnt_docs, amnt_queries, term_field, document_field, query_terms_ratio, memory_size, index_size_kb]
  df.to_csv(summary_filename, mode='a', header=False, index=False)

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
    tmp_df = pd.DataFrame(columns=['iter', 'num_documents', 'num_query_terms', 'document_field', 'term_field', 'strgy',
                                  'top1_acc', 'top5_acc', 'top10_acc',
                                  'num_train_documents', 'num_test_documents', 'query_terms_ratio', 'num_query_terms_median'])
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
  tokenization_method = terms_ngram(memory_size, train_dataset, query_dataset)
  project.tokenize(term_field, tokenization_method)

  build_start = time.time()
  evaluator.build(train_dataset)
  build_time = time.time() - build_start

  match_start = time.time()
  evaluator.match(query_dataset, query_dataset_full, memory_size)
  match_time = time.time() - match_start

  #evaluator.evaluate(query_dataset_full)
  print('Train dataset: ', len(train_dataset.document.unique()), 'documents')
  print('Query dataset: ', len(query_dataset.document.unique()), 'documents')

  partials_dir = dataset_cache_folder + '/partials'
  p_filename = 'results_%s_%s_%s_%s_%s' % (itr, term_field, document_field, query_terms_ratio, memory_size)
  partials_result_file = partials_dir + '/' + p_filename + '.csv'
  #partials_result_info_file = partials_dir + '/' + p_filename + '_info.csv'

  os.makedirs(partials_dir, exist_ok=True)
  evaluator.results = evaluator.results.astype(int)
  evaluator.results['iteration'] = int(itr)
  evaluator.results['term_field'] = term_field
  evaluator.results['document_field'] = document_field
  evaluator.results['query_terms_ratio'] = query_terms_ratio
  evaluator.results['memory_size'] = memory_size
  evaluator.results.to_csv(partials_result_file, index=False)
  index_size_kb = int(project.client.es.cat.indices(index=db_index_name, bytes='kb', format='json')[0]['store.size'])
  add_summary_row(itr, round(build_time, 3), round(match_time, 3), len(train_dataset), len(query_dataset), term_field, document_field, query_terms_ratio, memory_size, index_size_kb)
  logging.info("Created file: %s - %s documents" % (partials_result_file, len(evaluator.results)))


logging.info("Starting evaluation")

#dataset_file = '/dataset/pcounts_filtered_december_2013.csv'
#output_file = '/notebook/december2013_results_remaining2.csv'


cache_folder = '/cache'


dataset_file = '/dataset/sessions.md.csv'
output_file = '/notebook/sessions.md_results.csv'
summary_filename = '/notebook/sessions.md_summary.csv'

#preprocess_dataset(dataset_file, '/cache', 'session', 'track', 0.8)

#dataset = pd.read_csv('/dataset/pcounts_filtered_december_2013.csv')



create_summary()
num_processors = 3

#pool = Pool(num_processors)
for evaluation_instance in execution.evaluation_instances():
  if not already_done(output_file, *evaluation_instance):
    #pool.apply_async(exec_process, args=(dataset_file, cache_folder, output_file, *evaluation_instance))
    exec_process(dataset_file, cache_folder, output_file, *evaluation_instance)
    #print('skipping', *evaluation_instance)

#pool.close()
#pool.join()



#if already_done(final_df, itr, term_field, document_field, query_terms_ratio, memory_size):
  #print('skipping')
  #continue




print("Done!")
print("Summary file: ", summary_filename)
