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

def get_terms_with_seq(mem_size):
  def tokenizer(document):
    r = list(dataset[dataset[document_field]==int(document.filename)][term_field].astype(str))
    if mem_size==0:
      return ' '.join(r)
    else:
      grams = [':'.join(r[i:i+mem_size]) for i in range(len(r)-mem_size+1)]
      import code; code.interact(local=dict(globals(), **locals()))
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


logging.info("Starting evaluation")

output_file = '/notebook/all_results.csv'
db_index_name = 'recommendation_exp1'
#dataset_file = '/dataset/pcounts_filtered_december_2013.csv'
dataset_file = '/dataset/sessions.md.csv'
cache_folder = '/cache'

#preprocess_dataset(dataset_file, '/cache', 'session', 'track', 0.8)

#dataset = pd.read_csv('/dataset/pcounts_filtered_december_2013.csv')


def exec_process(cache_folder, output_file, itr, term_field, document_field, query_terms_ratio, memory_size):
  import code; code.interact(local=dict(globals(), **locals()))




for evaluation_instance in execution.evaluation_instances():
  if not already_done(output_file, *evaluation_instance):
    exec_process(cache_folder, output_file, *evaluation_instance)
  else:
    print('skipping', *evaluation_instance)
  #d = load_output_dataframe(output_file)



#if already_done(final_df, itr, term_field, document_field, query_terms_ratio, memory_size):
  #print('skipping')
  #continue



# Initialize project, db instance and evaluator
project = Project()
db.connect_to_elasticsearch(project, db_index_name, True)
project.client.set_scope(db_index_name, [term_field], 'tokens_by_spaces')
evaluator = Evaluator(project)

project.tokenize(term_field, get_terms_with_seq(memory_size))
#project.tokenize('tracks', session_artists)

evaluator.build(document_field, train_dataset)
evaluator.match(document_field, term_field, query_dataset)
evaluator.evaluate(document_field, term_field, query_dataset_full)
print('Train dataset: ', len(train_dataset[document_field].unique()), 'documents')
print('Query dataset: ', len(query_dataset[document_field].unique()), 'documents')
print(evaluator.results.mean())

complete_row = [itr, 0, document_field, term_field,
  memory_size, evaluator.results.mean().top1_match, evaluator.results.mean().top_5_match,
  evaluator.results.mean().top_10_match, len(train_dataset),
  len(query_dataset_full), query_terms_ratio]

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

final_df.to_csv(output_file, index=False)

print("Done!")

#['iter', 'num_query_terms', 'document_field', 'term_field', 'strgy',
#                              'top1_acc', 'top5_acc', 'top10_acc',
#                              'num_train_documents', 'num_test_documents', 'query_terms_ratio']

# tesoura
# import code; code.interact(local=dict(globals(), **locals()))

# import threading
# import pandas as pd
# import concurrent.futures
#
# ll = []
#
# def add(l, a, aa):
#   print(a, aa)
#   for ii in range(1000000):
#     pass
#   l.append((a, aa))
#   print(a, aa, 'ok')
#
#
# with concurrent.futures.ThreadPoolExecutor(8) as executor:
#   for i in range(30):
#     a =  i
#     aa = i*10
#     executor.submit(add, [ll, a, aa])
