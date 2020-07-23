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


def already_done(d, itr, term_field, document_field, query_terms_ratio, memory_size):
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

output_file = '/notebook/all_results.csv'
try:
  final_df = pd.read_csv(output_file)
except FileNotFoundError:
  final_df = pd.DataFrame(columns=['iter', 'num_query_terms', 'document_field', 'term_field', 'strgy',
                                'top1_acc', 'top5_acc', 'top10_acc',
                                'num_train_documents', 'num_test_documents', 'query_terms_ratio'])


logging.info("Starting evaluation")
db_index_name = 'recommendation_exp1'
dataset = pd.read_csv('/dataset/sessions_eval.sm.csv')
#dataset = pd.read_csv('/dataset/pcounts_filtered_december_2013.csv')
itr = 1
term_field = 'track'
for query_terms_ratio in [0.8, 0.5, 0.2, 0.4]:
  for document_field in ['session', 'user']:
    for memory_size in [0, 1, 2, 3, 4]:
      if already_done(final_df, itr, term_field, document_field, query_terms_ratio, memory_size):
        print('skipping')
        continue
      ds_counts = dataset.groupby([document_field]).count()
      #cond = (ds_counts['track'] >= 9) & (ds_counts['track'] <= 15)
      cond = (ds_counts[term_field] >= 5)
      dataset = dataset[dataset[document_field].isin(ds_counts[cond].index)]

      factor = 0.8

      documents = dataset.sort_values(['time'])[document_field].unique()
      train_documents = documents[0:int(len(documents) * factor)]
      test_documents = documents[int(len(documents) * factor):]

      train_dataset = dataset[dataset[document_field].isin(train_documents)]
      query_dataset_full = dataset[dataset[document_field].isin(test_documents)]



      #msk = np.random.rand(len(dataset[document_field].unique())) < 0.8
      #train_dataset = dataset[dataset[document_field].isin(dataset[document_field].unique()[msk])]
      #query_dataset_full = dataset[~dataset[document_field].isin(dataset[document_field].unique()[msk])]




      recs_to_remove = query_dataset_full.groupby([document_field]).apply(lambda x: x.sample(int(len(x)*query_terms_ratio)))[term_field]
      ds_counts = dataset.groupby([document_field]).count()

      # create a new dataset by removing 1 track (randomly) from each session
      query_dataset = query_dataset_full[~query_dataset_full[term_field].isin(recs_to_remove)].copy().reset_index()




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
