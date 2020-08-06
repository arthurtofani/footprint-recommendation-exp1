import glob
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import sklearn
import os.path
from footprint import util as util
import uuid
import csv
import logging

class Evaluator:
  dataset = None
  query_dataset = None
  project = None
  match_results = None
  ranking_size = None
  freqs = None
  results = None

  def __init__(self, project):
    self.project = project
    self.results = None

  def build(self, dataset):
    logging.info('Building dataset')
    self.dataset = dataset
    amnt = len(self.dataset['document'].unique())
    for idx, document in enumerate(self.dataset['document'].unique()):
      logging.info('Adding document %s/%s' % (idx, amnt))
      self.project.add(str(document))

  def match(self, query_dataset, query_dataset_full, ngrams_size, amnt_results_per_query=10, top_n_recs=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    df = self.query_dataset = query_dataset
    amnt = len(df['document'].unique())
    self.results = pd.DataFrame(columns=['document', 'candidates', 'top1_match', 'top_5_match', 'top_10_match', 'pos_1st_match', 'query_terms', 'full_query_terms'])
    self.results = self.results.astype(int)

    for idx, document in enumerate(df['document'].unique()):
      logging.info('Matching document %s/%s' % (idx, amnt))
      # Perform query on database
      doc, payload = self.project.match(str(document), amnt_results_per_query)
      key = list(doc.tokens.keys())[0]
      #import code; code.interact(local=dict(globals(), **locals()))
      # Use the returned document ids to gather document info on self.dataset

      #returned_records = self.dataset[self.dataset.document.isin([i.filename for i in payload])]
      #all_returned_terms = ' '.join(returned_records.terms).split(' ')
      query_terms = doc.tokens[key].split(' ')
      try:
        query_terms.remove('')
      except:
        pass

      all_returned_terms = (' '.join([payload[i].tokens[key] for i in range(len(payload))])).split(' ')
      try:
        all_returned_terms.remove('')
      except:
        pass

      results_df = pd.DataFrame(all_returned_terms, columns=['terms'])
      results_df = results_df[~results_df.terms.isin(query_terms)]
      results_df = results_df.terms.value_counts().reset_index()
      results_df.columns = ['term', 'ct']
      #results_df['query_document'] = document

      terms_histogram = self.calc_terms_histogram(results_df)

      query = df[df['document']==int(document)]
      unique_query_terms = query['terms'].values[0].split(' ')
      recs = terms_histogram[~terms_histogram.index.isin(unique_query_terms)]

      #try:
      #  if list(results_df.ct)[0] > 1:
      #    import code; code.interact(local=dict(globals(), **locals()))
      #except:
      #  print('eee', document)

      full_query_terms = (query_dataset_full[query_dataset_full['document']==int(document)].terms.values[0]).split(' ')

      # The songs recommended by the system
      #recs = results_df[results_df.query_document==document]

      try:
        # gets the first occurrence that is found into full_query_terms
        pos_1st_match = ((1-recs.index.isin(full_query_terms).astype(int)).argsort()[0])+1
      except:
        pos_1st_match = -1
      xx = [document, len(recs), int(pos_1st_match==1), int(pos_1st_match<=5), int(pos_1st_match<=10), pos_1st_match, len(unique_query_terms), len(full_query_terms)]
      self.results.loc[len(self.results)] = (xx)

  def calc_terms_histogram(self, results_df):
    terms = ''.join(list(((results_df.term + ':')*results_df.ct)))[:-1].split(':')
    ctdf = pd.DataFrame(terms)
    return ctdf[0].value_counts()
