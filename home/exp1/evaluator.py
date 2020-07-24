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

  def match(self, query_dataset, amnt_results_per_query=10, top_n_recs=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    df = self.query_dataset = query_dataset
    amnt = len(df['document'].unique())
    for idx, document in enumerate(df['document'].unique()):
      logging.info('Matching document %s/%s' % (idx, amnt))
      query = df[df['document']==int(document)]
      query_terms = query['terms'].values[0].split(' ')
      # Perform query on database
      payload = self.project.match(str(document), amnt_results_per_query)[1]

      # Use the returned document ids to gather document info on self.dataset
      returned_records = self.dataset[self.dataset.document.isin([i.filename for i in payload])]
      all_returned_terms = ' '.join(returned_records.terms).split(' ')
      results_df = pd.DataFrame(all_returned_terms, columns=['terms'])
      results_df = results_df[~results_df.terms.isin(query_terms)]
      results_df = results_df.terms.value_counts().reset_index()
      results_df.columns = ['term', 'ct']
      results_df['query_document'] = document
      freqs = results_df[:top_n_recs]
      if self.freqs is None:
        self.freqs = freqs
      else:
        self.freqs = pd.concat([self.freqs, freqs])

  def evaluate(self, query_dataset_full):
    self.results = pd.DataFrame(columns=['document', 'returned_candidates', 'unique_candidates', 'top1_match', 'top_5_match', 'top_10_match', 'terms_in_document', 'candidate_recs', 'rank_ratio_5', 'rank_ratio_10'])
    for idx, document in enumerate(self.query_dataset['document'].unique()):
      full_document = query_dataset_full[query_dataset_full['document']==int(document)]

      # The songs recommended by the system
      query_terms = self.query_dataset[self.query_dataset['document']==document].terms.values[0]
      recs = self.freqs[self.freqs.query_document==document]
      full_document_terms = full_document['terms'].values[0].split(' ')
      recs_present_top_1 = recs['term'][:1].isin(full_document_terms).astype(int).sum()
      recs_present_top_5 = recs['term'][:5].isin(full_document_terms).astype(int).sum()>0
      recs_present_top_5 = recs_present_top_5.astype(int)
      recs_present_top_10 = recs['term'][:10].isin(full_document_terms).astype(int).sum()>0
      recs_present_top_10 = recs_present_top_10.astype(int)
      terms_in_query = len(query_terms.split(' '))
      candidate_recs = len(recs)
      try:
        rank_ratio_5 = recs_present_top_5/candidate_recs
        rank_ratio_10 = recs_present_top_10/candidate_recs
      except:
        rank_ratio_5 = 0
        rank_ratio_10 = 0
      xx = [document,    len(recs),  len(recs),           recs_present_top_1, recs_present_top_5, recs_present_top_10, terms_in_query,    candidate_recs,   rank_ratio_5, rank_ratio_10]
      self.results.loc[len(self.results)] = (xx)
