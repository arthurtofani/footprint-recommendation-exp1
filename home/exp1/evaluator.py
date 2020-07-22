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

  def build(self, document_field, dataset):
    logging.info('Building dataset')
    self.dataset = dataset
    amnt = len(self.dataset[document_field].unique())
    for idx, document in enumerate(self.dataset[document_field].unique()):
      logging.info('Adding document %s/%s' % (idx, amnt))
      self.project.add(str(document))


  def match(self, document_field, term_field, query_dataset, amnt_results_per_query=10, top_n_recs=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    df = self.query_dataset = query_dataset
    amnt = len(df[document_field].unique())
    for idx, document in enumerate(df[document_field].unique()):
      logging.info('Matching document %s/%s' % (idx, amnt))
      query = df[df[document_field]==int(document)]

      # Perform query on database
      payload = self.project.match(str(document), amnt_results_per_query)[1]


      # Use the returned document ids to gather document info on self.dataset
      returned_records = self.dataset[self.dataset[document_field].isin([i.filename for i in payload])]
      returned_records = returned_records[~returned_records[term_field].isin(query[term_field])] # remove songs already in query document

      # Calculate the most frequent recs associated to that query document (but not in the document)
      freqs = returned_records.groupby([term_field]).count()
      freqs = freqs.reset_index()
      freqs['track_num'] = freqs[term_field].astype(int)
      freqs = freqs.sort_values([document_field, 'track_num'], ascending=[False, True])
      freqs = freqs[[term_field, 'user']]
      freqs.columns = [term_field, 'count']
      freqs = freqs[:top_n_recs]
      freqs['query_document'] = document
      if self.freqs is None:
        self.freqs = freqs
      else:
        self.freqs = pd.concat([self.freqs, freqs])

  def evaluate(self, document_field, term_field, query_dataset_full):
    self.results = pd.DataFrame(columns=[document_field, 'returned_candidates', 'unique_candidates', 'top1_match', 'top_5_match', 'top_10_match', 'terms_in_document', 'candidate_recs', 'rank_ratio_5', 'rank_ratio_10'])
    for idx, document in enumerate(self.query_dataset[document_field].unique()):
      full_document = query_dataset_full[query_dataset_full[document_field]==int(document)]

      # The songs recommended by the system
      query = self.query_dataset[self.query_dataset[document_field]==document]
      recs = self.freqs[self.freqs.query_document==document]
      recs_present_top_1 = recs[term_field][:1].isin(full_document[term_field]).astype(int).sum()
      recs_present_top_5 = recs[term_field][:5].isin(full_document[term_field]).astype(int).sum()
      recs_present_top_10 = recs[term_field][:10].isin(full_document[term_field]).astype(int).sum()
      terms_in_query = len(query)
      candidate_recs = len(recs)
      try:
        rank_ratio_5 = recs_present_top_5/candidate_recs
        rank_ratio_10 = recs_present_top_10/candidate_recs
      except:
        rank_ratio_5 = 0
        rank_ratio_10 = 0
      #import code; code.interact(local=dict(globals(), **locals()))
      xx = [document,    recs['count'].sum(),  len(recs),           recs_present_top_1, recs_present_top_5, recs_present_top_10, terms_in_query,    candidate_recs,   rank_ratio_5, rank_ratio_10]

      self.results.loc[len(self.results)] = (xx)
