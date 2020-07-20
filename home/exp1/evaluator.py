
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
    amnt = len(self.dataset.session.unique())
    for idx, session in enumerate(self.dataset.session.unique()):
      logging.info('Adding session %s/%s' % (idx, amnt))
      self.project.add(str(session))


  def match(self, field_name, query_dataset, amnt_results_per_query=10, top_n_recs=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    df = self.query_dataset = query_dataset
    amnt = len(df.session.unique())
    for idx, session in enumerate(df.session.unique()):
      logging.info('Matching session %s/%s' % (idx, amnt))
      query = df[df.session==int(session)]

      # Perform query on database
      payload = self.project.match(str(session), amnt_results_per_query)[1]


      # Use the returned session ids to gather session info on self.dataset
      returned_records = self.dataset[self.dataset.session.isin([i.filename for i in payload])]
      returned_records = returned_records[~returned_records[field_name].isin(query[field_name])] # remove songs already in query session

      # Calculate the most frequent recs associated to that query session (but not in the session)
      freqs = returned_records.groupby([field_name]).count()
      freqs = freqs.reset_index()
      freqs['track_num'] = freqs[field_name].astype(int)
      freqs = freqs.sort_values(['session', 'track_num'], ascending=[False, True])
      freqs = freqs[[field_name, 'user']]
      freqs.columns = [field_name, 'count']
      freqs = freqs[:top_n_recs]
      freqs['query_session'] = session
      if self.freqs is None:
        self.freqs = freqs
      else:
        self.freqs = pd.concat([self.freqs, freqs])

  def evaluate(self, field_name, query_dataset_full):
    self.results = pd.DataFrame(columns=['session', 'returned_candidates', 'unique_candidates', 'top1_match', 'top_n_match', 'terms_in_session', 'candidate_recs', 'rank_ratio'])
    for idx, session in enumerate(self.query_dataset.session.unique()):
      full_session = query_dataset_full[query_dataset_full.session==int(session)]

      # The songs recommended by the system
      query = self.query_dataset[self.query_dataset.session==session]
      recs = self.freqs[self.freqs.query_session==session]
      recs_present_top_1 = recs[field_name][:1].isin(full_session[field_name]).astype(int).sum()
      recs_present_top_n = recs[field_name].isin(full_session[field_name]).astype(int).sum()
      terms_in_query = len(query)
      candidate_recs = len(recs)
      try:
        rank_ratio = recs_present_top_n/candidate_recs
      except:
        rank_ratio = 0
      #import code; code.interact(local=dict(globals(), **locals()))
      self.results.loc[len(self.results)] = ([session, recs['count'].sum(), len(recs), recs_present_top_1, recs_present_top_n, terms_in_query, candidate_recs, rank_ratio])
