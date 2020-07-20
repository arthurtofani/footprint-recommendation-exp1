
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
  track_frequencies = None
  results = None

  def __init__(self, project):
    self.project = project
    self.results = None

  def build(self, dataset):
    logging.info('Building dataset')
    self.dataset = dataset
    for idx, session in enumerate(self.dataset.session.unique()):
      logging.info('Adding session %s' % session)
      self.project.add(str(session))


  def match(self, query_dataset, amnt_results_per_query=10, top_n_tracks=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    df = self.query_dataset = query_dataset
    for idx, session in enumerate(df.session.unique()):
      logging.info('matching session %s' % session)
      query = df[df.session==int(session)]

      # Perform query on database
      payload = self.project.match(str(session), amnt_results_per_query)[1]


      # Use the returned session ids to gather session info on self.dataset
      returned_tracks = self.dataset[self.dataset.session.isin([i.filename for i in payload])]
      returned_tracks = returned_tracks[~returned_tracks.track.isin(query.track)] # remove songs already in query session

      # Calculate the most frequent tracks associated to that query session (but not in the session)
      track_frequencies = returned_tracks.groupby(['track']).count()
      track_frequencies = track_frequencies.reset_index()
      track_frequencies['track_num'] = track_frequencies['track'].astype(int)
      track_frequencies = track_frequencies.sort_values(['session', 'track_num'], ascending=[False, True])
      track_frequencies = track_frequencies[['track', 'user']]
      track_frequencies.columns = ['track', 'count']
      track_frequencies = track_frequencies[:top_n_tracks]
      track_frequencies['query_session'] = session
      if self.track_frequencies is None:
        self.track_frequencies = track_frequencies
      else:
        self.track_frequencies = pd.concat([self.track_frequencies, track_frequencies])

  def evaluate(self, query_dataset_full):
    self.results = pd.DataFrame(columns=['session', 'returned_candidates', 'unique_candidates', 'top1_match', 'top_n_match', 'terms_in_session', 'candidate_tracks', 'rank_ratio'])
    for idx, session in enumerate(self.query_dataset.session.unique()):
      full_session = query_dataset_full[query_dataset_full.session==int(session)]

      # The songs recommended by the system
      query = self.query_dataset[self.query_dataset.session==session]
      tracks = self.track_frequencies[self.track_frequencies.query_session==session]
      tracks_present_top_1 = tracks.track[:1].isin(full_session.track).astype(int).sum()
      tracks_present_top_n = tracks.track.isin(full_session.track).astype(int).sum()
      terms_in_query = len(query)
      candidate_tracks = len(tracks)
      try:
        rank_ratio = tracks_present_top_n/candidate_tracks
      except:
        rank_ratio = 0
      #import code; code.interact(local=dict(globals(), **locals()))
      self.results.loc[len(self.results)] = ([session, tracks['count'].sum(), len(tracks), tracks_present_top_1, tracks_present_top_n, terms_in_query, candidate_tracks, rank_ratio])
