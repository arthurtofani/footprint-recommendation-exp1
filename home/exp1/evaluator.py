import numpy as np
import glob
import random
from collections import defaultdict
import pandas as pd
import sklearn
import os.path
from footprint import util as util
import uuid
import csv
import logging

class Evaluator:
  records_list_path = None
  db_files = None
  query_files = None
  project = None
  match_results = None
  ranking_size = None
  distance_matrix = None

  def __init__(self, project):
    self.project = project
    self.distance_matrix = None

  def build(self, dataset):
    logging.info('Building dataset')
    df = pd.read_csv(dataset)
    for idx, session in enumerate(df.session.unique()):
      logging.info('Adding session %s' % session)
      self.project.add(str(session))


  def match(self, query_files, amnt_results_per_query=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    df = pd.read_csv(query_files)

    dataset = '/dataset/sessions.sm.csv'
    query_datasets = '/dataset/sessions_eval.sm.csv'
    df  = pd.read_csv(dataset)
    df2 = pd.read_csv(query_datasets)

    for idx, session in enumerate(df.session.unique()):
      logging.info('matching session %s' % session)
      res = self.project.match(str(session), amnt_results_per_query)

      query = df2[df2.session==int(session)]
      results = [(i.filename, i.score) for i in res[1]]

      import code; code.interact(local=dict(globals(), **locals()))
      x = df[df.session.isin([i.filename for i in res[1]])]
      x.groupby(['track']).count().sort_values(['user'], ascending=False).user


      def check(xx):
        return df[df.session==int(xx)]



    #    ct = 0
    #    self.query_files = query_files
    #    self.match_results = []
    #    self.amnt_results_per_query = amnt_results_per_query
    #    self.queries = self.get_filenames(self.query_files)
    #
    #    sz = len(self.queries)
    #    for f in self.queries:
    #      ct+=1
    #      print('%s of %s => %s' % (ct, sz, f))
    #      res = self.project.match(f, self.amnt_results_per_query)
    #      #yield res
    #      res[0].features = dict()
    #      self.match_results.append(res)

  def evaluate(self):
    pass

  def results(self, clique_map, ranking_size=10):
    pass
