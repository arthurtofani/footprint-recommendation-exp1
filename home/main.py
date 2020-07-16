import time
from multiprocessing import Pool
from itertools import chain, combinations
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



logging.info("Starting evaluation")
db_index_name = 'recommendation_exp1'
dataset = '/dataset/sessions.sm.csv'
query_datasets = '/dataset/sessions_eval.sm.csv'
df = pd.read_csv(dataset)
df2 = pd.read_csv(query_datasets)
def session_tracks(document):
  tokens = df[df.session==int(document.filename)].track.astype(str)
  return ' '.join(tokens)

def session_artists(document):
  tokens = df[df.session==int(document.filename)].artist.astype(str)
  return ' '.join(tokens)

# Initialize project, db instance and evaluator
project = Project()
db.connect_to_elasticsearch(project, db_index_name, True)
project.client.set_scope(db_index_name, ['tracks'], 'tokens_by_spaces')
evaluator = Evaluator(project)

project.tokenize('tracks', session_tracks)
#project.tokenize('tracks', session_artists)

evaluator.build(dataset)
evaluator.match(query_datasets)

#import code; code.interact(local=dict(globals(), **locals()))

#p.process_feature('track_ids', features.session_tracks)

#max_processors = 3

#evaluator = evaluators.CSI(p)


#keys = [k for k in p.tokenization_methods.keys() if k not in already_processed_keys]

#import code; code.interact(local=dict(globals(), **locals()))
