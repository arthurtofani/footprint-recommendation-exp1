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

def consolidate_results(output_df, partial_results_file):
  results_mean = results.mean()
  results_median = results.median()
  print(results_mean)

  complete_row = [itr, len(results), 0, document_field, term_field,
    memory_size, results_mean.top1_match, results_mean.top_5_match,
    results_mean.top_10_match, len(train_dataset),
    len(query_dataset_full), query_terms_ratio]

  final_df = load_output_dataframe(output_file)
  try:
    final_df.loc[len(final_df)] = complete_row
  except:
    pass
    #import code; code.interact(local=dict(globals(), **locals()))

  for i, r in results.groupby(['terms_in_document']).mean().iterrows():
    #import code; code.interact(local=dict(globals(), **locals()))
    row = [itr, i, document_field, term_field,
    memory_size, r.top1_match, r.top_5_match,
    r.top_10_match, len(train_dataset),
    len(query_dataset_full), query_terms_ratio, results_median.query_terms]
    final_df.loc[len(final_df)] = row
  safe_sleep = random.random() * 10
  time.sleep(safe_sleep)
  final_df.to_csv(output_file, index=False)
