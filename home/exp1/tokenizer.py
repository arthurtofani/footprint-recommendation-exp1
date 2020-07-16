import footprint.tokenizers as tokenizers
from itertools import zip_longest
from datasketch import MinHash
import hashlib
import numpy as np

def exp1_tokenizer(feature_name, min_hash_fns=10, shingle_size=3):
  def tokenize_method(audio):
    feature = audio.features[feature_name]
    try:
      t = tokenizers.magic_hash(feature, min_hash_fns=min_hash_fns, shingle_size=shingle_size)
    except:
      print('error tokenizing %s for %s' % (feature_name, audio.filename))
      t = 'asd sdf dfg'
    return t
  return tokenize_method
