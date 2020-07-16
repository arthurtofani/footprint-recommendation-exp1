import time
import footprint.evaluators as evaluators
from footprint.features import ocmi
from footprint.models import Project
from multiprocessing import Pool
from itertools import chain, combinations
import os
import numpy as np
import csv
import pandas as pd
import random
import mazurkas

def generate_clique_map(entries_path, filename):
  f = open(entries_path, 'r', encoding='utf-8')
  files = f.read().split("\n")[0:-1]
  with open(filename, 'w') as f2:
    writer = csv.writer(f2, delimiter='\t')
    for file in files:
      writer.writerow([os.path.dirname(file), file])
  f.close()

def read_clique_map(filename):
  f = open(filename, 'r', encoding='utf-8')
  s = list(csv.reader(f, delimiter='\t'))
  f.close()
  return dict([[x[1], x[0]] for x in s])

def abs_path(path):
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirname, path)

def calc_neighborhood_matrix(D, k):
    sz = len(D)
    m = np.zeros(D.shape).astype(int)
    for i in range(len(D)):
        close_friends_indices = np.argsort(D[i])[0:k]
        for j in close_friends_indices:
            m[i, j] = 1
    return m

def calc_expanded_neighborhood_matrix(m, n, k):
    def amnt_common_neighbors(arr, song_a_idx, song_b_idx):
        # Gets the indexes of songs present in the top-k ranking
        if len(arr) <= song_a_idx or len(arr) <= song_b_idx:
            return 0
        song_a_results_idx = np.argwhere(arr[song_a_idx]==True)
        song_b_results_idx = np.argwhere(arr[song_b_idx]==True)
        return len(np.intersect1d(song_a_results_idx, song_b_results_idx))

    prox_arr = np.zeros(m.shape).astype(int)

    # TODO: make it run faster
    for i in range(len(prox_arr)):
        for j in range(len(prox_arr[i])):
            rr = (amnt_common_neighbors(m, i, j) >= n)
            prox_arr[i, j] = int(rr)
            if len(prox_arr) <= i:
                prox_arr[j, i] = int(rr)
    return prox_arr

def enhance(D, k=10, k2=40, n=7, f=0.6):
  D2 = D.copy()
  k2 = 40
  k = 10
  n = 7
  vls = [1, f]
  for i_orig in range(len(D)):
      topk_idx = np.argsort(D[i_orig])[0:k2]
      M = np.zeros((k2, k2))
      for i in range(k2):
          for j in range(k2):
              try:
                  M[i, j] = D[topk_idx[i], topk_idx[j]]
              except:
                  pass
      M2 = calc_neighborhood_matrix(M,k)
      M3 = calc_expanded_neighborhood_matrix(M2, n, k).astype(int)
      for j in range(k2):
          for j2 in range(k2):
              try:
                  v = (D2[topk_idx[j], topk_idx[j2]] * vls[M3[j, j2]])
                  D2[topk_idx[j], topk_idx[j2]]  = v
              except:
                  pass
  return D2
#generate_clique_map(entries_path, expect_path)

def register_feature_extraction(main_feature, feat_name, coefs, method):
  name = main_feature + ':' + feat_name + ':' + str(coefs)
  p.process_feature(name,
                    mazurkas.features.proc_ocmi_feature(main_feature,
                                                        method,
                                                        reshape=(0, coefs)))

def add_ocmi_features(main_feature):
  for coefs in [3, 4, 6, 8, 12]:
    register_feature_extraction(main_feature, 'ocmi', coefs, ocmi.ocmi)
    register_feature_extraction(main_feature, 'docmi', coefs, ocmi.docmi)
    #register_feature_extraction(main_feature, 'ocmi_rel', coefs, ocmi.ocmi_rel)
    #register_feature_extraction(main_feature, 'docmi_rel', coefs, ocmi.docmi_rel)


def add_tokens():
  for ft in p.feature_methods.keys():
    for shingle_size in [1, 2, 4, 8, 16]:
      tk_name =  ft + ':' + str(shingle_size)
      p.tokenize(tk_name, mazurkas.tokenizers.magic_tokenizer(ft, shingle_size=shingle_size))

def preprocess_audio(filename):
  p.load_audio(filename)

def all_subsets(ss):
    r = [list(i) for i in chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))]
    return [i for i in r if len(i)>0]


max_processors = 3
clique_path = abs_path('mazurkas/configs/mazurka_cliques.csv')
#entries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')
#queries_path = abs_path('mazurkas/configs/mazurka_test_entries2.txt')
clique_map = read_clique_map(clique_path)
entries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')
queries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')


p = Project(cache_signal=True, cache_features=True, cache_tokens=True, cache_folder='/cache')
p.process_feature('chroma_cens_12', mazurkas.features.chroma_cens)
p.process_feature('crema', mazurkas.features.crema)
#p.process_feature('hpcp', mazurkas.features.hpcp)

#add_ocmi_features('chroma_cens_12')
#add_ocmi_features('crema')
#add_tokens()

register_feature_extraction('crema', 'ocmi', 4, ocmi.ocmi)
register_feature_extraction('crema', 'ocmi', 6, ocmi.ocmi)
register_feature_extraction('crema', 'docmi', 3, ocmi.docmi)
register_feature_extraction('crema', 'docmi', 4, ocmi.docmi)
p.tokenize('crema:ocmi:4:16', mazurkas.tokenizers.magic_tokenizer('crema:ocmi:4', shingle_size=16))
p.tokenize('crema:ocmi:6:1', mazurkas.tokenizers.magic_tokenizer('crema:ocmi:6', shingle_size=1))
p.tokenize('crema:docmi:3:4', mazurkas.tokenizers.magic_tokenizer('crema:docmi:3', shingle_size=4))
p.tokenize('crema:docmi:4:8', mazurkas.tokenizers.magic_tokenizer('crema:docmi:4', shingle_size=8))




evaluator = evaluators.CSI(p)

mazurkas.db.connect_to_elasticsearch(p, True)
#for total_files in evaluator.preprocess(entries_path):
#  with Pool(max_processors) as pool:
#    pool.map(preprocess_audio , total_files)

print('building')
p.client.set_scope('csi', p.tokenization_methods.keys(), 'tokens_by_spaces')
evaluator.build(entries_path)

time.sleep(3)

best_keys = ['crema:ocmi:4:16','crema:docmi:3:4','crema:docmi:4:8','crema:ocmi:6:1']
best_keys2 = [['crema:ocmi:4:16','crema:docmi:4:8','crema:ocmi:6:1'], ['crema:ocmi:4:16','crema:docmi:3:4','crema:docmi:4:8','crema:ocmi:6:1']]

already_processed_keys = list(pd.read_csv('/home/results.csv')['tokens'])
#keys = [k for k in p.tokenization_methods.keys() if k not in already_processed_keys]

keys = [k for k in all_subsets(best_keys) if len(k)>1]
keys = [k for k in all_subsets(best_keys) if ' '.join(k) not in already_processed_keys]
keys = best_keys2

results = dict()
df0 = None
df0 = pd.DataFrame.from_csv('/home/results.csv')
ct = 0
total = len(keys)
for tk in keys:
  ct+=1
  print("Running %s/%s" % (ct, total))
  print("token: %s" % tk)
  p.client.set_scope('csi', tk, 'tokens_by_spaces')
  print('matching')
  try:
    for res in evaluator.match(queries_path, amnt_results_per_query=20):
      pass
  except:
    print("Error while matching")
    import traceback
    traceback.print_exc()
    #tks = p.load_audio(evaluator.queries[0]).tokens.keys()
    import code; code.interact(local=dict(globals(), **locals()))

  print('evaluating')
  evaluator.evaluate()

  print('enhancing distance matrix...')
  clique_size = 10
  neighborhood_size = 4
  evaluator.distance_matrix = mazurkas.enhancer.enhance(evaluator.distance_matrix,
                                                        clique_size,
                                                        neighborhood_size)

  print("\n\n\n==== Results  ===")
  df1, df2 = evaluator.results(clique_map, 10)
  print('Token used: %s' % p.client.current_tokens_keys)
  print(df1.sum())
  results[','.join(tk)] = df1

  df1['tokens'] = ' '.join(tk)
  if df0 is None:
    print('none')
    df0 = df1.copy()
  else:
    print('append')
    df0 = df0.append(df1)
  df0.to_csv('results.csv')


## Medir tempo
## Medir volume médio de dados de tokens por música

  #print("==== Total correct covers at rank positions ===")
  #print(df2.sum())


#d1 = evaluator.distance_matrix.copy()
#for f in [0.9, 0.5]:
#  for k2 in [20, 30, 40]:
#    for n in [4, 7, 9]:
#      evaluator.distance_matrix = enhance(d1, k=10, k2=k2, n=n, f=f)
#      print("\n\n\n==== Results ===")
#      print("k2: %s; n: %s; f:  %s" % (k2, n, f))
#      df1, df2 = evaluator.results(clique_map, 10)
#      print(df1.sum())
#      print("==== Total correct covers at rank positions ===")
#      print(df2.sum())


#print("\n\n\n==== Results ===")
#df1, df2 = evaluator.results(clique_map, 10)
#print(df1.sum())
#print("==== Total correct covers at rank positions ===")
#print(df2.sum())

### header = 'csi'
### mazurkas.mirex.generate_output(evaluator.distance_matrix,
###                       evaluator.db_files,
###                       evaluator.queries,
###                       output_file='mazurkas/out.txt',
###                       header_content=header)


###Token used: ['crema:ocmi:4:16', 'crema:docmi:4:8', 'crema:ocmi:6:1']
###Mean Average Precision (MAP)                 0.908915
###Mean number of covers in top 10              9.205937
###Mean rank of first correct cover (MRR)       1.037106
###Total candidates                           539.000000
###Total cliques                               49.000000
###Total covers in top 10                    4962.000000
###Total queries                              539.000000
###dtype: float64
###append
###Running 2/2


## Uses 3 epochs of enhancing over the final results
## p.client.set_scope('csi', ['tk_chroma_cens', 'tk_chroma_ocmi', 'tk_crema', 'tk_crema_ocmi'], 'tokens_by_spaces')
## ==== Results ===
## Mean Average Precision (MAP)                 0.898412
## Mean number of covers in top 10              8.237477
## Mean rank of first correct cover (MRR)       1.326531
## Total candidates                           539.000000
## Total cliques                               49.000000
## Total covers in top 10                    4294.000000
## Total queries                              539.000000
## dtype: float64
## ==== Total correct covers at rank positions ===
## 1     497
## 2     493
## 3     491
## 4     491
## 5     502
## 6     498
## 7     493
## 8     492
## 9     483
## 10    460
## dtype: int64

#     import code; code.interact(local=dict(globals(), **locals()))

##p.process_feature('beat_chroma_cens', beat_sync_chroma_cens)
##p.process_feature('beat_chroma_ocmi', beat_chroma_ocmi)
##p.process_feature('chroma_ocmi_4b', proc_ocmi_feature('chroma_censx', ocmi.ocmi, reshape=(0, 4)))
##p.process_feature('crema', crema)
##p.process_feature('crema_ocmi_4b', proc_ocmi_feature('crema', ocmi.ocmi, reshape=(0, 4)))
#p.process_feature('mfcc', feat_mfcc)
#p.process_feature('mfcc_delta', feat_mfcc_delta)

##p.use_tokenizer('magic2', magic_tokenizer('beat_chroma_ocmi', min_hash_fns=20, shingle_size=2))
##p.use_tokenizer('magic1', magic_tokenizer('beat_chroma_cens', min_hash_fns=20, shingle_size=2))
##p.use_tokenizer('magic5', magic_tokenizer('crema', min_hash_fns=20, shingle_size=1))
##p.use_tokenizer('magic7', magic_tokenizer('crema_ocmi_4b', min_hash_fns=20, shingle_size=1))
#p.use_tokenizer('magic8', magic_tokenizer('mfcc', min_hash_fns=20, shingle_size=1))
#p.use_tokenizer('magic9', magic_tokenizer('mfcc_delta', min_hash_fns=20, shingle_size=1))

#p.client.set_scope('csi', ['magic4', 'magic3'], 'tokens_by_spaces') -- best
#p.add('/dataset/YTCdataset/letitbe/test.mp3')
#import code; code.interact(local=dict(globals(), **locals()))


