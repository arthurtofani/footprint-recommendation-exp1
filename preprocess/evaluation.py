import pandas as pd
import numpy as np

def recall_at_k(preds,target,k):
    if not isinstance(preds, float):
        if target in preds[:k]: return 1
        else: return 0
    else:
        if target == preds: return 1
        else: return 0

def intersec_at_k(preds,target,k):
    if not isinstance(preds, float):
        return len(np.intersect1d(preds[:k],target))/len(target)
    else:
        if target == preds: return 1
        else: return 0

df = pd.read_csv("sessions_tracks.queries.csv")

recall1=[]
recall5=[]
recall10=[]

for idx,item in df.iterrows():
    # next item in the session
    if not isinstance(item["missing_terms"], float):
        target=item["missing_terms"].split(" ")[0]
    else:
        target = item["missing_terms"]

    if not isinstance(item["recommended_terms"], float):
        preds = item["recommended_terms"].split(" ")
    else:
        preds = item["recommended_terms"]

    recall1.append(recall_at_k(preds,target,1))
    recall5.append(recall_at_k(preds,target,5))
    recall10.append(recall_at_k(preds,target,10))


print("----------  NEXT ITEM -------------")

print("RECALL@1: {}".format(np.mean(recall1)))
print("RECALL@5: {}".format(np.mean(recall5)))
print("RECALL@10: {}".format(np.mean(recall10)))


print("----------  INTERSECTION -------------")

for idx,item in df.iterrows():
    # next item in the session
    if not isinstance(item["missing_terms"], float):
        target=item["missing_terms"].split(" ")[0]
    else:
        target = item["missing_terms"]

    if not isinstance(item["recommended_terms"], float):
        preds = item["recommended_terms"].split(" ")
    else:
        preds = item["recommended_terms"]

    recall1.append(intersec_at_k(preds,target,1))
    recall5.append(intersec_at_k(preds,target,5))
    recall10.append(intersec_at_k(preds,target,10))

print("RECALL@1: {}".format(np.mean(recall1)))
print("RECALL@5: {}".format(np.mean(recall5)))
print("RECALL@10: {}".format(np.mean(recall10)))
