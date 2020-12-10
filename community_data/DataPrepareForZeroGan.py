import community_data.data_loader as data
import json
import numpy as np
import math
from collections import defaultdict
import copy


# load the data
A, D, X, actual_Z, actual_W = data.conditional_n_community([ 50, 50, 50 , 50, 50,49],p_inter=0.01, p_intera=0.4)


#----------------------------------------------
# relation2ids
relation2ids = {}
for i, rel in enumerate(A):
    relation2ids["r_"+str(i)] = i
with open('relation2ids', 'w') as outfile:
    json.dump(relation2ids, outfile)

#----------------------------------------------
# rela_matrix
np.savez("rela_matrix", relaM=np.stack( D, axis=0))
#----------------------------------------------

tasks = {}
for i  in range(len(A)):
    entities = []
    in1,in2 =  A[i].nonzero()
    for node1, node2 in zip(in1,in2):
        entities.append(["node_"+str(node1), "r_"+str(i), "node_"+str(node2)])
    tasks["r_"+str(i)] = entities

# e1rel_e2_all
e1rel_e2_all = {}
for key, valu in tasks.items():
    for n1, r, n2 in valu:
        e1rel_e2_all[n1+r] = n2
with open('e1rel_e2_all.json', 'w') as outfile: #["node_46r_5"]
    json.dump(e1rel_e2_all, outfile)

#-----------------------------------
np.random.seed(0)
indexes = list(range(len(A)))
np.random.shuffle(indexes)
val_size = math.ceil(.05*len(indexes))
test_size =  math.ceil(.1*len(indexes))

# train_tasks.json
train_tasks = {}
for i in indexes[test_size+val_size:]:
    train_tasks["r_"+str(i)] = tasks["r_"+str(i)]


with open('train_tasks.json', 'w') as outfile:
    json.dump(train_tasks, outfile)

graphet = []
for _,relations in train_tasks.items():
    for e1,r,e2 in relations:
        graphet.append(e1+'\t'+r+'\t'+e2)
        if e1=="node_45" and r=="r_5":

            print()
with open('path_graph', 'w') as outfile:
    for el in graphet:
        outfile.write("%s\n" % el)
#----------------------------------------------
# dev_tasks.json
dev_tasks = {}
for i in indexes[0:val_size]:
    dev_tasks["r_"+str(i)] = tasks["r_"+str(i)]

with open('dev_tasks', 'w') as outfile:
    json.dump(dev_tasks, outfile)
#----------------------------------------------
# test_tasks.json
test_tasks = {}
for i in indexes[val_size:test_size+val_size]:
    test_tasks["r_"+str(i)] = tasks["r_"+str(i)]
with open('test_tasks', 'w') as outfile:
    json.dump(test_tasks, outfile)

#----------------------------------------------
# entity2id
entity2id = {}
for i in range(X.shape[0]):
    entity2id["node_"+str(i)] = i
with open('entity2id', 'w') as outfile:
    json.dump(entity2id, outfile)

#----------------------------------------------
#e1rel_e2
e1rel_e2 = defaultdict()

for _, all_rel_type_i in train_tasks.items():
    for (e1, r, e2) in all_rel_type_i:
        e1rel_e2[e1+r] = e2
json.dump(e1rel_e2, open('e1rel_e2.json', 'w'))

#-----------------------------------------------
# rel2candidates_all
rel2candidates = {}
for rel_i in relation2ids.keys():
    rel2candidates[rel_i] = list(entity2id.keys())
json.dump(rel2candidates, open( 'rel2candidates_all.json', 'w'))

#-----------------------------------------------
# embeddings
rel_em = np.array(D)
node_em = X.todense()


rep = np.identity(X.shape[0] + len(D))
np.savez('DistMult_embed.npz', rM=rep[:len(D)], eM=rep[len(D):])