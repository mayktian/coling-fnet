
import random
import numpy as np
import sys
from random import randint
from data_load import *
from cwsabie_inner import *
from gensim.corpora import Dictionary
from cython_Wsabie_transE import *

in_dir= "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN"
a=MentionData('/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/train_x_new.txt',
              "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/train_y.txt",
             in_dir+"/feature.txt",in_dir+"/type.txt")

#compatible with wsabie
def getLabelRel(fn):
    return processLines(fn,lambda x,y:addFeature(x,y.rstrip().split('\t')),list())

tuples = getLabelRel('/Users/mayk/working/figer/baseline/PLE/Model/warp/train.txt')
#getLabelRel('/Users/mayk/working/rel_extractor/baseline/genie-kb/data_rel/train_text.txt')
#'/Users/mayk/working/rel_extractor/baseline/genie-kb/data_fb15/train.txt')
#label_dict = #dict([[t[0],int(t[1])] for t in getLabelRel('/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/type.txt')] )
tuples = [ tuple([t[0].split('#'),t[1],t[2].split('#')]) for t in tuples]
rel_dict = Dictionary([[t[1]] for t in tuples])


seqs = [rel_dict[i].split(':') for i in range(len(rel_dict))]
word_dict = Dictionary(seqs)
seqs = map(lambda x:[word_dict.token2id[w] for w in x],seqs)

tuples = [[[a.label2id[x] for x in t[0]],rel_dict.token2id[t[1]],[a.label2id[x] for x in t[2]]] for t in tuples]
d_tuple = set( tuple([l,t[1],r]) for t in tuples for l in t[0] for r in t[2])

np.random.seed(int(sys.argv[1]))
size= 50
A= np.random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(a.feature2id),size]).astype(np.float32)#np.random.rand(len(a.feature2id),size).astype(np.float32)
B= np.random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(a.label2id),size]).astype(np.float32)# np.random.rand(len(a.label2id),size).astype(np.float32)
R = np.random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(rel_dict),size]).astype(np.float32)
#R = np.zeros([len(rel_dict),size]).astype(np.float32)#random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(rel_dict),size]).astype(np.float32)
N =len(d_tuple)#len(rel_dict)*len(a.label2id)*len(a.label2id)/10000
next_random = 1
n_batches = 10
batch_size = len(tuples) / n_batches
wn_batches = 10
wbatch_size = len(a.data) / wn_batches
reg_err =0.
flag = int(sys.argv[2])
#random.seed(3)
for i in range(9):
    print "iteration:",i
    reg_err =0.
    ctrain(A,B,a.data,50,0.01,warp_gradient,it=i)
    if flag:
        loss,next_random = train(B,R,tuples,d_tuple,next_random,i,N,lr=0.0001,L2=True,Verbose = False)
ctrain(A,B,a.data,50,0.01,warp_gradient,it=i)

save2bin(A,a.id2feature,'/Users/mayk/working/figer/baseline/PLE/Results/warp_A.bin')
save2bin(B,a.id2label,'/Users/mayk/working/figer/baseline/PLE/Results/warp_B.bin')

