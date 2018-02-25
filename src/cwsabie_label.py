import random
import numpy as np
import sys
from random import randint
from data_load import *
from gensim.models import Word2Vec
from cwsabie_label import *
from gensim.corpora import Dictionary
from collections import Counter
import json
from mention import *
from math import log
in_dir= sys.argv[1]
out_dir = sys.argv[2]
a=MentionData(in_dir+'/train_x_new.txt',
              in_dir+"/train_y.txt",
             in_dir+"/feature.txt",in_dir+"/type.txt")



label_bin  = Word2Vec.load_word2vec_format(in_dir+'/labels.bin')
np.random.seed(12)
size= 300
A= np.random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(a.feature2id),size]).astype(np.float32)#np.random.rand(len(a.feature2id),size).astype(np.float32)
B = np.asarray([label_bin[a.id2label[i]] for i in range(len(label_bin.vocab))],dtype=np.float32)
next_random = 1
normalize(A)
normalize(B)
for i in range(15): 
    reg_err =0.
    print 'iteration:',i,'started'
    ctrain(A,B,a.data,size,0.001,warp_gradient,it=i,Verbose=True)
    print "iteration:",i,'done.'
save2bin(A,a.id2feature,out_dir+'/warp_A.bin')
save2bin(B,a.id2label,out_dir+'/warp_B.bin')
