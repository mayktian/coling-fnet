import random
import numpy as np
import sys
from random import randint
from data_load import *


in_dir= sys.argv[1]
out_dir = sys.argv[2]
a=MentionData(in_dir+'/train_x_new.txt',
              in_dir+'/train_y.txt',
              in_dir+"/feature.txt",in_dir+"/type.txt")
from cwsabie_inner import *
from gensim.corpora import Dictionary
from collections import Counter
def save2bin(mat,dct,fn):
    n,d  = mat.shape
    with open(fn,'w') as out:
        out.write("%d %d\n" % (n,d))
        for i in range(n):
            text = " ".join(map(str,mat[i]))
            out.write("%s %s\n" %(dct[i],text))
np.random.seed(12)
size= int(sys.argv[3])
A= np.random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(a.feature2id),size]).astype(np.float32)#np.random.rand(len(a.feature2id),size).astype(np.float32)
B= np.random.uniform(-6/np.sqrt(size),6/np.sqrt(size), [len(a.label2id),size]).astype(np.float32)# np.random.rand(len(a.label2id),size).astype(np.float32)
next_random = 1
normalize(A)
normalize(B)
for i in range(15): 
    ctrain(A,B,a.data,size,0.001,warp_gradient,it=i,Verbose=True)
save2bin(A,a.id2feature,out_dir+'/warp_A.bin')
save2bin(B,a.id2label,out_dir+'/warp_B.bin')
