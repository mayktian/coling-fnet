import random
import numpy as np
import sys
from random import randint
from data_load import *
from cwsabie_inner import *
from gensim.corpora import Dictionary

cimport numpy as np
import cython
cdef extern from "math.h":
    double sqrt(double m)
import math
from libc.stdlib cimport malloc, free

from libc.math cimport exp
from libc.math cimport log

from libc.string cimport memset

# scipy <= 0.15

import scipy.linalg.blas as fblas
ctypedef np.float32_t REAL_t
cdef int ONE = 1

cdef REAL_t ONEF = <REAL_t>1.0
REAL = np.float32
cdef extern from "/Users/mayk/working/figer/baseline/PLE/Model/warp/voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)
    
def tuple_sum(B,R,C,t):
    h_vec = B[t[0]]
    r_vec = R[t[1]]
    t_vec = C[t[2]]
    sum_vec = h_vec+ r_vec - t_vec
    return sum_vec
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return next_random[0]
def sample_neg(ts,t,label_size,left_size,rel_size,d_tuple,next_random):
    samples = []
    cdef unsigned long long  c_next_random = next_random
    rand = random.random()
    if rand <0.5:
        #sample head
        c_next_random = random_int32(&c_next_random)

        sample = (c_next_random%label_size,t[1],t[2])

        while sample in d_tuple or sample[0] in ts[0]:
            c_next_random = random_int32(&c_next_random)

            sample = (c_next_random%label_size,t[1],t[2])
        samples.append(sample)
    else:
        c_next_random = random_int32(&c_next_random)
        sample = (t[0],t[1],c_next_random%label_size)
        while sample in d_tuple or sample[2] in ts[2]:
            c_next_random = random_int32(&c_next_random)
            sample = (t[0],t[1],c_next_random%label_size)
        samples.append(sample)
    c_next_random = random_int32(&c_next_random)

    sample = (t[0],c_next_random%rel_size,t[2])
    while sample in d_tuple:
        c_next_random = random_int32(&c_next_random)

        sample = (t[0],c_next_random%rel_size,t[2])
    samples.append(sample)
    return c_next_random,samples
            #sample rel
def sample_neg_n(ts,t,label_size,left_size,rel_size,d_tuple,next_random,sample_n=50):
    samples =[]
    
    for i in range(sample_n):
        next_random,sample =sample_neg(ts,t,label_size,left_size,rel_size,d_tuple,next_random)
        samples.extend(sample)
    return next_random,samples
cdef REAL_t csqr(REAL_t *vec,int *size):
    cdef REAL_t product = 0.
    cdef int i = 0
    for i in range(size[0]):
        product += vec[i]**2
    return product
def sqr(v):
    cdef int size= len(v)
    return csqr(<REAL_t *>(np.PyArray_DATA(v)),&size)
def l1_norm(vec):
    product = 0
    signs = np.ones(len(vec),dtype=np.float32)
    for i,v in enumerate(vec):
        if v <0:
            signs[i] =-1.
        product += abs(v)
    return product,signs
def normalize(mat):
    for v in mat:
        norm = np.linalg.norm(v)
        if norm >=1:
            v /= norm
def norm(v):
    norm = cnorm(v)
    if norm >=1:
        cdivide(v,norm)
def train(B,R,tuples,d_tuple,next_random,it,N,lr = 0.01,L2=True,Verbose =True):
  #  random.seed(1)
    n1 = len(B)
    n2 = len(R)

    batchsize = len(tuples)
    loss = 0.
    dR = np.copy(R)
    dB = np.copy(B)
    #dC = np.copy(C)
    cnt = 0
    lam =0.1
    ra = 1
    for _ in xrange(batchsize):
        ts = tuples[random.randint(0,len(tuples)-1)]
        cnt +=1
        for l in ts[0]:
            for r in ts[2]:
                if l == r:
                    continue
                ra =1
                t = tuple([l,ts[1],r])
                pos_vec = tuple_sum(B,R,B,t)
                if L2:
                    pos_scr= sqr(pos_vec)
                else:
                    pos_scr,sgns_pos = l1_norm(pos_vec)
                next_random,negs = sample_neg_n(ts,t,n1,n1,n2,d_tuple,next_random,sample_n=10)
                for neg in negs:
                    neg_vec = tuple_sum(B,R,B,neg)
                    if L2:
                        neg_scr= sqr(neg_vec)
                    else:
                        neg_scr,sgns_neg = l1_norm(neg_vec)
                    if  pos_scr +1 >neg_scr:
                        factor = lr*rank(N/ra)
                        loss += 1 + pos_scr-neg_scr
                        if L2:
                            dB[t[0]] -= factor*pos_vec
                            dR[t[1]] -= factor*pos_vec
                            dB[t[2]] += factor*pos_vec
                            dR[neg[1]] +=factor*neg_vec
                            dB[neg[0]] +=factor*neg_vec
                            dB[neg[2]] -= factor*neg_vec
                        else:
                            dB[t[0]] -= lr*sgns_pos
                            dR[t[1]] -= lr*sgns_pos
                            dB[t[2]] += lr*sgns_pos
                            dR[neg[1]] +=lr*sgns_neg
                            dB[neg[0]] +=lr*sgns_neg
                            dB[neg[2]] -= lr*sgns_neg
                        norm(dB[t[0]])
                        norm(dR[t[1]])
                        norm(dB[t[2]])
                        norm(dR[neg[1]])
                        norm(dB[neg[0]])
                        norm(dB[neg[2]])
                        break
                    else:ra+=1
        if cnt % 1000 ==0 and Verbose:
            sys.stdout.write("\rIteration %d " % (it)+ "trained {0:.0f}%".format(float(cnt)*100/len(tuples))+" Loss:{0:.2f}".format(loss))
            sys.stdout.flush()
    for i in range(len(B)):
        B[i] = dB[i]-lam
    for i in range(len(R)):
        R[i] = dR[i]-lam
    if Verbose:
        sys.stdout.write("\n")
    return loss,next_random
def cnn_train(B,W,V,b,seqs,tuples,d_tuple,next_random,it,lr = 0.01,Verbose =True):
    random.seed(1)
    n1 = len(B)
    n2 = len(seqs)
    d_tuple = set( tuple(t) for t in tuples)
    n_batches = 100
    batchsize = len(tuples)/n_batches
    loss = 0.
    window = 3
    dB = np.copy(B)
    nsize =50
    size =50
    cnt = 0
    dV = np.zeros(V.shape,dtype=np.float32)
    dW = np.zeros(W.shape,dtype=np.float32)
    db = np.zeros(nsize,dtype=np.float32)
    for _ in xrange(n_batches):
        for _ in xrange(batchsize):
            t = tuples[random.randint(0,len(tuples)-1)]
            cnt +=1
            rel,max_pool_pos,H_pos = cal_cnn_vec(seqs[t[1]],window,W,nsize,V,b)
            pos_vec = B[t[0]]+rel- B[t[2]]
            pos_scr= sqr(pos_vec)
            next_random,negs = sample_neg_n(t,n1,n1,n2,d_tuple,next_random)
            for neg in negs:
                rel,max_pool,H = cal_cnn_vec(seqs[neg[1]],window,W,nsize,V,b,sample_n=10)
                neg_vec =B[neg[0]]+rel- B[neg[2]]
                neg_scr= sqr(neg_vec)
                if  pos_scr +1 >neg_scr:
                    
                    loss += 1 + pos_scr-neg_scr
                    dB[t[0]] -= lr*pos_vec
                    gradient(-3*lr*pos_vec,dV,dW,db,seqs[t[1]],window,W,nsize,size,V,max_pool_pos,H_pos)
                    dB[t[2]] += lr*pos_vec 
                    gradient(3*lr*neg_vec,dV,dW,db,seqs[neg[1]],window,W,nsize,size,V,max_pool,H)
                    dB[neg[0]] +=lr*neg_vec
                    dB[neg[2]] -= lr*neg_vec
                    norm(dB[t[0]])
#                     norm(dR[t[1]])
                    norm(dB[t[2]])
                    # norm(dR[neg[1]])
                    norm(dB[neg[0]])
                    norm(dB[neg[2]])
            if cnt % 1000 ==0 and Verbose:
                sys.stdout.write("\rIteration %d " % (it)+ "trained {0:.0f}%".format(float(cnt)*100/len(tuples))+" Loss:{0:.2f}".format(loss))
                sys.stdout.flush()
        for i in range(len(B)):
            B[i] = dB[i]
        for i in range(len(W)):
            W[i] += dW[i]
        for i in range(len(V)):
            V[i] += dV[i]
        for i in range(len(b)):
            b[i] += db[i]
    if Verbose:
        sys.stdout.write("\n")
    return loss,next_random
def cal_cnn_vec(seq,window,W,nsize,V,b):
    H =[]
#     mid_H = []
    max_pool = np.zeros(nsize,dtype=np.int32)
    for i in range(0,len(seq)-window+1):
        vec = np.dot(W[:nsize],V[seq[i]])
        for j in range(1,window): #convolution layer
            vec += np.dot(W[j*nsize:(j+1)*nsize],V[seq[i+j]])
        vec += b # bias
        np.tanh(vec,vec,dtype=np.float32)
        H.append(vec)
    h = np.zeros(nsize)
    for i in range(nsize):
        h[i] = -float('inf')
        for j,vec in enumerate(H):
            if vec[i] > h[i]:
                h[i] = vec[i]
                max_pool[i] = j

    return h.astype(np.float32),max_pool,H
def gradient(err_vec,dV,dW,db,seq,window,W,nsize,size,V,max_pool,H):
   
    for i in range(nsize):
        non_linear = err_vec[i]*(1- H[max_pool[i]][i]*H[max_pool[i]][i])
        for j in range(0,window):
            dV[seq[j]] += non_linear*W[j*nsize:(j+1)*nsize][i]
            dW[j*nsize:(j+1)*nsize][i] += non_linear*V[seq[j]]
        db[i] += non_linear
def save2bin(mat,dct,fn):
    n,d  = mat.shape
    with open(fn,'w') as out:
        out.write("%d %d\n" % (n,d))
        for i in range(n):
            text = " ".join(map(str,mat[i]))
            out.write("%s %s\n" %(dct[i],text))
            
cdef REAL_t crank(int k):
    cdef REAL_t loss = 0.
    cdef int i = 1
    for i in range(1,k+1):
        loss += ONEF/i
    return loss
def rank(i):
    return crank(i)
