import random
import numpy as np
import sys
from data_load import *
from cwsabie_inner import *


cimport numpy as np
from random import randint
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
def sample_neg(t,label_size,left_size,rel_size,d_tuple,next_random):
    samples = []
    cdef unsigned long long  c_next_random = next_random
    rand = random.random()
    if rand <0.5:
        #sample head
        c_next_random = random_int32(&c_next_random)

        sample = (c_next_random%label_size,t[1],t[2])
        while sample in d_tuple:
            c_next_random = random_int32(&c_next_random)

            sample = (c_next_random%label_size,t[1],t[2])
        samples.append(sample)
    else:
        c_next_random = random_int32(&c_next_random)
        sample = (t[0],t[1],c_next_random%label_size)
        while sample in d_tuple:
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
    for v in vec:
        product += math.fabs(v)
    return product
def normalize(mat):
    for v in mat:
        norm = np.linalg.norm(v)
        if norm >=1:
            v /= norm
def norm(v):
    norm = cnorm(v)
    if norm >=1:
        cdivide(v,norm)
def train(B,R,tuples,d_tuple,next_random,it,lr = 0.01,Verbose =True):
	
    random.seed(1)
    n1 = len(B)
    n2 = len(R)
    d_tuple = set( tuple(t) for t in tuples)
    n_batches = 100
    batchsize = len(tuples)/n_batches
    loss = 0.
    dR = np.copy(R)
    dB = np.copy(B)
    #dC = np.copy(C)
  
    cnt = 0
    for _ in xrange(n_batches):
        for _ in xrange(batchsize):
            t = tuples[random.randint(0,len(tuples)-1)]
            cnt +=1
            pos_vec = tuple_sum(B,R,B,t)
            pos_scr= sqr(pos_vec)
            next_random,negs = sample_neg(t,n1,n1,n2,d_tuple,next_random)
            #print t, negs
            for neg in negs:
                neg_vec = tuple_sum(B,R,B,neg)
                neg_scr= sqr(neg_vec)
                if  pos_scr +1 >neg_scr:
                    loss += 1 + pos_scr-neg_scr
                    dB[t[0]] -= lr*pos_vec
                    dR[t[1]] -= lr*pos_vec
                    dB[t[2]] += lr*pos_vec
                    dR[neg[1]] +=lr*neg_vec
                    dB[neg[0]] +=lr*neg_vec
                    dB[neg[2]] -= lr*neg_vec
                    norm(dB[t[0]])
                    norm(dR[t[1]])
                    norm(dB[t[2]])
                    norm(dR[neg[1]])
                    norm(dB[neg[0]])
                    norm(dB[neg[2]])
            if cnt % 1000 ==0 and Verbose:
                sys.stdout.write("\rIteration %d " % (it)+ "trained {0:.0f}%".format(float(cnt)*100/len(tuples))+" Loss:{0:.2f}".format(loss))
                sys.stdout.flush()
        for i in range(len(B)):
            B[i] = dB[i]
        for i in range(len(R)):
            R[i] = dR[i]
    if Verbose:
        sys.stdout.write("\n")
    return loss,next_random
