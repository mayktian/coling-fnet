import random
import numpy as np
import sys
from random import randint
from data_load import *
from gensim.corpora import Dictionary
from collections import Counter

cimport numpy as np
from random import randint

import cython
cdef extern from "math.h":
    double sqrt(double m)
import math
from libc.stdlib cimport malloc, free

from libc.math cimport exp
from libc.math cimport log
from gensim.matutils import argsort

from libc.string cimport memset
import random
# scipy <= 0.15

import scipy.linalg.blas as fblas
ctypedef np.float32_t REAL_t
cdef int ONE = 1


REAL = np.float32
cdef extern from "/Users/mayk/working/figer/baseline/PLE/Model/warp/voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)
DEF MAX_SENTENCE_LEN = 10000
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil


cdef scopy_ptr scopy = <scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
DEF EXP_TABLE_SIZE = 10000
DEF MAX_EXP = 50

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef REAL_t ONEF = <REAL_t>1.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < 50 by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]
cdef REAL_t cvdot(vec1,vec2,size):
    cdef int csize = size
    f= dsdot(&csize,<REAL_t *>(np.PyArray_DATA(vec1)),&ONE,<REAL_t *>(np.PyArray_DATA(vec2)),&ONE)
    return f
def csaxpy(vec1,vec2,alpha,size):
    cdef int csize = size
    cdef float calpha = alpha
    f= our_saxpy_noblas(&csize,&calpha,<REAL_t *>(np.PyArray_DATA(vec1)),&ONE,<REAL_t *>(np.PyArray_DATA(vec2)),&ONE)
    return f
cdef REAL_t crank(int k):
    cdef REAL_t loss = 0.
    cdef int i = 1
    for i in range(1,k+1):
        loss += ONEF/i
    return loss
cdef REAL_t vsum(REAL_t *vec,int *size):
    cdef int i
    cdef REAL_t product
    product = <REAL_t>0.0
    for i from 0 <= i < size[0] by 1:
        product += vec[i] **2
    return sqrt(product)
def cnorm(vec):
    cdef int size
    size  = len(vec)
    return vsum(<REAL_t *>(np.PyArray_DATA(vec)),&size)
def init():
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
#init()


def ctrain(A,B,insts,size,lr,gradient,it,Verbose=False):
    cdef float error
    next_random = 1
    error = 0.
    for i,inst in enumerate(insts):
        err,next_random =gradient(A,B,inst,size,next_random,lr=lr)
        error += err
        if i % 1000 ==0 and Verbose:
            sys.stdout.write("\rIteration %d " % (it)+ "trained {0:.0f}%".format(float(i)*100/len(insts))+" Loss:{0:.2f}".format(error))
            sys.stdout.flush()
    if Verbose:
        sys.stdout.write("\n")
    return error

cdef void divide(REAL_t *vec, const float *alpha, const int *size):
    cdef int i
    for i from 0 <= i < size[0] by 1:
        vec[i] = vec[i]/alpha[0]
def cdivide(vec,alpha):
    cdef int size
    size  = len(vec)
    cdef float r = alpha
    divide(<REAL_t *>(np.PyArray_DATA(vec)),&r,&size)

    


def warp_gradient(A,B,inst,size,next_random,lr=0.01):
    #print B
    #print B[0]-B[9]
    cdef unsigned long long  c_next_random = next_random
    dA = np.zeros(size,dtype=REAL)
    dB = np.zeros([len(inst.labels),size],dtype=REAL)
    x = np.sum(A[inst.features],axis=0)
    cdef REAL_t error = 0.
    cdef REAL_t clr = lr
    cdef int N,n_sample 
    cdef int neg_num = len(inst.negative_labels)
    cdef REAL_t norm
    cdef int cSize = size
    cdef REAL_t floats
    scores = [ cvdot(x,B[l],cSize) for l in inst.sparse_labels]
   # ranks = argsort(scores,reverse=True)
    M = len(inst.sparse_labels)
    for i,l in enumerate(inst.sparse_labels):
        s1= scores[i]#cvdot(x,B[l],50)
        N=1
        n_sample  = -1
        for k in range(neg_num):
            c_next_random = random_int32(&c_next_random)
            nl = inst.negative_labels[c_next_random%neg_num]#randint(0,neg_num-1)]
            s2 = cvdot(x,B[nl],cSize)
            
            if s1 - s2<1:
                n_sample = nl
                N = k+1
                break
        if n_sample!=-1:

            L = crank(len(inst.negative_labels)/N)#*(crank(M/(ranks[i]+1)))
            negL = -L
            error += (1+s2-s1)*L

            csaxpy(B[l]-B[n_sample],dA,L,cSize)
            
#             csaxpy(x,dB[l],L,cSize)
#             csaxpy(x,dB[n_sample],-L,cSize)
            #print dB[l][0]
    for f in inst.features:
        csaxpy(dA,A[f],clr,cSize)
        norm = cnorm(A[f])
        if norm >1:
            cdivide(A[f],norm)

#     for i in range(len(B)):
#         csaxpy(dB[i],B[i],clr,cSize)

#         #B[i] += lr*dB[i]
#         norm =  cnorm(B[i])
#         if norm >1:
#             cdivide(B[i],norm)
#             B[i] /=norm
    return error,c_next_random
def save_to_text(matrix,output):
    shape = matrix.shape
    with open(output,'wb') as out:
        out.write("%d %d\n" % (shape))
        for row in matrix:
            x = " ".join(map(lambda x:"{0:.5}".format(x),row))
            out.write(x+"\n")

cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return next_random[0]
def crand(sed):
    cdef unsigned long long csed = sed
    return random_int32(&csed)
def save2bin(mat,dct,fn):
    n,d  = mat.shape
    with open(fn,'w') as out:
        out.write("%d %d\n" % (n,d))
        for i in range(n):
            text = " ".join(map(str,mat[i]))
            out.write("%s %s\n" %(dct[i],text))
def normalize(mat):
    for v in mat:
        norm = np.linalg.norm(v)
        if norm >=1:
            v /= norm
