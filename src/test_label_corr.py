import random
import numpy as np
import sys
from random import randint
from data_load import *
from cwsabie_inner import *
in_dir= "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN"
a=MentionData('/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/train_x_new.txt',
              "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/train_y.txt",
             in_dir+"/feature.txt",in_dir+"/type.txt")
def getLabelLinks(fn):
    return processLines(fn,lambda x,y:addFeature(x,y.rstrip().split('\t')),list())
def corrReg(B,tup,weight,size):
    diff = B[tup[0]] - B[tup[1]]
    B[tup[0]] -= weight*tup[2]*diff
    B[tup[1]] += weight*tup[2]*diff
    norm = np.linalg.norm(diff)
    return norm * norm * weight*tup[2]
tuples = getLabelLinks('/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/type_type_kb.txt')
tuples = [ (int(t[0]),int(t[1]),float(t[2]))for t in tuples]
size= 50
np.random.seed(1)

A= np.random.rand(len(a.feature2id),size).astype(np.float32)
B= np.random.rand(len(a.label2id),size).astype(np.float32)
reg_err =0.
weight = float(sys.argv[1])
print "weight",weight
for i in range(10): 
    reg_err =0.
    ctrain(A,B,a.data,50,0.01,warp_gradient,it=i)
    for tup in tuples:
        reg_err += corrReg(B,tup,weight,50)
    print "reg error:",reg_err
save_to_text(A,'/Users/mayk/working/figer/baseline/PLE/Results/warp_py_A.txt')
save_to_text(B,'/Users/mayk/working/figer/baseline/PLE/Results/warp_py_B.txt')
