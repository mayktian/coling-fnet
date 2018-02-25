from data_load import *
from TypeHierarchy import *
import numpy as np
from cwsabie_label import save2bin
import sys
def readTypeFunc(x,ln): 
    splited=ln.rstrip().split()
    x[splited[0]]=int(splited[1])
    return x
def readTypeRevFunc(x,ln): 
    splited=ln.rstrip().split()
    x[int(splited[1])]=splited[0]
    return x
def makeHierVec(tier,vocab):
    vecs = np.zeros([len(vocab),len(vocab)],dtype=np.float32)
    for k,v in vocab.iteritems():
        path = tier.get_type_path(v)
        vecs[v][path] =1.0
    return vecs
readType=lambda fn:processLines(fn,readTypeFunc,dict())
readTypeID=lambda fn:processLines(fn,readTypeRevFunc,dict())

indir = sys.argv[1]  # "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/"
num_labels = int(sys.argv[2]) #47
tier = TypeHierarchy(indir+"supertype.txt",num_labels)
label2id =readType(indir+'/type.txt')
id2label =readTypeID(indir+'/type.txt')
embedding = makeHierVec(tier,label2id)
save2bin(embedding,id2label,indir+'labels_hier.bin')
