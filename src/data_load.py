import numpy as np

def lazy_readlines(fn):
    with open(fn,'r') as f:
        for ln in f:
            yield ln
def processLines(fn,reduce_fn,init):
    return reduce(reduce_fn,lazy_readlines(fn),init)
def count_lines(fn):
    return processLines(fn,lambda x,y:x+1,0)
def addFeature(x,y):
    x.append(y)
    return x
def getAllFeatures(fn):
    return processLines(fn,lambda x,y:addFeature(x,[y.rstrip().split()[0],int(y.rstrip().split()[1])]),list())
def getAllFeaturesRev(fn):
    return processLines(fn,lambda x,y:addFeature(x,[int(y.rstrip().split()[1]),y.rstrip().split()[0]]),list())
def getVectors(fn):
    return processLines(fn,lambda x,y:addFeature(x,y.rstrip().split('\t')[1].split(',')),list())
def getIntVectors(fn):
    return processLines(fn,lambda x,y:addFeature(x,map(int,y.rstrip().split('\t')[1].split(','))),list())

class MentionData:
    def __init__(self,x_file,y_file,feature_dict,label_dict):
        self.feature2id = dict(getAllFeatures(feature_dict))
        self.id2feature = dict(getAllFeaturesRev(feature_dict))
        self.id2label = dict(getAllFeaturesRev(label_dict))
        self.label2id = dict(getAllFeatures(label_dict))        
        self.data = []
        for x,y in self.readData(x_file,y_file):
            labels = np.zeros(len(self.id2label),dtype=float)
            labels[y]=1.
            self.data.append(Instance([f for f in x if f in self.id2feature] ,labels,y))
    def readData(self,x_file,y_file):
        
        assert count_lines(x_file) == count_lines(y_file)
        return zip(getIntVectors(x_file),getIntVectors(y_file))
class Instance:
    def __init__(self,features,labels,sparse_labels):
        self.features = np.asarray(features,dtype=int)
        self.labels = labels
        self.sparse_labels = sparse_labels
        self.negative_labels = self.get_negatives()
	self.old_labels = self.sparse_labels
    def get_negatives(self):
        return [i for i in range(len(self.labels)) if self.labels[i]==0.]
