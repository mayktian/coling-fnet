
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
embed = sys.argv[3]
def total_count(d):
    return reduce(lambda x,y:x+y,d.values(),0)
def lambda_p(l,e,labels,entities,label_entity,t_l):
    if "%s_%s" % (l,e) not in label_entity: return -1.
    c = log(t_l)
    return (log(labels[l]) + log(entities[e]) - 2*c)/(log(label_entity["%s_%s" % (l,e)])-c) -1
def proto(l,es,labels,entities,label_entity,t_l,topn=10):
    scores = [lambda_p(l,e,labels,entities,label_entity,t_l) for e in es]
    return [es[i] for i in argsort(scores,topn=topn,reverse=True)]
stop = set([ ln.rstrip() for ln in open('stop.list')])



filename = in_dir+'/train_new.json'
relation_id = dict()
label_id = dict()
words = set()
labels = Counter()
entities = Counter()
label_entity = Counter()
with open(filename,'r') as f:
        for line in f:
            sent = json.loads(line.strip('\r\n'))
            mentions = [Mention(int(m['start']), int(m['end']), m['labels']," ".join(sent['tokens'][m['start']:m['end']])) for m in sent['mentions']]
            for m in mentions:
                for l in m.labels:
                   # if m.name.lower() in stop:continue
                    for e in [m.name.lower()]:
                        if e not in stop:
                            labels[l]+=1

                            entities[e]+=1
                            label_entity["%s_%s" % (l,e)]+=1



model = Word2Vec.load_word2vec_format(embed,binary=False)
t_l = total_count(labels)
es = [e for e in entities]
topk= int(sys.argv[4])
with open(out_dir+'/labels.bin','w') as lout:
    lout.write("%d %d\n" % (len(labels),300))
    for l in labels:
        protoes = proto(l,es,labels,entities,label_entity,t_l,topn=topk)
        avg = np.zeros(300,dtype=np.float32)
        hit = 0
        for w in protoes:
            if w in model.vocab:
                hit+=1
                avg += model[w]
        avg /= hit
        text =  " ".join(map(str,avg))

        lout.write("%s %s\n" % (l,text))




