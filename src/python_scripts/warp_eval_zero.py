from evaluation_zero import *
import sys
from TypeHierarchy import TypeHierarchy

def main():
    gold = load_labels(sys.argv[1])
    pred = load_labels(sys.argv[2])
    tier = TypeHierarchy(sys.argv[3],47)
    depth= int(sys.argv[4])
    acc,macro_p,macro_r,macro_f,micro_p,micro_r,micro_f= evaluate(pred,gold,tier,depth)
#    tier = TypeHierarchy(sys.argv[3],47)
    print "\t".join(map(lambda x:"%.2f" % (x),map(lambda x:x*100,[macro_p,macro_r,macro_f,micro_p,micro_r,micro_f,acc])))
    print "Accuracy:",acc
    print "macro Precision:",macro_p
    print "macro Recall:",macro_r
    print "macro F-score:", macro_f
    print "micro Precision:",micro_p
    print "micro Recall:",micro_r
    print "micro F-score:", micro_f
if __name__ == "__main__":
	    main() 
