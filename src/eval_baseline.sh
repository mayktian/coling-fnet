raw_data=../../Data/BBN/
data=../../Intermediate/BBN/
output=../../Results/BBN/

dim=47
lr=0.01
max_iter=2
thread=1
delimter='/'
max_dep=3
threshold=0.5
method=$1
#echo "###########Baseline Performance"
python python_scripts/warp_eval.py $data/mention_type_test.txt $output/prediction_null_null_$method.txt
