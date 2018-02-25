raw_data=../Data/BBN/
data=../Intermediate/BBN/
output=../Results/BBN/
delimter='/'
max_dep=$2
depth=$3
topk=40
word_embed=$raw_data/embd
threshold=$1
num_label=47
echo "######generating label embedding using PMI"
python ./pmi_label.py $data $data $word_embed $topk
python ./label_hier_embd_gen.py $data $num_label
echo "######Running WARP optimization"
python ./cwsabie_label_hier.py $data $output
python python_scripts/warp_pred.py $output/warp_B.bin $output/warp_A.bin $data/test_x.txt $data/mention.txt $output/warp_predictions $max_dep $delimter $threshold $data
echo "###########Warp Performance"
python python_scripts/warp_eval.py $data/mention_type_test.txt $output/warp_predictions
#python python_scripts/prediction_intext.py $raw_data $output/warp_predictions_intext $data $output/warp_predictions
#python python_scripts/test_y2map.py $data $data/test_y.txt $output/gold.txt
#python python_scripts/prediction_intext.py $raw_data $output/gold_intext $data $output/gold.txt
#diff $output/warp_predictions_intext $output/gold_intext > $output/warp.comp
